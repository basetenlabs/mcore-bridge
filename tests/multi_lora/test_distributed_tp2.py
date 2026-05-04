"""
Gate 2 — Multi-LoRA routing works correctly under tensor-parallelism (TP=2).

Verifies that per-sequence LoRA routing and per-adapter gradient isolation
hold when the model is sharded across 2 GPUs via tensor parallelism.

Run with:
    torchrun --nproc-per-node=2 -m pytest tests/multi_lora/test_distributed_tp2.py -v -s
"""
import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig

import mcore_bridge.patcher as _patcher
from mcore_bridge.config import ModelConfig
from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.register import get_mcore_model
from mcore_bridge.tuners.lora import LoraParallelLinear
from tests.multi_lora.conftest import requires_cuda

MODEL_ID = "/tmp/qwen3-0.6b"
LORA_RANK = 4
LORA_ALPHA = 8
LORA_TARGET = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
TP_SIZE = 2


def _init_megatron():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=1,
        )


def _lora_config():
    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET,
                      lora_dropout=0.0, bias="none")


def _build_model():
    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    cfg_dict = hf_to_mcore_config(hf_config)
    cfg_dict.update(
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    config = ModelConfig(**cfg_dict)
    mg_models = get_mcore_model(config)
    bridge = config.bridge
    mg_model = mg_models[0].cuda()
    bridge.load_weights(mg_models, MODEL_ID)
    mg_model = get_peft_model(mg_model, _lora_config())
    mg_models[0] = mg_model
    return mg_models, bridge, config


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


# ─────────────────────────────────────────────────────────────────────────────

@requires_cuda
def test_tp2_preallocate_and_registry():
    """preallocate_adapters and registry methods work correctly under TP=2."""
    _init_megatron()
    _patcher.apply_patch()

    mg_models, bridge, config = _build_model()
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "slot_a", slot_index=0)
    bridge.register_adapter(mg_models, "slot_b", slot_index=1)

    listing = bridge.list_adapters()
    assert listing == {"slot_a": "__slot_0__", "slot_b": "__slot_1__"}, \
        f"[rank {_rank()}] Wrong listing: {listing}"

    indices = bridge.resolve_routing(["slot_a", None, "slot_b", "slot_a"])
    expected = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    assert torch.equal(indices, expected), \
        f"[rank {_rank()}] Got {indices}, expected {expected}"

    lora_mods = [m for m in mg_models[0].modules() if isinstance(m, LoraParallelLinear)]
    assert lora_mods, f"[rank {_rank()}] No LoraParallelLinear found"
    for m in lora_mods:
        assert m._idx_to_name == {1: '__slot_0__', 2: '__slot_1__'}, \
            f"[rank {_rank()}] Wrong _idx_to_name: {m._idx_to_name}"


@requires_cuda
def test_tp2_routing_changes_output():
    """Non-zero LoRA adapter changes model output compared to base-only, under TP=2."""
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(42 + _rank())
    mg_models, bridge, config = _build_model()
    bridge.preallocate_adapters(mg_models, num_slots=1, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "adapter_a", slot_index=0)
    mg_models[0].train()

    # Force non-zero lora_B on __slot_0__
    for m in mg_models[0].modules():
        if isinstance(m, LoraParallelLinear):
            for name, param in m.named_parameters():
                if "lora_B" in name and "__slot_0__" in name and param.requires_grad:
                    torch.nn.init.kaiming_uniform_(param)

    vocab_size = config.padded_vocab_size
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (8, 2), device="cuda")
    position_ids = torch.arange(8, device="cuda").unsqueeze(0).expand(2, -1)

    with bridge.set_routing(mg_models, [None, None]):
        with torch.no_grad():
            out0 = mg_models[0](input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            logits0 = out0[0] if isinstance(out0, (tuple, list)) else out0

    with bridge.set_routing(mg_models, ["adapter_a", "adapter_a"]):
        with torch.no_grad():
            out1 = mg_models[0](input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            logits1 = out1[0] if isinstance(out1, (tuple, list)) else out1

    diff = (logits0 - logits1).abs().max().item()
    assert diff > 1e-4, \
        f"[rank {_rank()}] Non-zero lora_B should change output; max diff={diff:.2e}"


@requires_cuda
def test_tp2_grad_isolation():
    """Under TP=2, adapter 2 receives no gradient when batch routes only to adapter 1."""
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(2)
    mg_models, bridge, config = _build_model()
    mg_model = mg_models[0]

    mg_model.add_adapter("adapter_2", _lora_config())
    mg_model.train()

    seq_len, batch = 8, 2
    vocab_size = config.padded_vocab_size
    input_ids = torch.randint(0, vocab_size, (seq_len, batch), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1)

    root = mg_model.base_model.model
    lora_mods = [m for m in mg_model.modules() if isinstance(m, LoraParallelLinear)]

    for m in lora_mods:
        m._idx_to_name = {1: "default", 2: "adapter_2"}
    root._lora_adapter_indices = torch.tensor([1, 1], device="cuda")

    labels = torch.roll(input_ids, -1, dims=0)
    out = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None, labels=labels)
    raw_loss = out[0] if isinstance(out, (tuple, list)) else out
    raw_loss.mean().backward()

    del root._lora_adapter_indices
    for m in lora_mods:
        m._idx_to_name = {}

    default_grads = [
        p.grad for m in lora_mods
        for name, p in m.named_parameters()
        if "default" in name and p.requires_grad
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in default_grads), \
        f"[rank {_rank()}] adapter 'default' should have non-zero gradients"

    adapter2_grads = [
        p.grad for m in lora_mods
        for name, p in m.named_parameters()
        if "adapter_2" in name and p.requires_grad
    ]
    assert all(g is None for g in adapter2_grads), \
        f"[rank {_rank()}] adapter_2 should have None grads; got non-None on rank {_rank()}"


@requires_cuda
def test_tp2_determinism():
    """Same input + same routing is bit-identical across two calls, under TP=2."""
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(3)
    mg_models, bridge, config = _build_model()
    bridge.preallocate_adapters(mg_models, num_slots=1, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "alpha", slot_index=0)
    mg_models[0].train()

    vocab_size = config.padded_vocab_size
    input_ids = torch.randint(0, vocab_size, (8, 4), device="cuda")
    position_ids = torch.arange(8, device="cuda").unsqueeze(0).expand(4, -1)

    with bridge.set_routing(mg_models, ["alpha", None, "alpha", None]):
        with torch.no_grad():
            out1 = mg_models[0](input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            logits1 = out1[0] if isinstance(out1, (tuple, list)) else out1
        with torch.no_grad():
            out2 = mg_models[0](input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            logits2 = out2[0] if isinstance(out2, (tuple, list)) else out2

    assert torch.equal(logits1, logits2), \
        f"[rank {_rank()}] Non-deterministic output; max diff={(logits1 - logits2).abs().max():.2e}"


@requires_cuda
def test_tp2_loss_decreases():
    """Under TP=2, training loss decreases after a few optimizer steps with per-task routing."""
    from tests.multi_lora.conftest import synthetic_task

    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(7)
    mg_models, bridge, config = _build_model()
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "incr", slot_index=0)
    bridge.register_adapter(mg_models, "decr", slot_index=1)
    mg_models[0].train()

    lora_params = [p for p in mg_models[0].parameters() if p.requires_grad]
    assert lora_params, f"[rank {_rank()}] No trainable LoRA parameters"
    optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

    seq_len, batch_per_task, vocab_size = 16, 4, 32
    position_ids = (
        torch.arange(seq_len, device="cuda")
        .unsqueeze(0)
        .expand(2 * batch_per_task, -1)
    )

    N_STEPS = 30
    losses = []
    for step in range(N_STEPS):
        optimizer.zero_grad()
        inc_in, inc_lb, dec_in, dec_lb = [], [], [], []
        torch.manual_seed(step)
        for _ in range(batch_per_task):
            t, l = synthetic_task("increment", seq_len=seq_len, vocab_size=vocab_size)
            inc_in.append(t); inc_lb.append(l)
            t, l = synthetic_task("decrement", seq_len=seq_len, vocab_size=vocab_size)
            dec_in.append(t); dec_lb.append(l)

        input_ids = torch.stack(inc_in + dec_in, dim=1).cuda()
        labels = torch.stack(inc_lb + dec_lb, dim=1).cuda()
        adapter_names = ["incr"] * batch_per_task + ["decr"] * batch_per_task

        with bridge.set_routing(mg_models, adapter_names):
            out = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                               attention_mask=None, labels=labels)
            raw_loss = out[0] if isinstance(out, (tuple, list)) else out
            loss = raw_loss.mean()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    early_avg = sum(losses[:5]) / 5
    late_avg = sum(losses[-5:]) / 5
    drop = (early_avg - late_avg) / early_avg
    if _rank() == 0:
        print(f"\n[TP=2] early_avg={early_avg:.4f} late_avg={late_avg:.4f} drop={drop:.1%}")

    assert drop >= 0.10, \
        f"[rank {_rank()}] Training loss did not drop ≥10% over {N_STEPS} steps (drop={drop:.1%})"
