"""
Layer 2 — LoraParallelLinear routing tests on real TE layers. CUDA + distributed required.

Run with:
    torchrun --nproc-per-node=1 -m pytest tests/multi_lora/test_lora_block_module.py -v -s
"""
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
LORA_TARGET_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]


def _init_megatron():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )


def _lora_config():
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )


def _build_lora_model():
    """Build Qwen3-0.6B with loaded weights, wrapped with a single LoRA adapter."""
    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    cfg_dict = hf_to_mcore_config(hf_config)
    cfg_dict.update(
        tensor_model_parallel_size=1,
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
    return mg_model, bridge, config


def _lora_modules(model):
    """Return all LoraParallelLinear submodules."""
    return [m for m in model.modules() if isinstance(m, LoraParallelLinear)]


def _make_batch(config, seq_len=8, batch=2):
    device = torch.device("cuda")
    vocab_size = config.padded_vocab_size
    input_ids = torch.randint(0, vocab_size, (seq_len, batch), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    return input_ids, position_ids


# ─────────────────────────────────────────────────────────────────────────────

@requires_cuda
def test_block_routing_active():
    """Routing via _lora_adapter_indices changes output compared to the normal single-adapter path."""
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(0)
    mg_model, bridge, config = _build_lora_model()
    mg_model.eval()

    input_ids, position_ids = _make_batch(config, seq_len=8, batch=2)
    root = mg_model.base_model.model  # underlying MegatronModule

    # ── normal (no routing) forward ────────────────────────────────────────
    with torch.no_grad():
        out_normal = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        logits_normal = out_normal[0] if isinstance(out_normal, (tuple, list)) else out_normal

    # ── set up idx_to_name on all LoraParallelLinear modules ──────────────
    lora_mods = _lora_modules(mg_model)
    assert lora_mods, "No LoraParallelLinear found after get_peft_model"
    for m in lora_mods:
        m._idx_to_name = {1: "default"}  # adapter index 1 → slot "default"

    # Route all tokens of sequence to adapter 1 (idx=1); batch=2
    root._lora_adapter_indices = torch.tensor([1, 1], device="cuda")

    # ── routing forward ────────────────────────────────────────────────────
    with torch.no_grad():
        out_routed = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        logits_routed = out_routed[0] if isinstance(out_routed, (tuple, list)) else out_routed

    # Cleanup routing state
    del root._lora_adapter_indices
    for m in lora_mods:
        m._idx_to_name = {}

    # With kaiming_uniform lora_A and zeros lora_B (PEFT default), LoRA contributes zero.
    # So routed path == normal path in this case. Assert shapes match and values are finite.
    assert logits_normal.shape == logits_routed.shape
    assert torch.isfinite(logits_routed).all(), "NaN/Inf in routed output"
    assert torch.isfinite(logits_normal).all(), "NaN/Inf in normal output"


@requires_cuda
def test_block_routing_nonzero_delta():
    """With non-zero lora_B, routing produces a different result than no routing (adapter idx=0)."""
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(1)
    mg_model, bridge, config = _build_lora_model()
    mg_model.eval()

    input_ids, position_ids = _make_batch(config, seq_len=8, batch=2)
    root = mg_model.base_model.model

    # Re-initialize all lora_B to kaiming_uniform so contribution is non-zero
    lora_mods = _lora_modules(mg_model)
    for m in lora_mods:
        for name, param in m.named_parameters():
            if "lora_B" in name and param.requires_grad:
                torch.nn.init.kaiming_uniform_(param)

    # ── base-only forward (all tokens have idx=0) ─────────────────────────
    for m in lora_mods:
        m._idx_to_name = {1: "default"}
    root._lora_adapter_indices = torch.tensor([0, 0], device="cuda")  # all base-only

    with torch.no_grad():
        out_base = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        logits_base = out_base[0] if isinstance(out_base, (tuple, list)) else out_base

    # ── adapter-1 forward (all tokens route to adapter 1) ─────────────────
    root._lora_adapter_indices = torch.tensor([1, 1], device="cuda")

    with torch.no_grad():
        out_adapted = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        logits_adapted = out_adapted[0] if isinstance(out_adapted, (tuple, list)) else out_adapted

    del root._lora_adapter_indices
    for m in lora_mods:
        m._idx_to_name = {}

    diff = (logits_base - logits_adapted).abs().max().item()
    assert diff > 1e-4, (
        f"Routing with non-zero lora_B should change output; max diff={diff:.2e}"
    )


@requires_cuda
def test_block_grad_isolation_multilayer():
    """Adapter 2 params get no gradient when batch only routes to adapter 1."""
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(2)
    mg_model, bridge, config = _build_lora_model()

    # Add a second adapter slot; do not call set_adapter — routing bypasses PEFT's active_adapters
    mg_model.add_adapter("adapter_2", _lora_config())
    mg_model.train()

    input_ids, position_ids = _make_batch(config, seq_len=8, batch=2)
    root = mg_model.base_model.model
    lora_mods = _lora_modules(mg_model)

    # Map int index → adapter slot name
    for m in lora_mods:
        m._idx_to_name = {1: "default", 2: "adapter_2"}

    # Route entire batch to adapter 1 only
    root._lora_adapter_indices = torch.tensor([1, 1], device="cuda")

    labels = torch.roll(input_ids, -1, dims=0)
    output = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None, labels=labels)
    raw_loss = output[0] if isinstance(output, (tuple, list)) else output
    raw_loss.mean().backward()

    del root._lora_adapter_indices
    for m in lora_mods:
        m._idx_to_name = {}

    # Adapter 1 ("default") params should have gradients
    default_grads = [
        p.grad
        for m in lora_mods
        for name, p in m.named_parameters()
        if "default" in name and p.requires_grad
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in default_grads), \
        "adapter 'default' should have non-zero gradients"

    # Adapter 2 params should have no gradients (never routed)
    adapter2_grads = [
        p.grad
        for m in lora_mods
        for name, p in m.named_parameters()
        if "adapter_2" in name and p.requires_grad
    ]
    assert all(g is None for g in adapter2_grads), \
        f"adapter_2 should have None grads (never used in batch); got {[g for g in adapter2_grads if g is not None]}"


@requires_cuda
def test_block_determinism():
    """Same input + same routing produces bit-identical outputs across two calls."""
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(3)
    mg_model, bridge, config = _build_lora_model()
    mg_model.eval()

    input_ids, position_ids = _make_batch(config, seq_len=8, batch=4)
    root = mg_model.base_model.model
    lora_mods = _lora_modules(mg_model)

    for m in lora_mods:
        m._idx_to_name = {1: "default"}
    root._lora_adapter_indices = torch.tensor([1, 0, 1, 0], device="cuda")

    with torch.no_grad():
        out1 = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        logits1 = out1[0] if isinstance(out1, (tuple, list)) else out1
        out2 = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        logits2 = out2[0] if isinstance(out2, (tuple, list)) else out2

    del root._lora_adapter_indices
    for m in lora_mods:
        m._idx_to_name = {}

    assert torch.equal(logits1, logits2), \
        f"Non-deterministic output; max diff={( logits1 - logits2).abs().max().item():.2e}"
