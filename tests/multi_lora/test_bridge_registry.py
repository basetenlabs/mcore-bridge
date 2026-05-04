"""
Layer 3 — GPTBridge multi-LoRA registry tests. CUDA + distributed required.

Run with:
    torchrun --nproc-per-node=1 -m pytest tests/multi_lora/test_bridge_registry.py -v -s
"""
import tempfile

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


def _lora_config():
    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET,
                      lora_dropout=0.0, bias="none")


def _init_megatron():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )


def _build_model():
    """Build Qwen3-0.6B with loaded weights, wrapped with LoRA."""
    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    cfg_dict = hf_to_mcore_config(hf_config)
    cfg_dict.update(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                    sequence_parallel=False, params_dtype=torch.bfloat16, bf16=True)
    config = ModelConfig(**cfg_dict)
    mg_models = get_mcore_model(config)
    bridge = config.bridge
    mg_model = mg_models[0].cuda()
    bridge.load_weights(mg_models, MODEL_ID)
    mg_model = get_peft_model(mg_model, _lora_config())
    mg_models[0] = mg_model
    return mg_models, bridge, config


# ─────────────────────────────────────────────────────────────────────────────

@requires_cuda
def test_preallocate_sets_idx_to_name():
    """preallocate_adapters creates slots and populates _idx_to_name on all modules."""
    _init_megatron()
    _patcher.apply_patch()

    mg_models, bridge, config = _build_model()
    bridge.preallocate_adapters(mg_models, num_slots=3, lora_config=_lora_config())

    lora_mods = [m for m in mg_models[0].modules() if isinstance(m, LoraParallelLinear)]
    assert lora_mods, "No LoraParallelLinear found"

    for m in lora_mods:
        assert m._idx_to_name == {1: '__slot_0__', 2: '__slot_1__', 3: '__slot_2__'}, \
            f"Wrong _idx_to_name: {m._idx_to_name}"

    # Verify all slot adapters were added to the PEFT model
    adapter_keys = set(lora_mods[0].lora_A.keys())
    assert '__slot_0__' in adapter_keys
    assert '__slot_1__' in adapter_keys
    assert '__slot_2__' in adapter_keys


@requires_cuda
def test_register_and_list_adapters():
    """register_adapter and list_adapters maintain the name→slot mapping."""
    _init_megatron()
    _patcher.apply_patch()

    mg_models, bridge, config = _build_model()
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())

    assert bridge.list_adapters() == {}

    bridge.register_adapter(mg_models, "user_a", slot_index=0)
    bridge.register_adapter(mg_models, "user_b", slot_index=1)

    listing = bridge.list_adapters()
    assert listing == {"user_a": "__slot_0__", "user_b": "__slot_1__"}, \
        f"Unexpected listing: {listing}"

    bridge.unload_adapter("user_a")
    assert "user_a" not in bridge.list_adapters()
    assert "user_b" in bridge.list_adapters()


@requires_cuda
def test_resolve_routing():
    """resolve_routing maps names to correct integer indices; None → 0."""
    _init_megatron()
    _patcher.apply_patch()

    mg_models, bridge, config = _build_model()
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "user_a", slot_index=0)
    bridge.register_adapter(mg_models, "user_b", slot_index=1)

    indices = bridge.resolve_routing(["user_a", None, "user_b", "unknown", "user_a"])
    expected = torch.tensor([1, 0, 2, 0, 1], dtype=torch.long)
    assert torch.equal(indices, expected), f"Got {indices}, expected {expected}"


@requires_cuda
def test_set_routing_stamps_and_clears():
    """set_routing stamps _lora_adapter_indices inside the block and removes it after."""
    _init_megatron()
    _patcher.apply_patch()

    mg_models, bridge, config = _build_model()
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "user_a", slot_index=0)
    bridge.register_adapter(mg_models, "user_b", slot_index=1)

    # Find the root Megatron model (same path as bridge.set_routing)
    mg_model = mg_models[0]
    root = mg_model.base_model.model

    assert not hasattr(root, '_lora_adapter_indices'), "Should not exist before context"

    with bridge.set_routing(mg_models, ["user_a", "user_b"]):
        assert hasattr(root, '_lora_adapter_indices'), "Should exist inside context"
        expected = torch.tensor([1, 2], dtype=torch.long)
        assert torch.equal(root._lora_adapter_indices, expected), \
            f"Got {root._lora_adapter_indices}"

    assert not hasattr(root, '_lora_adapter_indices'), "Should be cleaned up after context"


@requires_cuda
def test_set_routing_forward_changes_output():
    """With non-zero LoRA weights, set_routing changes the model output vs base-only."""
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(42)
    mg_models, bridge, config = _build_model()
    bridge.preallocate_adapters(mg_models, num_slots=1, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "user_a", slot_index=0)
    mg_models[0].eval()

    # Non-zero lora_B
    for m in mg_models[0].modules():
        if isinstance(m, LoraParallelLinear):
            for name, param in m.named_parameters():
                if "lora_B" in name and param.requires_grad:
                    torch.nn.init.kaiming_uniform_(param)

    vocab_size = config.padded_vocab_size
    input_ids = torch.randint(0, vocab_size, (8, 2), device="cuda")
    position_ids = torch.arange(8, device="cuda").unsqueeze(0).expand(2, -1)

    # Base-only (all idx=0)
    with bridge.set_routing(mg_models, [None, None]):
        with torch.no_grad():
            out0 = mg_models[0](input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            logits0 = out0[0] if isinstance(out0, (tuple, list)) else out0

    # Adapter routing (all idx=1)
    with bridge.set_routing(mg_models, ["user_a", "user_a"]):
        with torch.no_grad():
            out1 = mg_models[0](input_ids=input_ids, position_ids=position_ids, attention_mask=None)
            logits1 = out1[0] if isinstance(out1, (tuple, list)) else out1

    diff = (logits0 - logits1).abs().max().item()
    assert diff > 1e-4, \
        f"Non-zero lora_B should change output; max diff={diff:.2e}"


@requires_cuda
def test_register_adapter_after_training():
    """register_adapter into a pre-allocated slot works correctly after fwd/bwd/optim steps.

    Production pattern: a slot is reserved at startup, training runs for several steps
    on already-registered adapters, then a new adapter is hot-loaded into the spare slot
    without restarting the process.

    Verifies:
    - The newly registered adapter can be routed to and changes output vs base-only.
    - The previously trained adapter's weights are intact (its loss is still lower
      than it was before training, meaning the optimizer state was not corrupted).
    - Both adapters can be routed simultaneously in the same batch.
    """
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(7)
    mg_models, bridge, config = _build_model()

    # Pre-allocate 2 slots but only register one at startup
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "trained", slot_index=0)
    # slot 1 is reserved but unregistered — simulates capacity held for a future adapter

    mg_models[0].train()
    lora_params = [p for p in mg_models[0].parameters() if p.requires_grad]
    assert lora_params, "No trainable LoRA parameters"
    optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

    vocab_size = config.padded_vocab_size
    seq_len, batch = 8, 4
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1)

    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (seq_len, batch), device="cuda")
    labels = torch.roll(input_ids, -1, dims=0)

    # ── train "trained" adapter for several steps ────────────────────────
    N_TRAIN = 20
    for _ in range(N_TRAIN):
        optimizer.zero_grad()
        with bridge.set_routing(mg_models, ["trained"] * batch):
            out = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                               attention_mask=None, labels=labels)
            loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
            loss.backward()
        optimizer.step()

    loss_after_train = loss.item()

    # ── hot-register a new adapter into slot 1 (simulates loading from checkpoint) ──
    bridge.register_adapter(mg_models, "new", slot_index=1)

    listing = bridge.list_adapters()
    assert "trained" in listing and "new" in listing, \
        f"Both adapters should be listed; got {listing}"

    # "new" adapter has zero lora_B (PEFT default init) — force non-zero so it's detectable
    for m in mg_models[0].modules():
        if isinstance(m, LoraParallelLinear):
            for name, param in m.named_parameters():
                if "__slot_1__" in name and "lora_B" in name and param.requires_grad:
                    torch.nn.init.kaiming_uniform_(param)

    # ── new adapter changes output vs base-only ──────────────────────────
    eval_ids = torch.randint(0, vocab_size, (seq_len, 2), device="cuda")
    eval_pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(2, -1)

    with bridge.set_routing(mg_models, [None, None]):
        with torch.no_grad():
            out_base = mg_models[0](input_ids=eval_ids, position_ids=eval_pos, attention_mask=None)
            logits_base = out_base[0] if isinstance(out_base, (tuple, list)) else out_base

    with bridge.set_routing(mg_models, ["new", "new"]):
        with torch.no_grad():
            out_new = mg_models[0](input_ids=eval_ids, position_ids=eval_pos, attention_mask=None)
            logits_new = out_new[0] if isinstance(out_new, (tuple, list)) else out_new

    diff = (logits_base - logits_new).abs().max().item()
    assert diff > 1e-4, \
        f"'new' adapter (non-zero lora_B) should change output; max diff={diff:.2e}"

    # ── trained adapter's loss is still lower than initial (weights intact) ──
    with bridge.set_routing(mg_models, ["trained"] * batch):
        with torch.no_grad():
            out_trained = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                                       attention_mask=None, labels=labels)
            loss_trained_eval = (out_trained[0] if isinstance(out_trained, (tuple, list))
                                 else out_trained).mean().item()

    # Run with a fresh (untrained) model to get a baseline loss for comparison
    with bridge.set_routing(mg_models, [None] * batch):
        with torch.no_grad():
            out_base_eval = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                                         attention_mask=None, labels=labels)
            loss_base_eval = (out_base_eval[0] if isinstance(out_base_eval, (tuple, list))
                              else out_base_eval).mean().item()

    print(f"\nLoss after {N_TRAIN} steps: {loss_after_train:.4f}")
    print(f"Eval loss — trained adapter: {loss_trained_eval:.4f}  base-only: {loss_base_eval:.4f}")

    assert loss_trained_eval < loss_base_eval, (
        f"Trained adapter should have lower loss than base-only after {N_TRAIN} steps; "
        f"trained={loss_trained_eval:.4f}, base={loss_base_eval:.4f}. "
        f"Optimizer state may have been corrupted by hot-registration."
    )

    # ── both adapters work correctly in the same mixed batch ────────────
    mixed_ids = torch.randint(0, vocab_size, (seq_len, 2), device="cuda")
    mixed_pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(2, -1)

    with bridge.set_routing(mg_models, ["trained", "new"]):
        with torch.no_grad():
            out_mixed = mg_models[0](input_ids=mixed_ids, position_ids=mixed_pos, attention_mask=None)
            logits_mixed = out_mixed[0] if isinstance(out_mixed, (tuple, list)) else out_mixed

    assert torch.isfinite(logits_mixed).all(), \
        "Mixed-adapter forward produced NaN/Inf"
