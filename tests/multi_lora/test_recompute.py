"""
Activation recompute compatibility tests for multi-LoRA per-sequence routing.

Two recompute modes are tested:
  - selective:  only attention core is recomputed (recompute_granularity='selective')
  - full/block: entire transformer layers recomputed (recompute_granularity='full',
                recompute_method='block', recompute_num_layers=N)

CRITICAL INVARIANT — backward must be inside set_routing context
-----------------------------------------------------------------
With activation recompute enabled, PyTorch re-runs the forward pass during
backward to reconstruct activations. If loss.backward() is called AFTER the
set_routing context exits, _lora_adapter_indices has already been cleared and
the recomputed forward silently runs with base weights only — producing wrong
gradients without raising any error.

Correct pattern:
    with bridge.set_routing(mg_models, adapter_names):
        out  = model(...)
        loss = compute_loss(out)
        loss.backward()   # ← inside context; recompute sees routing

Wrong pattern (silent bug with recompute):
    with bridge.set_routing(mg_models, adapter_names):
        out  = model(...)
        loss = compute_loss(out)
    loss.backward()        # ← outside context; recompute skips LoRA routing

test_recompute_backward_outside_context_is_wrong explicitly documents and
verifies this footgun so callers know what NOT to do.

Run with:
    torchrun --nproc-per-node=1 -m pytest tests/multi_lora/test_recompute.py -v -s
"""
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig

from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed

import mcore_bridge.patcher as _patcher
from mcore_bridge.config import ModelConfig
from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.register import get_mcore_model
from mcore_bridge.tuners.lora import LoraParallelLinear
from tests.multi_lora.conftest import requires_cuda

MODEL_ID = "/tmp/qwen3-0.6b"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
NUM_LAYERS = 28   # Qwen3-0.6B layer count


def _init_megatron():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )


def _reset_rng(seed: int = 42):
    """Re-initialize Megatron's tensor-parallel CUDA RNG tracker.

    Full activation recompute saves and restores the 'model-parallel-rng' state
    during backward. After backward completes, the tracker is in a different
    position than fresh initialization expects, causing the next model build
    to fail with 'cuda rng state model-parallel-rng is not added'. Resetting
    and re-seeding before each model build avoids this.
    """
    get_cuda_rng_tracker().reset()
    model_parallel_cuda_manual_seed(seed)


def _lora_config():
    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET,
                      lora_dropout=0.0, bias="none")


def _build_model(recompute_granularity=None, recompute_method=None, recompute_num_layers=None):
    _reset_rng()
    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    cfg_dict = hf_to_mcore_config(hf_config)
    cfg_dict.update(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    if recompute_granularity is not None:
        cfg_dict["recompute_granularity"] = recompute_granularity
    if recompute_method is not None:
        cfg_dict["recompute_method"] = recompute_method
    if recompute_num_layers is not None:
        cfg_dict["recompute_num_layers"] = recompute_num_layers

    config = ModelConfig(**cfg_dict)
    mg_models = get_mcore_model(config)
    bridge = config.bridge
    mg_model = mg_models[0].cuda()
    bridge.load_weights(mg_models, MODEL_ID)
    mg_model = get_peft_model(mg_model, _lora_config())
    mg_models[0] = mg_model
    return mg_models, bridge, config


def _make_batch(config, seq_len=16, batch=4):
    vocab_size = config.padded_vocab_size
    input_ids = torch.randint(0, vocab_size, (seq_len, batch), device="cuda")
    labels = torch.roll(input_ids, -1, dims=0)
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1)
    return input_ids, labels, position_ids


def _collect_lora_grads(mg_model):
    """Return {param_name: grad_clone} for all trainable LoRA params."""
    grads = {}
    for name, p in mg_model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[name] = p.grad.clone()
    return grads


# ─────────────────────────────────────────────────────────────────────────────

@requires_cuda
def test_selective_recompute_forward_backward():
    """selective recompute: forward + backward complete without error."""
    _init_megatron()
    _patcher.apply_patch()

    mg_models, bridge, config = _build_model(recompute_granularity="selective")
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "a", slot_index=0)
    bridge.register_adapter(mg_models, "b", slot_index=1)
    mg_models[0].train()

    input_ids, labels, position_ids = _make_batch(config)
    adapter_names = ["a", "a", "b", "b"]

    with bridge.set_routing(mg_models, adapter_names):
        out = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                           attention_mask=None, labels=labels)
        loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
        loss.backward()

    assert loss.item() > 0 and torch.isfinite(torch.tensor(loss.item())), \
        f"Loss invalid: {loss.item()}"


@requires_cuda
def test_full_recompute_forward_backward():
    """full/block recompute: forward + backward complete without error."""
    _init_megatron()
    _patcher.apply_patch()

    mg_models, bridge, config = _build_model(
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=NUM_LAYERS,
    )
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "a", slot_index=0)
    bridge.register_adapter(mg_models, "b", slot_index=1)
    mg_models[0].train()

    input_ids, labels, position_ids = _make_batch(config)
    adapter_names = ["a", "a", "b", "b"]

    with bridge.set_routing(mg_models, adapter_names):
        out = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                           attention_mask=None, labels=labels)
        loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
        loss.backward()

    assert loss.item() > 0 and torch.isfinite(torch.tensor(loss.item())), \
        f"Loss invalid: {loss.item()}"


@requires_cuda
def test_recompute_lora_grads_match_no_recompute():
    """Gradients with full recompute match gradients without recompute (backward inside context).

    Tolerance is 1e-2 in bfloat16 — recomputation in bfloat16 can accumulate
    small floating-point differences versus the non-recomputed path.
    """
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(42)
    input_ids, labels, position_ids = None, None, None

    # ── no recompute (reference) ──────────────────────────────────────────
    mg_models_ref, bridge_ref, config_ref = _build_model()
    bridge_ref.preallocate_adapters(mg_models_ref, num_slots=1, lora_config=_lora_config())
    bridge_ref.register_adapter(mg_models_ref, "task", slot_index=0)
    mg_models_ref[0].train()

    torch.manual_seed(0)
    input_ids, labels, position_ids = _make_batch(config_ref)
    adapter_names = ["task"] * 4

    with bridge_ref.set_routing(mg_models_ref, adapter_names):
        out = mg_models_ref[0](input_ids=input_ids, position_ids=position_ids,
                               attention_mask=None, labels=labels)
        loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
        loss.backward()

    ref_grads = _collect_lora_grads(mg_models_ref[0])
    assert ref_grads, "No LoRA gradients collected (reference run)"

    # ── full recompute ────────────────────────────────────────────────────
    torch.manual_seed(42)
    mg_models_rc, bridge_rc, config_rc = _build_model(
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=NUM_LAYERS,
    )
    bridge_rc.preallocate_adapters(mg_models_rc, num_slots=1, lora_config=_lora_config())
    bridge_rc.register_adapter(mg_models_rc, "task", slot_index=0)
    mg_models_rc[0].train()

    # Same input, same routing
    torch.manual_seed(0)
    input_ids, labels, position_ids = _make_batch(config_rc)

    with bridge_rc.set_routing(mg_models_rc, adapter_names):
        out = mg_models_rc[0](input_ids=input_ids, position_ids=position_ids,
                              attention_mask=None, labels=labels)
        loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
        loss.backward()

    rc_grads = _collect_lora_grads(mg_models_rc[0])
    assert rc_grads, "No LoRA gradients collected (recompute run)"

    # Compare matching param names
    common = set(ref_grads) & set(rc_grads)
    assert common, "No common gradient parameter names between runs"

    max_diff = max(
        (ref_grads[n] - rc_grads[n]).abs().max().item()
        for n in common
    )
    print(f"\nMax grad diff (no-recompute vs full-recompute): {max_diff:.2e}")
    assert max_diff < 1e-2, \
        f"Gradient mismatch between recompute and no-recompute: max_diff={max_diff:.2e}"


@requires_cuda
def test_recompute_lora_grads_nonzero():
    """With full recompute, LoRA adapter params receive non-zero gradients (routing active during recompute)."""
    _init_megatron()
    _patcher.apply_patch()

    mg_models, bridge, config = _build_model(
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=NUM_LAYERS,
    )
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "a", slot_index=0)
    bridge.register_adapter(mg_models, "b", slot_index=1)
    mg_models[0].train()

    input_ids, labels, position_ids = _make_batch(config)
    # Route half to "a", half to "b"
    adapter_names = ["a", "a", "b", "b"]

    with bridge.set_routing(mg_models, adapter_names):
        out = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                           attention_mask=None, labels=labels)
        loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
        loss.backward()

    lora_mods = [m for m in mg_models[0].modules() if isinstance(m, LoraParallelLinear)]

    slot0_grads = [
        p.grad for m in lora_mods
        for name, p in m.named_parameters()
        if "__slot_0__" in name and p.requires_grad
    ]
    slot1_grads = [
        p.grad for m in lora_mods
        for name, p in m.named_parameters()
        if "__slot_1__" in name and p.requires_grad
    ]

    assert any(g is not None and g.abs().sum() > 0 for g in slot0_grads), \
        "slot 0 (adapter 'a') should have non-zero gradients — routing may have been skipped during recompute"
    assert any(g is not None and g.abs().sum() > 0 for g in slot1_grads), \
        "slot 1 (adapter 'b') should have non-zero gradients — routing may have been skipped during recompute"


@requires_cuda
def test_recompute_backward_outside_context_is_wrong():
    """FOOTGUN: backward OUTSIDE set_routing context gives wrong gradients with full recompute.

    When recompute is enabled, loss.backward() re-runs the forward pass. If the
    set_routing context has already exited, _lora_adapter_indices is gone and the
    recomputed forward runs with base weights only. The LoRA adapter parameters
    then receive incorrect (usually smaller) gradients.

    This test documents the bug by verifying that gradients from the wrong pattern
    (backward outside context) differ from the correct pattern (backward inside context).
    The difference is only visible with activation recompute — without recompute
    both patterns produce the same result because no re-run occurs.
    """
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(42)
    adapter_names = ["a", "a", "b", "b"]

    def _run(backward_inside: bool):
        mg_models, bridge, config = _build_model(
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=NUM_LAYERS,
        )
        bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
        bridge.register_adapter(mg_models, "a", slot_index=0)
        bridge.register_adapter(mg_models, "b", slot_index=1)
        mg_models[0].train()

        torch.manual_seed(0)
        input_ids, labels, position_ids = _make_batch(config)

        if backward_inside:
            with bridge.set_routing(mg_models, adapter_names):
                out = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                                   attention_mask=None, labels=labels)
                loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
                loss.backward()   # correct: recomputed forward sees routing
        else:
            with bridge.set_routing(mg_models, adapter_names):
                out = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                                   attention_mask=None, labels=labels)
                loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
            loss.backward()       # wrong: recomputed forward skips routing

        return _collect_lora_grads(mg_models[0])

    grads_correct = _run(backward_inside=True)
    grads_wrong = _run(backward_inside=False)

    lora_mods_probe, bridge_probe, config_probe = _build_model(
        recompute_granularity="full", recompute_method="block", recompute_num_layers=NUM_LAYERS
    )
    bridge_probe.preallocate_adapters(lora_mods_probe, num_slots=2, lora_config=_lora_config())
    bridge_probe.register_adapter(lora_mods_probe, "a", slot_index=0)
    bridge_probe.register_adapter(lora_mods_probe, "b", slot_index=1)
    probe_mods = [m for m in lora_mods_probe[0].modules() if isinstance(m, LoraParallelLinear)]

    def _slot_grad_sum(grads, slot_key):
        return sum(
            g.abs().sum().item()
            for name, g in grads.items()
            if slot_key in name
        )

    # In the correct run, both adapters should receive gradients because each is
    # routed to exactly half the batch.
    slot0_correct = _slot_grad_sum(grads_correct, "__slot_0__")
    slot1_correct = _slot_grad_sum(grads_correct, "__slot_1__")
    print(f"\nCorrect (backward inside context):")
    print(f"  slot_0 grad sum: {slot0_correct:.4f}")
    print(f"  slot_1 grad sum: {slot1_correct:.4f}")

    assert slot0_correct > 0, "slot_0 ('a') should have gradients in correct run"
    assert slot1_correct > 0, "slot_1 ('b') should have gradients in correct run"

    # In the wrong run, PEFT's default active adapter fires uniformly during recompute,
    # causing incorrect gradient distribution across slots.
    slot0_wrong = _slot_grad_sum(grads_wrong, "__slot_0__")
    slot1_wrong = _slot_grad_sum(grads_wrong, "__slot_1__")
    print(f"Wrong (backward outside context — routing replaced by PEFT default during recompute):")
    print(f"  slot_0 grad sum: {slot0_wrong:.4f}")
    print(f"  slot_1 grad sum: {slot1_wrong:.4f}")

    # The per-slot distribution must differ between correct and wrong runs.
    # (Correct: both slots get equal share. Wrong: PEFT default adapter takes all.)
    slot0_diff = abs(slot0_correct - slot0_wrong)
    slot1_diff = abs(slot1_correct - slot1_wrong)
    assert slot0_diff > 0 or slot1_diff > 0, (
        "Gradient distribution is identical between correct and wrong backward placement. "
        "Either recompute is not active or the routing context is being preserved across "
        "backward — if the latter, this footgun no longer exists and this test can be removed."
    )
    print("Confirmed: backward outside context corrupts per-adapter gradient distribution.")
    print("Always place loss.backward() INSIDE the set_routing context when using activation recompute.")
