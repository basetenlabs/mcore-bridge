"""
Pipeline Parallelism (PP=2) — multi-LoRA per-sequence routing tests.

Standard PP=2: rank 0 holds pipeline stage 0 (first half of layers), rank 1
holds stage 1 (second half).  The routing fix (set_routing stamps ALL elements
of mg_models, not just index 0) ensures virtual-pipeline parallelism (where one
rank holds multiple interleaved chunks) also gets correctly stamped.

Single-rank stamp test exercises the multi-chunk code path directly without
requiring virtual-PP infrastructure.

Run with:
    torchrun --nproc-per-node=2 -m pytest tests/multi_lora/test_pp2.py -v -s
"""
import torch
import mcore_bridge.patcher as _patcher
from mcore_bridge.tuners.lora import LoraParallelLinear
from tests.multi_lora.conftest import (
    build_model, init_megatron, make_lora_config, rank, requires_cuda,
)

PP_SIZE = 2


# ─────────────────────────────────────────────────────────────────────────────

@requires_cuda
def test_pp2_preallocate_and_registry():
    """preallocate_adapters and registry methods work correctly under PP=2."""
    init_megatron(pp=PP_SIZE)
    _patcher.apply_patch()

    mg_models, bridge, config = build_model(pp=PP_SIZE)
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=make_lora_config())
    bridge.register_adapter(mg_models, "alpha", slot_index=0)
    bridge.register_adapter(mg_models, "beta", slot_index=1)

    listing = bridge.list_adapters()
    assert listing == {"alpha": "__slot_0__", "beta": "__slot_1__"}, \
        f"[rank {rank()}] Wrong listing: {listing}"

    indices = bridge.resolve_routing(["alpha", None, "beta", "alpha"])
    expected = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    assert torch.equal(indices, expected), \
        f"[rank {rank()}] Got {indices}, expected {expected}"

    lora_mods = [m for m in mg_models[0].modules() if isinstance(m, LoraParallelLinear)]
    assert lora_mods, f"[rank {rank()}] No LoraParallelLinear found"
    for m in lora_mods:
        assert m._idx_to_name == {1: '__slot_0__', 2: '__slot_1__'}, \
            f"[rank {rank()}] Wrong _idx_to_name: {m._idx_to_name}"


@requires_cuda
def test_pp2_set_routing_stamps_and_clears():
    """set_routing stamps _lora_adapter_indices on this rank's chunk and clears it after."""
    init_megatron(pp=PP_SIZE)
    _patcher.apply_patch()

    mg_models, bridge, config = build_model(pp=PP_SIZE)
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=make_lora_config())
    bridge.register_adapter(mg_models, "alpha", slot_index=0)
    bridge.register_adapter(mg_models, "beta", slot_index=1)

    root = mg_models[0].base_model.model

    assert not hasattr(root, '_lora_adapter_indices'), \
        f"[rank {rank()}] Should not exist before context"

    with bridge.set_routing(mg_models, ["alpha", "beta"]):
        assert hasattr(root, '_lora_adapter_indices'), \
            f"[rank {rank()}] Should exist inside context"
        expected = torch.tensor([1, 2], dtype=torch.long)
        assert torch.equal(root._lora_adapter_indices, expected), \
            f"[rank {rank()}] Got {root._lora_adapter_indices}"

    assert not hasattr(root, '_lora_adapter_indices'), \
        f"[rank {rank()}] Should be cleaned up after context"


@requires_cuda
def test_pp2_all_chunks_stamped():
    """set_routing stamps ALL model chunks when mg_models contains multiple elements.

    This directly exercises the virtual-pipeline fix: with interleaved PP
    one rank holds N chunks (mg_models has N elements). We simulate this by
    building two independent model instances and passing both to set_routing.
    Each chunk's root must receive _lora_adapter_indices, and both must be
    cleaned up when the context exits.
    """
    init_megatron(pp=PP_SIZE)
    _patcher.apply_patch()

    mg_models_a, bridge, config = build_model(pp=PP_SIZE)
    mg_models_b, _, _ = build_model(pp=PP_SIZE)

    # Simulate a rank that owns two pipeline chunks (virtual PP).
    # Preallocate on the combined list so both chunks get the slot adapters.
    mg_models_both = [mg_models_a[0], mg_models_b[0]]
    bridge.preallocate_adapters(mg_models_both, num_slots=1, lora_config=make_lora_config())
    bridge.register_adapter(mg_models_both, "adapter_a", slot_index=0)

    root_a = mg_models_a[0].base_model.model
    root_b = mg_models_b[0].base_model.model

    with bridge.set_routing(mg_models_both, ["adapter_a", None]):
        assert hasattr(root_a, '_lora_adapter_indices'), \
            f"[rank {rank()}] chunk 0 missing _lora_adapter_indices"
        assert hasattr(root_b, '_lora_adapter_indices'), \
            f"[rank {rank()}] chunk 1 missing _lora_adapter_indices"
        expected = torch.tensor([1, 0], dtype=torch.long)
        assert torch.equal(root_a._lora_adapter_indices, expected), \
            f"[rank {rank()}] chunk 0 wrong indices: {root_a._lora_adapter_indices}"
        assert torch.equal(root_b._lora_adapter_indices, expected), \
            f"[rank {rank()}] chunk 1 wrong indices: {root_b._lora_adapter_indices}"

    assert not hasattr(root_a, '_lora_adapter_indices'), \
        f"[rank {rank()}] chunk 0 not cleaned up"
    assert not hasattr(root_b, '_lora_adapter_indices'), \
        f"[rank {rank()}] chunk 1 not cleaned up"


# Note: grad isolation (adapter 2 gets no grad when routed to adapter 1 only) is
# tested in test_distributed_tp2.py and test_cp2.py. Those tests work because
# TP/CP preserve the full pipeline (every rank runs a complete forward pass).
# Under PP=2, a direct mg_model(input_ids=...) call on a non-first pipeline stage
# fails (it expects activations from the prior stage via pipeline comms, not raw
# tokens). Exercising that path requires the full Megatron pipeline scheduler,
# which is outside the scope of this unit-test suite.
