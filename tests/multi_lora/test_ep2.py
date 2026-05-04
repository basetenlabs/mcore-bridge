"""
Gate 5 — Multi-LoRA routing works correctly under expert parallelism (EP=2).

Verifies that RouterReplay carries per-sequence LoRA adapter indices through
the MoE AlltoAll dispatch so expert linear layers apply the right adapter
post-dispatch, and that per-adapter gradient isolation holds under EP=2.

Run with:
    torchrun --nproc-per-node=2 -m pytest tests/multi_lora/test_ep2.py -v -s
"""
import torch
import mcore_bridge.patcher as _patcher
from mcore_bridge.tuners.lora import LoraParallelLinear
from tests.multi_lora.conftest import (
    build_moe_model, init_megatron, make_lora_config, rank, requires_cuda,
    synthetic_task,
)

EP_SIZE = 2


# ─────────────────────────────────────────────────────────────────────────────

@requires_cuda
def test_ep2_preallocate_and_registry():
    """preallocate_adapters and registry methods work correctly under EP=2."""
    init_megatron(ep=EP_SIZE)
    _patcher.apply_patch()

    mg_models, bridge, config = build_moe_model(ep=EP_SIZE)
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=make_lora_config())
    bridge.register_adapter(mg_models, "slot_a", slot_index=0)
    bridge.register_adapter(mg_models, "slot_b", slot_index=1)

    listing = bridge.list_adapters()
    assert listing == {"slot_a": "__slot_0__", "slot_b": "__slot_1__"}, \
        f"[rank {rank()}] Wrong listing: {listing}"

    indices = bridge.resolve_routing(["slot_a", None, "slot_b", "slot_a"])
    expected = torch.tensor([1, 0, 2, 1], dtype=torch.long)
    assert torch.equal(indices, expected), \
        f"[rank {rank()}] Got {indices}, expected {expected}"

    lora_mods = [m for m in mg_models[0].modules() if isinstance(m, LoraParallelLinear)]
    assert lora_mods, f"[rank {rank()}] No LoraParallelLinear found"
    expert_lora = [m for m in lora_mods if m.is_expert]
    assert expert_lora, f"[rank {rank()}] No expert LoraParallelLinear found (EP routing cannot be tested)"
    for m in lora_mods:
        assert m._idx_to_name == {1: '__slot_0__', 2: '__slot_1__'}, \
            f"[rank {rank()}] Wrong _idx_to_name: {m._idx_to_name}"


@requires_cuda
def test_ep2_routing_changes_output():
    """Non-zero LoRA adapter changes model output compared to base-only, under EP=2."""
    init_megatron(ep=EP_SIZE)
    _patcher.apply_patch()

    torch.manual_seed(42 + rank())
    mg_models, bridge, config = build_moe_model(ep=EP_SIZE)
    bridge.preallocate_adapters(mg_models, num_slots=1, lora_config=make_lora_config())
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
    seq_len, batch = 8, 2
    input_ids = torch.randint(0, vocab_size, (seq_len, batch), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1)

    with bridge.set_routing(mg_models, [None, None]):
        with torch.no_grad():
            out0 = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                                attention_mask=None)
            logits0 = out0[0] if isinstance(out0, (tuple, list)) else out0

    with bridge.set_routing(mg_models, ["adapter_a", "adapter_a"]):
        with torch.no_grad():
            out1 = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                                attention_mask=None)
            logits1 = out1[0] if isinstance(out1, (tuple, list)) else out1

    diff = (logits0 - logits1).abs().max().item()
    assert diff > 1e-4, \
        f"[rank {rank()}] Non-zero lora_B should change output; max diff={diff:.2e}"


@requires_cuda
def test_ep2_grad_isolation():
    """Under EP=2, adapter 2 receives no gradient when batch routes only to adapter 1."""
    init_megatron(ep=EP_SIZE)
    _patcher.apply_patch()

    torch.manual_seed(2)
    mg_models, bridge, config = build_moe_model(ep=EP_SIZE)
    mg_model = mg_models[0]

    mg_model.add_adapter("adapter_2", make_lora_config())
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
    out = mg_model(input_ids=input_ids, position_ids=position_ids,
                   attention_mask=None, labels=labels)
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
        f"[rank {rank()}] adapter 'default' should have non-zero gradients"

    adapter2_grads = [
        p.grad for m in lora_mods
        for name, p in m.named_parameters()
        if "adapter_2" in name and p.requires_grad
    ]
    assert all(g is None for g in adapter2_grads), \
        f"[rank {rank()}] adapter_2 should have None grads; got non-None on rank {rank()}"


@requires_cuda
def test_ep2_determinism():
    """Same input + same routing is bit-identical across two calls, under EP=2."""
    init_megatron(ep=EP_SIZE)
    _patcher.apply_patch()

    torch.manual_seed(3)
    mg_models, bridge, config = build_moe_model(ep=EP_SIZE)
    bridge.preallocate_adapters(mg_models, num_slots=1, lora_config=make_lora_config())
    bridge.register_adapter(mg_models, "alpha", slot_index=0)
    mg_models[0].train()

    vocab_size = config.padded_vocab_size
    seq_len, batch = 8, 4
    input_ids = torch.randint(0, vocab_size, (seq_len, batch), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1)

    with bridge.set_routing(mg_models, ["alpha", None, "alpha", None]):
        with torch.no_grad():
            out1 = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                                attention_mask=None)
            logits1 = out1[0] if isinstance(out1, (tuple, list)) else out1
        with torch.no_grad():
            out2 = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                                attention_mask=None)
            logits2 = out2[0] if isinstance(out2, (tuple, list)) else out2

    assert torch.equal(logits1, logits2), \
        f"[rank {rank()}] Non-deterministic output; max diff={(logits1 - logits2).abs().max():.2e}"


@requires_cuda
def test_ep2_expert_dispatch_indices():
    """RouterReplay sets _expert_lora_indices on the root model during expert compute.

    Hooks MoELayer.routed_experts_compute to capture whether the attribute is present
    and records the range of indices — verifies that RouterReplay correctly carries
    per-sequence adapter indices through the AlltoAll dispatch.
    """
    init_megatron(ep=EP_SIZE)
    _patcher.apply_patch()

    torch.manual_seed(7)
    mg_models, bridge, config = build_moe_model(ep=EP_SIZE)
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=make_lora_config())
    bridge.register_adapter(mg_models, "ada_0", slot_index=0)
    bridge.register_adapter(mg_models, "ada_1", slot_index=1)
    mg_models[0].eval()

    # Force non-zero lora_B so routing actually matters
    for m in mg_models[0].modules():
        if isinstance(m, LoraParallelLinear) and m.is_expert:
            for name, param in m.named_parameters():
                if "lora_B" in name and param.requires_grad:
                    torch.nn.init.kaiming_uniform_(param)

    observed = {"found": False, "min_idx": None, "max_idx": None}

    try:
        from megatron.core.transformer.moe.moe_layer import MoELayer
    except ImportError:
        import pytest; pytest.skip("MoELayer not available")

    _orig_rec = MoELayer.routed_experts_compute

    def _hook_rec(self, hidden_states, probs):
        root = mg_models[0].base_model.model
        idx = getattr(root, '_expert_lora_indices', None)
        if idx is not None:
            observed["found"] = True
            observed["min_idx"] = int(idx.min().item())
            observed["max_idx"] = int(idx.max().item())
        return _orig_rec(self, hidden_states, probs)

    MoELayer.routed_experts_compute = _hook_rec
    try:
        vocab_size = config.padded_vocab_size
        seq_len, batch = 4, 4
        input_ids = torch.randint(0, vocab_size, (seq_len, batch), device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1)

        with bridge.set_routing(mg_models, ["ada_0", "ada_1", "ada_0", None]):
            with torch.no_grad():
                mg_models[0](input_ids=input_ids, position_ids=position_ids,
                             attention_mask=None)
    finally:
        MoELayer.routed_experts_compute = _orig_rec

    assert observed["found"], \
        f"[rank {rank()}] _expert_lora_indices was never set during forward — RouterReplay not firing"
    # Indices should be in [0, 2] (0=no adapter, 1=ada_0, 2=ada_1)
    assert 0 <= observed["min_idx"] <= 2, \
        f"[rank {rank()}] Unexpected min index {observed['min_idx']}"
    assert 0 <= observed["max_idx"] <= 2, \
        f"[rank {rank()}] Unexpected max index {observed['max_idx']}"
    if rank() == 0:
        print(f"\n[EP=2] _expert_lora_indices range: [{observed['min_idx']}, {observed['max_idx']}]")


@requires_cuda
def test_ep2_loss_decreases():
    """Under EP=2, training loss decreases after optimizer steps with per-task routing."""
    init_megatron(ep=EP_SIZE)
    _patcher.apply_patch()

    torch.manual_seed(7)
    mg_models, bridge, config = build_moe_model(ep=EP_SIZE)
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=make_lora_config())
    bridge.register_adapter(mg_models, "incr", slot_index=0)
    bridge.register_adapter(mg_models, "decr", slot_index=1)
    mg_models[0].train()

    lora_params = [p for p in mg_models[0].parameters() if p.requires_grad]
    assert lora_params, f"[rank {rank()}] No trainable LoRA parameters"
    optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

    seq_len, batch_per_task = 8, 2
    vocab_size = config.padded_vocab_size
    position_ids = (
        torch.arange(seq_len, device="cuda")
        .unsqueeze(0)
        .expand(2 * batch_per_task, -1)
    )

    N_STEPS = 20
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

        with bridge.set_routing(mg_models,
                                ["incr"] * batch_per_task + ["decr"] * batch_per_task):
            out = mg_models[0](input_ids=input_ids, position_ids=position_ids,
                               attention_mask=None, labels=labels)
            raw_loss = out[0] if isinstance(out, (tuple, list)) else out
            loss = raw_loss.mean()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    early_avg = sum(losses[:4]) / 4
    late_avg = sum(losses[-4:]) / 4
    drop = (early_avg - late_avg) / early_avg
    if rank() == 0:
        print(f"\n[EP=2] early_avg={early_avg:.4f} late_avg={late_avg:.4f} drop={drop:.1%}")

    assert drop >= 0.05, \
        f"[rank {rank()}] Training loss did not drop ≥5% over {N_STEPS} steps (drop={drop:.1%})"
