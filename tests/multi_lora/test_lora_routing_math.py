"""
Layer 1 — apply_routed_lora pure math tests.  CPU only, no Megatron required.

Run with:
    pytest tests/multi_lora/test_lora_routing_math.py -v
"""
import torch
import torch.nn as nn
import pytest

from tests.multi_lora.conftest import stub_adapter_pair
from mcore_bridge.tuners.lora import apply_routed_lora

IN = 16
RANK = 4
OUT = 8


def _make_dicts(adapters: dict):
    """Build callable dicts from {idx: (lora_A, lora_B)} mapping."""
    lora_A_by_idx = {i: ab[0] for i, ab in adapters.items()}
    lora_B_by_idx = {i: ab[1] for i, ab in adapters.items()}
    scaling_by_idx = {i: 1.0 for i in adapters}
    dropout_by_idx = {i: nn.Identity() for i in adapters}
    return lora_A_by_idx, lora_B_by_idx, scaling_by_idx, dropout_by_idx


def test_routing_forward_matches_hand_calc():
    """Output matches manual per-token LoRA calculation."""
    torch.manual_seed(0)
    adapter_indices = torch.tensor([1, 1, 2, 0])
    tokens = adapter_indices.shape[0]
    x = torch.randn(tokens, IN)
    result = torch.zeros(tokens, OUT)

    # hand-set weights
    A1 = torch.randn(RANK, IN)
    B1 = torch.randn(OUT, RANK)
    A2 = torch.randn(RANK, IN)
    B2 = torch.randn(OUT, RANK)

    lA1 = nn.Linear(IN, RANK, bias=False); lA1.weight.data.copy_(A1)
    lB1 = nn.Linear(RANK, OUT, bias=False); lB1.weight.data.copy_(B1)
    lA2 = nn.Linear(IN, RANK, bias=False); lA2.weight.data.copy_(A2)
    lB2 = nn.Linear(RANK, OUT, bias=False); lB2.weight.data.copy_(B2)

    A_by, B_by, sc_by, do_by = _make_dicts({1: (lA1, lB1), 2: (lA2, lB2)})
    apply_routed_lora(result, x, adapter_indices, A_by, B_by, sc_by, do_by)

    expected = torch.zeros(tokens, OUT)
    expected[[0, 1]] += (x[[0, 1]] @ A1.T) @ B1.T
    expected[[2]] += (x[[2]] @ A2.T) @ B2.T
    # token 3 (idx=0): unchanged

    assert torch.allclose(result, expected, atol=1e-5), \
        f'Max diff: {(result - expected).abs().max().item():.2e}'


def test_base_only_short_circuit():
    """All-zero adapter_indices leaves result unchanged."""
    torch.manual_seed(1)
    tokens = 6
    adapter_indices = torch.zeros(tokens, dtype=torch.long)
    x = torch.randn(tokens, IN)
    result = torch.randn(tokens, OUT)
    result_orig = result.clone()

    A_by, B_by, sc_by, do_by = _make_dicts({1: stub_adapter_pair(RANK, IN, OUT)})
    apply_routed_lora(result, x, adapter_indices, A_by, B_by, sc_by, do_by)

    assert torch.equal(result, result_orig)


def test_grad_isolation():
    """Gradients flow only through adapters whose tokens appear in the batch."""
    torch.manual_seed(2)
    adapter_indices = torch.tensor([1, 2, 1, 2])  # adapter 3 never used
    tokens = adapter_indices.shape[0]
    x = torch.randn(tokens, IN)
    result = torch.zeros(tokens, OUT)

    a1, b1 = stub_adapter_pair(RANK, IN, OUT)
    a2, b2 = stub_adapter_pair(RANK, IN, OUT)
    a3, b3 = stub_adapter_pair(RANK, IN, OUT)
    # give B non-zero init so output is non-zero
    nn.init.kaiming_uniform_(b1.weight)
    nn.init.kaiming_uniform_(b2.weight)
    nn.init.kaiming_uniform_(b3.weight)

    for m in [a1, b1, a2, b2, a3, b3]:
        for p in m.parameters():
            p.requires_grad_(True)

    A_by, B_by, sc_by, do_by = _make_dicts({1: (a1, b1), 2: (a2, b2), 3: (a3, b3)})
    apply_routed_lora(result, x, adapter_indices, A_by, B_by, sc_by, do_by)
    result.sum().backward()

    assert a1.weight.grad is not None and a1.weight.grad.abs().sum() > 0, 'adapter 1 should have grad'
    assert a2.weight.grad is not None and a2.weight.grad.abs().sum() > 0, 'adapter 2 should have grad'
    assert a3.weight.grad is None, 'adapter 3 was never used: grad should be None'


def test_dtype_preserve():
    """Output dtype matches input dtype for float32 and bfloat16."""
    for dtype in [torch.float32, torch.bfloat16]:
        adapter_indices = torch.tensor([1, 1, 2])
        tokens = adapter_indices.shape[0]
        x = torch.randn(tokens, IN).to(dtype)
        result = torch.zeros(tokens, OUT).to(dtype)

        lA1 = nn.Linear(IN, RANK, bias=False).to(dtype)
        lB1 = nn.Linear(RANK, OUT, bias=False).to(dtype)
        lA2 = nn.Linear(IN, RANK, bias=False).to(dtype)
        lB2 = nn.Linear(RANK, OUT, bias=False).to(dtype)

        apply_routed_lora(
            result, x, adapter_indices,
            {1: lA1, 2: lA2}, {1: lB1, 2: lB2},
            {1: 1.0, 2: 1.0}, {1: nn.Identity(), 2: nn.Identity()},
        )
        assert result.dtype == dtype, f'Expected {dtype}, got {result.dtype}'


def test_unload_then_forward():
    """Missing adapter index in dict is silently skipped (no error)."""
    torch.manual_seed(3)
    a2, b2 = stub_adapter_pair(RANK, IN, OUT)
    nn.init.kaiming_uniform_(b2.weight)

    # Only adapter 2 is present; tokens with idx=1 are "orphaned"
    A_by = {2: a2}
    B_by = {2: b2}
    sc_by = {2: 1.0}
    do_by = {2: nn.Identity()}

    adapter_indices = torch.tensor([1, 2, 2, 0])
    tokens = adapter_indices.shape[0]
    x = torch.randn(tokens, IN)
    result = torch.zeros(tokens, OUT)

    # Must not raise even though idx=1 is missing from dicts
    apply_routed_lora(result, x, adapter_indices, A_by, B_by, sc_by, do_by)

    # token 0 (idx=1, missing) → unchanged
    assert torch.equal(result[0], torch.zeros(OUT))
    # tokens 1,2 (idx=2) → modified
    assert result[[1, 2]].abs().sum() > 0
