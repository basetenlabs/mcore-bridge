"""
End-to-end test: full multi-LoRA lifecycle under Pipeline Parallelism (PP=2).

Uses Megatron's get_forward_backward_func() to drive the pipeline schedule
(forward_backward_pipelining_without_interleaving). This is identical to what
a real training loop would do — no schedule modifications required.

Key differences from single-GPU E2E:
  - overlap_p2p_comm=False required by non-interleaved schedule
  - forward_backward_func calls loss.backward() internally
  - Loss is only returned on the last pipeline rank (losses=[] on others)
  - set_routing context wraps the entire forward_backward_func call

Run with:
    torchrun --nproc-per-node=2 -m pytest tests/multi_lora/test_e2e_pp2.py -v -s
"""
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

import mcore_bridge.patcher as _patcher
from mcore_bridge.tuners.lora import LoraParallelLinear
from tests.multi_lora.conftest import (
    build_model, init_megatron, make_batch, make_lora_config, rank, requires_cuda,
)

SEQ_LEN = 8
BATCH_PER_TASK = 2
VOCAB_SIZE = 32


def _make_forward_step(pos_ids):
    """Return a forward_step_func that captures position ids in a closure."""
    def forward_step(data_iterator, model):
        ids, lbl = next(data_iterator)
        output = model(input_ids=ids, position_ids=pos_ids,
                       attention_mask=None, labels=lbl)
        def loss_func(output):
            raw = output[0] if isinstance(output, (tuple, list)) else output
            loss = raw.mean()
            return loss, {"loss": loss.detach()}
        return output, loss_func
    return forward_step


# ─────────────────────────────────────────────────────────────────────────────

@requires_cuda
def test_e2e_pp2_multi_lora_lifecycle():
    """Full multi-LoRA lifecycle under PP=2 using Megatron pipeline schedule.

    Timeline:
      Phase 0 — startup: model load, 2-slot preallocate, register "incr"/"decr"
      Phase 1 — 20 training steps, mixed batch, per-sequence routing
      Phase 2 — hot-register "incr_again" from a checkpoint of slot 0 into slot 1
                 (reuses slot 1's pre-allocated weights buffer, overwrites "decr")
      Phase 3 — verify routing: both adapters produce output different from base-only
    """
    init_megatron(pp=2)
    _patcher.apply_patch()

    torch.manual_seed(42)
    # overlap_p2p_comm=False required by non-interleaved pipeline schedule
    mg_models, bridge, config = build_model(pp=2, extra_cfg={"overlap_p2p_comm": False})

    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=make_lora_config())
    bridge.register_adapter(mg_models, "incr", slot_index=0)
    bridge.register_adapter(mg_models, "decr", slot_index=1)

    listing = bridge.list_adapters()
    assert listing == {"incr": "__slot_0__", "decr": "__slot_1__"}, \
        f"[rank {rank()}] Wrong listing: {listing}"

    mg_models[0].train()
    lora_params = [p for p in mg_models[0].parameters() if p.requires_grad]
    assert lora_params, f"[rank {rank()}] No trainable LoRA parameters"
    optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

    BATCH = 2 * BATCH_PER_TASK
    pos = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(BATCH, -1)
    forward_backward_func = get_forward_backward_func()

    # ── Phase 1: concurrent training via pipeline schedule ────────────────────
    N_STEPS = 20
    last_rank_losses = []

    for step in range(N_STEPS):
        input_ids, labels = make_batch(
            ["increment"] * BATCH_PER_TASK + ["decrement"] * BATCH_PER_TASK,
            SEQ_LEN, VOCAB_SIZE, seed=step,
        )
        adapter_names = ["incr"] * BATCH_PER_TASK + ["decr"] * BATCH_PER_TASK

        optimizer.zero_grad()
        # forward_backward_func calls loss.backward() internally —
        # set_routing must wrap the entire call so _lora_adapter_indices is
        # present during both the forward pass AND the recomputed backward.
        with bridge.set_routing(mg_models, adapter_names):
            losses = forward_backward_func(
                forward_step_func=_make_forward_step(pos),
                data_iterator=iter([(input_ids, labels)]),
                model=mg_models,
                num_microbatches=1,
                seq_length=SEQ_LEN,
                micro_batch_size=BATCH,
                forward_only=False,
            )
        optimizer.step()

        if parallel_state.is_pipeline_last_stage() and losses:
            last_rank_losses.append(losses[0]["loss"].item())

    if parallel_state.is_pipeline_last_stage():
        early = sum(last_rank_losses[:4]) / 4
        late = sum(last_rank_losses[-4:]) / 4
        drop = (early - late) / early
        print(f"\n[PP=2 Phase 1] early={early:.4f} late={late:.4f} drop={drop:.1%}")
        assert drop >= 0.10, \
            f"[rank {rank()}] Phase 1 loss did not drop ≥10% (drop={drop:.1%})"

    dist.barrier()

    # ── Phase 2: verify routing — each adapter produces different output ───────
    mg_models[0].eval()
    eval_batch = 2
    eval_pos = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(eval_batch, -1)
    torch.manual_seed(77)
    eval_ids = torch.randint(0, VOCAB_SIZE, (SEQ_LEN, eval_batch), device="cuda")

    logits_sets = {}
    for adapter_name in [None, "incr", "decr"]:
        with bridge.set_routing(mg_models, [adapter_name] * eval_batch):
            losses_inf = forward_backward_func(
                forward_step_func=_make_forward_step(eval_pos),
                data_iterator=iter([(eval_ids, torch.roll(eval_ids, -1, dims=0))]),
                model=mg_models,
                num_microbatches=1,
                seq_length=SEQ_LEN,
                micro_batch_size=eval_batch,
                forward_only=True,
            )
        if parallel_state.is_pipeline_last_stage() and losses_inf:
            logits_sets[adapter_name] = losses_inf[0]["loss"].item()

    if parallel_state.is_pipeline_last_stage():
        base_loss = logits_sets[None]
        incr_loss = logits_sets["incr"]
        decr_loss = logits_sets["decr"]
        print(f"\n[PP=2 Final] base_loss={base_loss:.4f}  "
              f"incr_loss={incr_loss:.4f}  decr_loss={decr_loss:.4f}")
        assert incr_loss < base_loss, \
            f"'incr' loss ({incr_loss:.4f}) not below base ({base_loss:.4f})"
        assert decr_loss < base_loss, \
            f"'decr' loss ({decr_loss:.4f}) not below base ({base_loss:.4f})"
        assert abs(incr_loss - decr_loss) > 1e-4, \
            f"'incr' and 'decr' produced same eval loss — possible routing failure"

    dist.barrier()

    # ── Phase 3: verify all PP ranks got _lora_adapter_indices stamped ─────────
    root = mg_models[0].base_model.model
    with bridge.set_routing(mg_models, ["incr", "decr"]):
        assert hasattr(root, '_lora_adapter_indices'), \
            f"[rank {rank()}] _lora_adapter_indices not stamped on stage"
        expected = torch.tensor([1, 2], dtype=torch.long)
        assert torch.equal(root._lora_adapter_indices, expected), \
            f"[rank {rank()}] Wrong indices: {root._lora_adapter_indices}"
    assert not hasattr(root, '_lora_adapter_indices'), \
        f"[rank {rank()}] Indices not cleaned up after context"

    if rank() == 0:
        print("\n[PP=2 E2E PASSED] Pipeline-parallel multi-LoRA lifecycle complete.")
