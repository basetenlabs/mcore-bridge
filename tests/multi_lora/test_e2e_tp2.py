"""
End-to-end test: full multi-LoRA lifecycle under Tensor Parallelism (TP=2).

Exercises the complete production workflow with weights sharded across 2 GPUs:

  Phase 0 — startup: Megatron TP=2, weights loaded and sharded, 3 slots preallocated
  Phase 1 — 25 training steps, mixed batch, per-sequence routing, selective recompute
  Phase 2 — checkpoint save (master rank gathers + writes) → hot-register into slot 2
             via register_adapter(weights_dir=...) which re-shards on load
  Phase 3 — 15 training steps with all 3 adapters
  Final    — routing verification: adapters produce distinct outputs vs base-only;
             eval loss for each adapter below base-only

Run with:
    torchrun --nproc-per-node=2 -m pytest tests/multi_lora/test_e2e_tp2.py -v -s
"""
import shutil
import tempfile

import torch
import torch.distributed as dist
import mcore_bridge.patcher as _patcher
from mcore_bridge.tuners.lora import LoraParallelLinear
from tests.multi_lora.conftest import (
    build_model, eval_loss, init_megatron, lora_weight, make_batch,
    make_lora_config, rank, requires_cuda,
)

SEQ_LEN = 16
BATCH_PER_TASK = 4
VOCAB_SIZE = 32


@requires_cuda
def test_e2e_tp2_multi_lora_lifecycle():
    """Full multi-LoRA lifecycle: startup → training → checkpoint → hot-register → verify (TP=2)."""
    init_megatron(tp=2)
    _patcher.apply_patch()

    torch.manual_seed(42)
    mg_models, bridge, config = build_model(tp=2, recompute=True)

    # 3 slots: 2 registered at startup, slot 2 held as spare
    bridge.preallocate_adapters(mg_models, num_slots=3, lora_config=make_lora_config())
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

    # ── Phase 1: concurrent training, selective recompute ─────────────────────
    N_PHASE1 = 25
    phase1_losses = []
    for step in range(N_PHASE1):
        input_ids, labels = make_batch(
            ["increment"] * BATCH_PER_TASK + ["decrement"] * BATCH_PER_TASK,
            SEQ_LEN, VOCAB_SIZE, seed=step,
        )
        optimizer.zero_grad()
        with bridge.set_routing(mg_models, ["incr"] * BATCH_PER_TASK + ["decr"] * BATCH_PER_TASK):
            out = mg_models[0](input_ids=input_ids, position_ids=pos,
                               attention_mask=None, labels=labels)
            loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
            loss.backward()
        optimizer.step()
        phase1_losses.append(loss.item())

    early = sum(phase1_losses[:5]) / 5
    late = sum(phase1_losses[-5:]) / 5
    drop = (early - late) / early
    if rank() == 0:
        print(f"\n[TP=2 Phase 1] early={early:.4f} late={late:.4f} drop={drop:.1%}")
    assert drop >= 0.15, \
        f"[rank {rank()}] Phase 1 loss did not drop ≥15% (drop={drop:.1%})"

    # ── Phase 2: save slot 0, hot-register into slot 2 ────────────────────────
    # Each torchrun rank is a separate process — broadcast rank-0's path so all
    # ranks read the same checkpoint that rank 0 wrote.
    if rank() == 0:
        ckpt_dir = tempfile.mkdtemp()
    else:
        ckpt_dir = None
    path_list = [ckpt_dir]
    dist.broadcast_object_list(path_list, src=0)
    ckpt_dir = path_list[0]

    try:
        bridge.save_weights(mg_models, ckpt_dir, peft_format=True, adapter_name="__slot_0__")
        bridge.register_adapter(mg_models, "incr_ckpt", slot_index=2, weights_dir=ckpt_dir)
    finally:
        dist.barrier()
        if rank() == 0:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    listing = bridge.list_adapters()
    assert set(listing.keys()) == {"incr", "decr", "incr_ckpt"}, \
        f"[rank {rank()}] Listing after hot-register: {listing}"

    # Verify reloaded weights are bit-identical on this rank's shard
    lora_mods = [m for m in mg_models[0].modules() if isinstance(m, LoraParallelLinear)]
    for m in lora_mods:
        for ab in ("lora_A", "lora_B"):
            module_dict = getattr(m, ab)
            if "__slot_0__" not in module_dict or "__slot_2__" not in module_dict:
                continue
            src_w = lora_weight(module_dict["__slot_0__"])
            dst_w = lora_weight(module_dict["__slot_2__"])
            assert torch.equal(src_w, dst_w), \
                f"[rank {rank()}] Reloaded shard differs for {ab}: " \
                f"max_diff={(src_w - dst_w).abs().max():.2e}"

    if rank() == 0:
        print(f"[TP=2 Phase 2] slot 0 → checkpoint → slot 2 reload: bit-identical on all shards")

    # ── Phase 3: train all 3 adapters ─────────────────────────────────────────
    N_PHASE3 = 15
    pos3 = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(3 * BATCH_PER_TASK, -1)
    for step in range(N_PHASE3):
        input_ids, labels = make_batch(
            ["increment"] * BATCH_PER_TASK
            + ["decrement"] * BATCH_PER_TASK
            + ["increment"] * BATCH_PER_TASK,
            SEQ_LEN, VOCAB_SIZE, seed=1000 + step,
        )
        optimizer.zero_grad()
        with bridge.set_routing(mg_models,
                                ["incr"] * BATCH_PER_TASK
                                + ["decr"] * BATCH_PER_TASK
                                + ["incr_ckpt"] * BATCH_PER_TASK):
            out = mg_models[0](input_ids=input_ids, position_ids=pos3,
                               attention_mask=None, labels=labels)
            loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
            loss.backward()
        optimizer.step()

    # ── Final: routing verification ────────────────────────────────────────────
    mg_models[0].eval()
    eval_batch = 4
    eval_pos = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(eval_batch, -1)
    torch.manual_seed(77)
    eval_ids = torch.randint(0, VOCAB_SIZE, (SEQ_LEN, eval_batch), device="cuda")

    def _infer(adapter_name_or_none):
        with bridge.set_routing(mg_models, [adapter_name_or_none] * eval_batch):
            with torch.no_grad():
                out = mg_models[0](input_ids=eval_ids, position_ids=eval_pos,
                                   attention_mask=None)
        return out[0] if isinstance(out, (tuple, list)) else out

    logits_base = _infer(None)
    logits_incr = _infer("incr")
    logits_decr = _infer("decr")
    logits_ckpt = _infer("incr_ckpt")

    for name, logits in [("incr", logits_incr), ("decr", logits_decr), ("incr_ckpt", logits_ckpt)]:
        diff = (logits_base - logits).abs().max().item()
        assert diff > 1e-4, \
            f"[rank {rank()}] Adapter '{name}' indistinguishable from base (diff={diff:.2e})"

    cross_diff = (logits_incr - logits_decr).abs().max().item()
    assert cross_diff > 1e-4, \
        f"[rank {rank()}] 'incr' and 'decr' outputs identical — routing failure (diff={cross_diff:.2e})"

    base_incr = eval_loss(mg_models, bridge, None, SEQ_LEN, eval_batch, VOCAB_SIZE, "increment")
    base_decr = eval_loss(mg_models, bridge, None, SEQ_LEN, eval_batch, VOCAB_SIZE, "decrement")
    incr_l = eval_loss(mg_models, bridge, "incr", SEQ_LEN, eval_batch, VOCAB_SIZE, "increment")
    decr_l = eval_loss(mg_models, bridge, "decr", SEQ_LEN, eval_batch, VOCAB_SIZE, "decrement")

    if rank() == 0:
        print(f"\n[TP=2 Final] incr={incr_l:.4f} (base={base_incr:.4f})  "
              f"decr={decr_l:.4f} (base={base_decr:.4f})")
    assert incr_l < base_incr, \
        f"[rank {rank()}] 'incr' loss ({incr_l:.4f}) not below base ({base_incr:.4f})"
    assert decr_l < base_decr, \
        f"[rank {rank()}] 'decr' loss ({decr_l:.4f}) not below base ({base_decr:.4f})"

    if rank() == 0:
        print("\n[TP=2 E2E PASSED] Full multi-LoRA lifecycle under TP=2 complete.")
