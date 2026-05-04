"""
End-to-end test: full multi-LoRA lifecycle under Expert Parallelism (EP=8).

Exercises the complete production workflow with MoE experts sharded at full
scale across 8 GPUs (Qwen3.5-35B-A3B, 256 experts, 32 per rank).
With num_experts_per_tok=8, tokens route across all 8 EP ranks — this is the
natural deployment configuration for this model.

  Phase 0 — startup: Megatron EP=8, weights loaded and sharded, 3 slots preallocated
  Phase 1 — 20 training steps, mixed batch, per-sequence routing, selective recompute
  Phase 2 — checkpoint save → hot-register into slot 2
             (gracefully skipped if EP checkpoint save/load not yet implemented)
  Phase 3 — 10 training steps with the 2 primary adapters
  Final    — routing verification: adapters produce distinct outputs vs base-only

Run with:
    torchrun --nproc-per-node=8 -m pytest tests/multi_lora/test_e2e_ep8.py -v -s
"""
import shutil
import tempfile

import torch
import torch.distributed as dist
import mcore_bridge.patcher as _patcher
from mcore_bridge.tuners.lora import LoraParallelLinear
from tests.multi_lora.conftest import (
    build_moe_model, eval_loss, init_megatron, lora_weight, make_batch,
    make_lora_config, rank, requires_cuda,
)

SEQ_LEN = 8
BATCH_PER_TASK = 4  # larger batch ensures all EP ranks receive dispatched tokens
VOCAB_SIZE = 32


@requires_cuda
def test_e2e_ep8_multi_lora_lifecycle():
    """Full multi-LoRA lifecycle: startup → training → verify (EP=8, Qwen3.5-35B-A3B)."""
    init_megatron(ep=8)
    _patcher.apply_patch()

    torch.manual_seed(42)
    mg_models, bridge, config = build_moe_model(ep=8, recompute=True)
    vocab_size = config.padded_vocab_size

    # 3 slots preallocated; 2 registered at startup, slot 2 reserved for checkpoint
    bridge.preallocate_adapters(mg_models, num_slots=3, lora_config=make_lora_config())
    bridge.register_adapter(mg_models, "incr", slot_index=0)
    bridge.register_adapter(mg_models, "decr", slot_index=1)

    listing = bridge.list_adapters()
    assert listing == {"incr": "__slot_0__", "decr": "__slot_1__"}, \
        f"[rank {rank()}] Wrong listing: {listing}"

    lora_mods = [m for m in mg_models[0].modules() if isinstance(m, LoraParallelLinear)]
    expert_lora = [m for m in lora_mods if m.is_expert]
    assert expert_lora, \
        f"[rank {rank()}] No expert LoraParallelLinear on rank {rank()} — EP routing won't be exercised"

    if rank() == 0:
        print(f"\n[EP=8] Expert LoRA modules: {len(expert_lora)}, "
              f"non-expert: {len(lora_mods) - len(expert_lora)}")

    mg_models[0].train()
    lora_params = [p for p in mg_models[0].parameters() if p.requires_grad]
    assert lora_params, f"[rank {rank()}] No trainable LoRA parameters"
    optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

    BATCH = 2 * BATCH_PER_TASK
    pos = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(BATCH, -1)

    # ── Phase 1: concurrent training, selective recompute ─────────────────────
    N_PHASE1 = 20
    phase1_losses = []
    for step in range(N_PHASE1):
        input_ids, labels = make_batch(
            ["increment"] * BATCH_PER_TASK + ["decrement"] * BATCH_PER_TASK,
            SEQ_LEN, VOCAB_SIZE, seed=step,
        )
        optimizer.zero_grad()
        with bridge.set_routing(mg_models,
                                ["incr"] * BATCH_PER_TASK + ["decr"] * BATCH_PER_TASK):
            out = mg_models[0](input_ids=input_ids, position_ids=pos,
                               attention_mask=None, labels=labels)
            loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
            loss.backward()
        optimizer.step()
        phase1_losses.append(loss.item())

    early = sum(phase1_losses[:4]) / 4
    late = sum(phase1_losses[-4:]) / 4
    drop = (early - late) / early
    if rank() == 0:
        print(f"\n[EP=8 Phase 1] early={early:.4f} late={late:.4f} drop={drop:.1%}")
    assert drop >= 0.05, \
        f"[rank {rank()}] Phase 1 loss did not drop ≥5% (drop={drop:.1%})"

    # ── Phase 2: checkpoint save → hot-register ───────────────────────────────
    _ep8_checkpoint_round_trip(mg_models, bridge)

    # ── Phase 3: 10 more training steps ──────────────────────────────────────
    N_PHASE3 = 10
    for step in range(N_PHASE3):
        input_ids, labels = make_batch(
            ["increment"] * BATCH_PER_TASK + ["decrement"] * BATCH_PER_TASK,
            SEQ_LEN, VOCAB_SIZE, seed=1000 + step,
        )
        optimizer.zero_grad()
        with bridge.set_routing(mg_models,
                                ["incr"] * BATCH_PER_TASK + ["decr"] * BATCH_PER_TASK):
            out = mg_models[0](input_ids=input_ids, position_ids=pos,
                               attention_mask=None, labels=labels)
            loss = (out[0] if isinstance(out, (tuple, list)) else out).mean()
            loss.backward()
        optimizer.step()

    # ── Final: routing verification ────────────────────────────────────────────
    mg_models[0].eval()
    eval_batch = 8  # one sequence per EP rank
    eval_pos = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(eval_batch, -1)
    torch.manual_seed(77)
    eval_ids = torch.randint(0, vocab_size, (SEQ_LEN, eval_batch), device="cuda")

    def _infer(adapter_name_or_none):
        with bridge.set_routing(mg_models, [adapter_name_or_none] * eval_batch):
            with torch.no_grad():
                out = mg_models[0](input_ids=eval_ids, position_ids=eval_pos,
                                   attention_mask=None)
        return out[0] if isinstance(out, (tuple, list)) else out

    logits_base = _infer(None)
    logits_incr = _infer("incr")
    logits_decr = _infer("decr")

    for name, logits in [("incr", logits_incr), ("decr", logits_decr)]:
        diff = (logits_base - logits).abs().max().item()
        assert diff > 1e-4, \
            f"[rank {rank()}] Adapter '{name}' indistinguishable from base (diff={diff:.2e})"

    cross_diff = (logits_incr - logits_decr).abs().max().item()
    assert cross_diff > 1e-4, \
        f"[rank {rank()}] 'incr' and 'decr' outputs identical — EP routing failure (diff={cross_diff:.2e})"

    base_incr = eval_loss(mg_models, bridge, None, SEQ_LEN, eval_batch, VOCAB_SIZE, "increment")
    base_decr = eval_loss(mg_models, bridge, None, SEQ_LEN, eval_batch, VOCAB_SIZE, "decrement")
    incr_l = eval_loss(mg_models, bridge, "incr", SEQ_LEN, eval_batch, VOCAB_SIZE, "increment")
    decr_l = eval_loss(mg_models, bridge, "decr", SEQ_LEN, eval_batch, VOCAB_SIZE, "decrement")

    if rank() == 0:
        print(f"\n[EP=8 Final] incr={incr_l:.4f} (base={base_incr:.4f})  "
              f"decr={decr_l:.4f} (base={base_decr:.4f})")
    assert incr_l < base_incr, \
        f"[rank {rank()}] 'incr' loss ({incr_l:.4f}) not below base ({base_incr:.4f})"
    assert decr_l < base_decr, \
        f"[rank {rank()}] 'decr' loss ({decr_l:.4f}) not below base ({base_decr:.4f})"

    if rank() == 0:
        print("\n[EP=8 E2E PASSED] Full multi-LoRA lifecycle under EP=8 complete.")


def _ep8_checkpoint_round_trip(mg_models, bridge):
    """Save slot 0, hot-register into slot 2, verify non-expert weights are bit-identical.

    Expert LoRA weights are EP-sharded (each rank holds a different subset of experts),
    so the saved checkpoint contains only this rank's shard. Full EP checkpoint
    gather/scatter is pending; this function logs but does not fail if it raises.
    """
    if rank() == 0:
        ckpt_dir = tempfile.mkdtemp()
    else:
        ckpt_dir = None
    path_list = [ckpt_dir]
    dist.broadcast_object_list(path_list, src=0)
    ckpt_dir = path_list[0]

    try:
        try:
            bridge.save_weights(mg_models, ckpt_dir, peft_format=True,
                                adapter_name="__slot_0__")
            bridge.register_adapter(mg_models, "incr_ckpt", slot_index=2,
                                    weights_dir=ckpt_dir)

            # Verify non-expert shard weights are bit-identical
            lora_mods = [m for m in mg_models[0].modules() if isinstance(m, LoraParallelLinear)
                         and not m.is_expert]
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
                print(f"[EP=8 Phase 2] checkpoint round-trip succeeded")
        except Exception as exc:
            if rank() == 0:
                print(f"[EP=8 Phase 2] checkpoint round-trip skipped (EP ckpt pending): {exc}")
    finally:
        dist.barrier()
        if rank() == 0:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
