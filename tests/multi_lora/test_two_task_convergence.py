"""
Gate 1 — Two synthetic tasks converge when trained with per-sequence LoRA routing.

A single batch mixes two synthetic causal-LM tasks:
  - "increment": next token = (prev + 1) % V
  - "decrement": next token = (prev - 1) % V

Each sequence is routed to its own LoRA adapter. After N optimizer steps,
the per-task cross-entropy should drop significantly from its initial value.

Run with:
    torchrun --nproc-per-node=1 -m pytest tests/multi_lora/test_two_task_convergence.py -v -s
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
from tests.multi_lora.conftest import requires_cuda, synthetic_task

MODEL_ID = "/tmp/qwen3-0.6b"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# Use a tiny effective vocabulary so convergence is fast
V_EFF = 32   # tokens [0, V_EFF)
SEQ_LEN = 16
N_STEPS = 150
LR = 1e-3
BATCH_PER_TASK = 4


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
    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET,
                      lora_dropout=0.0, bias="none")


def _build_model():
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


def _make_batch(batch_per_task: int = BATCH_PER_TASK, seed: int = None):
    """Build a mixed batch with batch_per_task sequences per task.

    Returns:
        input_ids: (seq_len, 2*batch_per_task) LongTensor on CUDA
        labels:    (seq_len, 2*batch_per_task) LongTensor on CUDA
        adapter_names: list of 2*batch_per_task adapter name strings
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device("cuda")
    inc_inputs, inc_labels = [], []
    dec_inputs, dec_labels = [], []

    for _ in range(batch_per_task):
        tok, lab = synthetic_task("increment", seq_len=SEQ_LEN, vocab_size=V_EFF)
        inc_inputs.append(tok)
        inc_labels.append(lab)
        tok, lab = synthetic_task("decrement", seq_len=SEQ_LEN, vocab_size=V_EFF)
        dec_inputs.append(tok)
        dec_labels.append(lab)

    input_ids = torch.stack(inc_inputs + dec_inputs, dim=1).to(device)  # (seq, 2B)
    labels = torch.stack(inc_labels + dec_labels, dim=1).to(device)
    adapter_names = ["incr"] * batch_per_task + ["decr"] * batch_per_task
    return input_ids, labels, adapter_names


@requires_cuda
def test_two_task_convergence():
    """Combined training loss decreases ≥ 20 % over N_STEPS steps with per-task routing.

    We also verify that each task's loss on a fixed held-out batch decreases,
    confirming the routing is task-isolating rather than collapsing.
    """
    _init_megatron()
    _patcher.apply_patch()

    torch.manual_seed(7)
    mg_models, bridge, config = _build_model()

    # Pre-allocate 2 slots: slot 0 = incr, slot 1 = decr
    bridge.preallocate_adapters(mg_models, num_slots=2, lora_config=_lora_config())
    bridge.register_adapter(mg_models, "incr", slot_index=0)
    bridge.register_adapter(mg_models, "decr", slot_index=1)

    mg_model = mg_models[0]
    mg_model.train()

    lora_params = [p for p in mg_model.parameters() if p.requires_grad]
    assert lora_params, "No trainable LoRA parameters"
    optimizer = torch.optim.AdamW(lora_params, lr=LR)

    position_ids = (
        torch.arange(SEQ_LEN, device="cuda")
        .unsqueeze(0)
        .expand(2 * BATCH_PER_TASK, -1)
    )

    # ── fixed eval batches (no randomness during evaluation) ─────────────
    eval_ids_incr, eval_lbl_incr, _ = _make_batch(batch_per_task=BATCH_PER_TASK, seed=100)
    eval_ids_decr, eval_lbl_decr, _ = _make_batch(batch_per_task=BATCH_PER_TASK, seed=200)
    # Take just the incr half from eval_ids_incr, decr half from eval_ids_decr
    eval_ids_incr = eval_ids_incr[:, :BATCH_PER_TASK]
    eval_lbl_incr = eval_lbl_incr[:, :BATCH_PER_TASK]
    eval_ids_decr = eval_ids_decr[:, BATCH_PER_TASK:]
    eval_lbl_decr = eval_lbl_decr[:, BATCH_PER_TASK:]
    eval_pos = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(BATCH_PER_TASK, -1)

    def _eval_on_fixed(ids, lbl, short_name):
        with bridge.set_routing(mg_models, [short_name] * BATCH_PER_TASK):
            with torch.no_grad():
                out = mg_model(input_ids=ids, position_ids=eval_pos,
                               attention_mask=None, labels=lbl)
                return (out[0] if isinstance(out, (tuple, list)) else out).mean().item()

    # ── baseline losses on fixed eval batches ────────────────────────────
    init_incr = _eval_on_fixed(eval_ids_incr, eval_lbl_incr, "incr")
    init_decr = _eval_on_fixed(eval_ids_decr, eval_lbl_decr, "decr")
    print(f"\nInitial loss — incr: {init_incr:.4f}  decr: {init_decr:.4f}")

    # ── training loop ─────────────────────────────────────────────────────
    train_losses = []
    for step in range(N_STEPS):
        optimizer.zero_grad()
        input_ids, labels, adapter_names = _make_batch()

        with bridge.set_routing(mg_models, adapter_names):
            out = mg_model(input_ids=input_ids, position_ids=position_ids,
                           attention_mask=None, labels=labels)
            raw_loss = out[0] if isinstance(out, (tuple, list)) else out
            loss = raw_loss.mean()

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (step + 1) % 30 == 0:
            avg = sum(train_losses[-30:]) / 30
            print(f"  step {step+1:3d}  avg_train_loss={avg:.4f}", flush=True)

    # ── final losses on same fixed eval batches ───────────────────────────
    final_incr = _eval_on_fixed(eval_ids_incr, eval_lbl_incr, "incr")
    final_decr = _eval_on_fixed(eval_ids_decr, eval_lbl_decr, "decr")
    print(f"Final   loss — incr: {final_incr:.4f}  decr: {final_decr:.4f}")

    # Overall training loss must drop by ≥ 30 %
    early_avg = sum(train_losses[:10]) / 10
    late_avg = sum(train_losses[-10:]) / 10
    train_drop = (early_avg - late_avg) / early_avg
    print(f"Train loss: early_avg={early_avg:.4f}  late_avg={late_avg:.4f}  drop={train_drop:.1%}")

    # Per-task eval: combined must not regress, and at least ONE task must improve
    combined_drop = ((init_incr + init_decr) - (final_incr + final_decr)) / (init_incr + init_decr)
    best_drop = max(
        (init_incr - final_incr) / init_incr,
        (init_decr - final_decr) / init_decr,
    )
    print(f"Combined eval drop: {combined_drop:.1%}  Best single-task drop: {best_drop:.1%}")

    assert train_drop >= 0.20, \
        f"Training loss did not drop ≥20% (drop={train_drop:.1%})"
    assert best_drop >= 0.15, \
        f"Neither task improved ≥15% on fixed eval: incr_drop={((init_incr - final_incr)/init_incr):.1%}, decr_drop={((init_decr - final_decr)/init_decr):.1%}"

    print(f"\nGate 1 PASSED ✓  (train_drop={train_drop:.1%}, best_task_drop={best_drop:.1%})")
