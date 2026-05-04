"""
Gate 0 — Basic LoRA training smoke test on Qwen3-0.6B, 1 GPU.

Verifies that mcore-bridge can:
  1. Init Megatron-Core (TP=1, PP=1)
  2. Load Qwen3-0.6B weights (HF → Megatron via bridge.load_weights)
  3. Inject LoRA on all linear modules via PEFT + dispatch_megatron
  4. Forward pass on a synthetic batch
  5. Backward pass
  6. Adam optimizer step
  7. Save PEFT checkpoint (bridge.save_weights peft_format=True)
  8. Reload into a fresh model and assert weights are bit-identical

Run with:
    torchrun --nproc-per-node=1 -m pytest tests/multi_lora/test_gate0_smoke.py -v -s
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

MODEL_ID = "/tmp/qwen3-0.6b"
LORA_RANK = 8
LORA_ALPHA = 16
# All TE linear modules in a standard Qwen3 transformer layer
LORA_TARGET_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]


# ── helpers ─────────────────────────────────────────────────────────────────

def _init_megatron():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )


def _make_lora_config():
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )


def _build(apply_lora: bool = True):
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
    if apply_lora:
        mg_model = get_peft_model(mg_model, _make_lora_config())
        mg_models[0] = mg_model
    return mg_models, bridge, config


# ── test ────────────────────────────────────────────────────────────────────

def test_gate0_lora_training_smoke():
    # Step 1: init Megatron-Core distributed
    _init_megatron()
    _patcher.apply_patch()

    # Step 2: build model and load base weights
    mg_models, bridge, config = _build(apply_lora=False)
    bridge.load_weights(mg_models, MODEL_ID)

    # Step 3: inject LoRA on all linear modules
    mg_model = get_peft_model(mg_models[0], _make_lora_config())
    mg_models[0] = mg_model
    mg_model.train()

    # Step 4 & 5: forward + backward on a synthetic batch
    vocab_size = config.padded_vocab_size
    seq_len, batch = 32, 2
    device = torch.device("cuda")
    input_ids = torch.randint(0, vocab_size, (seq_len, batch), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    labels = torch.roll(input_ids, -1, dims=0)

    # attention_mask=None lets Megatron build the causal mask internally.
    # GPTModel with labels returns per-token loss [seq, batch] as first element.
    output = mg_model(input_ids=input_ids, position_ids=position_ids, attention_mask=None, labels=labels)
    raw_loss = output[0] if isinstance(output, (tuple, list)) else output
    loss = raw_loss.mean()
    loss.backward()

    # Step 6: Adam step on LoRA params only
    lora_params = [p for p in mg_model.parameters() if p.requires_grad]
    assert lora_params, "No trainable LoRA parameters found — LoRA injection likely failed"
    optimizer = torch.optim.Adam(lora_params, lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()

    print(f"\nStep 1-6 passed. loss={loss.item():.4f}, "
          f"trainable params={sum(p.numel() for p in lora_params):,}")

    # Step 7: save PEFT checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        bridge.save_weights(mg_models, tmpdir, peft_format=True, adapter_name="default")

        # snapshot saved LoRA weights from the trained model
        saved = {
            k: v.clone().cpu()
            for k, v in mg_model.state_dict().items()
            if "lora_A" in k or "lora_B" in k
        }
        assert saved, "No lora_A/lora_B keys in state_dict — save likely wrote nothing"
        print(f"Saved {len(saved)} LoRA tensors to {tmpdir}")

        # Step 8: fresh model + reload → bit-identical
        mg_models2, bridge2, _ = _build(apply_lora=True)
        bridge2.load_weights(mg_models2, tmpdir, peft_format=True, adapter_name="default")

        reloaded = {
            k: v.cpu()
            for k, v in mg_models2[0].state_dict().items()
            if "lora_A" in k or "lora_B" in k
        }
        assert set(saved) == set(reloaded), (
            f"Key mismatch after reload.\n"
            f"  saved keys: {sorted(saved)[:5]}\n"
            f"  reloaded keys: {sorted(reloaded)[:5]}"
        )
        for k in saved:
            assert torch.equal(saved[k], reloaded[k]), f"Weight mismatch at {k}"

    print("Gate 0 PASSED ✓")
