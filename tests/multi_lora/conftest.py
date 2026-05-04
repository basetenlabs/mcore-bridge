"""Shared fixtures and helpers for multi-LoRA tests."""
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import List, Optional, Tuple

from megatron.core import parallel_state
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig

from mcore_bridge.config import ModelConfig
from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.register import get_mcore_model

MODEL_ID = "/tmp/qwen3-0.6b"
MODEL_ID_MOE = "/tmp/qwen3-35b-a3b"
LORA_RANK = 4
LORA_ALPHA = 8
LORA_TARGET = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]


# ── Process group helpers ────────────────────────────────────────────────────

def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def init_megatron(tp: int = 1, pp: int = 1, cp: int = 1, ep: int = 1) -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            expert_model_parallel_size=ep,
        )


# ── Model / LoRA construction ────────────────────────────────────────────────

def make_lora_config() -> LoraConfig:
    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET,
                      lora_dropout=0.0, bias="none")


def build_model(tp: int = 1, pp: int = 1, cp: int = 1, ep: int = 1,
                recompute: bool = False, extra_cfg: Optional[dict] = None,
                model_id: str = MODEL_ID):
    """Build a PEFT-wrapped Megatron model for the given parallelism config.

    Returns (mg_models, bridge, config).
    """
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cfg_dict = hf_to_mcore_config(hf_config)
    cfg_dict.update(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    if recompute:
        cfg_dict["recompute_granularity"] = "selective"
    if extra_cfg:
        cfg_dict.update(extra_cfg)
    config = ModelConfig(**cfg_dict)
    mg_models = get_mcore_model(config)
    bridge = config.bridge
    mg_models[0] = mg_models[0].cuda()
    bridge.load_weights(mg_models, model_id)
    mg_models[0] = get_peft_model(mg_models[0], make_lora_config())
    return mg_models, bridge, config


def build_moe_model(ep: int = 2, tp: int = 1, recompute: bool = False,
                    extra_cfg: Optional[dict] = None):
    """Build a PEFT-wrapped Megatron MoE model (Qwen3.5-35B-A3B) for EP tests.

    Returns (mg_models, bridge, config).
    """
    return build_model(tp=tp, ep=ep, recompute=recompute, extra_cfg=extra_cfg,
                       model_id=MODEL_ID_MOE)


# ── Batch / data helpers ─────────────────────────────────────────────────────

def synthetic_task(kind: str, seq_len: int = 16, vocab_size: int = 64):
    """Return (input_ids, labels) for a simple synthetic token task.

    kind: 'increment', 'decrement', or 'double'.
    """
    if kind == 'increment':
        tokens = torch.randint(0, vocab_size - 1, (seq_len,))
        labels = (tokens + 1) % vocab_size
    elif kind == 'decrement':
        tokens = torch.randint(1, vocab_size, (seq_len,))
        labels = (tokens - 1) % vocab_size
    elif kind == 'double':
        tokens = torch.randint(0, vocab_size // 2, (seq_len,))
        labels = (tokens * 2) % vocab_size
    else:
        raise ValueError(f'Unknown task kind: {kind!r}')
    return tokens, labels


def make_batch(tasks: List[str], seq_len: int, vocab_size: int, seed: int):
    """Stack per-task (input_ids, labels) into a single GPU batch.

    Returns (input_ids, labels) each of shape (seq_len, batch).
    """
    inputs, labels_list = [], []
    torch.manual_seed(seed)
    for task in tasks:
        t, l = synthetic_task(task, seq_len=seq_len, vocab_size=vocab_size)
        inputs.append(t)
        labels_list.append(l)
    return torch.stack(inputs, dim=1).cuda(), torch.stack(labels_list, dim=1).cuda()


def make_position_ids(config, seq_len: int, batch: int) -> torch.Tensor:
    """Return position_ids with the shape expected by the model.

    Standard models: [batch, seq_len].
    MRoPE models (e.g. Qwen3.5-35B-A3B): [3, batch, seq_len] — the three
    components correspond to the mrope_section axes; for text-only use they
    are identical.
    """
    pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1)
    if getattr(config, 'position_embedding_type', None) == 'mrope':
        pos = pos.unsqueeze(0).expand(3, -1, -1)
    return pos


def eval_loss(mg_models, bridge, adapter_name: Optional[str],
              seq_len: int, batch: int, vocab_size: int, task: str,
              config=None) -> float:
    """Single no-grad forward pass; returns mean cross-entropy loss."""
    input_ids, labels = make_batch([task] * batch, seq_len, vocab_size, seed=999)
    if config is None:
        pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1)
    else:
        pos = make_position_ids(config, seq_len, batch)
    with bridge.set_routing(mg_models, [adapter_name] * batch):
        with torch.no_grad():
            out = mg_models[0](input_ids=input_ids, position_ids=pos,
                               attention_mask=None, labels=labels)
    raw = out[0] if isinstance(out, (tuple, list)) else out
    return raw.mean().item()


def lora_weight(module: nn.Module) -> torch.Tensor:
    """Extract the weight tensor from a lora_A / lora_B module."""
    return module.weight if hasattr(module, 'weight') else next(module.parameters())


# ── Stub helpers ─────────────────────────────────────────────────────────────

def stub_adapter_pair(r: int, in_features: int, out_features: int) -> Tuple[nn.Linear, nn.Linear]:
    """Two nn.Linear layers representing lora_A and lora_B for CPU tests."""
    lora_A = nn.Linear(in_features, r, bias=False)
    lora_B = nn.Linear(r, out_features, bias=False)
    nn.init.kaiming_uniform_(lora_A.weight)
    nn.init.zeros_(lora_B.weight)
    return lora_A, lora_B


# ── Pytest markers ───────────────────────────────────────────────────────────

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='requires CUDA GPU',
)
