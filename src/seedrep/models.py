"""Loading a checkpoint as a merged model: base + adapter chain.

The original stages are reconstructed with ``STAGE_ADAPTER_CHAINS``; a new
seed checkpoint is the M1 (or M1_alt) chain followed by one local adapter
directory. Adapters are merged one at a time with ``merge_and_unload``, the
same way ``src.training.model.load_stage_model`` does it, so a re-extraction
of an original stage reproduces the stored activations.

``base_name`` is a parameter rather than a constant so the CPU smoke test can
run the identical code path on a tiny randomly initialised Qwen2 model.
"""
from __future__ import annotations

from src.training.model import STAGE_ADAPTER_CHAINS

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
START_STAGE = {"alpaca": "M1", "dolly": "M1_alt"}


def default_dtype(device: str):
    import torch

    # T4 has no bf16; training ran in fp16, CPU stays fp32 (as load_stage_model)
    return torch.float16 if device.startswith("cuda") else torch.float32


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(base_name: str = BASE_MODEL):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def chain_for(stage: str | None = None, adapters: list[str] | None = None,
              branch: str | None = None) -> list[str]:
    """Adapter chain for an original stage, or for ``start(branch) + adapters``."""
    if stage is not None:
        if stage not in STAGE_ADAPTER_CHAINS:
            raise ValueError(f"unknown stage {stage!r}")
        return list(STAGE_ADAPTER_CHAINS[stage])
    if branch not in START_STAGE:
        raise ValueError(f"branch must be one of {sorted(START_STAGE)}, got {branch!r}")
    return list(STAGE_ADAPTER_CHAINS[START_STAGE[branch]]) + list(adapters or [])


def load_merged(chain: list[str], base_name: str = BASE_MODEL, device: str | None = None,
                dtype=None):
    """Base model with every adapter in ``chain`` merged in order, in eval mode."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device = device or pick_device()
    dtype = dtype or default_dtype(device)
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
    model = model.to(device)
    for adapter in chain:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model


def free(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
