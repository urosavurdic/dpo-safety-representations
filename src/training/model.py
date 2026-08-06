import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig


def load_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name"],
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def create_lora_config(cfg):
    return LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def load_model(cfg):
    # Explicit fp16, not dtype="auto" - Qwen2.5 checkpoints default to
    # bf16, which T4 (Turing, no native bf16 tensor cores) handles poorly.
    return AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        dtype=torch.float16,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )