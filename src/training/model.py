import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel


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
    """
    Loads the base model. If cfg["model"]["init_from_adapter"] is set (an HF
    Hub repo id), loads that adapter on top of the base weights and merges it
    in before returning - this is how M2 is initialized from M1, and later
    M3 from M2. The returned model is always a plain dense model; the
    trainer attaches a FRESH LoRA adapter for the new stage via
    create_lora_config + get_peft_model. Each stage gets its own fresh LoRA
    delta relative to the merged state of the prior stage - not LoRA-on-LoRA.
    """
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        dtype=torch.float16,
        trust_remote_code=cfg["model"]["trust_remote_code"],
    )

    init_from_adapter = cfg["model"].get("init_from_adapter")
    if init_from_adapter:
        print(f"Merging prior-stage adapter into base model: {init_from_adapter}")
        model = PeftModel.from_pretrained(model, init_from_adapter)
        model = model.merge_and_unload()

    return model