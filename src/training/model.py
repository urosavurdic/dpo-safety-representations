import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel


STAGE_ADAPTER_CHAINS = {
    "M0": [],
    "M1": ["urosavurdic/qwen2.5-1.5b-m1-helpful"],
    "M2": ["urosavurdic/qwen2.5-1.5b-m1-helpful", "urosavurdic/qwen2.5-1.5b-m2-safety"],
    "M3": [
        "urosavurdic/qwen2.5-1.5b-m1-helpful",
        "urosavurdic/qwen2.5-1.5b-m2-safety",
        "urosavurdic/qwen2.5-1.5b-m3-dpo",
    ],
    "M3_direct": [
        "urosavurdic/qwen2.5-1.5b-m1-helpful",
        "urosavurdic/qwen2.5-1.5b-m3-direct-dpo",
    ],
    # Data-dependence branch: M1_alt trained on Dolly-15k instead of Alpaca;
    # M2_alt/M3_alt/M3_direct_alt all keep training on the SAME PKU-SafeRLHF
    # data as M2/M3/M3_direct (single-variable design - only M1's source data
    # differs). See configs/m*_alt_*.yaml and PROJECT_CONTEXT.md M1_alt notes.
    "M1_alt": ["urosavurdic/qwen2.5-1.5b-m1-alt-helpful"],
    "M2_alt": [
        "urosavurdic/qwen2.5-1.5b-m1-alt-helpful",
        "urosavurdic/qwen2.5-1.5b-m2-alt-safety",
    ],
    "M3_alt": [
        "urosavurdic/qwen2.5-1.5b-m1-alt-helpful",
        "urosavurdic/qwen2.5-1.5b-m2-alt-safety",
        "urosavurdic/qwen2.5-1.5b-m3-alt-dpo",
    ],
    "M3_direct_alt": [
        "urosavurdic/qwen2.5-1.5b-m1-alt-helpful",
        "urosavurdic/qwen2.5-1.5b-m3-direct-alt-dpo",
    ],
}


def load_stage_model(stage_name: str, dtype=None):
    """
    Reconstructs the ACTUAL trained state of a given stage (M0-M3) by
    replaying the same adapter-merge cascade used during training. Each
    stage's saved adapter is a delta relative to the PREVIOUS stage's merged
    weights, not the raw base model.

    Auto-detects device: CUDA + fp16 when a GPU is available (matches
    training precision - T4 has no native bf16 support), CPU + fp32
    otherwise (fp16 has poor CPU kernel support). Originally written
    CPU-only for the local qualitative check; reused unmodified on Colab it
    silently stayed on CPU even with a GPU present - this fixes that.
    """
    if stage_name not in STAGE_ADAPTER_CHAINS:
        raise ValueError(f"Unknown stage: {stage_name}. Expected one of {list(STAGE_ADAPTER_CHAINS)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=dtype)
    model = model.to(device)
    for adapter_repo in STAGE_ADAPTER_CHAINS[stage_name]:
        model = PeftModel.from_pretrained(model, adapter_repo)
        model = model.merge_and_unload()
    model.eval()
    return model


def try_load_stage_model(stage_name: str, dtype=None):
    """
    Same as load_stage_model, but returns None (with a printed warning)
    instead of raising if any adapter in the chain hasn't been pushed to HF
    yet - e.g. urosavurdic/qwen2.5-1.5b-m2-alt-safety doesn't exist until
    M2_alt has actually been really trained AND pushed.

    Needed once STAGES lists span multiple branches that train and push
    independently across separate Colab sessions (the alt branch, on a
    Drive-space-constrained schedule) - a batch loop over 9 stages
    shouldn't die on stage 6 just because stage 6 isn't ready yet, losing
    the results already produced for stages 1-5 in the same run.

    Used by eval_behavioral.py / eval_extract_activations.py (the two
    scripts that create model-derived results from scratch); the CPU-only
    scripts that consume already-extracted activations (eval_probes.py,
    eval_refusal_direction.py, etc.) don't need this - they check for the
    activation FILE's existence instead, which is cheaper and doesn't need
    a network round-trip.
    """
    try:
        return load_stage_model(stage_name, dtype=dtype)
    except Exception as e:
        print(f"  [{stage_name}] SKIPPED - could not load: {type(e).__name__}: {e}")
        print(f"  [{stage_name}] (this usually means one of {STAGE_ADAPTER_CHAINS.get(stage_name)} "
              "hasn't been pushed to HF yet - not necessarily an error)")
        return None

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