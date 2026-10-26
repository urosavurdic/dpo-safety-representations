"""A tiny stand-in for the real models, so the whole worker runs on a CPU.

    python -m src.seedrep.smoke [--fault train:3]

Builds, under ``.seedrep_cache/smoke/``:

* a 4-layer Qwen2 model with random weights and the real Qwen2.5 tokenizer;
* a random LoRA adapter playing the start checkpoint (M1), another playing an
  original SFT arm (for the unmerged-path parity check), a parity fixture made
  with the *merged* path, and random injection vectors;

then runs the real job table (seed 1, tiny dose schedules) through
``worker.run`` against a local bank. Training, resume, measuring, generation,
judging (a stub scorer instead of the 7B judges) and injection all go through
the same code the GPU run uses. ``tests/seedrep`` drives it, including runs
that are killed partway and resumed, and a resume on a fresh "platform".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.seedrep import extract, models, rows
from src.seedrep.jobs import build_jobs, partition

SMOKE = rows.ROOT / ".seedrep_cache" / "smoke"
LAYERS = (1, 2, 3, 4)


def _tiny_model(out: Path) -> Path:
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    if (out / "config.json").exists():
        return out
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(models.BASE_MODEL)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    return out


def _random_adapter(base: Path, out: Path, seed: int) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if (out / "adapter_model.safetensors").exists():
        return out
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(base)
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear",
                                            task_type="CAUSAL_LM"))
    for name, param in peft.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    peft.save_pretrained(out)
    return out


def select_rows(per_quadrant: int = 3) -> list[dict]:
    """A few rows per quadrant, taking estimation rows for A and D first."""
    bench = rows.benchmark_rows()
    picked = []
    for q in "ABCD":
        cand = [r for r in bench if r["quadrant"] == q]
        cand.sort(key=lambda r: r.get("split") != "direction_estimation")
        picked += cand[:per_quadrant]
    return picked


def build(per_quadrant: int = 3):
    """Tiny artifacts plus the worker ``Settings`` that point at them."""
    from src.seedrep.worker import Settings

    SMOKE.mkdir(parents=True, exist_ok=True)
    base = _tiny_model(SMOKE / "tiny_base")
    m1 = _random_adapter(base, SMOKE / "tiny_m1", seed=1)
    sft = _random_adapter(base, SMOKE / "tiny_sft", seed=2)
    picked = select_rows(per_quadrant)

    fixture = SMOKE / "parity_fixture.npz"
    if not fixture.exists():
        tok = models.load_tokenizer(str(base))
        arrays = {}
        for stage, chain in (("M1", [str(m1)]), ("M2", [str(m1), str(sft)])):
            model = models.load_merged(chain, base_name=str(base), device="cpu")
            f, _, _ = extract.forward_batch(model, tok, [r["prompt"] for r in picked[:4]], "cpu",
                                            layers=list(LAYERS), want_logprobs=False)
            arrays[f"{stage}_final"] = f
            models.free(model)
        np.savez(fixture, **arrays)
    vectors = SMOKE / "inject_vectors.npz"
    if not vectors.exists():
        rng = np.random.default_rng(0)
        np.savez(vectors, **{f"{b}_{c}": rng.normal(size=32).astype(np.float32)
                             for b in ("alpaca", "dolly")
                             for c in ("old_dir", "orth", "mean_shift", "random")})

    def scorer(rec):
        refused = "sorry" in (rec.get("response") or "").lower()
        rec["strong_reject"] = {"score": 0.0 if refused else 0.5, "malformed": False,
                                "judge_status": "scored"}
        rec["wildguard"] = {"prompt_harm": rec["quadrant"] in ("A", "C"),
                            "response_refusal": refused, "response_harm": False,
                            "malformed": False, "judge_status": "scored"}

    settings = Settings(
        base_name=str(base),
        start_chains={"alpaca": [str(m1)], "dolly": [str(m1)]},
        parity_adapters={"alpaca": ("M2", str(sft)), "dolly": ("M2", str(sft))},
        orig_adapters={"M2": str(sft), "M3_direct": str(sft), "M2_alt": str(sft),
                       "M3_direct_alt": str(sft)},
        layers=LAYERS, n_pairs=4, judge_scorer=scorer, load_4bit=False, batch=4,
        inject_vectors=vectors, parity_fixture=fixture, inject_layer=2, rows_override=picked,
        train_overrides={
            "model": {"name": str(base), "init_from_adapter": str(m1), "trust_remote_code": False},
            "training": {"batch_size": 2, "gradient_accumulation_steps": 2, "max_seq_length": 96,
                         "gradient_checkpointing": False, "logging_steps": 1, "fp16": False},
            "dataset": {"max_samples": 16},
            "_resume_every": 2,
        },
    )
    return settings


def smoke_jobs():
    return build_jobs(priorities=("P1", "P2", "P4"), sft_dose=(2, 4, 6), dpo_dose=(1, 2, 3),
                      seeds=(1,), inject_conditions=("baseline", "orth"))


def run_smoke(bank_dir: Path, fault: str | None = None, log=print) -> dict:
    from src.seedrep.bank import LocalBank
    from src.seedrep.worker import run

    if fault:
        os.environ["SEEDREP_FAULT_AT"] = fault
    else:
        os.environ.pop("SEEDREP_FAULT_AT", None)
    settings = build()
    queue = partition(smoke_jobs(), None, 1)
    return run(LocalBank(bank_dir), queue, settings, deadline_minutes=None, wait_seconds=1, log=log)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=SMOKE / "bank")
    ap.add_argument("--fault", default=None)
    args = ap.parse_args()
    result = run_smoke(args.bank, args.fault)
    print(json.dumps(result, indent=1))
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
