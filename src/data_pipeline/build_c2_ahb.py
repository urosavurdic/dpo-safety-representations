"""
Quadrant C secondary set: C2 (stylistic_displacement)
Source: AHB — Adversarial Harm Benchmark (icaro-lab/ahb on HuggingFace)
Paper: arXiv:2604.18487

Place this file at:
    src/data_pipeline/build_c2_ahb.py

Run from repo root as:
    python -m src.data_pipeline.build_c2_ahb

Output:
    data/quadrant_c_pipeline/secondary_c2_stylistic.jsonl
"""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

# Three levels up: src/data_pipeline/build_c2_ahb.py -> src/data_pipeline -> src -> repo root
REPO = Path(__file__).parent.parent.parent

C2_OUT = REPO / "data/quadrant_c_pipeline/secondary_c2_stylistic.jsonl"

TRAINING_FILES = {
    "sft_helpful.jsonl":     REPO / "data/processed/sft_helpful.jsonl",
    "sft_helpful_alt.jsonl": REPO / "data/processed/sft_helpful_alt.jsonl",
    "sft_safety.jsonl":      REPO / "data/processed/sft_safety.jsonl",
    "dpo_pairs.jsonl":       REPO / "data/processed/dpo_pairs.jsonl",
}

# AILuminate hazard codes — hard exclusions regardless of literary style.
# cse = child sexual exploitation, src = sex-related crimes.
EXCLUDED_HAZARD_CODES = {"cse", "src"}
EXCLUDED_HAZARD_NAMES = {
    "child sexual exploitation",
    "sex-related crimes",
    "sexual content involving minors",
}

PER_STYLE_CAP = 6  # max records per reformulation style


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha256("||".join(parts).encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


def load_train_prompts():
    result = {}
    for label, path in TRAINING_FILES.items():
        if not path.exists():
            print(f"  [skip] {label} not found")
            result[label] = None
            continue
        prompts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "prompt" in row:
                    prompts.append(row["prompt"])
        print(f"  {label}: {len(prompts)} prompts")
        result[label] = prompts
    return result


def exact_contamination_check(texts, train_prompts_by_label):
    results = {}
    for label, train_prompts in train_prompts_by_label.items():
        if train_prompts is None:
            results[label] = "unknown"
            continue
        train_normed = {normalize(p) for p in train_prompts}
        exact_hits = [t for t in texts if normalize(t) in train_normed]
        results[label] = {
            "exact": exact_hits,
            "near": "unknown_run_complete_neardup_check",
        }
    return results


def is_excluded_hazard(row, hazard_field):
    raw = str(row.get(hazard_field, "")).lower().strip()
    if any(raw == code or raw.startswith(code + "_") or raw.startswith(code + "/")
           for code in EXCLUDED_HAZARD_CODES):
        return True
    if any(name in raw for name in EXCLUDED_HAZARD_NAMES):
        return True
    return False


def pick_field(all_fields, *candidates):
    for c in candidates:
        if c in all_fields:
            return c
    return None


def main():
    print("Loading icaro-lab/ahb from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed.  Run:  pip install datasets")
        sys.exit(1)

    ds = load_dataset("icaro-lab/ahb", split="test")
    print(f"Loaded {len(ds)} rows.")

    print("\n=== Dataset schema ===")
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k!r}: {repr(str(v)[:120])}")

    all_fields = set(ds.column_names)
    hazard_field    = pick_field(all_fields, "hazard", "hazard_category", "category", "harm_category")
    style_field     = pick_field(all_fields, "method", "style", "reformulation_style", "transformation_type", "literary_style")
    rewritten_field = pick_field(all_fields, "prompt", "rewritten_prompt", "reformulated_prompt", "adversarial_prompt")
    original_field  = pick_field(all_fields, "original_prompt", "original", "source_prompt", "base_prompt")

    print(f"\nDetected fields:")
    print(f"  hazard    -> {hazard_field}")
    print(f"  style     -> {style_field}")
    print(f"  rewritten -> {rewritten_field}")
    print(f"  original  -> {original_field}")

    missing = [n for n, v in [("hazard", hazard_field), ("style", style_field), ("rewritten", rewritten_field)] if v is None]
    if missing:
        print(f"\nERROR: Could not auto-detect fields: {missing}")
        print("Available fields:", sorted(all_fields))
        print("Update the pick_field() calls in src/data_pipeline/build_c2_ahb.py and rerun.")
        sys.exit(1)

    print("\nLoading training files for contamination check...")
    train_prompts = load_train_prompts()

    print("\nBuilding C2 records...")
    records = []
    counted_by_style = defaultdict(int)
    skipped_hazard = 0
    skipped_cap = 0

    for row in ds:
        if hazard_field and is_excluded_hazard(row, hazard_field):
            skipped_hazard += 1
            continue

        style    = str(row.get(style_field, "unknown")).strip()
        hazard   = str(row.get(hazard_field, "unknown")).strip()
        rewritten = str(row.get(rewritten_field, "")).strip()
        original  = str(row.get(original_field, "")).strip() if original_field else ""

        if not rewritten:
            continue
        if counted_by_style[style] >= PER_STYLE_CAP:
            skipped_cap += 1
            continue

        texts_to_check = [t for t in [rewritten, original] if t]
        contam = exact_contamination_check(texts_to_check, train_prompts)
        contaminated = any(
            isinstance(v, dict) and v["exact"]
            for v in contam.values()
        )

        rec = {
            "candidate_id": stable_id("AHB", rewritten),
            "source_dataset": "AHB",
            "source_version": "icaro-lab/ahb, split=test (arXiv:2604.18487)",
            "source_id": stable_id("AHB", rewritten),
            "source_category": hazard,
            "project_category": None,
            "transformation_family": "stylistic_displacement",
            "reformulation_style": style,
            "source_prompt": rewritten,
            "original_prompt": original if original else None,
            "source_url": "https://huggingface.co/datasets/icaro-lab/ahb",
            "source_validation_type": "published_academic_benchmark",
            "training_contamination": {
                k: ("unknown" if v == "unknown" else {"exact": len(v["exact"]), "near": v["near"]})
                for k, v in contam.items()
            },
            "agent_pre_screen": "exclude" if contaminated else "secondary_only",
            "agent_reason": (
                "flagged in exact-match training-data contamination check" if contaminated
                else "C2 is secondary/exploratory — not eligible for C1 promotion without explicit review"
            ),
            "needs_review": True,
        }
        records.append(rec)
        counted_by_style[style] += 1

    C2_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(C2_OUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n=== C2 done ===")
    print(f"  Records written:    {len(records)}")
    print(f"  By style:           {dict(counted_by_style)}")
    print(f"  Skipped (hazard):   {skipped_hazard}")
    print(f"  Skipped (cap):      {skipped_cap}")
    print(f"  Contaminated/excl:  {sum(1 for r in records if r['agent_pre_screen'] == 'exclude')}")
    print(f"  Output:             {C2_OUT}")
    print(f"\nNext: python -m src.diagnostics.complete_neardup_check")


if __name__ == "__main__":
    main()