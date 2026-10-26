"""Behavioural levels and training-data facts for the matched-start comparison.

    python -m src.stagecmp.behaviour_levels

Writes ``results/stagecmp/behaviour_levels.json`` (+ binding):

* absolute WildGuard refusal and harmful-response rates per quadrant and
  checkpoint, with Wilson intervals (the recipe *terms* already exist in
  ``stage_comparison_wg_refusal.json``; this adds the levels they move from);
* the cross-fitted direction-ablation effect for the four DPO checkpoints,
  copied from ``final_token_repair/summaries/final_token_endpoints.json``;
* an audit of the safety-SFT targets: what share are refusals by the frozen
  keyword classifier (which under-counts refusals, so a lower bound), lengths,
  and confirmation that every SFT target is the DPO chosen response.
"""
from __future__ import annotations

import datetime as _dt
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.common.refusal_classifier import classify_completion
from src.stagecmp import STATUS_EXPLORATORY
from src.stagecmp.loaders import behavior_rows
from src.stagecmp.run_wp5 import JUDGE_COMPLETE
from src.v2_io import sha256_file, write_json_lf

OUT = Path("results/stagecmp/behaviour_levels.json")
ENDPOINTS = Path("results/final_token_repair/summaries/final_token_endpoints.json")
SFT_DATA = Path("data/processed/sft_safety.jsonl")
DPO_DATA = Path("data/processed/dpo_pairs.jsonl")
STAGES = ("M1", "M2", "M3_direct", "M1_alt", "M2_alt", "M3_direct_alt", "M3", "M3_alt")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def wilson(k: int, n: int, z: float = 1.959964) -> dict:
    if n == 0:
        return {"rate": None, "ci_low": None, "ci_high": None, "k": 0, "n": 0}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"rate": p, "ci_low": float(centre - half), "ci_high": float(centre + half),
            "k": int(k), "n": int(n)}


def absolute_rates() -> dict:
    out = {}
    for stage in STAGES:
        rows = behavior_rows(stage, JUDGE_COMPLETE).values()
        cell = {}
        for q in "ABCD":
            wg = [r["wildguard"] for r in rows
                  if r["quadrant"] == q and isinstance(r.get("wildguard"), dict)]
            ref = [w["response_refusal"] for w in wg if w.get("response_refusal") is not None]
            harm = [w["response_harm"] for w in wg if w.get("response_harm") is not None]
            cell[q] = {"refusal": wilson(sum(ref), len(ref)), "harmful_response": wilson(sum(harm), len(harm))}
        out[stage] = cell
    return out


def ablation() -> dict:
    data = json.loads(ENDPOINTS.read_text(encoding="utf-8"))["CF2_by_stage"]
    keep = ("cf2", "ci_low", "ci_high", "E_AD", "E_random", "n_effective_triples", "population")
    return {stage: {k: v["cross_fitted"].get(k) for k in keep} for stage, v in data.items()}


def sft_audit() -> dict:
    def read(path):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    sft, dpo = read(SFT_DATA), read(DPO_DATA)
    classes = Counter(classify_completion(r["response"]) for r in sft)
    rejected = Counter(classify_completion(r["rejected"]) for r in dpo)
    n = len(sft)
    return {
        "n_examples": n,
        "n_distinct_prompts": len({r["prompt"] for r in sft}),
        "sft_target_equals_dpo_chosen": sum(a["response"] == b["chosen"] and a["prompt"] == b["prompt"]
                                            for a, b in zip(sft, dpo)),
        "chosen_classes_keyword": {k: v / n for k, v in classes.items()},
        "rejected_classes_keyword": {k: v / n for k, v in rejected.items()},
        "chosen_words_median": float(np.median([len(r["response"].split()) for r in sft])),
        "rejected_words_median": float(np.median([len(r["rejected"].split()) for r in dpo])),
        "note": ("keyword classifier (src.common.refusal_classifier) under-counts refusals "
                 "relative to WildGuard on model outputs (base rate 0.101 vs 0.265), so the "
                 "refusal share is a lower bound"),
    }


def main() -> int:
    payload = {
        "status": STATUS_EXPLORATORY,
        "produced_utc": _now(),
        "judge_file": str(JUDGE_COMPLETE),
        "absolute_rates_wildguard": absolute_rates(),
        "ablation_cross_fitted_strongreject_A": ablation(),
        "sft_data_audit": sft_audit(),
    }
    write_json_lf(OUT, payload)
    write_json_lf(OUT.with_name(OUT.stem + "_binding.json"), {
        "status": STATUS_EXPLORATORY, "produced_utc": _now(),
        "sha256": {str(p): sha256_file(p) for p in (JUDGE_COMPLETE, ENDPOINTS, SFT_DATA, DPO_DATA)},
    })
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
