"""Producer for results/summaries/mcnemar_direction_specificity.json (audit P14 / CPU-2).

The committed summary came from an uncommitted Colab bundle (commit b7cb9bd) with
no producer and no binding sidecar, and its M3 quadrant-C p (8.9e-7) matched no
standard McNemar variant. This script regenerates every cell **from the committed
raw causal files** with the in-repo regex classifier, and reports the McNemar
statistic under one documented variant: the **two-sided exact binomial** test on
the discordant pairs (``scipy.stats.binomtest(min(b, c), b + c, 0.5,
alternative="two-sided")``), which is the exact McNemar test. ``b`` = discordant
pairs where ``ablated_AD`` flagged the category and ``ablated_random`` did not;
``c`` = the reverse.

Held-out quadrant C/D come from the frozen ``causal_ablation_v2_{stage}_L24-28.json``
files; quadrant A (n=150, sensitivity) from the ``_fullAD`` files. Pooling of the
underlying direction is the mean-pooled (last-5-token) contrast the whole causal
core uses (see the manuscript's deviations table) - this test reads only the
generated response text, not the direction, so it is pooling-agnostic in practice.

CPU only. Writes the summary JSON + a ``*_binding.json`` sidecar pinning the raw
files, their sha256, the classifier module commit, and this script's commit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from scipy.stats import binomtest

from src.analysis.summarize_causal_ablation import classify_completion

RAW = Path("results/raw")
OUT = Path("results/summaries/mcnemar_direction_specificity.json")
BRANCHES = ("M3", "M3_alt", "M3_direct", "M3_direct_alt")


def _git_commit(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", path], text=True
        ).strip() or "uncommitted"
    except Exception:
        return "unknown"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def paired_counts(rows, quadrant, category, ad_cond, rand_cond):
    """b/c discordant counts for `category` under ad_cond vs rand_cond on `quadrant`."""
    by_prompt = defaultdict(dict)
    for row in rows:
        if row.get("quadrant") != quadrant:
            continue
        by_prompt[row["prompt"]][row.get("stage")] = (
            classify_completion(row["response"]) == category
        )
    b = c = n = 0
    for conds in by_prompt.values():
        if ad_cond not in conds or rand_cond not in conds:
            continue
        n += 1
        ad, rd = conds[ad_cond], conds[rand_cond]
        if ad and not rd:
            b += 1
        elif rd and not ad:
            c += 1
    return b, c, n


def mcnemar_exact(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)


def cell(rows, stage, quadrant, category, infix=""):
    ad = f"{stage}_{infix}ablated_AD"
    rand = f"{stage}_{infix}ablated_random"
    b, c, n = paired_counts(rows, quadrant, category, ad, rand)
    return {"b": b, "c": c, "n_discordant": b + c, "n_paired": n,
            "p_two_sided_exact_binomial": round(mcnemar_exact(b, c), 10)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    raw = Path(args.raw_dir)

    report = {
        "producer": "src/analysis/mcnemar_direction_specificity.py",
        "test": "two-sided exact binomial on discordant pairs (exact McNemar)",
        "b_def": "flagged the category under ablated_AD only",
        "c_def": "flagged the category under ablated_random only; c >> b => "
                 "ablating the learned direction suppresses the category more "
                 "than a magnitude-matched random ablation (direction-specific)",
        "direction_pooling": "mean-pooled last-5 (the causal-core contrast); this "
                             "test reads response text only, not the direction",
        "quadrant_C_soft_deflection_n104": {},
        "quadrant_A_refusal_n150_SENSITIVITY": {},
        "quadrant_D_refusal_n150_SENSITIVITY": {},
    }
    used = {}
    for st in BRANCHES:
        ho = raw / f"causal_ablation_v2_{st}_L24-28.json"
        ad = raw / f"causal_ablation_v2_{st}_L24-28_fullAD.json"
        ho_rows = json.loads(ho.read_text(encoding="utf-8"))
        report["quadrant_C_soft_deflection_n104"][st] = cell(
            ho_rows, st, "C", "soft_deflection")
        used[ho.name] = _sha(ho)
        if ad.exists():
            ad_rows = json.loads(ad.read_text(encoding="utf-8"))
            report["quadrant_A_refusal_n150_SENSITIVITY"][st] = cell(
                ad_rows, st, "A", "refusal")
            report["quadrant_D_refusal_n150_SENSITIVITY"][st] = cell(
                ad_rows, st, "D", "refusal")
            used[ad.name] = _sha(ad)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    binding = {
        "produced_by": "src/analysis/mcnemar_direction_specificity.py",
        "producer_commit": _git_commit("src/analysis/mcnemar_direction_specificity.py"),
        "classifier_module": "src/analysis/eval_refusal_classifier.py",
        "classifier_commit": _git_commit("src/analysis/eval_refusal_classifier.py"),
        "summarize_module_commit": _git_commit("src/analysis/summarize_causal_ablation.py"),
        "raw_files_sha256": used,
        "test_variant": "scipy.stats.binomtest(min(b,c), b+c, 0.5, 'two-sided')",
    }
    Path(str(args.out).replace(".json", "_binding.json")).write_text(
        json.dumps(binding, indent=2), encoding="utf-8")

    print(f"wrote {args.out} (+ _binding.json)")
    for q, key in (("C", "quadrant_C_soft_deflection_n104"),
                   ("A", "quadrant_A_refusal_n150_SENSITIVITY"),
                   ("D", "quadrant_D_refusal_n150_SENSITIVITY")):
        print(f"  quadrant {q}:")
        for st, v in report[key].items():
            print(f"    {st:16s} b/c = {v['b']}/{v['c']}  "
                  f"p_exact = {v['p_two_sided_exact_binomial']:.2e}  (n_disc={v['n_discordant']})")


if __name__ == "__main__":
    main()
