"""C-vs-A and C-vs-training near-duplicate scan + 4-category coverage (WP-Leak).

Two questions the frozen plan needs answered before the T4 run (CPU-only,
runs now):

  1. Do quadrant-C candidate prompts overlap quadrant-A prompts, or the
     M1 helpful-SFT data (``sft_helpful.jsonl`` / ``sft_helpful_alt.jsonl``) or
     the safety-SFT / DPO data (``sft_safety.jsonl`` / ``dpo_pairs.jsonl``)?
     Exact-match always; near-duplicate via sentence-transformers if available,
     else a token-Jaccard fallback (>= 0.6) with that limitation stated.
  2. How are the 4 preregistered benchmark categories
     (misinformation_disinformation / harassment_bullying / illegal /
     cybercrime_intrusion) distributed across quadrant A and quadrant C in the
     frozen benchmark? (CF3 needs both label sets to match.)

Reads the frozen benchmark by SHA (``src.v2_io.resolve_benchmark``); never the
mutable ``data/processed/controlled_eval.jsonl``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.v2_io import load_jsonl, resolve_benchmark

CF3_CATEGORIES = (
    "misinformation_disinformation",
    "harassment_bullying",
    "illegal",
    "cybercrime_intrusion",
)
TRAINING_FILES = [
    "data/processed/sft_helpful.jsonl",
    "data/processed/sft_helpful_alt.jsonl",
    "data/processed/sft_safety.jsonl",
    "data/processed/dpo_pairs.jsonl",
]


def _norm(t: str) -> str:
    return " ".join((t or "").strip().lower().split())


def _tokens(t: str) -> set:
    return set(_norm(t).split())


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def exact_overlap(c_prompts, other_prompts):
    other = {_norm(p) for p in other_prompts}
    return [p for p in c_prompts if _norm(p) in other]


def near_duplicates(c_prompts, other_prompts, *, st_threshold=0.9, jac_threshold=0.6):
    """Returns (pairs, method). Tries sentence-transformers; falls back to
    token Jaccard with the limitation recorded in `method`."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        ce = model.encode(c_prompts, normalize_embeddings=True, show_progress_bar=False)
        oe = model.encode(other_prompts, normalize_embeddings=True, show_progress_bar=False)
        sims = ce @ oe.T
        pairs = []
        for i, cp in enumerate(c_prompts):
            for j, op in enumerate(other_prompts):
                s = float(sims[i][j])
                if s >= st_threshold:
                    pairs.append({"c_prompt": cp, "match": op, "similarity": round(s, 4)})
        return pairs, f"sentence-transformers all-MiniLM-L6-v2 (cos>={st_threshold})"
    except Exception as exc:
        pairs = []
        for cp in c_prompts:
            for op in other_prompts:
                s = jaccard(cp, op)
                if s >= jac_threshold:
                    pairs.append({"c_prompt": cp, "match": op, "jaccard": round(s, 4)})
        return pairs, (
            f"token-Jaccard fallback (>={jac_threshold}); sentence-transformers "
            f"unavailable ({type(exc).__name__}) - semantic near-dupes may be missed"
        )


def category_coverage(benchmark_rows):
    a = Counter()
    c = Counter()
    for row in benchmark_rows:
        cat = row.get("project_category") or row.get("source_category")
        if row.get("quadrant") == "A":
            a[cat] += 1
        elif row.get("quadrant") == "C":
            c[cat] += 1
    a_cf3 = {k: a.get(k, 0) for k in CF3_CATEGORIES}
    c_cf3 = {k: c.get(k, 0) for k in CF3_CATEGORIES}
    return {
        "A_all": dict(a), "C_all": dict(c),
        "A_cf3_only": a_cf3, "C_cf3_only": c_cf3,
        "cf3_label_sets_match": (
            {k for k, v in a_cf3.items() if v} == {k for k, v in c_cf3.items() if v}
        ),
        "A_non_cf3_categories": sorted(set(a) - set(CF3_CATEGORIES) - {None}),
    }


def run(latest_path=None):
    bench_path, bench_sha = resolve_benchmark(
        **({"latest_path": latest_path} if latest_path else {})
    )
    rows = load_jsonl(bench_path)
    c_prompts = [r["prompt"] for r in rows if r.get("quadrant") == "C"]
    a_prompts = [r["prompt"] for r in rows if r.get("quadrant") == "A"]

    report = {
        "benchmark_sha256": bench_sha,
        "n_C": len(c_prompts), "n_A": len(a_prompts),
        "C_vs_A": {},
        "C_vs_training": {},
        "category_coverage": category_coverage(rows),
    }

    exact_ca = exact_overlap(c_prompts, a_prompts)
    near_ca, method_ca = near_duplicates(c_prompts, a_prompts)
    report["C_vs_A"] = {"exact": exact_ca, "near_duplicates": near_ca, "method": method_ca}

    for tf in TRAINING_FILES:
        p = Path(tf)
        if not p.exists():
            report["C_vs_training"][tf] = {"status": "absent"}
            continue
        train_prompts = []
        for row in load_jsonl(p):
            train_prompts.append(row.get("prompt", row.get("instruction", "")))
        near, method = near_duplicates(c_prompts, train_prompts)
        report["C_vs_training"][tf] = {
            "n_train_prompts": len(train_prompts),
            "exact": exact_overlap(c_prompts, train_prompts),
            "near_duplicates": near,
            "method": method,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-path", default=None)
    parser.add_argument("--out", default="logs/c_vs_a_leakage.json")
    args = parser.parse_args()
    report = run(args.latest_path)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    cov = report["category_coverage"]
    print(f"benchmark {report['benchmark_sha256'][:12]}  C={report['n_C']} A={report['n_A']}")
    print(f"C-vs-A: {len(report['C_vs_A']['exact'])} exact, "
          f"{len(report['C_vs_A']['near_duplicates'])} near-dup  [{report['C_vs_A']['method']}]")
    for tf, res in report["C_vs_training"].items():
        if res.get("status") == "absent":
            print(f"  {tf}: absent")
        else:
            print(f"  {tf}: {len(res['exact'])} exact, {len(res['near_duplicates'])} near-dup")
    print(f"CF3 category coverage  A={cov['A_cf3_only']}  C={cov['C_cf3_only']}  "
          f"label sets match: {cov['cf3_label_sets_match']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
