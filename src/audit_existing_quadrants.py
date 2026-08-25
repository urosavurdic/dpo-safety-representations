"""
Audit A, B, and D quadrants of the controlled evaluation set.

Usage:
    python -m src.audit_existing_quadrants \\
        --eval-set data/processed/controlled_eval.jsonl \\
        [--a-path data/behavior_datasets/harmbench_behaviors_text_all.csv] \\
        [--out-dir logs/]

Reports per-quadrant: exact path, SHA-256, record count, unique prompt count,
schema, duplicates, source/category/domain/function distributions, and
contamination against all referenced training files.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def sha256file(path: str) -> str:
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return h


def normalize_prompt(text: str) -> str:
    """Lowercase + collapse whitespace for soft-dedup."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def load_training_prompts(path: str) -> set:
    """Load prompts from a JSONL training file. Tolerates missing files."""
    p = Path(path)
    if not p.exists():
        return set()
    prompts = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("prompt", "instruction", "chosen", "text"):
                if key in row and isinstance(row[key], str):
                    prompts.add(normalize_prompt(row[key]))
                    break
    return prompts


def audit_quadrant(rows: List[Dict], label: str) -> Dict:
    prompts = [r["prompt"] for r in rows]
    normalized = [normalize_prompt(p) for p in prompts]
    exact_dupes = len(prompts) - len(set(prompts))
    norm_dupes = len(prompts) - len(set(normalized))

    sources = Counter(r.get("source") for r in rows)
    categories = Counter(r.get("category") for r in rows)
    splits = Counter(r.get("split") for r in rows)
    word_lens = [len(p.split()) for p in prompts]
    char_lens = [len(p) for p in prompts]

    return {
        "quadrant": label,
        "total_rows": len(rows),
        "unique_prompts": len(set(prompts)),
        "unique_normalized_prompts": len(set(normalized)),
        "exact_duplicates": exact_dupes,
        "normalized_duplicates": norm_dupes,
        "near_duplicate_check": "not_run_model_unavailable",
        "sources": dict(sources),
        "categories": dict(categories),
        "splits": dict(splits),
        "word_length": {
            "min": min(word_lens) if word_lens else None,
            "max": max(word_lens) if word_lens else None,
            "mean": round(sum(word_lens) / len(word_lens), 2) if word_lens else None,
        },
        "char_length": {
            "min": min(char_lens) if char_lens else None,
            "max": max(char_lens) if char_lens else None,
            "mean": round(sum(char_lens) / len(char_lens), 2) if char_lens else None,
        },
    }


def check_contamination(rows: List[Dict], training_sets: Dict[str, set]) -> Dict:
    """Check each row's normalized prompt against all training prompt sets."""
    results = {}
    for ts_name, ts_prompts in training_sets.items():
        if not ts_prompts:
            results[ts_name] = {"status": "file_not_found", "hits": 0}
            continue
        hits = [r["prompt"] for r in rows
                if normalize_prompt(r["prompt"]) in ts_prompts]
        results[ts_name] = {"status": "checked", "hits": len(hits), "hit_prompts": hits[:5]}
    return results


def main():
    ap = argparse.ArgumentParser(description="Audit A/B/D quadrants")
    ap.add_argument("--eval-set", default="data/processed/controlled_eval.jsonl")
    ap.add_argument("--a-path", default=None, help="HarmBench source CSV (optional)")
    ap.add_argument("--b-path", default=None, help="XSTest source (optional)")
    ap.add_argument("--d-path", default=None, help="D source override (optional)")
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--d-review-seed", type=int, default=271828)
    args = ap.parse_args()

    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        print(f"ERROR: eval set not found: {eval_path}", file=sys.stderr)
        sys.exit(1)

    eval_sha = sha256file(str(eval_path))
    print(f"Eval set: {eval_path}")
    print(f"SHA-256:  {eval_sha}")

    with open(eval_path) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]

    print(f"Total rows: {len(all_rows)}")

    by_quad = {}
    for row in all_rows:
        q = row.get("quadrant", "MISSING")
        by_quad.setdefault(q, []).append(row)

    # Training files to check against
    training_files = {
        "sft_helpful": "data/processed/sft_helpful.jsonl",
        "sft_helpful_alt": "data/processed/sft_helpful_alt.jsonl",
        "sft_safety": "data/processed/sft_safety.jsonl",
        "dpo_pairs": "data/processed/dpo_pairs.jsonl",
    }
    training_prompts = {k: load_training_prompts(v) for k, v in training_files.items()}
    training_status = {k: ("found" if Path(v).exists() else "missing")
                       for k, v in training_files.items()}

    report = {
        "eval_set": str(eval_path),
        "eval_set_sha256": eval_sha,
        "total_rows": len(all_rows),
        "quadrant_counts": {q: len(rows) for q, rows in by_quad.items()},
        "training_files_checked": training_status,
        "quadrant_audits": {},
        "contamination": {},
    }

    for quad in ["A", "B", "D"]:
        rows = by_quad.get(quad, [])
        audit = audit_quadrant(rows, quad)
        report["quadrant_audits"][quad] = audit
        report["contamination"][quad] = check_contamination(rows, training_prompts)
        print(f"\n── Quadrant {quad}: {len(rows)} rows ──")
        print(f"  Unique prompts: {audit['unique_prompts']}")
        print(f"  Exact dupes: {audit['exact_duplicates']}, Norm dupes: {audit['normalized_duplicates']}")
        print(f"  Sources: {audit['sources']}")
        print(f"  Categories: {dict(list(audit['categories'].items())[:5])}")
        for ts, res in report["contamination"][quad].items():
            status = res["status"]
            hits = res.get("hits", 0)
            print(f"  Contamination vs {ts}: {status}, hits={hits}")

    # D spot-check CSV
    d_rows = by_quad.get("D", [])
    rng = random.Random(args.d_review_seed)
    spot_rows = rng.sample(d_rows, min(50, len(d_rows)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spot_path = Path("data/review/d_spot_check.csv")
    spot_path.parent.mkdir(parents=True, exist_ok=True)

    spot_fields = [
        "record_id", "source", "prompt", "word_count", "character_count",
        "formatting_flags", "duplicate_flags", "near_duplicate_flags",
        "review_status", "review_notes",
    ]
    with open(spot_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=spot_fields)
        writer.writeheader()
        for i, row in enumerate(spot_rows):
            p = row["prompt"]
            writer.writerow({
                "record_id": f"D_{i:04d}",
                "source": row.get("source", ""),
                "prompt": p,
                "word_count": len(p.split()),
                "character_count": len(p),
                "formatting_flags": "",
                "duplicate_flags": "",
                "near_duplicate_flags": "not_checked",
                "review_status": "pending",
                "review_notes": "",
            })
    spot_sha = sha256file(str(spot_path))
    report["d_spot_check"] = {
        "path": str(spot_path),
        "sha256": spot_sha,
        "rows": len(spot_rows),
        "seed": args.d_review_seed,
        "description": "Manual spot-check only. Not proof that D is benign.",
    }
    print(f"\nD spot-check: {spot_path} ({len(spot_rows)} rows, seed={args.d_review_seed})")

    out_path = out_dir / "audit_quadrants_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit report: {out_path}")
    return report


if __name__ == "__main__":
    main()
