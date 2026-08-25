"""
Benchmark finalization — run AFTER researcher review is complete.

Reads the reviewed c_review_queue.csv, accepts only rows where
review_status == 'accept', validates all provenance constraints, and writes
a timestamped frozen benchmark under data/frozen_v2/.

Usage:
    python -m src.finalize_benchmark \\
        --review-csv data/review/c_review_queue.csv \\
        --gate-config logs/benchmark_gate_config.json \\
        [--eval-set data/processed/controlled_eval.jsonl] \\
        [--provenance data/quadrant_c_pipeline/candidate_records_v2.jsonl]

Will ABORT if:
  - any review_status is 'pending'
  - duplicate record_ids or scored_prompts in accepted rows
  - accepted scored_prompt does not match provenance candidate_prompt
  - data/frozen_v2/ benchmark already exists at same timestamp (race guard)
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256str(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def abort(msg: str):
    print(f"\nABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Finalize v2 benchmark after review")
    ap.add_argument("--review-csv",  required=True)
    ap.add_argument("--gate-config", required=True)
    ap.add_argument("--eval-set",    default="data/processed/controlled_eval.jsonl")
    ap.add_argument("--provenance",  default="data/quadrant_c_pipeline/candidate_records_v2.jsonl")
    args = ap.parse_args()

    # ── Load gate config ───────────────────────────────────────────────────────
    gate_path = Path(args.gate_config)
    if not gate_path.exists():
        abort(f"Gate config not found: {gate_path}")
    gate = json.loads(gate_path.read_text())
    gate_sha = sha256file(str(gate_path))

    # ── Load review CSV ────────────────────────────────────────────────────────
    review_path = Path(args.review_csv)
    if not review_path.exists():
        abort(f"Review CSV not found: {review_path}")
    review_sha = sha256file(str(review_path))

    with open(review_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        review_rows = list(reader)

    print(f"Review CSV: {review_path} ({len(review_rows)} rows)")

    # ── Check for pending rows ─────────────────────────────────────────────────
    pending = [r for r in review_rows if r.get("review_status", "").strip() == "pending"]
    if pending:
        abort(
            f"{len(pending)} rows still have review_status='pending'. "
            f"Complete review before finalizing. First pending: {pending[0].get('record_id')}"
        )

    accepted = [r for r in review_rows if r.get("review_status", "").strip() == "accept"]
    rejected = [r for r in review_rows if r.get("review_status", "").strip() == "reject"]
    print(f"  accepted: {len(accepted)}, rejected: {len(rejected)}")

    if not accepted:
        abort("Zero accepted rows — cannot produce a benchmark.")

    # ── Check for duplicate record_ids ─────────────────────────────────────────
    acc_ids = [r["record_id"] for r in accepted]
    if len(acc_ids) != len(set(acc_ids)):
        from collections import Counter
        dupes = [k for k, v in Counter(acc_ids).items() if v > 1]
        abort(f"Duplicate record_ids in accepted rows: {dupes}")

    # ── Check for duplicate normalized prompts ─────────────────────────────────
    norm_prompts = [normalize_prompt(r["scored_prompt"]) for r in accepted]
    if len(norm_prompts) != len(set(norm_prompts)):
        abort("Duplicate normalized scored_prompts in accepted rows.")

    # ── Cross-check scored_prompt vs provenance ────────────────────────────────
    prov_path = Path(args.provenance)
    if prov_path.exists():
        with open(prov_path) as f:
            prov_recs = {rec["candidate_id"]: rec for rec in (json.loads(l) for l in f)}
        mismatch = []
        for r in accepted:
            cid = r.get("candidate_id", r.get("record_id"))
            prov = prov_recs.get(cid)
            if prov and r["scored_prompt"].strip() != prov["candidate_prompt"].strip():
                mismatch.append(cid)
        if mismatch:
            abort(f"scored_prompt ≠ provenance candidate_prompt for: {mismatch}")
    else:
        print(f"WARNING: provenance file not found ({prov_path}), skipping cross-check.")

    # ── Load A, B, D from eval set ────────────────────────────────────────────
    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        abort(f"Eval set not found: {eval_path}")
    eval_sha = sha256file(str(eval_path))

    with open(eval_path) as f:
        eval_rows = [json.loads(l) for l in f if l.strip()]

    abd_rows = [r for r in eval_rows if r.get("quadrant") in ("A", "B", "D")]

    # ── Build canonical benchmark rows ────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/frozen_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_rows = []

    # A/B/D from the eval set
    for row in abd_rows:
        p = row["prompt"]
        rec = {
            "record_id": hashlib.sha256(
                f"{row['quadrant']}|{row.get('source','')}|{p}".encode()
            ).hexdigest()[:16],
            "prompt":          p,
            "scored_prompt":   p,
            "quadrant":        row["quadrant"],
            "c_construction":  None,
            "ground_truth_intent": None,
            "expected_response_mode": None,
            "source_dataset":  row.get("source"),
            "source_id":       None,
            "source_category": row.get("category"),
            "project_category": row.get("category"),
            "domain":          None,
            "prompt_function": None,
            "word_count":      len(p.split()),
            "character_count": len(p),
            "formatting_metadata": None,
            "pair_id":         None,
            "source_prompt":   None,
            "candidate_prompt": None,
            "review_status":   None,
            "review_notes":    None,
            "split":           row.get("split"),
            "source_url":      None,
            "source_revision": None,
            "source_file_sha256": None,
            "source_prompt_sha256": None,
            "candidate_prompt_sha256": None,
            "record_sha256":   sha256str(p),
        }
        bench_rows.append(rec)

    # C accepted rows
    for r in accepted:
        cp = r["scored_prompt"].strip()
        sp = r.get("source_prompt", "").strip()
        rec = {
            "record_id":       r["record_id"],
            "prompt":          cp,
            "scored_prompt":   cp,
            "quadrant":        "C",
            "c_construction":  "c_paired",
            "ground_truth_intent": r.get("harmful_objective", ""),
            "expected_response_mode": "refusal",
            "source_dataset":  r.get("source_dataset", "StrongREJECT"),
            "source_id":       r.get("source_id", ""),
            "source_category": r.get("source_category", ""),
            "project_category": r.get("project_category", ""),
            "domain":          r.get("domain") or None,
            "prompt_function": r.get("prompt_function") or None,
            "word_count":      int(r.get("word_count_candidate", len(cp.split()))),
            "character_count": int(r.get("character_count_candidate", len(cp))),
            "formatting_metadata": None,
            "pair_id":         r.get("pair_id", ""),
            "source_prompt":   sp,
            "candidate_prompt": cp,
            "review_status":   "accept",
            "review_notes":    r.get("review_notes", ""),
            "split":           None,
            "source_url":      "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
            "source_revision": "Souly et al. 2024, main branch",
            "source_file_sha256": None,
            "source_prompt_sha256":    sha256str(sp) if sp else None,
            "candidate_prompt_sha256": sha256str(cp),
            "record_sha256":           sha256str(cp),
        }
        bench_rows.append(rec)

    # ── Write benchmark ────────────────────────────────────────────────────────
    bench_path = out_dir / f"benchmark_v2_{ts}.jsonl"
    if bench_path.exists():
        abort(f"Benchmark file already exists: {bench_path} (race guard)")

    with open(bench_path, "w") as f:
        for row in bench_rows:
            f.write(json.dumps(row) + "\n")

    bench_sha = sha256file(str(bench_path))
    print(f"\nBenchmark written: {bench_path}")
    print(f"SHA-256: {bench_sha}")
    print(f"Total rows: {len(bench_rows)}")

    # ── Counts ────────────────────────────────────────────────────────────────
    from collections import Counter
    q_counts = Counter(r["quadrant"] for r in bench_rows)
    c_arm_counts = Counter(
        r.get("c_construction") for r in bench_rows if r["quadrant"] == "C"
    )
    print(f"By quadrant: {dict(q_counts)}")
    print(f"C arms: {dict(c_arm_counts)}")

    # ── Manifest ───────────────────────────────────────────────────────────────
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    manifest = {
        "benchmark_path":   str(bench_path),
        "benchmark_sha256": bench_sha,
        "creation_timestamp_utc": ts,
        "git_commit": git_commit,
        "counts": dict(q_counts),
        "c_counts_by_construction": dict(c_arm_counts),
        "source_distribution": dict(Counter(r.get("source_dataset") for r in bench_rows)),
        "category_distribution": dict(Counter(r.get("project_category") for r in bench_rows)),
        "inputs": {
            "eval_set": str(eval_path),
            "eval_set_sha256": eval_sha,
            "review_csv": str(review_path),
            "review_csv_sha256": review_sha,
            "gate_config": str(gate_path),
            "gate_config_sha256": gate_sha,
            "provenance": str(prov_path),
        },
        "review_summary": {
            "total": len(review_rows),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "pending": 0,
        },
        "contamination_status": "checked_exact_zero_near_unknown",
        "direction_split_status": "existing_fields_preserved",
    }

    manifest_path = out_dir / f"benchmark_v2_{ts}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Stable latest pointer
    latest = {"benchmark_path": str(bench_path), "benchmark_sha256": bench_sha}
    latest_path = out_dir / "LATEST_BENCHMARK.json"
    latest_path.write_text(json.dumps(latest, indent=2))

    print(f"Manifest: {manifest_path}")
    print(f"Latest pointer: {latest_path}")
    return str(bench_path), bench_sha


if __name__ == "__main__":
    main()
