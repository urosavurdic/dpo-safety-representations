"""Finalize the reviewed CSV into a frozen, LF-stable benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_lf(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def abort(message: str) -> None:
    print(f"\nABORT: {message}", file=sys.stderr)
    raise SystemExit(1)


def as_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def first_value(*candidates):
    """First meaningfully-present value, treating "" and None alike.

    csv.DictReader yields "" for an absent cell while the provenance JSONL
    yields None. A benchmark field should be null in both cases rather than
    an empty string, which a later reader would mistake for a real value.
    """
    for value in candidates:
        if value not in (None, ""):
            return value
    return None


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--gate-config", required=True)
    parser.add_argument(
        "--eval-set",
        default="data/processed/controlled_eval.jsonl",
    )
    parser.add_argument(
        "--provenance",
        default="data/quadrant_c_pipeline/candidate_records_v2.jsonl",
    )
    args = parser.parse_args()

    gate_path = Path(args.gate_config)
    review_path = Path(args.review_csv)
    eval_path = Path(args.eval_set)
    provenance_path = Path(args.provenance)

    for path in (gate_path, review_path, eval_path):
        if not path.exists():
            abort(f"Required input does not exist: {path}")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    review_sha = sha256file(review_path)
    gate_sha = sha256file(gate_path)
    eval_sha = sha256file(eval_path)

    with review_path.open(newline="", encoding="utf-8") as handle:
        review_rows = list(csv.DictReader(handle))

    allowed_statuses = {"pending", "accept", "reject"}
    invalid = [
        row.get("record_id")
        for row in review_rows
        if row.get("review_status", "").strip()
        not in allowed_statuses
    ]
    if invalid:
        abort(f"Invalid review_status values: {invalid[:10]}")

    pending = [
        row for row in review_rows
        if row.get("review_status", "").strip() == "pending"
    ]
    if pending:
        abort(
            f"{len(pending)} rows remain pending; first row is "
            f"{pending[0].get('record_id')}"
        )

    accepted = [
        row for row in review_rows
        if row.get("review_status", "").strip() == "accept"
    ]
    rejected = [
        row for row in review_rows
        if row.get("review_status", "").strip() == "reject"
    ]

    if not accepted:
        abort("There are no accepted C rows.")

    accepted_ids = [row.get("record_id") for row in accepted]
    if not all(accepted_ids):
        abort("Every accepted row requires record_id.")
    if len(accepted_ids) != len(set(accepted_ids)):
        abort("Accepted rows contain duplicate record_id values.")

    accepted_prompts = [
        normalize_prompt(row.get("scored_prompt", ""))
        for row in accepted
    ]
    if not all(accepted_prompts):
        abort("Every accepted row requires scored_prompt.")
    if len(accepted_prompts) != len(set(accepted_prompts)):
        abort("Accepted rows contain duplicate normalized prompts.")

    provenance = {}
    if provenance_path.exists():
        with provenance_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    provenance[record["candidate_id"]] = record
    else:
        abort(f"Provenance file not found: {provenance_path}")

    for row in accepted:
        candidate_id = row.get("candidate_id") or row["record_id"]
        record = provenance.get(candidate_id)
        if record is None:
            abort(f"Accepted row missing from provenance: {candidate_id}")

        if row["scored_prompt"] != record.get("candidate_prompt"):
            abort(
                f"scored_prompt differs from provenance for {candidate_id}"
            )

        if (
            row.get("source_prompt")
            and record.get("source_prompt")
            and row["source_prompt"] != record["source_prompt"]
        ):
            abort(
                f"source_prompt differs from provenance for {candidate_id}"
            )

    with eval_path.open(encoding="utf-8") as handle:
        eval_rows = [
            json.loads(line)
            for line in handle
            if line.strip()
        ]

    benchmark_rows = []

    for row in eval_rows:
        if row.get("quadrant") not in {"A", "B", "D"}:
            continue

        prompt = row["prompt"]
        record_id = row.get("record_id") or sha256str(
            f"{row['quadrant']}|{row.get('source', '')}|{prompt}"
        )[:16]

        benchmark_rows.append(
            {
                "record_id": record_id,
                "prompt": prompt,
                "scored_prompt": prompt,
                "quadrant": row["quadrant"],
                "c_construction": None,
                "ground_truth_intent": None,
                "expected_response_mode": (
                    "refusal" if row["quadrant"] == "A"
                    else "comply"
                ),
                "source_dataset": row.get(
                    "source_dataset",
                    row.get("source"),
                ),
                "source_id": row.get("source_id"),
                "source_category": row.get(
                    "source_category",
                    row.get("category"),
                ),
                "project_category": row.get(
                    "project_category",
                    row.get("category"),
                ),
                "domain": row.get("domain"),
                "prompt_function": row.get("prompt_function"),
                "word_count": len(prompt.split()),
                "character_count": len(prompt),
                "formatting_metadata": row.get(
                    "formatting_metadata"
                ),
                "pair_id": None,
                "source_prompt": None,
                "candidate_prompt": None,
                "review_status": None,
                "review_notes": None,
                "split": row.get("split"),
                "source_url": row.get("source_url"),
                "source_revision": row.get("source_revision"),
                "source_file_sha256": row.get(
                    "source_file_sha256"
                ),
                "source_prompt_sha256": None,
                "candidate_prompt_sha256": None,
                "record_sha256": sha256str(prompt),
            }
        )

    for row in accepted:
        construction = row.get(
            "c_construction",
            "c_paired",
        ) or "c_paired"
        candidate = row["scored_prompt"]
        source = row.get("source_prompt") or candidate

        if construction == "c_source_authored":
            source = candidate
            pair_id = None
        else:
            pair_id = row.get("pair_id") or None

        benchmark_rows.append(
            {
                "record_id": row["record_id"],
                "prompt": candidate,
                "scored_prompt": candidate,
                "quadrant": "C",
                "c_construction": construction,
                "ground_truth_intent": row.get(
                    "harmful_objective"
                ),
                "expected_response_mode": "refusal",
                "source_dataset": row.get("source_dataset"),
                "source_id": row.get("source_id"),
                "source_category": row.get("source_category"),
                "project_category": row.get("project_category"),
                "domain": row.get("domain"),
                "prompt_function": row.get("prompt_function"),
                "word_count": as_int(
                    row.get("word_count_candidate"),
                    len(candidate.split()),
                ),
                "character_count": as_int(
                    row.get("character_count_candidate"),
                    len(candidate),
                ),
                "formatting_metadata": row.get(
                    "formatting_metadata"
                ),
                "pair_id": pair_id,
                "source_prompt": source,
                "candidate_prompt": candidate,
                "review_status": "accept",
                "review_notes": row.get("review_notes", ""),
                "split": None,
                "source_url": row.get("source_url"),
                "source_revision": row.get("source_revision"),
                "source_file_sha256": row.get(
                    "source_file_sha256"
                ),
                "source_prompt_sha256": (
                    sha256str(source) if source else None
                ),
                "candidate_prompt_sha256": sha256str(candidate),
                "record_sha256": sha256str(candidate),
            }
        )

    ids = [row["record_id"] for row in benchmark_rows]
    prompts = [
        normalize_prompt(row["prompt"])
        for row in benchmark_rows
    ]

    if len(ids) != len(set(ids)):
        abort("Final benchmark contains duplicate record IDs.")
    if len(prompts) != len(set(prompts)):
        abort("Final benchmark contains duplicate normalized prompts.")

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out_dir = Path("data/frozen_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = out_dir / f"benchmark_v2_{timestamp}.jsonl"
    manifest_path = out_dir / f"benchmark_v2_{timestamp}.manifest.json"

    if benchmark_path.exists() or manifest_path.exists():
        abort(f"Timestamp collision for {timestamp}")

    with benchmark_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in benchmark_rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    benchmark_sha = sha256file(benchmark_path)
    quadrant_counts = Counter(
        row["quadrant"] for row in benchmark_rows
    )
    arm_counts = Counter(
        row["c_construction"]
        for row in benchmark_rows
        if row["quadrant"] == "C"
    )

    manifest = {
        "benchmark_path": benchmark_path.as_posix(),
        "benchmark_sha256": benchmark_sha,
        "creation_timestamp_utc": timestamp,
        "git_commit": git_commit(),
        "counts": dict(quadrant_counts),
        "c_counts_by_construction": dict(arm_counts),
        "source_distribution": dict(
            Counter(row.get("source_dataset") for row in benchmark_rows)
        ),
        "category_distribution": dict(
            Counter(row.get("project_category") for row in benchmark_rows)
        ),
        "inputs": {
            "eval_set": eval_path.as_posix(),
            "eval_set_sha256": eval_sha,
            "review_csv": review_path.as_posix(),
            "review_csv_sha256": review_sha,
            "gate_config": gate_path.as_posix(),
            "gate_config_sha256": gate_sha,
            "provenance": provenance_path.as_posix(),
        },
        "review_summary": {
            "total": len(review_rows),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "pending": 0,
        },
        "contamination_status": (
            "preserved_from_review_metadata"
        ),
        "direction_split_status": (
            "preserved_from_eval_set_fields"
        ),
        "gate_config_snapshot": gate,
    }

    write_json_lf(manifest_path, manifest)
    write_json_lf(
        out_dir / "LATEST_BENCHMARK.json",
        {
            "benchmark_path": benchmark_path.as_posix(),
            "benchmark_sha256": benchmark_sha,
        },
    )

    print(f"Benchmark written: {benchmark_path}")
    print(f"Benchmark SHA-256: {benchmark_sha}")
    print(f"Rows: {len(benchmark_rows)}")
    print(f"Manifest: {manifest_path}")
    print("LATEST_BENCHMARK.json updated")


if __name__ == "__main__":
    main()
