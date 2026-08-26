from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


BASE_COMMIT = "faee3317200b6e04dfb351e62f2fda21caf84a91"
ROOT = Path.cwd()


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, check=check, text=True)


def write_file(relative: str, content: str) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = textwrap.dedent(content).lstrip()
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(normalized)
        if not normalized.endswith("\n"):
            handle.write("\n")
    print(f"updated {relative}")


def assert_clean_enough() -> None:
    head = run("git", "rev-parse", "HEAD").stdout.strip()
    if head != BASE_COMMIT:
        raise RuntimeError(
            f"Unexpected HEAD {head}; expected {BASE_COMMIT}. "
            "Switch to the original branch commit before running this script."
        )

    status = run("git", "status", "--short").stdout.splitlines()
    unexpected = [
        line for line in status
        if not line.startswith("?? artifacts")
        and not line.startswith("?? assets")
    ]
    if unexpected:
        raise RuntimeError(
            "Working tree contains changes outside artifacts/assets:\n"
            + "\n".join(unexpected)
        )


write_file(
    "src/v2_io.py",
    r'''
"""Strict benchmark-bound I/O for the v2 Colab rerun."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


LATEST_BENCHMARK = Path("data/frozen_v2/LATEST_BENCHMARK.json")
DEFAULT_SPLIT_MANIFEST = Path("logs/direction_split_manifest.json")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json(data: Any) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def write_json_lf(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write UTF-8 JSON without platform newline translation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def normalize_json_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def identity_snapshot(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row.get("record_id"),
            "prompt": row.get("prompt"),
            "scored_prompt": row.get("scored_prompt"),
            "quadrant": row.get("quadrant"),
            "source": row.get(
                "source",
                row.get("source_dataset"),
            ),
            "source_dataset": row.get("source_dataset"),
            "c_construction": row.get("c_construction"),
            "split": row.get("split"),
        }
        for row in rows
    ]


def resolve_benchmark(
    eval_set: str | Path | None = None,
    latest_path: str | Path = LATEST_BENCHMARK,
) -> tuple[Path, str]:
    latest_path = Path(latest_path)
    if not latest_path.exists():
        raise FileNotFoundError(
            f"Missing frozen benchmark pointer: {latest_path}"
        )

    latest = load_json(latest_path)
    pointer_path = normalize_json_path(
        latest.get("benchmark_path", "")
    )
    pointer_sha = latest.get("benchmark_sha256")

    if not pointer_path or not pointer_sha:
        raise RuntimeError(
            "LATEST_BENCHMARK.json must contain benchmark_path and "
            "benchmark_sha256."
        )

    requested_path = (
        normalize_json_path(eval_set)
        if eval_set is not None
        else pointer_path
    )

    if not requested_path.exists():
        raise FileNotFoundError(
            f"Frozen benchmark does not exist: {requested_path}"
        )

    actual_sha = sha256_file(requested_path)
    if actual_sha != pointer_sha:
        raise RuntimeError(
            "Frozen benchmark hash mismatch:\n"
            f"  path:     {requested_path}\n"
            f"  recorded: {pointer_sha}\n"
            f"  actual:   {actual_sha}"
        )

    if requested_path.resolve() != pointer_path.resolve():
        raise RuntimeError(
            "Requested eval set is not the benchmark referenced by "
            f"{latest_path}:\n"
            f"  requested: {requested_path}\n"
            f"  pointer:   {pointer_path}"
        )

    return requested_path, actual_sha


def split_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {
            "split_manifest_sha256",
            "split_file_sha256",
        }
    }


def load_run_inputs(
    eval_set: str | Path | None = None,
    benchmark_sha256: str | None = None,
    split_manifest: str | Path = DEFAULT_SPLIT_MANIFEST,
) -> tuple[Path, str, Path, str]:
    benchmark_path, actual_benchmark_sha = resolve_benchmark(eval_set)

    if (
        benchmark_sha256 is not None
        and benchmark_sha256 != actual_benchmark_sha
    ):
        raise RuntimeError(
            "Supplied benchmark SHA does not match the frozen benchmark: "
            f"{benchmark_sha256} != {actual_benchmark_sha}"
        )

    split_path = Path(split_manifest)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing direction split manifest: {split_path}"
        )

    manifest = load_json(split_path)

    if manifest.get("benchmark_sha256") != actual_benchmark_sha:
        raise RuntimeError(
            "Direction split manifest is bound to a different benchmark: "
            f"{manifest.get('benchmark_sha256')} != "
            f"{actual_benchmark_sha}"
        )

    recorded_split_sha = manifest.get("split_manifest_sha256")
    if not recorded_split_sha:
        raise RuntimeError(
            f"{split_path} has no split_manifest_sha256."
        )

    calculated_split_sha = sha256_bytes(
        canonical_json(split_payload(manifest))
    )
    if calculated_split_sha != recorded_split_sha:
        raise RuntimeError(
            "Direction split manifest content hash mismatch: "
            f"{calculated_split_sha} != {recorded_split_sha}"
        )

    return (
        benchmark_path,
        actual_benchmark_sha,
        split_path,
        recorded_split_sha,
    )


def binding(
    benchmark_path: str | Path,
    benchmark_sha256: str,
    split_path: str | Path,
    split_manifest_sha256: str,
) -> dict[str, str]:
    return {
        "benchmark_path": Path(benchmark_path).as_posix(),
        "benchmark_sha256": benchmark_sha256,
        "split_manifest_path": Path(split_path).as_posix(),
        "split_manifest_sha256": split_manifest_sha256,
    }


def assert_binding(
    path: str | Path,
    benchmark_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact binding: {path}")

    data = load_json(path)

    if data.get("benchmark_sha256") != benchmark_sha256:
        raise RuntimeError(
            f"Artifact {path} is bound to a different benchmark."
        )

    if data.get("split_manifest_sha256") != split_manifest_sha256:
        raise RuntimeError(
            f"Artifact {path} is bound to a different split manifest."
        )

    return data
''',
)

write_file(
    "src/create_direction_split_manifest.py",
    r'''
"""Bind the existing A/D split fields to the frozen benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.v2_io import (
    canonical_json,
    load_jsonl,
    resolve_benchmark,
    sha256_bytes,
    write_json_lf,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=None)
    parser.add_argument(
        "--out",
        default="logs/direction_split_manifest.json",
    )
    args = parser.parse_args()

    benchmark_path, benchmark_sha = resolve_benchmark(args.benchmark)
    rows = load_jsonl(benchmark_path)

    direction_ids = []
    held_out_ids = []
    source_counts = {
        "direction_estimation": Counter(),
        "held_out_behavioral": Counter(),
    }

    for row in rows:
        if row.get("quadrant") not in {"A", "D"}:
            continue

        record_id = row.get("record_id")
        split = row.get("split")

        if not record_id:
            raise RuntimeError(
                "Every A/D benchmark row must contain record_id."
            )

        if split not in {
            "direction_estimation",
            "held_out_behavioral",
        }:
            raise RuntimeError(
                f"A/D record {record_id} has invalid split {split!r}."
            )

        if split == "direction_estimation":
            direction_ids.append(record_id)
        else:
            held_out_ids.append(record_id)

        source = row.get(
            "source_dataset",
            row.get("source", "UNKNOWN"),
        )
        source_counts[split][source] += 1

    if not direction_ids or not held_out_ids:
        raise RuntimeError(
            "Both direction-estimation and held-out partitions must "
            "be non-empty."
        )

    if set(direction_ids) & set(held_out_ids):
        raise RuntimeError(
            "A record occurs in both direction split partitions."
        )

    payload = {
        "benchmark_path": benchmark_path.as_posix(),
        "benchmark_sha256": benchmark_sha,
        "direction_split_seed": 45,
        "direction_train_fraction": 0.80,
        "split_algorithm": (
            "preserved_split_fields_from_frozen_benchmark"
        ),
        "record_ids_direction_estimation": direction_ids,
        "record_ids_held_out_behavioral": held_out_ids,
        "counts": {
            "direction_estimation": len(direction_ids),
            "held_out_behavioral": len(held_out_ids),
        },
        "sources": {
            name: dict(counter)
            for name, counter in source_counts.items()
        },
        "split_hash_algorithm": (
            "sha256_canonical_json_without_hash_fields"
        ),
    }

    manifest = {
        **payload,
        "split_manifest_sha256": sha256_bytes(
            canonical_json(payload)
        ),
    }

    write_json_lf(args.out, manifest)

    print(f"Benchmark: {benchmark_path}")
    print(f"Benchmark SHA-256: {benchmark_sha}")
    print(
        "Direction rows: "
        f"{len(direction_ids)}; held-out rows: {len(held_out_ids)}"
    )
    print(f"Split manifest: {args.out}")
    print(
        "Split SHA-256: "
        f"{manifest['split_manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
''',
)

write_file(
    "src/finalize_benchmark.py",
    r'''
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
''',
)

write_file(
    "src/validate_benchmark_v2.py",
    r'''
"""Strict validation for the frozen v2 benchmark and model artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.v2_io import (
    canonical_json,
    identity_snapshot,
    load_json,
    load_jsonl,
    sha256_file,
    write_json_lf,
)


REQUIRED_STAGES = [
    "M0",
    "M1",
    "M2",
    "M3",
    "M3_direct",
    "M1_alt",
    "M2_alt",
    "M3_alt",
    "M3_direct_alt",
]

REQUIRED_FIELDS = [
    "record_id",
    "prompt",
    "scored_prompt",
    "quadrant",
    "c_construction",
    "ground_truth_intent",
    "expected_response_mode",
    "source_dataset",
    "source_id",
    "source_category",
    "project_category",
    "domain",
    "prompt_function",
    "word_count",
    "character_count",
    "formatting_metadata",
    "pair_id",
    "source_prompt",
    "candidate_prompt",
    "review_status",
    "review_notes",
    "split",
    "source_url",
    "source_revision",
    "source_file_sha256",
    "source_prompt_sha256",
    "candidate_prompt_sha256",
    "record_sha256",
]


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def sha256str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")


def bootstrap_mean(
    values: list[float],
    n_boot: int = 1000,
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value, value

    import random

    rng = random.Random(42)
    samples = []
    for _ in range(n_boot):
        sample = [
            rng.choice(values)
            for _ in range(len(values))
        ]
        samples.append(sum(sample) / len(sample))

    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return (
        sum(values) / len(values),
        lo,
        hi,
    )


def pooled_cohen_d(a: list[float], b: list[float]):
    if len(a) < 2 or len(b) < 2:
        return None

    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt(
        (
            (len(a) - 1) * var_a
            + (len(b) - 1) * var_b
        )
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return None
    return abs((mean_a - mean_b) / pooled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--gate-config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", default="logs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(args.benchmark)
    review_path = Path(args.review_csv)
    gate_path = Path(args.gate_config)
    split_path = Path(args.split_manifest)

    for path in (
        benchmark_path,
        review_path,
        gate_path,
        split_path,
    ):
        if not path.exists():
            print(
                f"ERROR: required file missing: {path}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    benchmark_sha = sha256_file(benchmark_path)
    review_sha = sha256_file(review_path)
    gate_sha = sha256_file(gate_path)
    split_file_sha = sha256_file(split_path)

    benchmark_rows = load_jsonl(benchmark_path)
    split_manifest = load_json(split_path)
    gate = load_json(gate_path)

    counts = Counter(
        row.get("quadrant", "MISSING")
        for row in benchmark_rows
    )
    c_rows = [
        row for row in benchmark_rows
        if row.get("quadrant") == "C"
    ]
    c_arms = Counter(
        row.get("c_construction")
        for row in c_rows
    )

    schema_errors = []
    for index, row in enumerate(benchmark_rows):
        for field in REQUIRED_FIELDS:
            if field not in row:
                schema_errors.append(
                    f"row {index}: missing {field}"
                )

    schema_integrity_pass = not schema_errors

    ids = [row.get("record_id") for row in benchmark_rows]
    prompts = [row.get("prompt", "") for row in benchmark_rows]
    normalized = [normalize_prompt(prompt) for prompt in prompts]

    duplicate_ids = len(ids) - len(set(ids))
    duplicate_prompts = len(prompts) - len(set(prompts))
    duplicate_normalized = len(normalized) - len(set(normalized))

    record_hash_errors = [
        row.get("record_id")
        for row in benchmark_rows
        if row.get("record_sha256")
        != sha256str(row.get("scored_prompt", ""))
    ]

    prompt_integrity_pass = (
        duplicate_ids == 0
        and duplicate_prompts == 0
        and duplicate_normalized == 0
        and not record_hash_errors
    )

    with review_path.open(newline="", encoding="utf-8") as handle:
        review_rows = list(csv.DictReader(handle))

    review_statuses = {
        row.get("review_status", "").strip()
        for row in review_rows
    }
    pending = [
        row for row in review_rows
        if row.get("review_status", "").strip() == "pending"
    ]
    invalid_review = review_statuses - {"pending", "accept", "reject"}
    c_review_pass = not pending and not invalid_review

    latest_path = Path("data/frozen_v2/LATEST_BENCHMARK.json")
    benchmark_hash_pass = False
    latest_warning = None

    if latest_path.exists():
        latest = load_json(latest_path)
        latest_benchmark = Path(
            str(latest.get("benchmark_path", "")).replace(
                "\\",
                "/",
            )
        )
        latest_sha = latest.get("benchmark_sha256")
        benchmark_hash_pass = (
            latest_benchmark.resolve()
            == benchmark_path.resolve()
            and latest_sha == benchmark_sha
            and sha256_file(benchmark_path) == latest_sha
        )
    else:
        latest_warning = (
            "LATEST_BENCHMARK.json is missing."
        )

    split_benchmark_hash_pass = (
        split_manifest.get("benchmark_sha256")
        == benchmark_sha
    )

    split_recorded_sha = split_manifest.get(
        "split_manifest_sha256"
    )
    split_computed_sha = hashlib.sha256(
        canonical_json(
            {
                key: value
                for key, value in split_manifest.items()
                if key not in {
                    "split_manifest_sha256",
                    "split_file_sha256",
                }
            }
        )
    ).hexdigest()

    split_hash_pass = (
        bool(split_recorded_sha)
        and split_recorded_sha == split_computed_sha
    )

    expected_snapshot = identity_snapshot(benchmark_rows)
    stale_artifacts = []

    for stage in REQUIRED_STAGES:
        activation_dir = Path("results/activations")
        final_path = activation_dir / f"{stage}_final.npy"
        pooled_path = activation_dir / f"{stage}_pooled.npy"
        metadata_path = activation_dir / f"{stage}_metadata.json"
        binding_path = (
            activation_dir
            / f"{stage}_metadata_binding.json"
        )

        missing = [
            str(path)
            for path in (
                final_path,
                pooled_path,
                metadata_path,
                binding_path,
            )
            if not path.exists()
        ]

        if missing:
            stale_artifacts.extend(
                f"{stage}: missing {path}"
                for path in missing
            )
            continue

        try:
            metadata = load_json(metadata_path)
            final_shape = __import__("numpy").load(
                final_path,
                mmap_mode="r",
            ).shape
            pooled_shape = __import__("numpy").load(
                pooled_path,
                mmap_mode="r",
            ).shape
            binding = load_json(binding_path)
        except Exception as exc:
            stale_artifacts.append(
                f"{stage}: unreadable artifact ({exc})"
            )
            continue

        if metadata != expected_snapshot:
            stale_artifacts.append(
                f"{stage}: metadata does not match benchmark"
            )

        if (
            len(metadata) != len(benchmark_rows)
            or final_shape[0] != len(benchmark_rows)
            or pooled_shape[0] != len(benchmark_rows)
        ):
            stale_artifacts.append(
                f"{stage}: activation row count mismatch"
            )

        if binding.get("benchmark_sha256") != benchmark_sha:
            stale_artifacts.append(
                f"{stage}: activation benchmark hash mismatch"
            )

        if (
            binding.get("split_manifest_sha256")
            != split_recorded_sha
        ):
            stale_artifacts.append(
                f"{stage}: activation split hash mismatch"
            )

    artifact_freshness_pass = not stale_artifacts

    accepted_review = [
        row for row in review_rows
        if row.get("review_status", "").strip() == "accept"
    ]
    accepted_ids = {
        row.get("record_id")
        for row in accepted_review
    }
    benchmark_c_ids = {
        row.get("record_id")
        for row in c_rows
    }
    c_review_mapping_pass = accepted_ids == benchmark_c_ids

    paired_diffs = []
    for row in accepted_review:
        value = row.get(
            "fightin_words_paired_difference",
            "",
        )
        try:
            paired_diffs.append(float(value))
        except (TypeError, ValueError):
            pass

    paired_mean, paired_lo, paired_hi = bootstrap_mean(
        paired_diffs
    )
    positive_count = sum(
        value > 0 for value in paired_diffs
    )

    if paired_diffs:
        reduced_status = (
            "SUPPORTED_OPERATIONALLY"
            if paired_mean is not None
            and paired_mean > 0
            and positive_count / len(paired_diffs) >= 0.5
            else "INCONCLUSIVE"
        )
    else:
        reduced_status = "INCONCLUSIVE"

    a_lengths = [
        row.get("word_count", len(row["prompt"].split()))
        for row in benchmark_rows
        if row.get("quadrant") == "A"
    ]
    c_lengths = [
        row.get("word_count", len(row["prompt"].split()))
        for row in c_rows
        if row.get("c_construction") == "c_paired"
    ]
    length_d = pooled_cohen_d(a_lengths, c_lengths)
    length_confound_pass = (
        None
        if not a_lengths or not c_lengths
        else length_d is not None and length_d < 0.5
    )

    warnings = []

    if schema_errors:
        warnings.append(
            f"Schema errors: {len(schema_errors)}"
        )
    if not prompt_integrity_pass:
        warnings.append(
            "Prompt or record-id integrity failed."
        )
    if not c_review_mapping_pass:
        warnings.append(
            "Accepted review IDs do not exactly match benchmark C IDs."
        )
    if not benchmark_hash_pass:
        warnings.append(
            "Benchmark does not match LATEST_BENCHMARK.json."
        )
    if not split_benchmark_hash_pass:
        warnings.append(
            "Direction split manifest targets another benchmark."
        )
    if not split_hash_pass:
        warnings.append(
            "Direction split manifest content hash failed."
        )
    if stale_artifacts:
        warnings.append(
            f"Stale or missing activation artifacts: "
            f"{len(stale_artifacts)}"
        )
    if length_confound_pass is False:
        warnings.append(
            f"Length confound: |d|={length_d:.3f}; "
            "A-versus-C comparison is source-confounded."
        )
    warnings.append(
        "source_cue_effect_status=not_identified: "
        "A and C-paired use different source datasets."
    )

    technical_pass = all(
        (
            schema_integrity_pass,
            prompt_integrity_pass,
            c_review_pass,
            c_review_mapping_pass,
            benchmark_hash_pass,
            split_benchmark_hash_pass,
            split_hash_pass,
            artifact_freshness_pass,
        )
    )

    status = {
        "technical_benchmark_status": (
            "PASS" if technical_pass else "FAIL"
        ),
        "reduced_cue_evidence_status": reduced_status,
        "wording_only_claim_status": "INCONCLUSIVE",
        "benchmark_path": benchmark_path.as_posix(),
        "benchmark_sha256": benchmark_sha,
        "counts": dict(counts),
        "c_counts_by_construction": dict(c_arms),
        "schema_integrity_pass": schema_integrity_pass,
        "prompt_integrity_pass": prompt_integrity_pass,
        "c_review_pass": c_review_pass,
        "c_review_mapping_pass": c_review_mapping_pass,
        "benchmark_hash_pass": benchmark_hash_pass,
        "split_benchmark_hash_pass": split_benchmark_hash_pass,
        "split_hash_pass": split_hash_pass,
        "artifact_freshness_pass": artifact_freshness_pass,
        "surface_separation_pass": (
            reduced_status == "SUPPORTED_OPERATIONALLY"
            if paired_diffs
            else None
        ),
        "length_confound_pass": length_confound_pass,
        "category_confound_pass": None,
        "prompt_function_confound_pass": None,
        "source_confound_pass": False,
        "wording_only_claim_pass": None,
        "stale_activation_files": stale_artifacts,
        "paired_diagnostics": {
            "n_pairs": len(paired_diffs),
            "n_positive_diff": positive_count,
            "mean_diff": paired_mean,
            "ci_95_lo": paired_lo,
            "ci_95_hi": paired_hi,
            "resampling_unit": "matched_pair",
            "n_bootstrap_replicates": 1000,
        },
        "precision_band": (
            "very_limited_exploratory"
            if len(c_rows) < 30
            else "exploratory"
            if len(c_rows) < 60
            else "preferred_minimum"
            if len(c_rows) < 80
            else "desirable_target"
        ),
        "warnings": warnings,
        "inputs": {
            "benchmark": benchmark_path.as_posix(),
            "benchmark_sha256": benchmark_sha,
            "review_csv": review_path.as_posix(),
            "review_csv_sha256": review_sha,
            "gate_config": gate_path.as_posix(),
            "gate_config_sha256": gate_sha,
            "split_manifest": split_path.as_posix(),
            "split_manifest_file_sha256": split_file_sha,
            "split_manifest_sha256": split_recorded_sha,
        },
        "outputs": {
            "validation_status": (
                "logs/benchmark_validation_status.json"
            ),
            "validation_report": (
                "logs/benchmark_validation_report.md"
            ),
        },
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    write_json_lf(
        out_dir / "benchmark_validation_status.json",
        status,
    )

    report = [
        "# Benchmark v2 Validation Report",
        "",
        f"Generated: {status['generated_at_utc']}",
        "",
        (
            "**Technical benchmark status:** "
            f"`{status['technical_benchmark_status']}`"
        ),
        (
            "**Reduced-cue evidence status:** "
            f"`{status['reduced_cue_evidence_status']}`"
        ),
        (
            "**Wording-only claim status:** "
            "`INCONCLUSIVE`"
        ),
        "",
        "## Counts",
        "",
        "| Quadrant | Count |",
        "|---|---:|",
    ]

    for quadrant, count in sorted(counts.items()):
        report.append(f"| {quadrant} | {count} |")

    report.extend(
        [
            "",
            "## C construction counts",
            "",
            "| Construction | Count |",
            "|---|---:|",
        ]
    )

    for arm, count in sorted(c_arms.items()):
        report.append(f"| {arm} | {count} |")

    report.extend(
        [
            "",
            "## Gate fields",
            "",
            "| Field | Value |",
            "|---|---|",
        ]
    )

    for field in (
        "schema_integrity_pass",
        "prompt_integrity_pass",
        "c_review_pass",
        "c_review_mapping_pass",
        "benchmark_hash_pass",
        "split_benchmark_hash_pass",
        "split_hash_pass",
        "artifact_freshness_pass",
    ):
        report.append(f"| {field} | {status[field]} |")

    report.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    report.extend(f"- {warning}" for warning in warnings)

    if stale_artifacts:
        report.extend(
            [
                "",
                "## Stale artifacts",
                "",
            ]
        )
        report.extend(f"- {item}" for item in stale_artifacts)

    write_report(
        out_dir / "benchmark_validation_report.md",
        report,
    )

    print(f"Benchmark rows: {len(benchmark_rows)}")
    print(
        "technical_benchmark_status: "
        f"{status['technical_benchmark_status']}"
    )
    print(
        "reduced_cue_evidence_status: "
        f"{status['reduced_cue_evidence_status']}"
    )
    print(
        "wording_only_claim_status: INCONCLUSIVE"
    )
    print(
        "Validation status: "
        "logs/benchmark_validation_status.json"
    )
    print(
        "Markdown report: "
        "logs/benchmark_validation_report.md"
    )

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
''',
)

write_file(
    "src/analysis/v2_pipeline.py",
    r'''
"""Strict benchmark-bound GPU pipeline for the v2 Colab rerun.

The legacy analysis scripts remain available for historical reproducibility.
This module is the only runner used by rerun_mechanistic_v2.sh and the v2
notebook. It always reads the frozen benchmark referenced by
LATEST_BENCHMARK.json and refuses stale or unbound artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.v2_io import (
    assert_binding,
    binding,
    identity_snapshot,
    load_json,
    load_jsonl,
    load_run_inputs,
    write_json_lf,
)


ALL_STAGES = [
    "M0",
    "M1",
    "M2",
    "M3",
    "M3_direct",
    "M1_alt",
    "M2_alt",
    "M3_alt",
    "M3_direct_alt",
]

INTERVENTION_STAGES = [
    "M3",
    "M3_direct",
    "M3_alt",
    "M3_direct_alt",
]

ABLATION_LAYERS = list(range(24, 29))
BATCH_SIZE = 8
MAX_NEW_TOKENS = 200
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

ACT_DIR = Path("results/activations")
DIRECTION_DIR = Path("results/refusal_direction")
RAW_DIR = Path("results/raw")
PROBE_DIR = Path("results/probes_v2")
MANIFEST_DIR = Path("results/manifests")


def ml_imports():
    import torch
    from transformers import AutoTokenizer
    from src.training.model import load_stage_model

    return torch, AutoTokenizer, load_stage_model


def snapshot_for(rows):
    return identity_snapshot(rows)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def load_bound_activation(
    stage: str,
    benchmark_path: Path,
    benchmark_sha: str,
    split_sha: str,
):
    pooled_path = ACT_DIR / f"{stage}_pooled.npy"
    final_path = ACT_DIR / f"{stage}_final.npy"
    metadata_path = ACT_DIR / f"{stage}_metadata.json"
    binding_path = ACT_DIR / f"{stage}_metadata_binding.json"

    assert_binding(binding_path, benchmark_sha, split_sha)

    if not pooled_path.exists() or not final_path.exists():
        raise FileNotFoundError(
            f"Missing activation arrays for {stage}."
        )

    metadata = load_json(metadata_path)
    expected = snapshot_for(load_jsonl(benchmark_path))

    if metadata != expected:
        raise RuntimeError(
            f"{metadata_path} does not match the frozen benchmark."
        )

    pooled = np.load(pooled_path)
    final = np.load(final_path)

    if pooled.shape[0] != len(metadata):
        raise RuntimeError(
            f"{stage}: pooled activation row count mismatch."
        )
    if final.shape[0] != len(metadata):
        raise RuntimeError(
            f"{stage}: final activation row count mismatch."
        )

    return final, pooled, metadata


def generation_batch(model, tokenizer, prompts, device):
    import torch
    from src.training.eval_generation import (
        build_generation_prompt,
        get_generation_eos_ids,
    )

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        texts = [
            build_generation_prompt(tokenizer, prompt)
            for prompt in prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=get_generation_eos_ids(tokenizer),
            )

        new_tokens = output_ids[
            :,
            inputs["input_ids"].shape[1]:,
        ]
        return tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )
    finally:
        tokenizer.padding_side = original_padding_side


def activation_batch(model, tokenizer, prompts, device):
    import torch
    from src.training.eval_generation import build_generation_prompt

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        texts = [
            build_generation_prompt(tokenizer, prompt)
            for prompt in prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        layer_count = len(hidden_states)
        hidden_dim = hidden_states[0].shape[-1]

        final = np.zeros(
            (len(prompts), layer_count, hidden_dim),
            dtype=np.float32,
        )
        pooled = np.zeros_like(final)

        for row_index in range(len(prompts)):
            attention_length = int(
                inputs["attention_mask"][row_index]
                .sum()
                .item()
            )
            window = min(5, attention_length)

            for layer in range(layer_count):
                hidden = hidden_states[layer][row_index]
                final[row_index, layer] = (
                    hidden[-1]
                    .float()
                    .cpu()
                    .numpy()
                )
                pooled[row_index, layer] = (
                    hidden[-window:]
                    .float()
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )

        return final, pooled
    finally:
        tokenizer.padding_side = original_padding_side


def cmd_extract(args):
    torch, AutoTokenizer, load_stage_model = ml_imports()

    benchmark_path, benchmark_sha, split_path, split_sha = (
        load_run_inputs(
            args.eval_set,
            args.benchmark_sha256,
            args.split_manifest,
        )
    )
    rows = load_jsonl(benchmark_path)

    if args.limit is not None:
        rows = rows[:args.limit]

    metadata = snapshot_for(rows)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_name = None

    for stage in args.stages:
        final_path = ACT_DIR / f"{stage}_final.npy"
        pooled_path = ACT_DIR / f"{stage}_pooled.npy"
        metadata_path = ACT_DIR / f"{stage}_metadata.json"
        binding_path = ACT_DIR / f"{stage}_metadata_binding.json"

        existing = [
            final_path,
            pooled_path,
            metadata_path,
            binding_path,
        ]

        if any(path.exists() for path in existing) and not args.force:
            raise FileExistsError(
                f"{stage} activation artifacts already exist. "
                "Pass --force explicitly."
            )

        print(f"\n=== Extracting activations: {stage} ===")
        model = load_stage_model(stage)
        device = next(model.parameters()).device
        device_name = str(device)

        final_batches = []
        pooled_batches = []

        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            final, pooled = activation_batch(
                model,
                tokenizer,
                [row["prompt"] for row in batch],
                device,
            )
            final_batches.append(final)
            pooled_batches.append(pooled)
            print(
                f"  {stage}: "
                f"{min(start + BATCH_SIZE, len(rows))}/{len(rows)}"
            )

        if not final_batches:
            raise RuntimeError(
                f"No activation batches produced for {stage}."
            )

        final_array = np.concatenate(final_batches, axis=0)
        pooled_array = np.concatenate(pooled_batches, axis=0)

        save_array(final_path, final_array)
        save_array(pooled_path, pooled_array)
        write_json_lf(metadata_path, metadata)
        write_json_lf(
            binding_path,
            {
                **binding(
                    benchmark_path,
                    benchmark_sha,
                    split_path,
                    split_sha,
                ),
                "stage": stage,
                "activation_shape_final": list(final_array.shape),
                "activation_shape_pooled": list(pooled_array.shape),
                "positions": {
                    "final": "last_nonpadding_prompt_token",
                    "pooled": "mean_last_five_nonpadding_tokens",
                },
                "device": device_name,
            },
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nActivation extraction completed.")


def cmd_direction(args):
    benchmark_path, benchmark_sha, split_path, split_sha = (
        load_run_inputs(
            args.eval_set,
            args.benchmark_sha256,
            args.split_manifest,
        )
    )
    benchmark_rows = load_jsonl(benchmark_path)
    expected = snapshot_for(benchmark_rows)

    directions = {}
    projections = {}

    for stage in args.stages:
        _, pooled, metadata = load_bound_activation(
            stage,
            benchmark_path,
            benchmark_sha,
            split_sha,
        )

        if metadata != expected:
            raise RuntimeError(
                f"{stage}: metadata differs from benchmark."
            )

        direction_path = (
            DIRECTION_DIR / f"{stage}_v2_direction.npy"
        )
        binding_path = (
            DIRECTION_DIR / f"{stage}_v2_direction_binding.json"
        )

        if direction_path.exists() and not args.force:
            raise FileExistsError(
                f"{direction_path} exists. Pass --force explicitly."
            )

        quadrants = np.asarray(
            [row.get("quadrant") for row in metadata]
        )
        splits = np.asarray(
            [row.get("split") for row in metadata]
        )

        a = pooled[
            (quadrants == "A")
            & (splits == "direction_estimation")
        ]
        d = pooled[
            (quadrants == "D")
            & (splits == "direction_estimation")
        ]

        if len(a) == 0 or len(d) == 0:
            raise RuntimeError(
                f"{stage}: A/D direction-estimation rows are required."
            )

        delta = a.mean(axis=0) - d.mean(axis=0)
        norms = np.linalg.norm(
            delta,
            axis=-1,
            keepdims=True,
        )
        direction = delta / np.where(norms == 0, 1.0, norms)

        save_array(direction_path, direction)
        write_json_lf(
            binding_path,
            {
                **binding(
                    benchmark_path,
                    benchmark_sha,
                    split_path,
                    split_sha,
                ),
                "stage": stage,
                "direction_shape": list(direction.shape),
                "construction": (
                    "mean(A_direction_estimation) - "
                    "mean(D_direction_estimation)"
                ),
            },
        )

        directions[stage] = direction

        by_quadrant = {}
        for quadrant in sorted(set(quadrants.tolist())):
            indices = [
                index
                for index, value in enumerate(quadrants)
                if value == quadrant
            ]
            by_quadrant[quadrant] = (
                np.einsum(
                    "nlh,lh->nl",
                    pooled[indices],
                    direction,
                )
                .mean(axis=0)
                .tolist()
            )

        projections[stage] = by_quadrant
        print(
            f"{stage}: direction shape {direction.shape}"
        )

    if not directions:
        raise RuntimeError("No directions were produced.")

    cosine = {}
    if "M0" in directions:
        cosine["vs_M0"] = {
            stage: np.sum(
                directions["M0"] * direction,
                axis=-1,
            ).tolist()
            for stage, direction in directions.items()
        }

    if "M3" in directions:
        cosine["vs_M3"] = {
            stage: np.sum(
                directions["M3"] * direction,
                axis=-1,
            ).tolist()
            for stage, direction in directions.items()
        }

    if not args.force:
        for path in (
            DIRECTION_DIR / "cosine_similarity_v2.json",
            DIRECTION_DIR / "quadrant_projections_v2.json",
        ):
            if path.exists():
                raise FileExistsError(
                    f"{path} exists. Pass --force explicitly."
                )

    write_json_lf(
        DIRECTION_DIR / "cosine_similarity_v2.json",
        cosine,
    )
    write_json_lf(
        DIRECTION_DIR / "quadrant_projections_v2.json",
        projections,
    )
    write_json_lf(
        DIRECTION_DIR / "v2_diagnostics_binding.json",
        {
            **binding(
                benchmark_path,
                benchmark_sha,
                split_path,
                split_sha,
            ),
            "stages": list(directions),
        },
    )


def cmd_behavior(args):
    torch, AutoTokenizer, load_stage_model = ml_imports()
    from src.analysis.eval_refusal_classifier import classify_refusal

    benchmark_path, benchmark_sha, split_path, split_sha = (
        load_run_inputs(
            args.eval_set,
            args.benchmark_sha256,
            args.split_manifest,
        )
    )
    rows = load_jsonl(benchmark_path)

    output_path = Path("results/behavioral_eval/v2_raw.json")
    binding_path = Path(
        "results/behavioral_eval/v2_binding.json"
    )

    if (
        (output_path.exists() or binding_path.exists())
        and not args.force
    ):
        raise FileExistsError(
            f"{output_path} exists. Pass --force explicitly."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    for stage in args.stages:
        print(f"\n=== Behavioral evaluation: {stage} ===")
        model = load_stage_model(stage)
        device = next(model.parameters()).device
        stage_rows = []

        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]
            completions = generation_batch(
                model,
                tokenizer,
                [row["prompt"] for row in batch],
                device,
            )

            for row, completion in zip(batch, completions):
                stage_rows.append(
                    {
                        "record_id": row.get("record_id"),
                        "prompt": row["prompt"],
                        "quadrant": row["quadrant"],
                        "source": row.get(
                            "source",
                            row.get("source_dataset"),
                        ),
                        "source_dataset": row.get(
                            "source_dataset"
                        ),
                        "c_construction": row.get(
                            "c_construction"
                        ),
                        "split": row.get("split"),
                        "stage": stage,
                        "completion": completion,
                        "refused": classify_refusal(completion),
                        "benchmark_sha256": benchmark_sha,
                        "split_manifest_sha256": split_sha,
                        "generation": {
                            "max_new_tokens": MAX_NEW_TOKENS,
                            "do_sample": False,
                            "repetition_penalty": 1.1,
                        },
                    }
                )

            print(
                f"  {stage}: "
                f"{min(start + BATCH_SIZE, len(rows))}/{len(rows)}"
            )

        results[stage] = stage_rows
        write_json_lf(output_path, results)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json_lf(
        binding_path,
        {
            **binding(
                benchmark_path,
                benchmark_sha,
                split_path,
                split_sha,
            ),
            "stages": list(results),
            "row_count_per_stage": len(rows),
        },
    )


def intervention_rows(rows):
    return [
        row
        for row in rows
        if (
            row.get("quadrant") not in {"A", "D"}
            or row.get("split") == "held_out_behavioral"
        )
    ]


def decoder_layers(model):
    try:
        return model.model.layers
    except AttributeError as exc:
        raise AttributeError(
            "Expected model.model.layers for Qwen-style checkpoint."
        ) from exc


def ablate_hidden(hidden, direction):
    import torch

    direction = direction.to(
        device=hidden.device,
        dtype=hidden.dtype,
    )
    projection = torch.einsum(
        "...h,h->...",
        hidden,
        direction,
    )
    return hidden - projection.unsqueeze(-1) * direction


def ablation_hook(direction):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (
                ablate_hidden(output[0], direction),
            ) + output[1:]
        return ablate_hidden(output, direction)

    return hook


def steering_hook(direction, alpha):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        direction_device = direction.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        steered = hidden + alpha * direction_device

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return hook


def run_condition(
    model,
    tokenizer,
    rows,
    device,
    condition,
    model_stage,
    benchmark_sha,
    split_sha,
):
    result = []

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        responses = generation_batch(
            model,
            tokenizer,
            [row["prompt"] for row in batch],
            device,
        )

        for row, response in zip(batch, responses):
            result.append(
                {
                    "record_id": row.get("record_id"),
                    "prompt": row["prompt"],
                    "quadrant": row["quadrant"],
                    "source": row.get(
                        "source",
                        row.get("source_dataset"),
                    ),
                    "source_dataset": row.get(
                        "source_dataset"
                    ),
                    "c_construction": row.get(
                        "c_construction"
                    ),
                    "split": row.get("split"),
                    "stage": condition,
                    "condition": condition,
                    "model_stage": model_stage,
                    "response": response,
                    "benchmark_sha256": benchmark_sha,
                    "split_manifest_sha256": split_sha,
                    "generation": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "do_sample": False,
                        "repetition_penalty": 1.1,
                    },
                }
            )

    return result


def cmd_causal(args):
    torch, AutoTokenizer, load_stage_model = ml_imports()

    benchmark_path, benchmark_sha, split_path, split_sha = (
        load_run_inputs(
            args.eval_set,
            args.benchmark_sha256,
            args.split_manifest,
        )
    )
    rows = intervention_rows(load_jsonl(benchmark_path))

    if not rows:
        raise RuntimeError("No rows remain for causal ablation.")

    direction_path = (
        DIRECTION_DIR
        / f"{args.stage}_v2_direction.npy"
    )
    binding_path = (
        DIRECTION_DIR
        / f"{args.stage}_v2_direction_binding.json"
    )
    assert_binding(binding_path, benchmark_sha, split_sha)

    if not direction_path.exists():
        raise FileNotFoundError(direction_path)

    output_path = (
        RAW_DIR
        / f"causal_ablation_v2_{args.stage}_L24-28.json"
    )
    binding_output = (
        RAW_DIR
        / f"causal_ablation_v2_{args.stage}_L24-28_binding.json"
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} exists. Pass --overwrite explicitly."
        )

    directions = np.load(direction_path)
    if directions.ndim != 2:
        raise RuntimeError(
            "Direction must have shape (layers, hidden_dim)."
        )
    if max(ABLATION_LAYERS) >= directions.shape[0]:
        raise RuntimeError(
            "Direction array does not contain all ablation layers."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_stage_model(args.stage)
    device = next(model.parameters()).device

    output_rows = run_condition(
        model,
        tokenizer,
        rows,
        device,
        f"{args.stage}_baseline",
        args.stage,
        benchmark_sha,
        split_sha,
    )

    handles = []
    try:
        blocks = decoder_layers(model)
        if max(ABLATION_LAYERS) > len(blocks):
            raise RuntimeError(
                "Requested ablation layer exceeds model depth."
            )

        for layer in ABLATION_LAYERS:
            handles.append(
                blocks[layer - 1].register_forward_hook(
                    ablation_hook(
                        torch.from_numpy(directions[layer])
                    )
                )
            )

        output_rows.extend(
            run_condition(
                model,
                tokenizer,
                rows,
                device,
                f"{args.stage}_ablated",
                args.stage,
                benchmark_sha,
                split_sha,
            )
        )
    finally:
        for handle in handles:
            handle.remove()

    write_json_lf(output_path, output_rows)
    write_json_lf(
        binding_output,
        {
            **binding(
                benchmark_path,
                benchmark_sha,
                split_path,
                split_sha,
            ),
            "stage": args.stage,
            "conditions": [
                f"{args.stage}_baseline",
                f"{args.stage}_ablated",
            ],
            "layers": ABLATION_LAYERS,
            "row_count": len(output_rows),
        },
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved {output_path}")


def calibration_alpha(
    stage,
    layer,
    direction,
    benchmark_sha,
    split_sha,
):
    pooled_path = ACT_DIR / f"{stage}_pooled.npy"
    metadata_path = ACT_DIR / f"{stage}_metadata.json"
    binding_path = (
        ACT_DIR
        / f"{stage}_metadata_binding.json"
    )

    assert_binding(binding_path, benchmark_sha, split_sha)

    pooled = np.load(pooled_path)
    metadata = load_json(metadata_path)

    indices = [
        index
        for index, row in enumerate(metadata)
        if (
            row.get("quadrant") == "A"
            and row.get("split")
            == "direction_estimation"
        )
    ]

    if not indices:
        raise RuntimeError(
            "No A direction-estimation rows are available "
            "for steering calibration."
        )

    if layer <= 0 or layer >= pooled.shape[1]:
        raise IndexError(
            f"Invalid steering layer {layer}."
        )

    return float(
        np.mean(
            pooled[indices, layer]
            @ direction[layer]
        )
    )


def cmd_steering(args):
    torch, AutoTokenizer, load_stage_model = ml_imports()

    benchmark_path, benchmark_sha, split_path, split_sha = (
        load_run_inputs(
            args.eval_set,
            args.benchmark_sha256,
            args.split_manifest,
        )
    )

    direction_path = (
        DIRECTION_DIR
        / f"{args.stage}_v2_direction.npy"
    )
    direction_binding = (
        DIRECTION_DIR
        / f"{args.stage}_v2_direction_binding.json"
    )
    assert_binding(
        direction_binding,
        benchmark_sha,
        split_sha,
    )

    direction = np.load(direction_path)
    layers = sorted(set(args.layers))

    if args.alpha_source == "fixed":
        if args.alpha_value is None:
            raise ValueError(
                "--alpha-value is required with fixed alpha."
            )
        base_alphas = {
            layer: args.alpha_value
            for layer in layers
        }
    else:
        base_alphas = {
            layer: calibration_alpha(
                args.stage,
                layer,
                direction,
                benchmark_sha,
                split_sha,
            )
            for layer in layers
        }

    alphas = {
        layer: value * args.alpha_coefficient
        for layer, value in base_alphas.items()
    }

    tag = args.tag or (
        f"{args.stage}_L"
        f"{'-'.join(str(layer) for layer in layers)}_"
        f"{args.alpha_source}_coef"
        f"{args.alpha_coefficient:g}_Q"
        f"{''.join(args.quadrants)}"
    )

    output_path = RAW_DIR / f"steering_v2_{tag}.json"
    binding_output = (
        RAW_DIR
        / f"steering_v2_{tag}_binding.json"
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} exists. Pass --overwrite explicitly."
        )

    rows = [
        row
        for row in load_jsonl(benchmark_path)
        if row.get("quadrant") in args.quadrants
        and (
            row.get("quadrant") not in {"A", "D"}
            or row.get("split") == "held_out_behavioral"
        )
    ]

    if not rows:
        raise RuntimeError("No rows remain for steering.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_stage_model(args.stage)
    device = next(model.parameters()).device

    baseline_name = f"{tag}_baseline"
    steered_name = f"{tag}_steered"

    output_rows = run_condition(
        model,
        tokenizer,
        rows,
        device,
        baseline_name,
        args.stage,
        benchmark_sha,
        split_sha,
    )

    handles = []
    try:
        blocks = decoder_layers(model)

        for layer in layers:
            if layer <= 0 or layer > len(blocks):
                raise IndexError(
                    f"Invalid steering layer {layer}."
                )

            handles.append(
                blocks[layer - 1].register_forward_hook(
                    steering_hook(
                        torch.from_numpy(direction[layer]),
                        alphas[layer],
                    )
                )
            )

        output_rows.extend(
            run_condition(
                model,
                tokenizer,
                rows,
                device,
                steered_name,
                args.stage,
                benchmark_sha,
                split_sha,
            )
        )
    finally:
        for handle in handles:
            handle.remove()

    write_json_lf(output_path, output_rows)
    write_json_lf(
        binding_output,
        {
            **binding(
                benchmark_path,
                benchmark_sha,
                split_path,
                split_sha,
            ),
            "stage": args.stage,
            "layers": layers,
            "alpha_source": args.alpha_source,
            "alpha_coefficient": args.alpha_coefficient,
            "alphas_by_layer": alphas,
            "quadrants": args.quadrants,
            "conditions": [
                baseline_name,
                steered_name,
            ],
            "row_count": len(output_rows),
        },
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved {output_path}")


def cmd_probes(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
    )

    benchmark_path, benchmark_sha, split_path, split_sha = (
        load_run_inputs(
            args.eval_set,
            args.benchmark_sha256,
            args.split_manifest,
        )
    )
    benchmark_rows = load_jsonl(benchmark_path)

    for stage in args.stages:
        final, _, metadata = load_bound_activation(
            stage,
            benchmark_path,
            benchmark_sha,
            split_sha,
        )

        by_quadrant = {
            quadrant: [
                index
                for index, row in enumerate(metadata)
                if row.get("quadrant") == quadrant
            ]
            for quadrant in ["A", "B", "C", "D"]
        }

        if len(by_quadrant["B"]) < 50:
            raise RuntimeError(
                f"{stage}: fewer than 50 B rows."
            )

        output_path = (
            PROBE_DIR / f"{stage}_probe_results.json"
        )
        binding_path = (
            PROBE_DIR / f"{stage}_probe_binding.json"
        )

        if output_path.exists() and not args.force:
            raise FileExistsError(
                f"{output_path} exists. Pass --force explicitly."
            )

        rng = np.random.RandomState(42)
        b_indices = np.asarray(by_quadrant["B"])
        rng.shuffle(b_indices)

        train_a = np.asarray(by_quadrant["A"])
        train_b = b_indices[:50]
        holdout_b = b_indices[50:]
        test_c = np.asarray(by_quadrant["C"])
        test_d = np.asarray(by_quadrant["D"])

        results = []

        for layer in range(final.shape[1]):
            x_train = np.concatenate(
                [
                    final[train_a, layer],
                    final[train_b, layer],
                ],
                axis=0,
            )
            y_train = np.concatenate(
                [
                    np.ones(len(train_a)),
                    np.zeros(len(train_b)),
                ]
            )

            classifier = LogisticRegression(
                max_iter=2000,
                random_state=42,
            )
            folds = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            )
            scores = cross_val_score(
                classifier,
                x_train,
                y_train,
                cv=folds,
            )
            classifier.fit(x_train, y_train)

            def flagged(indices):
                if len(indices) == 0:
                    return None
                return float(
                    classifier.predict(
                        final[indices, layer]
                    ).mean()
                )

            results.append(
                {
                    "layer": layer,
                    "cv_accuracy_mean": float(scores.mean()),
                    "cv_accuracy_std": float(scores.std()),
                    "cv_fold_scores": [
                        float(value)
                        for value in scores
                    ],
                    "holdout_b_flagged_unsafe_frac": (
                        flagged(holdout_b)
                    ),
                    "quadrant_c_flagged_unsafe_frac": (
                        flagged(test_c)
                    ),
                    "quadrant_d_flagged_unsafe_frac": (
                        flagged(test_d)
                    ),
                }
            )

        PROBE_DIR.mkdir(parents=True, exist_ok=True)
        write_json_lf(output_path, results)
        write_json_lf(
            binding_path,
            {
                **binding(
                    benchmark_path,
                    benchmark_sha,
                    split_path,
                    split_sha,
                ),
                "stage": stage,
                "layer_selection": (
                    "none; all layers retained; "
                    "C not used for selection"
                ),
            },
        )

        print(f"Saved {output_path}")


def command_common(
    benchmark_path,
    benchmark_sha,
    split_path,
):
    return [
        "--eval-set",
        str(benchmark_path),
        "--benchmark-sha256",
        benchmark_sha,
        "--split-manifest",
        str(split_path),
    ]


def gate_for_run(args):
    status_path = Path(
        "logs/benchmark_validation_status.json"
    )
    gate_path = Path(
        "logs/benchmark_gate_config.json"
    )

    if not status_path.exists():
        raise FileNotFoundError(status_path)
    if not gate_path.exists():
        raise FileNotFoundError(gate_path)

    status = load_json(status_path)
    gate = load_json(gate_path)

    benchmark_path, benchmark_sha, split_path, split_sha = (
        load_run_inputs(
            None,
            None,
            "logs/direction_split_manifest.json",
        )
    )

    static_fields = [
        "schema_integrity_pass",
        "prompt_integrity_pass",
        "c_review_pass",
        "c_review_mapping_pass",
        "benchmark_hash_pass",
        "split_benchmark_hash_pass",
        "split_hash_pass",
    ]

    failures = [
        f"{field}={status.get(field)!r}"
        for field in static_fields
        if status.get(field) is not True
    ]

    if failures:
        raise RuntimeError(
            "Static benchmark gate failed: "
            + ", ".join(failures)
        )

    if status.get("technical_benchmark_status") != "PASS":
        if not args.regenerate:
            raise RuntimeError(
                "technical_benchmark_status is not PASS. "
                "Use --regenerate to rebuild stale model artifacts."
            )

        if status.get("artifact_freshness_pass") is not False:
            raise RuntimeError(
                "Technical validation failed for a reason other "
                "than stale artifact freshness."
            )

        print(
            "Only artifact freshness is failing; "
            "explicit regeneration is permitted."
        )

    for field in gate.get("warning_only_gate_fields", []):
        print(
            f"warning-only {field}: "
            f"{status.get(field)!r}"
        )

    print(f"Frozen benchmark: {benchmark_path}")
    print(f"Benchmark SHA-256: {benchmark_sha}")
    print(f"Split manifest: {split_path}")
    print(f"Split SHA-256: {split_sha}")

    return (
        benchmark_path,
        benchmark_sha,
        split_path,
        split_sha,
    )


def build_run_commands(args, benchmark_path, benchmark_sha, split_path):
    common = command_common(
        benchmark_path,
        benchmark_sha,
        split_path,
    )

    commands = [
        [
            sys.executable,
            "-m",
            "src.analysis.v2_pipeline",
            "extract",
            *common,
            "--stages",
            *args.analysis_stage,
            "--force",
        ],
        [
            sys.executable,
            "-m",
            "src.analysis.v2_pipeline",
            "direction",
            *common,
            "--stages",
            *args.analysis_stage,
            "--force",
        ],
        [
            sys.executable,
            "-m",
            "src.analysis.v2_pipeline",
            "behavior",
            *common,
            "--stages",
            *args.analysis_stage,
            "--force",
        ],
    ]

    if args.with_probes:
        commands.append(
            [
                sys.executable,
                "-m",
                "src.analysis.v2_pipeline",
                "probes",
                *common,
                "--stages",
                *args.analysis_stage,
                "--force",
            ]
        )

    commands.append(
        [
            sys.executable,
            "-m",
            "src.validate_benchmark_v2",
            "--benchmark",
            str(benchmark_path),
            "--review-csv",
            "data/review/c_review_queue.csv",
            "--gate-config",
            "logs/benchmark_gate_config.json",
            "--split-manifest",
            str(split_path),
        ]
    )

    for stage in args.stage:
        if not args.no_causal:
            command = [
                sys.executable,
                "-m",
                "src.analysis.v2_pipeline",
                "causal",
                *common,
                "--stage",
                stage,
            ]
            if args.overwrite:
                command.append("--overwrite")
            commands.append(command)

        if not args.no_steering:
            command = [
                sys.executable,
                "-m",
                "src.analysis.v2_pipeline",
                "steering",
                *common,
                "--stage",
                stage,
                "--layers",
                "24",
                "--alpha-source",
                "direction_estimation_only",
                "--quadrants",
                "A",
                "B",
                "C",
                "D",
            ]
            if args.overwrite:
                command.append("--overwrite")
            commands.append(command)

    return commands


def main_run(args):
    (
        benchmark_path,
        benchmark_sha,
        split_path,
        split_sha,
    ) = gate_for_run(args)

    commands = build_run_commands(
        args,
        benchmark_path,
        benchmark_sha,
        split_path,
    )

    print("\nExact commands:")
    for command in commands:
        print("$ " + " ".join(str(value) for value in command))

    if args.dry_run:
        print("\nDry run complete. No model code was executed.")
        return

    if not args.regenerate:
        raise RuntimeError(
            "A live run requires --regenerate."
        )

    for command in commands:
        print("\nExecuting:")
        print("$ " + " ".join(str(value) for value in command))
        subprocess.run(command, check=True)

        if "validate_benchmark_v2" in command:
            status = load_json(
                "logs/benchmark_validation_status.json"
            )
            if status.get("technical_benchmark_status") != "PASS":
                raise RuntimeError(
                    "Fresh activation validation did not pass."
                )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    manifest_path = (
        MANIFEST_DIR
        / f"v2_run_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    write_json_lf(
        manifest_path,
        {
            "component": "v2_pipeline",
            "created_at_utc": timestamp.isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "benchmark_path": str(benchmark_path),
            "benchmark_sha256": benchmark_sha,
            "split_manifest_path": str(split_path),
            "split_manifest_sha256": split_sha,
            "analysis_stages": args.analysis_stage,
            "intervention_stages": args.stage,
            "commands": commands,
        },
    )

    print(f"\nRun manifest: {manifest_path}")
    print("v2 GPU pipeline completed.")


def add_common(parser):
    parser.add_argument("--eval-set", default=None)
    parser.add_argument("--benchmark-sha256", default=None)
    parser.add_argument(
        "--split-manifest",
        default="logs/direction_split_manifest.json",
    )


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    extract = subparsers.add_parser("extract")
    add_common(extract)
    extract.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
    )
    extract.add_argument("--force", action="store_true")
    extract.add_argument("--limit", type=int)

    direction = subparsers.add_parser("direction")
    add_common(direction)
    direction.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
    )
    direction.add_argument("--force", action="store_true")

    behavior = subparsers.add_parser("behavior")
    add_common(behavior)
    behavior.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
    )
    behavior.add_argument("--force", action="store_true")

    probes = subparsers.add_parser("probes")
    add_common(probes)
    probes.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
    )
    probes.add_argument("--force", action="store_true")

    causal = subparsers.add_parser("causal")
    add_common(causal)
    causal.add_argument(
        "--stage",
        required=True,
        choices=ALL_STAGES,
    )
    causal.add_argument("--overwrite", action="store_true")

    steering = subparsers.add_parser("steering")
    add_common(steering)
    steering.add_argument(
        "--stage",
        required=True,
        choices=ALL_STAGES,
    )
    steering.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[24],
    )
    steering.add_argument(
        "--alpha-source",
        choices=["direction_estimation_only", "fixed"],
        default="direction_estimation_only",
    )
    steering.add_argument("--alpha-value", type=float)
    steering.add_argument(
        "--alpha-coefficient",
        type=float,
        default=1.0,
    )
    steering.add_argument(
        "--quadrants",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
    )
    steering.add_argument("--tag")
    steering.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument(
        "--analysis-stage",
        action="append",
        choices=ALL_STAGES,
        default=None,
    )
    run.add_argument(
        "--stage",
        action="append",
        choices=INTERVENTION_STAGES,
        default=None,
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--regenerate",
        "--force-regen",
        action="store_true",
    )
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--with-probes", action="store_true")
    run.add_argument("--no-causal", action="store_true")
    run.add_argument("--no-steering", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "direction":
        cmd_direction(args)
    elif args.command == "behavior":
        cmd_behavior(args)
    elif args.command == "probes":
        cmd_probes(args)
    elif args.command == "causal":
        cmd_causal(args)
    elif args.command == "steering":
        cmd_steering(args)
    elif args.command == "run":
        if args.analysis_stage is None:
            args.analysis_stage = list(ALL_STAGES)
        if args.stage is None:
            args.stage = list(INTERVENTION_STAGES)
        main_run(args)


if __name__ == "__main__":
    main()
''',
)

write_file(
    "rerun_mechanistic_v2.sh",
    r'''
#!/usr/bin/env bash
set -euo pipefail
exec python -m src.analysis.v2_pipeline run "$@"
''',
)

write_file(
    "tests/test_v2_io.py",
    r'''
import json

from src.v2_io import (
    canonical_json,
    identity_snapshot,
    sha256_bytes,
    write_json_lf,
)


def test_identity_snapshot():
    rows = [
        {
            "record_id": "r1",
            "prompt": "hello",
            "quadrant": "A",
            "source": "HarmBench",
            "split": "direction_estimation",
        }
    ]

    assert identity_snapshot(rows) == [
        {
            "record_id": "r1",
            "prompt": "hello",
            "scored_prompt": None,
            "quadrant": "A",
            "source": "HarmBench",
            "source_dataset": None,
            "c_construction": None,
            "split": "direction_estimation",
        }
    ]


def test_canonical_json_is_key_order_independent():
    first = {"a": 1, "b": 2}
    second = {"b": 2, "a": 1}

    assert canonical_json(first) == canonical_json(second)
    assert sha256_bytes(canonical_json(first)) == (
        sha256_bytes(canonical_json(second))
    )


def test_write_json_is_lf(tmp_path):
    path = tmp_path / "result.json"
    write_json_lf(path, {"text": "ž"})

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8")) == {"text": "ž"}
''',
)

notebook_path = ROOT / "notebooks" / "colab_unified_analysis.ipynb"
if not notebook_path.exists():
    notebook_path = ROOT / "colab_unified_analysis.ipynb"

if not notebook_path.exists():
    raise RuntimeError(
        "Could not find colab_unified_analysis.ipynb."
    )

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Frozen v2 mechanistic rerun\n",
                "\n",
                "This notebook uses only the strict benchmark-bound v2 "
                "runner. It does not use the legacy mutable-eval-set "
                "pipeline.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import subprocess\n",
                "\n",
                "REPO_URL = "
                "'https://github.com/urosavurdic/"
                "dpo-safety-representations.git'\n",
                "REPO_DIR = "
                "'/content/dpo-safety-representations'\n",
                "BRANCH = "
                "'agent/c-quadrant-end-to-end-e0e2317a'\n",
                "PINNED_COMMIT = "
                "'REPLACE_AFTER_PUSH_WITH_COMMIT_SHA'\n",
                "\n",
                "if not os.path.exists(REPO_DIR):\n",
                "    subprocess.run([\n",
                "        'git', 'clone', '-b', BRANCH,\n",
                "        REPO_URL, REPO_DIR,\n",
                "    ], check=True)\n",
                "\n",
                "os.chdir(REPO_DIR)\n",
                "subprocess.run(['git', 'fetch', 'origin'], check=True)\n",
                "subprocess.run([\n",
                "    'git', 'checkout', BRANCH,\n",
                "], check=True)\n",
                "subprocess.run([\n",
                "    'git', 'pull', '--ff-only', 'origin', BRANCH,\n",
                "], check=True)\n",
                "\n",
                "commit = subprocess.check_output([\n",
                "    'git', 'rev-parse', 'HEAD',\n",
                "], text=True).strip()\n",
                "assert PINNED_COMMIT != "
                "'REPLACE_AFTER_PUSH_WITH_COMMIT_SHA', "
                "'Set PINNED_COMMIT after pushing the patch.'\n",
                "assert commit == PINNED_COMMIT, "
                "f'Wrong commit: {commit}'\n",
                "print('Checked out exact commit:', commit)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python -m pip install -q -r requirements.txt\n",
                "!python -m pip uninstall -y torchao || true\n",
                "!python -m compileall src\n",
                "!pytest tests/ -q\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import subprocess\n",
                "\n",
                "latest = json.load(open(\n",
                "    'data/frozen_v2/LATEST_BENCHMARK.json'\n",
                "))\n",
                "bench = latest['benchmark_path']\n",
                "\n",
                "subprocess.run([\n",
                "    'python', '-m',\n",
                "    'src.create_direction_split_manifest',\n",
                "    '--benchmark', bench,\n",
                "], check=True)\n",
                "\n",
                "subprocess.run([\n",
                "    'python', '-m',\n",
                "    'src.validate_benchmark_v2',\n",
                "    '--benchmark', bench,\n",
                "    '--review-csv',\n",
                "    'data/review/c_review_queue.csv',\n",
                "    '--gate-config',\n",
                "    'logs/benchmark_gate_config.json',\n",
                "    '--split-manifest',\n",
                "    'logs/direction_split_manifest.json',\n",
                "], check=True)\n",
                "\n",
                "status = json.load(open(\n",
                "    'logs/benchmark_validation_status.json'\n",
                "))\n",
                "static_fields = [\n",
                "    'schema_integrity_pass',\n",
                "    'prompt_integrity_pass',\n",
                "    'c_review_pass',\n",
                "    'c_review_mapping_pass',\n",
                "    'benchmark_hash_pass',\n",
                "    'split_benchmark_hash_pass',\n",
                "    'split_hash_pass',\n",
                "]\n",
                "assert all(status.get(k) is True for k in static_fields), status\n",
                "print('Static benchmark checks passed.')\n",
                "print('Artifact freshness:', status['artifact_freshness_pass'])\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!bash rerun_mechanistic_v2.sh "
                "--dry-run --regenerate --with-probes\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "RUN_GPU = False\n",
                "if RUN_GPU:\n",
                "    !bash rerun_mechanistic_v2.sh "
                "--regenerate --with-probes\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if RUN_GPU:\n",
                "    import json\n",
                "    import subprocess\n",
                "    latest = json.load(open(\n",
                "        'data/frozen_v2/LATEST_BENCHMARK.json'\n",
                "    ))\n",
                "    subprocess.run([\n",
                "        'python', '-m',\n",
                "        'src.validate_benchmark_v2',\n",
                "        '--benchmark', latest['benchmark_path'],\n",
                "        '--review-csv',\n",
                "        'data/review/c_review_queue.csv',\n",
                "        '--gate-config',\n",
                "        'logs/benchmark_gate_config.json',\n",
                "        '--split-manifest',\n",
                "        'logs/direction_split_manifest.json',\n",
                "    ], check=True)\n",
                "    status = json.load(open(\n",
                "        'logs/benchmark_validation_status.json'\n",
                "    ))\n",
                "    assert status['technical_benchmark_status'] == 'PASS', status\n",
                "    print('Fresh v2 validation passed.')\n",
            ],
        },
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with notebook_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    json.dump(
        notebook,
        handle,
        ensure_ascii=False,
        indent=1,
    )
    handle.write("\n")

print(f"updated {notebook_path.relative_to(ROOT)}")

gitattributes = ROOT / ".gitattributes"
existing = (
    gitattributes.read_text(encoding="utf-8")
    if gitattributes.exists()
    else ""
)
required_lines = [
    "* text=auto eol=lf",
    "*.json text eol=lf",
    "*.jsonl text eol=lf",
    "*.csv text eol=lf",
    "*.md text eol=lf",
    "*.py text eol=lf",
    "*.sh text eol=lf",
    "*.ipynb text eol=lf",
]
for line in required_lines:
    if line not in existing.splitlines():
        existing += line + "\n"

with gitattributes.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    handle.write(existing)

print("updated .gitattributes")

run("git", "diff", "--check")
run("python", "-m", "compileall", "src")

patch_dir = ROOT / "assets"
patch_dir.mkdir(parents=True, exist_ok=True)
patch_path = patch_dir / "c_quadrant_colab_fixes.patch"

diff = subprocess.check_output(
    [
        "git",
        "diff",
        "--binary",
        "HEAD",
        "--",
        ".",
        ":(exclude)assets/c_quadrant_colab_fixes.patch",
        ":(exclude)artifacts/patches/*.patch",
    ],
    cwd=ROOT,
)

if not diff:
    raise RuntimeError(
        "Git produced an empty diff. No fixes were generated."
    )

patch_path.write_bytes(diff)
print(f"\nValid Git patch written to: {patch_path}")
print(f"Patch bytes: {patch_path.stat().st_size}")
print("\nRun this verification on the current tree:")
print("  git diff --check")
print("  git apply --check --reverse assets/c_quadrant_colab_fixes.patch")
