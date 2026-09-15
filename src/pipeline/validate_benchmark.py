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
    normalize_json_path,
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


def _stage_artifact_missing(activation_dir: Path, stage: str) -> list[str]:
    """Return the repository-contract (POSIX-style) paths of any of the
    four required per-stage activation artifacts that do not exist.

    Uses .as_posix(), not str(): str(Path(...)) renders with the host OS
    separator, so on a Windows host this would silently emit
    "results\\activations\\M0_metadata.json" instead of the stable
    "results/activations/M0_metadata.json" the rest of the repository
    (e.g. benchmark_path.as_posix() below) already commits to. That
    divergence is invisible on a POSIX CI host, where str() and
    as_posix() happen to coincide -- see
    test_missing_stage_artifacts_reports_posix_paths_even_on_windows_like_path
    for a platform-independent regression guard.
    """
    final_path = activation_dir / f"{stage}_final.npy"
    pooled_path = activation_dir / f"{stage}_pooled.npy"
    metadata_path = activation_dir / f"{stage}_metadata.json"
    binding_path = activation_dir / f"{stage}_metadata_binding.json"

    return [
        path.as_posix()
        for path in (final_path, pooled_path, metadata_path, binding_path)
        if not path.exists()
    ]


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

    # normalize_json_path, not Path: these values are routinely copied
    # straight out of LATEST_BENCHMARK.json / a manifest, which may have
    # been written on Windows with backslash separators. Path() would
    # treat "data\frozen_v2\x.jsonl" as a single literal filename on
    # Linux and fail with FileNotFoundError on Colab.
    benchmark_path = normalize_json_path(args.benchmark)
    review_path = normalize_json_path(args.review_csv)
    gate_path = normalize_json_path(args.gate_config)
    split_path = normalize_json_path(args.split_manifest)

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

        missing = _stage_artifact_missing(activation_dir, stage)

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
                out_dir / "benchmark_validation_status.json"
            ).as_posix(),
            "validation_report": (
                out_dir / "benchmark_validation_report.md"
            ).as_posix(),
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
        f"{status['outputs']['validation_status']}"
    )
    print(
        "Markdown report: "
        f"{status['outputs']['validation_report']}"
    )

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
