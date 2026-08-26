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
