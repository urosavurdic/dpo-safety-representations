"""Build the C ``source_overt`` adjunct companion set (WP-Adjunct).

For every quadrant-C row in the FROZEN benchmark, emit a companion row whose
prompt is the row's overt StrongREJECT ``source_prompt`` (not the reduced-cue
candidate), tagged ``judged_prompt_variant = "source_overt"`` and sharing the
C row's ``pair_id`` / ``record_id`` stem. This companion set is:

  * a SEPARATE namespaced file with its OWN pointer + SHA - it is hash-bound
    just as strictly (``src.v2_io.load_run_inputs`` supports a companion
    ``latest_path``);
  * NEVER written into ``data/processed/controlled_eval.jsonl`` or the frozen
    654-row benchmark - this script asserts both are untouched;
  * used only for the ``source_overt`` labelled SECONDARY judge pass and the
    matched-pair representation analysis (``matched_pair_representation.py``).

CPU-only. Idempotent: re-running with an unchanged frozen benchmark produces a
byte-identical file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2_io import (
    canonical_json,
    load_jsonl,
    resolve_benchmark,
    sha256_bytes,
    sha256_file,
    split_payload,
)

DEFAULT_OUT = Path("data/frozen_v2/adjunct_c_source_overt.jsonl")
DEFAULT_POINTER = Path("data/frozen_v2/adjunct_c_source_overt.LATEST.json")
# LATEST_BENCHMARK-shaped pointer + split manifest so `v2_pipeline extract
# --latest-pointer ... --split-manifest ... --namespace c_source_overt` binds
# the companion set the same strict way as the main benchmark.
DEFAULT_LATEST_BENCHMARK = Path("data/frozen_v2/adjunct_c_source_overt.LATEST_BENCHMARK.json")
DEFAULT_SPLIT_MANIFEST = Path("data/frozen_v2/adjunct_c_source_overt.split_manifest.json")
FROZEN_C_COUNT = 104
CONTROLLED_EVAL = Path("data/processed/controlled_eval.jsonl")


def build_rows(benchmark_rows):
    rows = []
    for row in benchmark_rows:
        if row.get("quadrant") != "C":
            continue
        overt = row.get("source_prompt")
        if not overt:
            raise RuntimeError(
                f"C row {row.get('record_id')} has no source_prompt - cannot "
                "build the source_overt adjunct."
            )
        rows.append({
            "record_id": f"{row['record_id']}__source_overt",
            "companion_of": row["record_id"],
            "pair_id": row.get("pair_id"),
            "quadrant": "C",
            "judged_prompt_variant": "source_overt",
            "prompt": overt,
            "scored_prompt": overt,
            "project_category": row.get("project_category") or row.get("source_category"),
            "source_dataset": row.get("source_dataset"),
            "source_id": row.get("source_id"),
            "split": None,
        })
    return rows


def run(latest_path=None, out_path=DEFAULT_OUT, pointer_path=DEFAULT_POINTER,
        latest_benchmark_path=DEFAULT_LATEST_BENCHMARK,
        split_manifest_path=DEFAULT_SPLIT_MANIFEST):
    latest_benchmark_path = Path(latest_benchmark_path)
    split_manifest_path = Path(split_manifest_path)
    bench_path, bench_sha = resolve_benchmark(
        **({"latest_path": latest_path} if latest_path else {})
    )
    before_controlled = sha256_file(CONTROLLED_EVAL) if CONTROLLED_EVAL.exists() else None
    before_bench = sha256_file(bench_path)

    benchmark_rows = load_jsonl(bench_path)
    c_count = sum(1 for r in benchmark_rows if r.get("quadrant") == "C")
    if bench_sha == "e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b" \
            and c_count != FROZEN_C_COUNT:
        raise RuntimeError(f"frozen benchmark C count is {c_count}, expected {FROZEN_C_COUNT}")

    rows = build_rows(benchmark_rows)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    adjunct_sha = sha256_file(out_path)

    pointer = {
        "kind": "c_source_overt_adjunct",
        "adjunct_path": str(out_path).replace("\\", "/"),
        "adjunct_sha256": adjunct_sha,
        "benchmark_sha256": bench_sha,
        "n_rows": len(rows),
        "note": (
            "Companion set for the source_overt SECONDARY judge pass + matched-"
            "pair representation. NOT part of the frozen 654-row benchmark; never "
            "in controlled_eval.jsonl."
        ),
    }
    Path(pointer_path).write_text(json.dumps(pointer, indent=2), encoding="utf-8")

    # LATEST_BENCHMARK-shaped pointer (so resolve_benchmark accepts it) ------
    latest_benchmark = {
        "benchmark_path": str(out_path).replace("\\", "/"),
        "benchmark_sha256": adjunct_sha,
        "kind": "c_source_overt_adjunct",
        "frozen_benchmark_sha256": bench_sha,
    }
    latest_benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    latest_benchmark_path.write_text(json.dumps(latest_benchmark, indent=2), encoding="utf-8")

    # companion split manifest bound to the adjunct sha. matched-pair rep does
    # NOT use a direction-estimation split, so every row goes in one bucket;
    # the manifest exists only so load_run_inputs' strict binding passes.
    ids = [r["record_id"] for r in rows]
    split_manifest = {
        "benchmark_sha256": adjunct_sha,
        "direction_split_seed": None,
        "direction_train_fraction": None,
        "split_algorithm": "companion_all_in_direction_estimation (unused for matched-pair rep)",
        "record_ids_direction_estimation": ids,
        "record_ids_held_out_behavioral": [],
        "counts": {"direction_estimation": len(ids), "held_out_behavioral": 0},
        "split_hash_algorithm": "sha256_canonical_json_without_hash_fields",
    }
    split_manifest["split_manifest_sha256"] = sha256_bytes(
        canonical_json(split_payload(split_manifest))
    )
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    pointer["latest_benchmark_pointer"] = str(latest_benchmark_path).replace("\\", "/")
    pointer["split_manifest"] = str(split_manifest_path).replace("\\", "/")

    # assertions: the frozen benchmark and controlled_eval were NOT modified
    assert sha256_file(bench_path) == before_bench, "frozen benchmark changed!"
    if before_controlled is not None:
        assert sha256_file(CONTROLLED_EVAL) == before_controlled, "controlled_eval.jsonl changed!"
    assert out_path.resolve() != CONTROLLED_EVAL.resolve()

    return pointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-path", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--pointer", default=str(DEFAULT_POINTER))
    args = parser.parse_args()
    pointer = run(args.latest_path, args.out, args.pointer)
    print(f"wrote {pointer['n_rows']} source_overt rows -> {pointer['adjunct_path']}")
    print(f"adjunct sha256 = {pointer['adjunct_sha256']}")
    print("\nextract its activations with:")
    print(f"  python -m src.analysis.v2_pipeline extract --stage M3 \\")
    print(f"    --latest-pointer {pointer['latest_benchmark_pointer']} \\")
    print(f"    --split-manifest {pointer['split_manifest']} \\")
    print(f"    --namespace c_source_overt")


if __name__ == "__main__":
    main()
