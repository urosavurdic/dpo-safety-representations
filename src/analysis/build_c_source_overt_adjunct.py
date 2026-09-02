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

from src.v2_io import canonical_json, load_jsonl, resolve_benchmark, sha256_bytes, sha256_file

DEFAULT_OUT = Path("data/frozen_v2/adjunct_c_source_overt.jsonl")
DEFAULT_POINTER = Path("data/frozen_v2/adjunct_c_source_overt.LATEST.json")
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


def run(latest_path=None, out_path=DEFAULT_OUT, pointer_path=DEFAULT_POINTER):
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


if __name__ == "__main__":
    main()
