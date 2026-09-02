"""CPU-only verifier for frozen-v2 activation artifacts (WP-Repro).

`v2_pipeline.py` already refuses to *use* an activation set that is not bound
to the frozen benchmark (`activations_bound`). This standalone script reports
that binding status for every stage without importing torch, so a researcher
can confirm on a laptop that `results/activations/` (as pulled from Drive) is
the frozen-v2 set before running any downstream CPU analysis.

Per stage it checks:
  * the 4 files exist: {stage}_final.npy, {stage}_pooled.npy,
    {stage}_metadata.json, {stage}_metadata_binding.json;
  * the binding sidecar's benchmark_sha256 / split_manifest_sha256 match the
    frozen inputs (via src.v2_io.assert_binding);
  * the metadata row identity (record_id / quadrant / split list) matches the
    frozen benchmark exactly, in order;
  * the .npy row counts equal the metadata row count (header-only read, arrays
    not loaded into memory).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2_io import (
    assert_binding,
    identity_snapshot,
    load_json,
    load_jsonl,
    load_run_inputs,
)

ALL_STAGES = [
    "M0", "M1", "M2", "M3", "M3_direct",
    "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt",
]
ACT_DIR = Path("results/activations")


def _npy_rows(path: Path) -> int:
    """Row count from a .npy file without materialising the array (mmap)."""
    import numpy as np

    return int(np.load(path, mmap_mode="r").shape[0])


def verify_stage(
    stage: str,
    benchmark_rows: list[dict],
    benchmark_sha256: str,
    split_manifest_sha256: str,
    act_dir: Path = ACT_DIR,
) -> dict:
    problems: list[str] = []
    final_p = act_dir / f"{stage}_final.npy"
    pooled_p = act_dir / f"{stage}_pooled.npy"
    meta_p = act_dir / f"{stage}_metadata.json"
    bind_p = act_dir / f"{stage}_metadata_binding.json"

    present = {p.name: p.exists() for p in (final_p, pooled_p, meta_p, bind_p)}
    if not all(present.values()):
        return {"stage": stage, "status": "absent",
                "missing": [n for n, ok in present.items() if not ok]}

    try:
        assert_binding(bind_p, benchmark_sha256, split_manifest_sha256)
    except (RuntimeError, FileNotFoundError) as exc:
        problems.append(f"binding: {exc}")

    metadata = load_json(meta_p)
    expected = identity_snapshot(benchmark_rows)
    if metadata != expected:
        problems.append(
            f"metadata identity mismatch (rows: got {len(metadata)}, "
            f"expected {len(expected)})"
        )

    for arr in (final_p, pooled_p):
        try:
            n = _npy_rows(arr)
        except Exception as exc:  # pragma: no cover - corrupt file
            problems.append(f"{arr.name}: unreadable header ({exc})")
            continue
        if n != len(metadata):
            problems.append(f"{arr.name}: {n} rows != metadata {len(metadata)}")

    return {
        "stage": stage,
        "status": "ok" if not problems else "mismatch",
        "problems": problems,
        "n_rows": len(metadata),
    }


def verify_all(act_dir: Path = ACT_DIR, *, latest_path=None, split_manifest=None) -> dict:
    kwargs = {}
    if latest_path is not None:
        kwargs["latest_path"] = latest_path
    if split_manifest is not None:
        kwargs["split_manifest"] = split_manifest
    bench_path, bench_sha, _split_path, split_sha = load_run_inputs(**kwargs)
    benchmark_rows = load_jsonl(bench_path)

    reports = [
        verify_stage(stage, benchmark_rows, bench_sha, split_sha, act_dir)
        for stage in ALL_STAGES
    ]
    return {
        "benchmark_sha256": bench_sha,
        "split_manifest_sha256": split_sha,
        "n_benchmark_rows": len(benchmark_rows),
        "stages": reports,
        "all_present_ok": all(r["status"] == "ok" for r in reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-dir", default=str(ACT_DIR))
    parser.add_argument("--latest-path", default=None,
                        help="Override LATEST_BENCHMARK.json (fixture tests).")
    parser.add_argument("--split-manifest", default=None,
                        help="Override the direction split manifest path.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args()

    report = verify_all(
        Path(args.act_dir),
        latest_path=args.latest_path,
        split_manifest=args.split_manifest,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"frozen benchmark : {report['benchmark_sha256']}  ({report['n_benchmark_rows']} rows)")
    print(f"split manifest   : {report['split_manifest_sha256']}\n")
    for r in report["stages"]:
        line = f"  {r['stage']:16s} {r['status']}"
        if r["status"] == "absent":
            line += f"  (missing: {', '.join(r['missing'])})"
        elif r["status"] == "mismatch":
            line += "".join(f"\n      - {p}" for p in r["problems"])
        print(line)
    print(f"\nall present stages bound & consistent: {report['all_present_ok']}")


if __name__ == "__main__":
    main()
