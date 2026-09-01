"""
Idempotency regression test for src/finalize_benchmark.py against the
current R104/C-paired review inputs.

Task 2 (finalize R104/C-paired) requires comparing the desired final
content against the current frozen benchmark before writing anything
new, and NOT regenerating a new frozen artifact merely because a
logically equivalent serialization is possible. This test codifies
that check: it re-runs the real finalization script, unmodified,
against a scratch copy of the repository's current committed inputs
and asserts the resulting benchmark JSONL is byte-for-byte identical
(same SHA-256) to the benchmark currently pointed to by
data/frozen_v2/LATEST_BENCHMARK.json.

This does not run any GPU workload; it only re-runs the existing,
deterministic, CPU-only finalization script against already-committed
CSV/JSONL inputs.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.v2_io import load_json, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]

# Only the inputs finalize_benchmark.py actually reads are needed in
# the scratch copy; keeping this list explicit (rather than copying
# the whole repository) makes the test's dependency surface visible
# and keeps it fast.
REQUIRED_RELATIVE_PATHS = [
    "src",
    "data/review/c_review_queue.csv",
    "logs/benchmark_gate_config.json",
    "data/processed/controlled_eval.jsonl",
    "data/quadrant_c_pipeline/candidate_records_v2.jsonl",
]


def test_finalize_benchmark_is_idempotent_against_current_inputs(tmp_path):
    latest = load_json(REPO_ROOT / "data/frozen_v2/LATEST_BENCHMARK.json")
    current_benchmark_path = REPO_ROOT / latest["benchmark_path"]
    current_sha256 = latest["benchmark_sha256"]

    # Sanity: the pointer must actually match the file on disk before
    # we trust it as the thing we're comparing against.
    assert sha256_file(current_benchmark_path) == current_sha256

    scratch = tmp_path / "scratch_repo"
    for rel in REQUIRED_RELATIVE_PATHS:
        src = REPO_ROOT / rel
        dst = scratch / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.finalize_benchmark",
            "--review-csv",
            "data/review/c_review_queue.csv",
            "--gate-config",
            "logs/benchmark_gate_config.json",
        ],
        cwd=scratch,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    regenerated = list((scratch / "data/frozen_v2").glob("benchmark_v2_*.jsonl"))
    assert len(regenerated) == 1
    regenerated_sha256 = sha256_file(regenerated[0])

    assert regenerated_sha256 == current_sha256, (
        "Re-running finalize_benchmark against the current committed "
        "inputs produced a different benchmark than the one currently "
        "frozen. Either the frozen benchmark is stale (needs a real "
        "regeneration + new commit) or finalize_benchmark's behavior "
        "changed. Do not resolve this by silently overwriting either "
        "artifact."
    )

    regenerated_manifest_paths = list(
        (scratch / "data/frozen_v2").glob("benchmark_v2_*.manifest.json")
    )
    assert len(regenerated_manifest_paths) == 1
    regenerated_manifest = json.loads(regenerated_manifest_paths[0].read_text())
    assert regenerated_manifest["counts"] == {"A": 150, "B": 250, "D": 150, "C": 104}
    assert regenerated_manifest["c_counts_by_construction"] == {"c_paired": 104}
