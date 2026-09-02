"""Toy tests for src/analysis/verify_activations.py (WP-Repro).

Builds a tiny frozen-benchmark trio (via tests/fixtures/_generate output) and
a matching / mismatching activation set under tmp_path, then checks the
per-stage verdict. No torch, no real model.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.verify_activations import verify_stage, verify_all
from src.v2_io import identity_snapshot, load_jsonl

FIX = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def frozen_inputs():
    latest = FIX / "benchmark_654.LATEST_BENCHMARK.json"
    split = FIX / "benchmark_654.split_manifest.json"
    bench_rows = load_jsonl(FIX / "benchmark_654.jsonl")
    bench_sha = json.loads(latest.read_text())["benchmark_sha256"]
    split_sha = json.loads(split.read_text())["split_manifest_sha256"]
    return latest, split, bench_rows, bench_sha, split_sha


def _write_stage(act_dir, stage, bench_rows, bench_sha, split_sha, *, n_override=None):
    act_dir.mkdir(parents=True, exist_ok=True)
    n = n_override if n_override is not None else len(bench_rows)
    np.save(act_dir / f"{stage}_final.npy", np.zeros((n, 3, 4), dtype=np.float32))
    np.save(act_dir / f"{stage}_pooled.npy", np.zeros((n, 3, 4), dtype=np.float32))
    (act_dir / f"{stage}_metadata.json").write_text(
        json.dumps(identity_snapshot(bench_rows)), encoding="utf-8"
    )
    (act_dir / f"{stage}_metadata_binding.json").write_text(json.dumps({
        "benchmark_path": "tests/fixtures/benchmark_654.jsonl",
        "benchmark_sha256": bench_sha,
        "split_manifest_path": "tests/fixtures/benchmark_654.split_manifest.json",
        "split_manifest_sha256": split_sha,
    }), encoding="utf-8")


def test_absent_stage_reported(tmp_path, frozen_inputs):
    _, _, bench_rows, bench_sha, split_sha = frozen_inputs
    r = verify_stage("M0", bench_rows, bench_sha, split_sha, act_dir=tmp_path)
    assert r["status"] == "absent"
    assert set(r["missing"]) == {
        "M0_final.npy", "M0_pooled.npy", "M0_metadata.json", "M0_metadata_binding.json",
    }


def test_consistent_stage_is_ok(tmp_path, frozen_inputs):
    _, _, bench_rows, bench_sha, split_sha = frozen_inputs
    _write_stage(tmp_path, "M2", bench_rows, bench_sha, split_sha)
    r = verify_stage("M2", bench_rows, bench_sha, split_sha, act_dir=tmp_path)
    assert r["status"] == "ok", r["problems"]
    assert r["n_rows"] == len(bench_rows)


def test_wrong_binding_sha_flagged(tmp_path, frozen_inputs):
    _, _, bench_rows, bench_sha, split_sha = frozen_inputs
    _write_stage(tmp_path, "M3", bench_rows, "deadbeef" * 8, split_sha)
    r = verify_stage("M3", bench_rows, bench_sha, split_sha, act_dir=tmp_path)
    assert r["status"] == "mismatch"
    assert any("binding" in p for p in r["problems"])


def test_row_count_mismatch_flagged(tmp_path, frozen_inputs):
    _, _, bench_rows, bench_sha, split_sha = frozen_inputs
    _write_stage(tmp_path, "M1", bench_rows, bench_sha, split_sha, n_override=len(bench_rows) - 1)
    r = verify_stage("M1", bench_rows, bench_sha, split_sha, act_dir=tmp_path)
    assert r["status"] == "mismatch"
    assert any("rows != metadata" in p for p in r["problems"])


def test_verify_all_against_fixture_pointer(tmp_path, frozen_inputs):
    latest, split, bench_rows, bench_sha, split_sha = frozen_inputs
    _write_stage(tmp_path, "M0", bench_rows, bench_sha, split_sha)
    report = verify_all(tmp_path, latest_path=latest, split_manifest=split)
    assert report["benchmark_sha256"] == bench_sha
    by_stage = {r["stage"]: r["status"] for r in report["stages"]}
    assert by_stage["M0"] == "ok"
    assert by_stage["M1"] == "absent"
    assert report["all_present_ok"] is False  # M1..M3_direct_alt absent
