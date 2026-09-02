"""Fail-closed contract tests for src/v2_io.py's benchmark/split/artifact binding.

Covers the failure modes the frozen-v2 T4 run depends on: a wrong-SHA benchmark,
a split manifest bound to the wrong benchmark or with a tampered content hash,
and artifact binding sidecars that disagree with the run. Also a cheap
stage-graph regression guard. Everything runs under tmp_path; nothing is written
to results/.
"""

import json

import pytest

from src.v2_io import (
    assert_binding,
    binding,
    canonical_json,
    load_run_inputs,
    resolve_benchmark,
    sha256_bytes,
    sha256_file,
    split_payload,
)

BENCHMARK_BYTES = (
    b'{"record_id": "r1", "quadrant": "A", "prompt": "x"}\n'
    b'{"record_id": "r2", "quadrant": "D", "prompt": "y"}\n'
)


def _write_pointer(tmp_path, benchmark_path, sha):
    pointer = tmp_path / "LATEST_BENCHMARK.json"
    pointer.write_text(
        json.dumps({"benchmark_path": str(benchmark_path), "benchmark_sha256": sha}),
        encoding="utf-8",
    )
    return pointer


def _write_split_manifest(tmp_path, benchmark_sha, *, name="split.json"):
    manifest = {
        "benchmark_sha256": benchmark_sha,
        "counts": {"direction_estimation": 1, "held_out_behavioral": 1},
        "record_ids_direction_estimation": ["r1"],
        "record_ids_held_out_behavioral": ["r2"],
    }
    manifest["split_manifest_sha256"] = sha256_bytes(
        canonical_json(split_payload(manifest))
    )
    path = tmp_path / name
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def _consistent_trio(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_bytes(BENCHMARK_BYTES)
    sha = sha256_file(benchmark)
    pointer = _write_pointer(tmp_path, benchmark, sha)
    split_path, _ = _write_split_manifest(tmp_path, sha)
    return benchmark, sha, pointer, split_path


# --- resolve_benchmark -----------------------------------------------------


def test_resolve_benchmark_happy_path(tmp_path):
    benchmark, sha, pointer, _ = _consistent_trio(tmp_path)
    resolved_path, resolved_sha = resolve_benchmark(latest_path=pointer)
    assert resolved_path.resolve() == benchmark.resolve()
    assert resolved_sha == sha


def test_resolve_benchmark_missing_pointer(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_benchmark(latest_path=tmp_path / "nope.json")


def test_resolve_benchmark_missing_benchmark_file(tmp_path):
    pointer = _write_pointer(tmp_path, tmp_path / "gone.jsonl", "0" * 64)
    with pytest.raises(FileNotFoundError):
        resolve_benchmark(latest_path=pointer)


def test_resolve_benchmark_rejects_sha_mismatch(tmp_path):
    benchmark, _, _, _ = _consistent_trio(tmp_path)
    stale_pointer = _write_pointer(tmp_path, benchmark, "1" * 64)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        resolve_benchmark(latest_path=stale_pointer)


def test_resolve_benchmark_rejects_eval_set_that_is_not_the_pointer_target(tmp_path):
    benchmark, sha, pointer, _ = _consistent_trio(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_bytes(BENCHMARK_BYTES)  # same content, same SHA, different path
    with pytest.raises(RuntimeError, match="not the benchmark referenced"):
        resolve_benchmark(eval_set=other, latest_path=pointer)


# --- load_run_inputs -----------------------------------------------------------


def test_load_run_inputs_happy_path(tmp_path):
    benchmark, sha, pointer, split_path = _consistent_trio(tmp_path)
    b_path, b_sha, s_path, s_sha = load_run_inputs(
        split_manifest=split_path, latest_path=pointer
    )
    assert b_sha == sha
    assert s_path == split_path
    assert s_sha


def test_load_run_inputs_missing_split_manifest(tmp_path):
    _, _, pointer, _ = _consistent_trio(tmp_path)
    with pytest.raises(FileNotFoundError, match="split manifest"):
        load_run_inputs(split_manifest=tmp_path / "absent.json", latest_path=pointer)


def test_load_run_inputs_rejects_split_bound_to_other_benchmark(tmp_path):
    _, _, pointer, _ = _consistent_trio(tmp_path)
    wrong_split, _ = _write_split_manifest(
        tmp_path, "a" * 64, name="wrong_bench_split.json"
    )
    with pytest.raises(RuntimeError, match="different benchmark"):
        load_run_inputs(split_manifest=wrong_split, latest_path=pointer)


def test_load_run_inputs_rejects_tampered_split_content_hash(tmp_path):
    _, sha, pointer, _ = _consistent_trio(tmp_path)
    path, manifest = _write_split_manifest(tmp_path, sha, name="tampered_split.json")
    manifest["record_ids_direction_estimation"] = ["r1", "r2"]  # hash now stale
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        load_run_inputs(split_manifest=path, latest_path=pointer)


# --- assert_binding ----------------------------------------------------------


def test_assert_binding_roundtrip_and_rejections(tmp_path):
    side = tmp_path / "artifact_binding.json"
    side.write_text(
        json.dumps(binding("b.jsonl", "BENCH", "s.json", "SPLIT")), encoding="utf-8"
    )

    assert assert_binding(side, "BENCH", "SPLIT")["benchmark_sha256"] == "BENCH"

    with pytest.raises(RuntimeError, match="different benchmark"):
        assert_binding(side, "OTHER", "SPLIT")
    with pytest.raises(RuntimeError, match="different split manifest"):
        assert_binding(side, "BENCH", "OTHER")
    with pytest.raises(FileNotFoundError):
        assert_binding(tmp_path / "missing_binding.json", "BENCH", "SPLIT")


# --- stage-graph regression guard ------------------------------------------


def test_stage_graph_shape_is_frozen():
    from src.analysis import v2_pipeline as vp

    assert vp.ALL_STAGES == [
        "M0", "M1", "M2", "M3", "M3_direct",
        "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt",
    ]
    assert vp.INTERVENTION_STAGES == ["M3", "M3_direct", "M3_alt", "M3_direct_alt"]
    assert vp.STEERING_STAGES == [s for s in vp.ALL_STAGES if s != "M0"]
    # No direct-DPO endpoint may leak into a sequential (adjacent) chain.
    for chain in (vp.SEQUENTIAL_STAGES, vp.ALT_SEQUENTIAL_STAGES):
        assert "M3_direct" not in chain
        assert "M3_direct_alt" not in chain
    assert vp.SEQUENTIAL_STAGES == ["M0", "M1", "M2", "M3"]
    assert vp.ALT_SEQUENTIAL_STAGES == ["M0", "M1_alt", "M2_alt", "M3_alt"]
    assert vp.ABLATION_LAYERS == list(range(24, 29))
    assert vp.DEFAULT_STEER_LAYERS == [24]
