"""Regression tests for src.validate_benchmark_v2's artifact-freshness gate.

This is the check the v2 run gate (src.analysis.v2_pipeline.gate_for_run) leans
on to decide whether a T4 session may reuse existing activations or must
regenerate them. Before the T4 rerun has produced any activations, or between
sessions while a stage is only partially written, the activation directory is
expected to be incomplete. That incompleteness must always show up as an
explicit, per-stage "missing" reason and must always resolve to
artifact_freshness_pass=False - never silently to PASS. A run gate that turns
a half-written directory into a false PASS would let a resumed T4 session
skip real extraction work and analyze stale or partial activations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import src.validate_benchmark_v2 as vbv
from src.v2_io import canonical_json, identity_snapshot, sha256_bytes, sha256_file


_BENCH_ROWS_TEMPLATE = [
    {
        "record_id": "r001",
        "prompt": "one two three four",
        "scored_prompt": "one two three four",
        "quadrant": "A",
        "c_construction": None,
        "source_dataset": "HarmBench",
        "split": "direction_estimation",
    },
    {
        "record_id": "r002",
        "prompt": "alpha beta gamma",
        "scored_prompt": "alpha beta gamma",
        "quadrant": "C",
        "c_construction": "c_paired",
        "source_dataset": "AdvBench",
        "split": "held_out",
    },
    {
        "record_id": "r003",
        "prompt": "one two three four five six seven eight",
        "scored_prompt": "one two three four five six seven eight",
        "quadrant": "A",
        "c_construction": None,
        "source_dataset": "HarmBench",
        "split": "direction_estimation",
    },
    {
        "record_id": "r004",
        "prompt": "alpha beta gamma delta epsilon zeta eta theta iota",
        "scored_prompt": "alpha beta gamma delta epsilon zeta eta theta iota",
        "quadrant": "C",
        "c_construction": "c_paired",
        "source_dataset": "AdvBench",
        "split": "held_out",
    },
]
# Note: pooled_cohen_d (and therefore length_confound_pass) needs >=2 rows
# with non-zero within-group variance in both the quadrant-A and c_paired
# groups, or it returns None. The real frozen benchmark always satisfies
# this, but a too-small or same-length fixture hits a latent bug where
# validate_benchmark_v2.py's f"{length_d:.3f}" warning formatting crashes on
# a None length_d. That bug is pre-existing, out of scope for this
# milestone, and not triggered by the real benchmark - so the fixture is
# sized/varied to avoid it rather than patch unrelated code.


def _make_bench_rows():
    rows = copy.deepcopy(_BENCH_ROWS_TEMPLATE)
    for row in rows:
        row["record_sha256"] = hashlib.sha256(
            row["scored_prompt"].encode("utf-8")
        ).hexdigest()
    return rows


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle)


def build_fixture_repo(root: Path) -> dict:
    """Lay out the minimal frozen-benchmark inputs validate_benchmark_v2
    needs, independent of the results/activations directory under test.

    The benchmark rows deliberately omit most of REQUIRED_FIELDS: schema
    completeness is a different check (schema_integrity_pass) and is not
    what these tests exercise. Nothing in main() short-circuits on schema
    failure before reaching the activation-freshness loop, so this keeps
    the fixture minimal without weakening the assertions below.
    """
    bench_rows = _make_bench_rows()
    bench_path = root / "data" / "frozen_v2" / "benchmark_v2_test.jsonl"
    _write_jsonl(bench_path, bench_rows)
    bench_sha = sha256_file(bench_path)

    _write_json(
        root / "data" / "frozen_v2" / "LATEST_BENCHMARK.json",
        {
            "benchmark_path": bench_path.as_posix(),
            "benchmark_sha256": bench_sha,
        },
    )

    review_path = root / "data" / "review" / "c_review_queue.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(
        "record_id,review_status\nr002,accept\nr004,accept\n",
        encoding="utf-8",
        newline="\n",
    )

    gate_path = root / "logs" / "benchmark_gate_config.json"
    _write_json(gate_path, {})

    split_payload = {
        "benchmark_sha256": bench_sha,
        "direction_ids": ["r001", "r003"],
        "held_out_ids": ["r002", "r004"],
    }
    split_sha = sha256_bytes(canonical_json(split_payload))
    split_path = root / "logs" / "direction_split_manifest.json"
    _write_json(split_path, dict(split_payload, split_manifest_sha256=split_sha))

    return {
        "bench_rows": bench_rows,
        "benchmark_path": bench_path,
        "benchmark_sha256": bench_sha,
        "review_path": review_path,
        "gate_path": gate_path,
        "split_path": split_path,
        "split_manifest_sha256": split_sha,
    }


def write_activation_set(
    root: Path,
    fixture: dict,
    stage: str,
    *,
    final: bool = True,
    pooled: bool = True,
    metadata: bool = True,
    binding: bool = True,
    metadata_content=None,
    binding_content=None,
    row_count: int | None = None,
) -> None:
    act_dir = root / "results" / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    n = row_count if row_count is not None else len(fixture["bench_rows"])

    if final:
        np.save(act_dir / f"{stage}_final.npy", np.zeros((n, 4)))
    if pooled:
        np.save(act_dir / f"{stage}_pooled.npy", np.zeros((n, 4)))
    if metadata:
        content = (
            metadata_content
            if metadata_content is not None
            else identity_snapshot(fixture["bench_rows"])
        )
        _write_json(act_dir / f"{stage}_metadata.json", content)
    if binding:
        content = (
            binding_content
            if binding_content is not None
            else {
                "benchmark_sha256": fixture["benchmark_sha256"],
                "split_manifest_sha256": fixture["split_manifest_sha256"],
            }
        )
        _write_json(act_dir / f"{stage}_metadata_binding.json", content)


def run_validate(root: Path, fixture: dict, monkeypatch) -> dict:
    monkeypatch.chdir(root)
    argv = [
        "validate_benchmark_v2",
        "--benchmark",
        str(fixture["benchmark_path"]),
        "--review-csv",
        str(fixture["review_path"]),
        "--gate-config",
        str(fixture["gate_path"]),
        "--split-manifest",
        str(fixture["split_path"]),
        "--out-dir",
        "logs",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    vbv.main()
    return json.loads(
        (root / "logs" / "benchmark_validation_status.json").read_text(
            encoding="utf-8"
        )
    )


# --------------------------------------------------------------------------
# Missing activation metadata / binding metadata -> stale, with a reason
# --------------------------------------------------------------------------


def test_missing_activation_metadata_is_stale_with_explicit_reason(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vbv, "REQUIRED_STAGES", ["M0"])
    fixture = build_fixture_repo(tmp_path)
    write_activation_set(tmp_path, fixture, "M0", metadata=False)

    status = run_validate(tmp_path, fixture, monkeypatch)

    assert status["artifact_freshness_pass"] is False
    assert (
        "M0: missing results/activations/M0_metadata.json"
        in status["stale_activation_files"]
    )
    # A missing artifact must never be reported as a content mismatch instead
    # of a missing-file reason - the reason has to say what is actually wrong.
    assert not any(
        "does not match benchmark" in reason
        for reason in status["stale_activation_files"]
    )
    assert status["technical_benchmark_status"] == "FAIL"


def test_missing_activation_binding_metadata_is_stale_with_explicit_reason(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vbv, "REQUIRED_STAGES", ["M0"])
    fixture = build_fixture_repo(tmp_path)
    write_activation_set(tmp_path, fixture, "M0", binding=False)

    status = run_validate(tmp_path, fixture, monkeypatch)

    assert status["artifact_freshness_pass"] is False
    assert (
        "M0: missing results/activations/M0_metadata_binding.json"
        in status["stale_activation_files"]
    )
    assert status["technical_benchmark_status"] == "FAIL"


def test_completely_absent_stage_reports_all_four_missing_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vbv, "REQUIRED_STAGES", ["M0"])
    fixture = build_fixture_repo(tmp_path)
    # No activation files written for M0 at all - the pre-GPU-rerun state
    # for every stage on a fresh checkout.

    status = run_validate(tmp_path, fixture, monkeypatch)

    reasons = status["stale_activation_files"]
    for suffix in (
        "final.npy",
        "pooled.npy",
        "metadata.json",
        "metadata_binding.json",
    ):
        assert f"M0: missing results/activations/M0_{suffix}" in reasons
    assert status["artifact_freshness_pass"] is False
    assert status["technical_benchmark_status"] == "FAIL"


def test_multiple_stages_each_report_their_own_missing_reason(
    tmp_path, monkeypatch
):
    """Stale reasons must be per-stage, not a single aggregate flag, so a
    resumed session knows exactly which stages still need extraction."""
    monkeypatch.setattr(vbv, "REQUIRED_STAGES", ["M0", "M1"])
    fixture = build_fixture_repo(tmp_path)
    write_activation_set(tmp_path, fixture, "M0")  # fully fresh
    write_activation_set(tmp_path, fixture, "M1", metadata=False)  # stale

    status = run_validate(tmp_path, fixture, monkeypatch)

    reasons = status["stale_activation_files"]
    assert "M1: missing results/activations/M1_metadata.json" in reasons
    assert not any(reason.startswith("M0:") for reason in reasons)
    # One stale stage is enough to fail the whole gate.
    assert status["artifact_freshness_pass"] is False


# --------------------------------------------------------------------------
# A missing artifact can never be mistaken for freshness (PASS)
# --------------------------------------------------------------------------


_ARTIFACT_SUFFIX_BY_KW = {
    "final": "final.npy",
    "pooled": "pooled.npy",
    "metadata": "metadata.json",
    "binding": "metadata_binding.json",
}


@pytest.mark.parametrize(
    "missing_kw",
    ["final", "pooled", "metadata", "binding"],
)
def test_removing_any_single_required_artifact_flips_pass_to_fail(
    tmp_path, monkeypatch, missing_kw
):
    """Start from a fully fresh, matching activation set (freshness PASSes),
    then delete exactly one required file. This must never still PASS -
    the freshness check must not accidentally keep passing just because it
    passed on a previous, more-complete run."""
    monkeypatch.setattr(vbv, "REQUIRED_STAGES", ["M0"])
    fixture = build_fixture_repo(tmp_path)
    write_activation_set(tmp_path, fixture, "M0")

    baseline = run_validate(tmp_path, fixture, monkeypatch)
    assert baseline["artifact_freshness_pass"] is True
    assert baseline["stale_activation_files"] == []

    suffix = _ARTIFACT_SUFFIX_BY_KW[missing_kw]
    (tmp_path / "results" / "activations" / f"M0_{suffix}").unlink()

    status = run_validate(tmp_path, fixture, monkeypatch)
    assert status["artifact_freshness_pass"] is False
    assert f"M0: missing results/activations/M0_{suffix}" in (
        status["stale_activation_files"]
    )
    assert status["technical_benchmark_status"] == "FAIL"


def test_unreadable_metadata_is_stale_not_a_crash_or_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(vbv, "REQUIRED_STAGES", ["M0"])
    fixture = build_fixture_repo(tmp_path)
    write_activation_set(tmp_path, fixture, "M0")

    act_dir = tmp_path / "results" / "activations"
    (act_dir / "M0_metadata.json").write_text("{not valid json", encoding="utf-8")

    status = run_validate(tmp_path, fixture, monkeypatch)

    assert status["artifact_freshness_pass"] is False
    assert any(
        reason.startswith("M0: unreadable artifact")
        for reason in status["stale_activation_files"]
    )


# --------------------------------------------------------------------------
# Validation status contains every field the v2 run gate consumes
# --------------------------------------------------------------------------


def test_status_contains_every_field_the_v2_run_gate_reads(tmp_path, monkeypatch):
    """src.analysis.v2_pipeline.gate_for_run reads STATIC_GATE_FIELDS plus
    technical_benchmark_status and artifact_freshness_pass unconditionally
    via status.get(...). If any key were absent, .get() would silently
    return None instead of surfacing the real gap - so presence of every
    key (not just a non-crashing run) is asserted explicitly here."""
    monkeypatch.setattr(vbv, "REQUIRED_STAGES", ["M0"])
    fixture = build_fixture_repo(tmp_path)
    # Deliberately leave activations missing: the gate must still be able
    # to read every field it needs even in the least-complete state.

    status = run_validate(tmp_path, fixture, monkeypatch)

    static_gate_fields = [
        "schema_integrity_pass",
        "prompt_integrity_pass",
        "c_review_pass",
        "c_review_mapping_pass",
        "benchmark_hash_pass",
        "split_benchmark_hash_pass",
        "split_hash_pass",
    ]
    for field in static_gate_fields:
        assert field in status, f"missing gate field: {field}"

    assert "technical_benchmark_status" in status
    assert "artifact_freshness_pass" in status

    warning_only_gate_fields = [
        "source_confound_pass",
        "category_confound_pass",
        "prompt_function_confound_pass",
        "length_confound_pass",
        "surface_separation_pass",
        "wording_only_claim_pass",
    ]
    for field in warning_only_gate_fields:
        assert field in status, f"missing warning-only gate field: {field}"


def test_reported_output_paths_track_a_non_default_out_dir(
    tmp_path, monkeypatch
):
    """status['outputs'] and the printed summary must name wherever the
    files were actually written, not a hardcoded 'logs/...'. Every other
    test in this module passes --out-dir logs (the notebook's own default),
    which would let a hardcoded 'logs/...' string pass unnoticed - so this
    test deliberately points --out-dir somewhere else and checks the
    self-reported paths resolve to real files there."""
    monkeypatch.setattr(vbv, "REQUIRED_STAGES", ["M0"])
    fixture = build_fixture_repo(tmp_path)

    monkeypatch.chdir(tmp_path)
    argv = [
        "validate_benchmark_v2",
        "--benchmark",
        str(fixture["benchmark_path"]),
        "--review-csv",
        str(fixture["review_path"]),
        "--gate-config",
        str(fixture["gate_path"]),
        "--split-manifest",
        str(fixture["split_path"]),
        "--out-dir",
        "custom_reports",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    vbv.main()

    status = json.loads(
        (tmp_path / "custom_reports" / "benchmark_validation_status.json")
        .read_text(encoding="utf-8")
    )

    assert status["outputs"]["validation_status"] == (
        "custom_reports/benchmark_validation_status.json"
    )
    assert status["outputs"]["validation_report"] == (
        "custom_reports/benchmark_validation_report.md"
    )
    for key in ("validation_status", "validation_report"):
        assert (tmp_path / status["outputs"][key]).exists(), (
            f"self-reported path for {key} does not exist: "
            f"{status['outputs'][key]}"
        )
    # The default 'logs/...' location must NOT have been written to by a
    # run that requested a different --out-dir.
    assert not (tmp_path / "logs" / "benchmark_validation_status.json").exists()


# --------------------------------------------------------------------------
# Stable POSIX-style path formatting for reported missing-artifact paths
# --------------------------------------------------------------------------


class _FakeWindowsChildPath:
    """Path-like stand-in for a single artifact path under a fake Windows
    host: str() renders with backslashes (like a real WindowsPath), but
    .as_posix() still returns the stable forward-slash form. Delegates
    .exists() to a real Path so the missing/present logic under test is
    unchanged.
    """

    def __init__(self, real: Path, posix_str: str):
        self._real = real
        self._posix = posix_str

    def exists(self) -> bool:
        return self._real.exists()

    def as_posix(self) -> str:
        return self._posix

    def __str__(self) -> str:
        return self._posix.replace("/", "\\")


class _FakeWindowsDir:
    """Directory-like stand-in whose '/' operator yields
    _FakeWindowsChildPath children, so a helper that (incorrectly) does
    str(path) instead of path.as_posix() would produce backslash paths
    even though this test runs on a POSIX CI host."""

    def __init__(self, real_dir: Path, posix_prefix: str):
        self._real_dir = real_dir
        self._posix_prefix = posix_prefix

    def __truediv__(self, name: str) -> _FakeWindowsChildPath:
        return _FakeWindowsChildPath(
            self._real_dir / name, f"{self._posix_prefix}/{name}"
        )


def test_missing_stage_artifacts_reports_posix_paths_even_on_windows_like_path(
    tmp_path,
):
    """Regression guard for the reported-path formatting bug: on a real
    Windows host, str(pathlib.Path(...)) renders with backslashes, so
    building the "missing" message with str(path) instead of
    path.as_posix() silently produces
    "results\\activations\\M0_metadata.json" there -- invisible on this
    POSIX CI host, where str() and as_posix() happen to coincide. The
    fake directory below mimics Windows str() semantics without needing
    an actual Windows host, so a regression back to str(path) fails this
    assertion here too.
    """
    fake_dir = _FakeWindowsDir(tmp_path, "results/activations")

    missing = vbv._stage_artifact_missing(fake_dir, "M0")

    assert missing == [
        "results/activations/M0_final.npy",
        "results/activations/M0_pooled.npy",
        "results/activations/M0_metadata.json",
        "results/activations/M0_metadata_binding.json",
    ]
    assert all("\\" not in path for path in missing)


def test_missing_stage_artifacts_omits_present_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    act_dir = Path("results") / "activations"
    (tmp_path / act_dir).mkdir(parents=True)
    (tmp_path / act_dir / "M0_final.npy").write_bytes(b"\x00")
    (tmp_path / act_dir / "M0_metadata.json").write_text("{}", encoding="utf-8")

    missing = vbv._stage_artifact_missing(act_dir, "M0")

    assert missing == [
        "results/activations/M0_pooled.npy",
        "results/activations/M0_metadata_binding.json",
    ]
