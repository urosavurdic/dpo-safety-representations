"""Unit tests for src/v2_binding_guard.py (WP-Repro).

Covers: legacy-basename rejection, per-row binding-field enforcement, the
frozen-SHA check with an explicit-override escape hatch, and the fixture
round-trip ("654 v2-bound fixture passes, 370-era fixture rejected").
"""
import json
from pathlib import Path

import pytest

from src.v2_binding_guard import (
    LEGACY_370_BASENAMES,
    LegacyArtifactError,
    assert_not_legacy_basename,
    assert_rows_bound,
    iter_rows,
    load_guarded_raw,
)

FIX = Path(__file__).resolve().parent / "fixtures"
BENCH_SHA = json.loads((FIX / "benchmark_654.LATEST_BENCHMARK.json").read_text())["benchmark_sha256"]
SPLIT_SHA = json.loads((FIX / "benchmark_654.split_manifest.json").read_text())["split_manifest_sha256"]


def test_legacy_basenames_are_rejected():
    for name in ["causal_ablation_raw_wide.json", "causal_ablation_raw_narrow.json", "steering_raw_D.json"]:
        assert name in LEGACY_370_BASENAMES
        with pytest.raises(LegacyArtifactError):
            assert_not_legacy_basename(f"results/raw/{name}")


def test_legacy_prefix_pattern_rejected_even_if_not_in_the_set():
    with pytest.raises(LegacyArtifactError):
        assert_not_legacy_basename("results/raw/causal_ablation_raw_somethingnew.json")


def test_v2_basename_passes():
    assert_not_legacy_basename("results/raw/causal_ablation_v2_M3_L24-28.json")


def test_iter_rows_accepts_list_and_rows_object():
    assert iter_rows([{"a": 1}]) == [{"a": 1}]
    assert iter_rows({"rows": [{"a": 1}]}) == [{"a": 1}]
    with pytest.raises(LegacyArtifactError):
        iter_rows({"nope": 1})


def test_assert_rows_bound_requires_both_sha_fields_per_row():
    good = [{"benchmark_sha256": "X", "split_manifest_sha256": "Y"}]
    assert assert_rows_bound(good, benchmark_sha256="X") == good
    with pytest.raises(LegacyArtifactError, match="binding field"):
        assert_rows_bound([{"benchmark_sha256": "X"}], benchmark_sha256="X")
    with pytest.raises(LegacyArtifactError, match="binding field"):
        assert_rows_bound([{"split_manifest_sha256": "Y"}], benchmark_sha256="X")


def test_assert_rows_bound_enforces_expected_benchmark_sha():
    rows = [{"benchmark_sha256": "WRONG", "split_manifest_sha256": "Y"}]
    with pytest.raises(LegacyArtifactError, match="does not.*match"):
        assert_rows_bound(rows, benchmark_sha256="RIGHT")


def test_assert_rows_bound_enforces_split_sha_when_supplied():
    rows = [{"benchmark_sha256": "X", "split_manifest_sha256": "WRONG"}]
    with pytest.raises(LegacyArtifactError):
        assert_rows_bound(rows, benchmark_sha256="X", split_manifest_sha256="RIGHT")


def test_empty_file_is_rejected():
    with pytest.raises(LegacyArtifactError, match="no rows"):
        assert_rows_bound([], benchmark_sha256="X")


def test_fixture_v2_causal_file_passes_guard():
    rows = load_guarded_raw(
        FIX / "causal_ablation_v2_M3_L24-28.json", benchmark_sha256=BENCH_SHA,
        split_manifest_sha256=SPLIT_SHA,
    )
    assert len(rows) == 6  # 2 held-out A rows x 3 conditions
    assert {r["stage"] for r in rows} == {"M3_baseline", "M3_ablated_AD", "M3_ablated_random"}


def test_fixture_370_legacy_file_is_rejected():
    with pytest.raises(LegacyArtifactError):
        load_guarded_raw(FIX / "causal_ablation_raw_370_legacy.json")


def test_allow_unbound_bypasses_every_check():
    rows = load_guarded_raw(
        FIX / "causal_ablation_raw_370_legacy.json", allow_unbound=True,
    )
    assert rows and "model_stage" in rows[0]
