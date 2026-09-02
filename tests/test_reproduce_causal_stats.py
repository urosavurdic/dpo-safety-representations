"""Regression tests for the causal_stats component of src/reproduce.py (WP-Repro).

History: causal_stats used to point at results/raw/causal_ablation_raw_wide.json
(a pre-freeze / 370-era artifact keyed by `model_stage`, no benchmark/split
binding). The frozen-v2 T4 run instead writes
results/raw/causal_ablation_v2_M3_L24-28.json with a *_binding.json sidecar and
per-row benchmark_sha256 / split_manifest_sha256. This module pins the new
contract:

* causal_stats' three commands agree on one v2 file path;
* that path is the benchmark-bound v2 name, not a legacy causal_ablation_raw_*;
* causal_stats is BLOCKED until the T4 run produces that file (pre-T4 state);
* the guarded stat scripts accept a benchmark-bound v2 fixture and refuse a
  370-era fixture.
"""
import json
from pathlib import Path

import pytest

from src.reproduce import COMPONENTS, missing_requirements
from src.v2_binding_guard import LegacyArtifactError, load_guarded_raw

REPO_ROOT = Path(__file__).resolve().parents[1]
FIX = REPO_ROOT / "tests" / "fixtures"


def _file_arg(cmd):
    parts = cmd.split()
    return parts[parts.index("--file") + 1]


def test_causal_stats_commands_reference_one_consistent_v2_file():
    commands = COMPONENTS["causal_stats"]["commands"]
    input_paths = {_file_arg(cmd) for cmd in commands}
    assert len(input_paths) == 1, f"causal_stats commands disagree on --file: {input_paths}"
    (only,) = input_paths
    assert only in COMPONENTS["causal_stats"]["requires"]
    assert "causal_ablation_v2_" in only and "_L24-28" in only
    assert "causal_ablation_raw_" not in only, "must not point at a pre-freeze file"


def test_causal_stats_requires_the_binding_sidecar():
    requires = COMPONENTS["causal_stats"]["requires"]
    assert any(r.endswith("_binding.json") for r in requires), (
        "causal_stats must require the *_binding.json provenance sidecar"
    )


def test_causal_stats_is_blocked_until_t4(monkeypatch, tmp_path):
    """Pre-T4, the v2 causal file does not exist -> causal_stats is BLOCKED.
    This is the intended state, not a failure."""
    monkeypatch.chdir(tmp_path)
    assert missing_requirements("causal_stats"), (
        "expected causal_stats to be blocked pre-T4 (v2 causal file not yet generated)"
    )


# The three causal_stats scripts all route their --file through
# src.v2_binding_guard.load_guarded_raw, so the guard is exercised here at the
# library level (env-independent - no statsmodels/torch needed).

def test_v2_bound_fixture_passes_the_guard():
    bench_sha = json.loads((FIX / "benchmark_654.LATEST_BENCHMARK.json").read_text())["benchmark_sha256"]
    rows = load_guarded_raw(
        FIX / "causal_ablation_v2_M3_L24-28.json", benchmark_sha256=bench_sha,
    )
    assert len(rows) == 6
    assert {r["stage"] for r in rows} == {"M3_baseline", "M3_ablated_AD", "M3_ablated_random"}


def test_370_legacy_fixture_is_rejected_by_the_guard():
    with pytest.raises(LegacyArtifactError):
        load_guarded_raw(FIX / "causal_ablation_raw_370_legacy.json")


def test_allow_unbound_escape_hatch_lets_legacy_through():
    rows = load_guarded_raw(FIX / "causal_ablation_raw_370_legacy.json", allow_unbound=True)
    assert rows and "model_stage" in rows[0]
