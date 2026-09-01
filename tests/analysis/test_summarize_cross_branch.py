import json

import pytest

from src.analysis.summarize_cross_branch import (
    behavioral_rates_for_stage,
    probe_best_layer_for_stage,
    direction_cross_branch_similarity,
    build_comparison,
    CROSS_BRANCH_PAIRS,
)


def test_cross_branch_pairs_match_the_four_alt_stages():
    assert CROSS_BRANCH_PAIRS == [
        ("M1", "M1_alt"), ("M2", "M2_alt"), ("M3", "M3_alt"), ("M3_direct", "M3_direct_alt"),
    ]


def test_behavioral_rates_for_stage_returns_none_when_stage_missing():
    assert behavioral_rates_for_stage({}, "M1_alt") is None


def test_behavioral_rates_for_stage_computes_soft_deflection_not_just_refusal():
    raw = {
        "M1": [
            {"quadrant": "C", "completion": "I cannot help with something so dangerous."},  # refusal
            {"quadrant": "C", "completion": "That would be quite difficult to actually pull off in practice."},  # soft deflection
            {"quadrant": "C", "completion": "Sure, here's how you do it: step one..."},  # comply
        ],
    }
    result = behavioral_rates_for_stage(raw, "M1")
    assert result is not None
    assert "C" in result
    assert result["C"]["refusal"]["n"] == 3
    assert result["C"]["soft_deflection"]["n"] == 3


def test_probe_best_layer_for_stage_missing_file_returns_none(tmp_path):
    assert probe_best_layer_for_stage("M1_alt", probes_dir=str(tmp_path)) is None


def test_probe_best_layer_for_stage_picks_max_cv_accuracy(tmp_path):
    results = [
        {"layer": 0, "cv_accuracy_mean": 0.5, "quadrant_c_flagged_unsafe_frac": 0.1},
        {"layer": 14, "cv_accuracy_mean": 0.95, "quadrant_c_flagged_unsafe_frac": 0.8},
        {"layer": 28, "cv_accuracy_mean": 0.7, "quadrant_c_flagged_unsafe_frac": 0.3},
    ]
    (tmp_path / "M1_probe_results.json").write_text(json.dumps(results))
    best = probe_best_layer_for_stage("M1", probes_dir=str(tmp_path))
    assert best["layer"] == 14
    assert best["cv_accuracy_mean"] == 0.95


def test_direction_cross_branch_similarity_missing_file_returns_none(tmp_path):
    assert direction_cross_branch_similarity("M1", "M1_alt", cosine_path=str(tmp_path / "nope.json")) is None


def test_direction_cross_branch_similarity_missing_pair_returns_none(tmp_path):
    path = tmp_path / "cosine_similarity.json"
    path.write_text(json.dumps({"cross_branch": {"M2_vs_M2_alt": [0.9, 0.8]}}))
    assert direction_cross_branch_similarity("M1", "M1_alt", cosine_path=str(path)) is None


def test_direction_cross_branch_similarity_computes_mean(tmp_path):
    path = tmp_path / "cosine_similarity.json"
    path.write_text(json.dumps({"cross_branch": {"M1_vs_M1_alt": [0.8, 0.6, 1.0]}}))
    result = direction_cross_branch_similarity("M1", "M1_alt", cosine_path=str(path))
    assert result["per_layer"] == [0.8, 0.6, 1.0]
    assert result["mean"] == pytest.approx(0.8)


def test_build_comparison_omits_sections_with_missing_data(tmp_path, monkeypatch):
    # Nothing available at all for this pair -> only "pair" key present.
    # Must isolate cwd: the real repo now has completed M3_direct/M3_direct_alt
    # refusal-direction results on disk, so without this the test silently
    # picks up real data and no longer exercises the "missing data" path.
    monkeypatch.chdir(tmp_path)
    comp = build_comparison("M3_direct", "M3_direct_alt", raw_rows_by_stage={})
    assert comp == {"pair": "M3_direct_vs_M3_direct_alt"}


def test_build_comparison_never_raises_on_partial_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # No probes/, no refusal_direction/ directories at all
    raw = {
        "M1": [{"quadrant": "C", "completion": "I refuse to help with that."}],
        "M1_alt": [{"quadrant": "C", "completion": "Sure, here's the info."}],
    }
    comp = build_comparison("M1", "M1_alt", raw_rows_by_stage=raw)
    assert "behavioral" in comp
    assert "probes" not in comp
    assert "direction" not in comp
