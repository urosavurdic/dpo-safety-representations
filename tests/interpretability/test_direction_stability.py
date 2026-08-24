import json

import pytest

from src.interpretability.direction_stability import analyze_direction_stability


def _write_toy_cosine_sim(path):
    data = {
        "vs_M0": {
            "M0": [1.0, 1.0],
            "M1": [0.5, 0.6],
            "M2": [0.5, 0.55],
            "M3": [0.4, 0.5],
        },
        "adjacent": {
            "M0_vs_M1": [0.5, 0.6],
            "M1_vs_M2": [0.95, 0.9],
            "M2_vs_M3": [0.8, 0.85],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_per_layer_stability_reads_stage_keyed_lists_correctly(tmp_path):
    cosine_path = tmp_path / "cosine_similarity.json"
    _write_toy_cosine_sim(cosine_path)
    out_path = tmp_path / "out" / "stability_report.json"

    results = analyze_direction_stability(str(cosine_path), str(out_path))

    assert results["per_layer_stability"][0]["M3"] == 0.4
    assert results["per_layer_stability"][1]["M3"] == 0.5
    assert out_path.exists()


def test_biggest_drift_correctly_identified_as_M0_to_M1(tmp_path):
    cosine_path = tmp_path / "cosine_similarity.json"
    _write_toy_cosine_sim(cosine_path)
    out_path = tmp_path / "out" / "stability_report.json"

    results = analyze_direction_stability(str(cosine_path), str(out_path))

    assert "M0_vs_M1" in results["interpretation"]["summary"]


def test_only_m0_present_does_not_crash_and_reports_nothing_comparable(tmp_path):
    """Regression test for the real bug hit in production: a partial
    activation-extraction run (e.g. only M0's activations exist, everything
    else skipped due to an environment problem) writes cosine_similarity.json
    with only {"M0": [...]} in vs_M0 -- this used to hard-crash with
    KeyError('M1') and take down the whole `src.reproduce direction`
    pipeline. Must now degrade gracefully instead."""
    cosine_path = tmp_path / "cosine_similarity.json"
    with open(cosine_path, "w", encoding="utf-8") as f:
        json.dump({"vs_M0": {"M0": [1.0, 1.0]}, "adjacent": {}}, f)
    out_path = tmp_path / "out" / "stability_report.json"

    results = analyze_direction_stability(str(cosine_path), str(out_path))

    assert results["metadata"]["missing_stages"] == ["M1", "M2", "M3"]
    assert "note" in results["metadata"]
    assert results["per_layer_stability"] == {}  # nothing to compare with only one stage
    assert out_path.exists()


def test_partial_chain_m0_and_m1_only_compares_what_exists(tmp_path):
    """M2/M3 activations not yet extracted (only M0 and M1 are) -- should
    compare M0 vs M1 (the only pair available), not crash looking for M2/M3,
    and should NOT claim an M0_vs_M2 or M1_vs_M2 drift number that has no
    underlying data."""
    cosine_path = tmp_path / "cosine_similarity.json"
    data = {
        "vs_M0": {"M0": [1.0, 1.0], "M1": [0.6, 0.7]},
        "adjacent": {"M0_vs_M1": [0.6, 0.7]},
    }
    with open(cosine_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    out_path = tmp_path / "out" / "stability_report.json"

    results = analyze_direction_stability(str(cosine_path), str(out_path))

    assert results["metadata"]["missing_stages"] == ["M2", "M3"]
    assert results["per_layer_stability"][0] == {"M0": 1.0, "M1": 0.6}
    assert "mean_similarity_M0_vs_M1" in results["stability_summary"]
    assert "mean_similarity_M0_vs_M3" not in results["stability_summary"]
    assert results["drift_dynamics"]["aggregate"] == {"mean_drift_M0_vs_M1": pytest.approx(1.0 - 0.65, abs=1e-6)}
    assert "M0_vs_M1" in results["interpretation"]["summary"]


def test_missing_adjacent_data_skips_drift_dynamics_without_crashing(tmp_path):
    """vs_M0 has M0/M1/M2/M3 but 'adjacent' is empty (e.g. an older
    cosine_similarity.json format, or a run that only computed vs_M0) --
    per_layer_stability/stability_summary should still work; drift_dynamics
    and interpretation should be skipped, not KeyError on a missing pair."""
    cosine_path = tmp_path / "cosine_similarity.json"
    data = {
        "vs_M0": {"M0": [1.0, 1.0], "M1": [0.6, 0.65], "M2": [0.55, 0.6], "M3": [0.5, 0.55]},
        "adjacent": {},
    }
    with open(cosine_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    out_path = tmp_path / "out" / "stability_report.json"

    results = analyze_direction_stability(str(cosine_path), str(out_path))

    assert "mean_similarity_M0_vs_M3" in results["stability_summary"]
    assert results["drift_dynamics"] == {}
    assert "interpretation" not in results