import json

import numpy as np
import pytest

from src.interpretability.paired_deep_layer_stability_test import (
    DIRECT_VS_MEDIATED_PAIRS,
    deep_layer_mean_sims,
    main,
    paired_stability_test,
)


def _toy_stage_data(deep_mean, seed, n_bootstrap=200, n_layers=29, deep_start=16):
    """Mimics bootstrap_direction_stability.json's out[stage] shape: keys
    are layer indices (as strings, matching what json.load produces), each
    a dict with the usual summary stats plus raw_sims for deep layers."""
    rng = np.random.default_rng(seed)
    data = {}
    for layer in range(n_layers):
        if layer >= deep_start:
            sims = np.clip(rng.normal(deep_mean, 0.01, size=n_bootstrap), -1, 1)
            data[str(layer)] = {"mean": float(sims.mean()), "raw_sims": sims.tolist()}
        else:
            data[str(layer)] = {"mean": 0.5}  # shallow layer, no raw_sims (matches real pipeline)
    return data


def test_deep_layer_mean_sims_shape_and_averages_across_deep_layers():
    stage_data = _toy_stage_data(deep_mean=0.98, seed=0, n_bootstrap=50)
    sims = deep_layer_mean_sims(stage_data)
    assert sims.shape == (50,)
    assert abs(sims.mean() - 0.98) < 0.01


def test_deep_layer_mean_sims_raises_clearly_when_raw_sims_missing():
    stage_data = {"16": {"mean": 0.9}}  # no raw_sims - simulates a stale pre-change results file
    with pytest.raises(ValueError, match="raw_sims missing"):
        deep_layer_mean_sims(stage_data, layers=[16])


def test_paired_stability_test_detects_direct_more_stable():
    direct_sims = np.clip(np.random.default_rng(1).normal(0.99, 0.003, size=500), -1, 1)
    mediated_sims = np.clip(np.random.default_rng(2).normal(0.96, 0.01, size=500), -1, 1)
    result = paired_stability_test(direct_sims, mediated_sims)
    assert result["direct_mean"] > result["mediated_mean"]
    d = result["difference_direct_minus_mediated"]
    assert d["mean"] > 0
    assert d["ci_low_2.5pct"] > 0  # CI excludes 0
    assert result["p_value"] < 0.05
    assert result["frac_replicates_direct_gt_mediated"] > 0.9


def test_paired_stability_test_not_significant_when_no_real_difference():
    rng = np.random.default_rng(0)
    direct_sims = np.clip(rng.normal(0.95, 0.02, size=500), -1, 1)
    mediated_sims = np.clip(rng.normal(0.95, 0.02, size=500), -1, 1)
    result = paired_stability_test(direct_sims, mediated_sims)
    assert result["p_value"] > 0.05


def test_paired_stability_test_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        paired_stability_test(np.ones(10), np.ones(20))


def test_direct_vs_mediated_pairs_are_within_branch():
    assert DIRECT_VS_MEDIATED_PAIRS == [("M3_direct", "M3"), ("M3_direct_alt", "M3_alt")]


def test_main_runs_per_branch_and_pooled_comparison(tmp_path, monkeypatch):
    import src.interpretability.paired_deep_layer_stability_test as pdlst

    data = {
        "M3": _toy_stage_data(deep_mean=0.96, seed=1),
        "M3_direct": _toy_stage_data(deep_mean=0.99, seed=2),
        "M3_alt": _toy_stage_data(deep_mean=0.965, seed=3),
        "M3_direct_alt": _toy_stage_data(deep_mean=0.99, seed=4),
    }
    in_path = tmp_path / "bootstrap_direction_stability.json"
    in_path.write_text(json.dumps(data))
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(pdlst, "OUT_PATH", out_path)
    monkeypatch.setattr(
        "sys.argv", ["paired_deep_layer_stability_test.py", "--file", str(in_path)]
    )

    main()

    result = json.loads(out_path.read_text())
    assert set(result["per_branch"].keys()) == {"M3_direct_vs_M3", "M3_direct_alt_vs_M3_alt"}
    assert "pooled_across_branches" in result
    assert result["pooled_across_branches"]["mean_diff"] > 0  # direct more stable in both toy branches


def test_main_skips_missing_branch_and_reports_available_one(tmp_path, monkeypatch):
    import src.interpretability.paired_deep_layer_stability_test as pdlst

    # Only the original-branch pair available - alt branch not trained yet.
    data = {
        "M3": _toy_stage_data(deep_mean=0.96, seed=1),
        "M3_direct": _toy_stage_data(deep_mean=0.99, seed=2),
    }
    in_path = tmp_path / "bootstrap_direction_stability.json"
    in_path.write_text(json.dumps(data))
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(pdlst, "OUT_PATH", out_path)
    monkeypatch.setattr(
        "sys.argv", ["paired_deep_layer_stability_test.py", "--file", str(in_path)]
    )

    main()

    result = json.loads(out_path.read_text())
    assert set(result["per_branch"].keys()) == {"M3_direct_vs_M3"}
    assert "pooled_across_branches" not in result  # only one branch available


def test_main_raises_clearly_when_input_file_missing(tmp_path, monkeypatch):
    import src.interpretability.paired_deep_layer_stability_test as pdlst

    monkeypatch.setattr(
        "sys.argv",
        ["paired_deep_layer_stability_test.py", "--file", str(tmp_path / "nope.json")],
    )
    with pytest.raises(FileNotFoundError):
        main()
