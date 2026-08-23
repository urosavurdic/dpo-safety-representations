import json

import numpy as np
import pytest

from src.interpretability.bootstrap_cross_branch_difference import (
    bootstrap_cross_branch_similarity,
    bootstrap_group_difference,
    main,
)


def _toy_branch_data(n_layers=2, hidden_dim=3, seed=0, direction_noise=0.0):
    """8 prompts per quadrant (A and D only, matching diff_in_means_direction's
    required quadrants), identical eval order convention used elsewhere in
    this repo's tests. `direction_noise` perturbs the D-quadrant activations
    so orig/alt directions aren't forced to be exactly identical."""
    rng = np.random.default_rng(seed)
    n_a, n_d = 8, 8
    pooled = np.zeros((n_a + n_d, n_layers, hidden_dim))
    pooled[:n_a, :, 0] = 5.0
    pooled[n_a:, :, 0] = -5.0
    pooled += rng.normal(scale=0.01, size=pooled.shape)
    if direction_noise:
        pooled[n_a:, :, 1] += direction_noise  # tilt D activations off-axis
    quadrants = np.array(["A"] * n_a + ["D"] * n_d)
    return pooled, quadrants


def test_bootstrap_cross_branch_similarity_shape():
    pooled_orig, quad_orig = _toy_branch_data(n_layers=4, seed=1)
    pooled_alt, quad_alt = _toy_branch_data(n_layers=4, seed=2)
    per_layer, mean_sims = bootstrap_cross_branch_similarity(
        pooled_orig, quad_orig, pooled_alt, quad_alt, n_bootstrap=15, seed=0
    )
    assert per_layer.shape == (15, 4)
    assert mean_sims.shape == (15,)


def test_bootstrap_cross_branch_similarity_near_one_when_branches_match():
    # Same quadrant structure, near-identical activations (tiny iid noise
    # only) -> both branches' directions should point the same way every
    # replicate, cosine sim ~1.0.
    pooled_orig, quad_orig = _toy_branch_data(seed=1)
    pooled_alt, quad_alt = _toy_branch_data(seed=2)
    _, mean_sims = bootstrap_cross_branch_similarity(
        pooled_orig, quad_orig, pooled_alt, quad_alt, n_bootstrap=30, seed=0
    )
    assert mean_sims.mean() > 0.99


def test_bootstrap_cross_branch_similarity_drops_when_branches_diverge():
    pooled_orig, quad_orig = _toy_branch_data(seed=1, direction_noise=0.0)
    pooled_alt, quad_alt = _toy_branch_data(seed=2, direction_noise=8.0)
    _, mean_sims = bootstrap_cross_branch_similarity(
        pooled_orig, quad_orig, pooled_alt, quad_alt, n_bootstrap=30, seed=0
    )
    assert mean_sims.mean() < 0.9


def test_bootstrap_cross_branch_similarity_excludes_layer_0():
    # Layer 0 forced to a wildly different direction on each side; if it
    # weren't excluded, mean similarity would be dragged down a lot.
    pooled_orig, quad_orig = _toy_branch_data(n_layers=3, seed=1)
    pooled_alt, quad_alt = _toy_branch_data(n_layers=3, seed=2)
    pooled_orig[:, 0, :] = 0.0  # degenerate layer-0 direction, like the real pipeline
    pooled_alt[:, 0, :] = 0.0
    per_layer, mean_sims = bootstrap_cross_branch_similarity(
        pooled_orig, quad_orig, pooled_alt, quad_alt, n_bootstrap=10, seed=0
    )
    # layer 0's own cosine sim is undefined/degenerate (both directions are
    # the zero vector normalized to itself -> diff_in_means_direction's
    # divide-by-zero guard returns a 0-vector there), but it must not affect
    # the reported mean, which should still reflect layers 1-2 only.
    np.testing.assert_allclose(mean_sims, per_layer[:, 1:].mean(axis=1))


def test_bootstrap_cross_branch_similarity_raises_on_mismatched_quadrant_order():
    pooled_orig, quad_orig = _toy_branch_data(seed=1)
    pooled_alt, _ = _toy_branch_data(seed=2)
    mismatched_quad_alt = np.array(["D"] * len(quad_orig))  # wrong order entirely
    with pytest.raises(ValueError, match="quadrant order must match"):
        bootstrap_cross_branch_similarity(pooled_orig, quad_orig, pooled_alt, mismatched_quad_alt, n_bootstrap=5)


def test_bootstrap_group_difference_ci_excludes_zero_when_clearly_separated():
    rng = np.random.default_rng(0)
    mediated = [rng.normal(0.92, 0.005, size=1000), rng.normal(0.90, 0.005, size=1000)]
    direct = [rng.normal(0.87, 0.005, size=1000)]
    result = bootstrap_group_difference(mediated, direct)
    d = result["difference_mediated_minus_direct"]
    assert d["mean"] > 0
    assert d["ci_low_2.5pct"] > 0  # CI excludes 0 - reliably separated
    assert result["frac_replicates_mediated_gt_direct"] > 0.95


def test_bootstrap_group_difference_ci_includes_zero_when_overlapping():
    rng = np.random.default_rng(0)
    mediated = [rng.normal(0.90, 0.05, size=1000)]
    direct = [rng.normal(0.90, 0.05, size=1000)]
    result = bootstrap_group_difference(mediated, direct)
    d = result["difference_mediated_minus_direct"]
    assert d["ci_low_2.5pct"] < 0 < d["ci_high_97.5pct"]


def _write_toy_stage(act_dir, stage, seed=0, direction_noise=0.0):
    pooled, quadrants = _toy_branch_data(seed=seed, direction_noise=direction_noise)
    np.save(act_dir / f"{stage}_pooled.npy", pooled)
    meta = [{"prompt": f"p{i}", "quadrant": q, "source": "toy", "split": "direction_estimation"}
            for i, q in enumerate(quadrants)]
    with open(act_dir / f"{stage}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def test_main_computes_group_comparison_when_all_pairs_available(tmp_path, monkeypatch):
    import src.analysis.eval_refusal_direction as erd
    import src.interpretability.bootstrap_cross_branch_difference as bccd

    act_dir = tmp_path / "activations"
    act_dir.mkdir()
    out_path = tmp_path / "out.json"

    for stage, noise in [
        ("M1", 0.0), ("M1_alt", 0.0),
        ("M2", 0.0), ("M2_alt", 0.0),
        ("M3", 0.0), ("M3_alt", 0.0),
        ("M3_direct", 0.0), ("M3_direct_alt", 5.0),  # bigger divergence, like the real result
    ]:
        _write_toy_stage(act_dir, stage, seed=hash(stage) % 1000, direction_noise=noise)

    monkeypatch.setattr(erd, "ACT_DIR", act_dir)
    monkeypatch.setattr(bccd, "OUT_PATH", out_path)
    monkeypatch.setattr(bccd, "N_BOOTSTRAP", 20)

    main()

    result = json.loads(out_path.read_text())
    assert set(result["per_pair"].keys()) == {
        "M1_vs_M1_alt", "M2_vs_M2_alt", "M3_vs_M3_alt", "M3_direct_vs_M3_direct_alt",
    }
    assert "group_comparison" in result
    assert result["group_comparison"]["mediated_pairs"] == ["M2_vs_M2_alt", "M3_vs_M3_alt"]
    assert result["group_comparison"]["direct_pairs"] == ["M3_direct_vs_M3_direct_alt"]


def test_main_skips_group_comparison_when_a_pair_missing(tmp_path, monkeypatch):
    import src.analysis.eval_refusal_direction as erd
    import src.interpretability.bootstrap_cross_branch_difference as bccd

    act_dir = tmp_path / "activations"
    act_dir.mkdir()
    out_path = tmp_path / "out.json"

    # M3_direct_alt missing entirely -> direct-DPO pair can't be computed
    for stage in ["M1", "M1_alt", "M2", "M2_alt", "M3", "M3_alt"]:
        _write_toy_stage(act_dir, stage, seed=hash(stage) % 1000)

    monkeypatch.setattr(erd, "ACT_DIR", act_dir)
    monkeypatch.setattr(bccd, "OUT_PATH", out_path)
    monkeypatch.setattr(bccd, "N_BOOTSTRAP", 10)

    main()

    result = json.loads(out_path.read_text())
    assert "group_comparison" not in result
    assert "M3_direct_vs_M3_direct_alt" not in result["per_pair"]
