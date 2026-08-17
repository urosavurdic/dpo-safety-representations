import json

import numpy as np

from src.analysis.eval_refusal_direction import (
    activations_available,
    cosine_similarity_per_layer,
    diff_in_means_direction,
    main,
    project_onto_direction,
)


def _toy_data():
    # 4 prompts, 2 layers, 3-dim hidden. A and D are trivially separable
    # along dim 0 so the expected direction is exactly known.
    pooled = np.array([
        [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],   # A
        [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],   # A
        [[-5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]], # D
        [[-5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]], # D
    ])
    quadrants = np.array(["A", "A", "D", "D"])
    return pooled, quadrants


def test_diff_in_means_direction_is_unit_norm_and_points_at_A():
    pooled, quadrants = _toy_data()
    direction = diff_in_means_direction(pooled, quadrants)
    assert direction.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(direction, axis=-1), 1.0, atol=1e-6)
    # A is at +x, D at -x -> direction should point along +x
    np.testing.assert_allclose(direction[:, 0], 1.0, atol=1e-6)


def test_cosine_similarity_identical_directions_is_one():
    pooled, quadrants = _toy_data()
    d = diff_in_means_direction(pooled, quadrants)
    sim = cosine_similarity_per_layer(d, d)
    np.testing.assert_allclose(sim, 1.0, atol=1e-6)


def test_project_onto_direction_shape_and_sign():
    pooled, quadrants = _toy_data()
    direction = diff_in_means_direction(pooled, quadrants)
    proj = project_onto_direction(pooled, direction)
    assert proj.shape == (4, 2)
    # A-quadrant rows should project positive, D-quadrant negative
    assert (proj[:2] > 0).all()
    assert (proj[2:] < 0).all()


def _write_toy_stage(act_dir, stage, seed=0):
    rng = np.random.default_rng(seed)
    pooled, quadrants = _toy_data()
    pooled = pooled + rng.normal(scale=0.01, size=pooled.shape)  # tiny per-stage variation
    np.save(act_dir / f"{stage}_pooled.npy", pooled)
    meta = [{"prompt": f"p{i}", "quadrant": q, "source": "toy"} for i, q in enumerate(quadrants)]
    with open(act_dir / f"{stage}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def test_activations_available_true_only_when_both_files_present(tmp_path):
    act_dir = tmp_path / "activations"
    act_dir.mkdir()
    assert activations_available("M0", act_dir=act_dir) is False
    _write_toy_stage(act_dir, "M0")
    assert activations_available("M0", act_dir=act_dir) is True
    assert activations_available("M1", act_dir=act_dir) is False  # different stage, not written


def test_main_computes_cross_branch_only_for_pairs_with_both_sides_available(tmp_path, monkeypatch):
    """The whole point of the alt branch: does main() actually compute
    M1_vs_M1_alt etc. once both activations exist, and correctly SKIP pairs
    where the alt side isn't ready yet (rather than crashing)."""
    import src.analysis.eval_refusal_direction as erd

    act_dir = tmp_path / "activations"
    out_dir = tmp_path / "refusal_direction"
    act_dir.mkdir()

    # Available: M0, M1, M3, M1_alt (M2, M3_alt, M3_direct, M3_direct_alt NOT ready yet)
    for stage in ["M0", "M1", "M3", "M1_alt"]:
        _write_toy_stage(act_dir, stage, seed=hash(stage) % 1000)

    monkeypatch.setattr(erd, "ACT_DIR", act_dir)
    monkeypatch.setattr(erd, "OUT_DIR", out_dir)

    main()

    with open(out_dir / "cosine_similarity.json", encoding="utf-8") as f:
        result = json.load(f)

    # Cross-branch: only M1_vs_M1_alt should be present (M2/M3/M3_direct's alt
    # counterparts aren't available)
    assert "M1_vs_M1_alt" in result["cross_branch"]
    assert "M2_vs_M2_alt" not in result["cross_branch"]
    assert "M3_vs_M3_alt" not in result["cross_branch"]
    assert "M3_direct_vs_M3_direct_alt" not in result["cross_branch"]

    # vs_M0 should include all 4 available stages
    assert set(result["vs_M0"].keys()) == {"M0", "M1", "M3", "M1_alt"}

    # adjacent (original sequential): only M0_vs_M1 possible (M2 missing breaks the rest)
    assert "M0_vs_M1" in result["adjacent"]
    assert "M1_vs_M2" not in result["adjacent"]
    assert "M2_vs_M3" not in result["adjacent"]

    # adjacent_alt: M0_vs_M1_alt possible; M1_alt_vs_M2_alt not (M2_alt missing)
    assert "M0_vs_M1_alt" in result["adjacent_alt"]
    assert "M1_alt_vs_M2_alt" not in result["adjacent_alt"]

    # direct_branch: M3_direct not available at all -> neither of its two entries present
    assert "M1_vs_M3_direct" not in result["direct_branch"]
    assert "M3_direct_vs_M3" not in result["direct_branch"]


def test_main_computes_all_cross_branch_pairs_when_everything_available(tmp_path, monkeypatch):
    import src.analysis.eval_refusal_direction as erd

    act_dir = tmp_path / "activations"
    out_dir = tmp_path / "refusal_direction"
    act_dir.mkdir()

    for stage in erd.STAGES:
        _write_toy_stage(act_dir, stage, seed=hash(stage) % 1000)

    monkeypatch.setattr(erd, "ACT_DIR", act_dir)
    monkeypatch.setattr(erd, "OUT_DIR", out_dir)

    main()

    with open(out_dir / "cosine_similarity.json", encoding="utf-8") as f:
        result = json.load(f)

    assert set(result["cross_branch"].keys()) == {
        "M1_vs_M1_alt", "M2_vs_M2_alt", "M3_vs_M3_alt", "M3_direct_vs_M3_direct_alt",
    }
    assert set(result["direct_branch"].keys()) == {
        "M1_vs_M3_direct", "M3_direct_vs_M3", "M1_alt_vs_M3_direct_alt", "M3_direct_alt_vs_M3_alt",
    }
    # every cross-branch cosine similarity should be a real per-layer list, not empty
    for pair_values in result["cross_branch"].values():
        assert len(pair_values) == 2  # n_layers from toy data


def test_main_raises_clearly_when_m0_activations_missing(tmp_path, monkeypatch):
    import src.analysis.eval_refusal_direction as erd
    import pytest

    act_dir = tmp_path / "activations"
    out_dir = tmp_path / "refusal_direction"
    act_dir.mkdir()
    _write_toy_stage(act_dir, "M1")  # M0 deliberately missing

    monkeypatch.setattr(erd, "ACT_DIR", act_dir)
    monkeypatch.setattr(erd, "OUT_DIR", out_dir)

    with pytest.raises(RuntimeError, match="M0"):
        main()