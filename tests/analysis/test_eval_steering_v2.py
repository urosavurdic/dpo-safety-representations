import json

import pytest

from src.analysis.eval_steering_v2 import (
    build_output_path,
    build_run_config,
    default_tag,
    resolve_alphas,
    steer_direction,
)


class _FakeArgs:
    """Minimal stand-in for argparse.Namespace, only the fields build_run_config reads."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_resolve_alphas_quadrant_a_projection_source(tmp_path):
    proj_path = tmp_path / "quadrant_projections.json"
    with open(proj_path, "w", encoding="utf-8") as f:
        json.dump({"M3": {"A": {str(i): float(i) * 2 for i in range(30)}}}, f)
    # NOTE: real file keys are ints via json list not dict-of-str in practice,
    # but resolve_alphas indexes quadrant_a_proj[layer] as a list - build that shape instead:
    with open(proj_path, "w", encoding="utf-8") as f:
        json.dump({"M3": {"A": [float(i) * 2 for i in range(30)]}}, f)

    alphas = resolve_alphas([24, 25], "M3", "quadrant_a_projection", None, 1.0,
                             quadrant_projections_path=str(proj_path))
    assert alphas == {24: 48.0, 25: 50.0}


def test_resolve_alphas_fixed_source_applies_same_value_to_all_layers():
    alphas = resolve_alphas([14, 21, 28], "M3", "fixed", 3.5, 1.0)
    assert alphas == {14: 3.5, 21: 3.5, 28: 3.5}


def test_resolve_alphas_fixed_source_requires_alpha_value():
    with pytest.raises(ValueError, match="alpha-value"):
        resolve_alphas([24], "M3", "fixed", None, 1.0)


def test_resolve_alphas_coefficient_scales_final_result():
    alphas = resolve_alphas([14], "M3", "fixed", 10.0, 0.2)
    assert alphas == {14: 2.0}


def test_resolve_alphas_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown alpha_source"):
        resolve_alphas([24], "M3", "bogus", None, 1.0)


def test_default_tag_is_descriptive_and_filesystem_safe():
    tag = default_tag("M3_direct", [14, 15, 28], "quadrant_a_projection", 0.2, ["A", "D"])
    assert tag == "M3_direct_L14-15-28_quadrant_a_projection_coef0p2_QAD"
    assert "/" not in tag and " " not in tag


def test_build_output_path_refuses_to_overwrite_existing_file_without_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "raw").mkdir(parents=True)
    (tmp_path / "results" / "raw" / "steering_v2_mytag.json").write_text("[]")

    with pytest.raises(FileExistsError, match="overwrite"):
        build_output_path("mytag", overwrite=False)

    # With --overwrite, must succeed
    path = build_output_path("mytag", overwrite=True)
    assert path.exists()


def test_build_output_path_creates_parent_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = build_output_path("newtag", overwrite=False)
    assert path.parent.exists()
    assert path.name == "steering_v2_newtag.json"


def test_steer_direction_adds_scaled_direction():
    import torch
    hidden = torch.zeros(2, 3, 4)  # (batch, seq, hidden)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    result = steer_direction(hidden, direction, alpha=5.0)
    assert torch.allclose(result[..., 0], torch.full((2, 3), 5.0))
    assert torch.allclose(result[..., 1:], torch.zeros(2, 3, 3))


def test_build_run_config_flags_layers_outside_causally_validated_range():
    args = _FakeArgs(
        stage="M3", alpha_source="quadrant_a_projection", alpha_value=None,
        alpha_coefficient=0.3, quadrants=["D"], skip_baseline=False, limit=None,
        direction_source=None,
    )
    cfg = build_run_config(args, layers=[14, 21, 26], alphas_by_layer={14: 1.0, 21: 2.0, 26: 3.0},
                            out_path=__import__("pathlib").Path("results/raw/steering_v2_test.json"))
    assert cfg["layers_outside_causally_validated_range"] == [14, 21]
    assert cfg["alpha_coefficient"] == 0.3
    assert cfg["generation"]["deterministic"] is True
    assert "git_commit" in cfg
    assert "timestamp_utc" in cfg


def test_build_run_config_no_warning_when_fully_inside_validated_range():
    args = _FakeArgs(
        stage="M3", alpha_source="fixed", alpha_value=1.0,
        alpha_coefficient=1.0, quadrants=["A", "D"], skip_baseline=True, limit=5,
        direction_source=None,
    )
    cfg = build_run_config(args, layers=[24, 25, 26], alphas_by_layer={24: 1.0, 25: 1.0, 26: 1.0},
                            out_path=__import__("pathlib").Path("results/raw/steering_v2_test2.json"))
    assert cfg["layers_outside_causally_validated_range"] == []
