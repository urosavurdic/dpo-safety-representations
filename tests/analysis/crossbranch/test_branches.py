"""Canonical condition table: the single source of truth the worker and
runner both key off. These tests pin the arithmetic behind "Stage 1 is 8
units" and the model/vector distinction that keeps the runner from demanding
a .npz for a condition that injects nothing."""
import pytest

from src.analysis.crossbranch import branches as B


def test_branch_map_matches_repo_stage_names():
    assert B.BRANCHES["A"]["pre"] == "M2"
    assert B.BRANCHES["A"]["post"] == "M3"
    assert B.BRANCHES["B"]["pre"] == "M2_alt"
    assert B.BRANCHES["B"]["post"] == "M3_alt"


def test_resolve_maps_roles_to_stages():
    r = B.resolve("A", "B")
    assert r["source_pre"] == "M2" and r["source_post"] == "M3"
    assert r["target_pre"] == "M2_alt" and r["target_post"] == "M3_alt"


def test_resolve_reciprocal_direction_needs_no_rename():
    r = B.resolve("B", "A")
    assert r["source_pre"] == "M2_alt" and r["target_pre"] == "M2"


def test_resolve_rejects_same_branch_and_unknown_branch():
    with pytest.raises(ValueError, match="must differ"):
        B.resolve("A", "A")
    with pytest.raises(ValueError, match="not a known branch"):
        B.resolve("A", "Z")


def test_p0_conditions_are_exactly_the_stage_one_four():
    assert set(B.P0_CONDITIONS) == {
        "baseline_target",
        "reference_target",
        "own_delta_target",
        "own_normmatched_random",
    }


def test_model_conditions_take_no_coefficient_and_no_artifact():
    for name in ("baseline_target", "reference_target"):
        c = B.get(name)
        assert c.kind == B.MODEL
        assert c.artifact is None, "model conditions must not require an .npz"
        assert c.takes_coefficient is False


def test_vector_conditions_all_declare_an_artifact():
    for c in B.CONDITIONS:
        if c.kind == B.VECTOR:
            assert c.artifact, f"{c.name} is a vector condition with no artifact"


def test_stage_one_is_eight_units():
    units = B.planned_units(list(B.P0_CONDITIONS))
    assert len(units) == 8
    assert sum(1 for _, c in units if c is None) == 2       # model conditions
    assert sum(1 for _, c in units if c is not None) == 6   # 2 x 3 coefficients


def test_reference_target_loads_the_post_dpo_checkpoint():
    assert B.checkpoint_for("reference_target", "A", "B") == "M3_alt"
    assert B.checkpoint_for("baseline_target", "A", "B") == "M2_alt"
    assert B.checkpoint_for("own_delta_target", "A", "B") == "M2_alt"


def test_stages_needed_is_the_four_activation_stages():
    assert set(B.stages_needed("A", "B")) == {"M2", "M3", "M2_alt", "M3_alt"}


def test_deferred_condition_raises_with_a_reason():
    with pytest.raises(KeyError, match="DEFERRED"):
        B.get("xfer_delta_source_decomposition")


def test_no_condition_name_encodes_a_branch():
    # Direction lives in metadata and filenames, never in a condition name,
    # so the reciprocal run is a flag rather than a rename of every artifact.
    for c in B.CONDITIONS:
        assert "_A_" not in c.name and "_B_" not in c.name
        assert not c.name.endswith(("_A", "_B"))
        assert "b2" not in c.name and "b3" not in c.name


def test_direction_tag():
    assert B.direction_tag("A", "B") == "AtoB"
    assert B.direction_tag("B", "A") == "BtoA"


def test_frozen_constants_mirror_the_plan():
    assert B.INJECT_LAYER == 24
    assert B.COEFFICIENTS == (0.5, 1.0, 2.0)
    assert B.INJECT_MODE == "last_prompt_only"
    assert B.POSITION == "final"
