"""Toy tests for src/analysis/intervention_conditions.py (WP-Causal / WP-Steer)."""
import pytest

from src.analysis import intervention_conditions as ic


def test_frozen_vocabulary():
    assert ic.CAUSAL_REQUIRED == ("baseline", "ablated_AD", "ablated_random")
    assert ic.CAUSAL_SECONDARY == ("ablated_AB",)
    assert ic.STEERING_ALPHA_COEFFICIENTS == (0.5, 1.0, 2.0)
    assert ic.STEERING_CONDITIONS == ("baseline", "steered_learned", "steered_random")


def test_parse_conditions_defaults_to_required_set():
    assert ic.parse_conditions_arg(None) == list(ic.CAUSAL_REQUIRED)
    assert ic.parse_conditions_arg([]) == list(ic.CAUSAL_REQUIRED)


def test_parse_conditions_always_includes_required_and_keeps_extras():
    out = ic.parse_conditions_arg(["ablated_AB"])
    assert set(ic.CAUSAL_REQUIRED).issubset(out)
    assert out[-1] == "ablated_AB"


def test_parse_conditions_rejects_unknown():
    with pytest.raises(ic.ConditionError):
        ic.parse_conditions_arg(["ablated_AD", "ablated_nonsense"])


def test_parse_alpha_coefficients_default_and_validation():
    assert ic.parse_alpha_coefficients_arg(None) == [0.5, 1.0, 2.0]
    assert ic.parse_alpha_coefficients_arg(["0.5", "1", "2"]) == [0.5, 1.0, 2.0]
    with pytest.raises(ic.ConditionError):
        ic.parse_alpha_coefficients_arg([1.0, -0.5])


def test_plan_causal_fits_all_when_budget_ample():
    plan = ic.plan_causal_conditions("M3", per_condition_minutes=30, budget_minutes=300,
                                     requested=["baseline", "ablated_AD", "ablated_random", "ablated_AB"])
    assert plan.scheduled == ["baseline", "ablated_AD", "ablated_random", "ablated_AB"]
    assert plan.omitted == []


def test_plan_causal_omits_ablated_AB_when_only_required_fits():
    plan = ic.plan_causal_conditions("M3", per_condition_minutes=80, budget_minutes=250,
                                     requested=["baseline", "ablated_AD", "ablated_random", "ablated_AB"])
    assert plan.scheduled == ["baseline", "ablated_AD", "ablated_random"]
    assert plan.omitted == ["ablated_AB"]
    js = plan.to_json()
    assert "NO causal safety-specificity claim" in js["ablated_AB_omission_effect"]


def test_plan_causal_flags_when_required_set_does_not_fit():
    plan = ic.plan_causal_conditions("M3", per_condition_minutes=200, budget_minutes=250)
    assert "ablated_random" in plan.omitted
    assert any("REQUIRED condition" in n for n in plan.notes)


def test_steering_cut_order_never_cuts_the_random_control():
    order = ic.steering_cut_order()
    assert order[0].startswith("M1")
    assert "never cut" in order[-1]


def test_provenance_blocks_shape():
    stage_rec = {
        "d_AB_vs_d_AD_cosine_per_layer": [0.1, 0.2],
        "d_AB_gate": "NONE - descriptive only",
        "random_direction_seed": 20260904,
        "ablation_control": {"seed": 20260904, "per_layer": {}},
    }
    ab = ic.ablation_provenance_block(stage_rec)
    assert ab["random_direction_seed"] == 20260904
    assert ab["reference"] == "analysis_plan.md §6.1"

    st = ic.steering_provenance_block("M3", 24, 120, 3.5, 2.0, 4.1, 0.02, 20260904)
    assert st["alpha"] == pytest.approx(7.0)
    assert st["degeneration_rate"] == 0.02
