"""Magnitude versus discrimination.

The additive quadrant contrasts confound the two, and the fix is not obvious:
the risk ratio -- the natural scale-free choice, and the one a reader is
likeliest to suggest -- is the WRONG statistic here, because it cannot exceed
``1 / p_benign`` and the strong arms are pinned against that ceiling. These
tests pin both facts down on hand-checkable rates, so the reason the analysis
reports log odds rather than risk ratios is a tested property and not a
preference.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.stagecmp.behav_metrics import (
    HALDANE,
    MIN_SUCCESSES_FOR_RATIO,
    discrimination_report,
    movement_report,
)

from tests.stagecmp.conftest import _records, judge_payload


def _judge(tmp_path, rates, name="judge.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps(judge_payload(rates, scorers=("regex", "wildguard", "strong_reject"))),
        encoding="utf-8",
    )
    return path


def _n_per_quadrant():
    counts = {}
    for row in _records():
        counts[row["quadrant"]] = counts.get(row["quadrant"], 0) + 1
    return counts


# ---- risk ratio and its ceiling ----------------------------------------

def test_risk_ratio_matches_the_hand_computed_rate_ratio(tmp_path):
    rates = {"M2": {"A": 0.5, "B": 0.25, "C": 0.5, "D": 0.25}}
    judge = _judge(tmp_path, rates)
    out = discrimination_report("M2", "wg_refusal", judge, b=200)
    rr = out["ratios"]["A_over_D"]["risk_ratio"]
    assert rr["point"] == pytest.approx(0.5 / 0.25)


def test_the_risk_ratio_ceiling_is_one_over_the_benign_rate(tmp_path):
    """The property that disqualifies the scale: with a benign rate of 0.25 no
    risk ratio above 4 is arithmetically reachable, however selective the
    model is."""
    rates = {"M2": {"A": 1.0, "B": 0.25, "C": 1.0, "D": 0.25}}
    judge = _judge(tmp_path, rates)
    out = discrimination_report("M2", "wg_refusal", judge, b=200)
    cell = out["ratios"]["A_over_D"]
    assert cell["risk_ratio_ceiling"] == pytest.approx(1 / 0.25)
    # saturated numerator: the point estimate IS the ceiling
    assert cell["risk_ratio"]["point"] == pytest.approx(cell["risk_ratio_ceiling"])


def test_log_odds_is_not_bounded_by_the_benign_rate(tmp_path):
    """Two models with the same pinned risk ratio separate on log odds."""
    judge_mid = _judge(tmp_path, {"M2": {"A": 0.5, "B": 0.25, "C": 0.5, "D": 0.25}}, "a.json")
    judge_sat = _judge(tmp_path, {"M2": {"A": 1.0, "B": 0.25, "C": 1.0, "D": 0.25}}, "b.json")
    mid = discrimination_report("M2", "wg_refusal", judge_mid, b=200)
    sat = discrimination_report("M2", "wg_refusal", judge_sat, b=200)
    lo_mid = mid["ratios"]["A_over_D"]["log_odds_ratio"]["point"]
    lo_sat = sat["ratios"]["A_over_D"]["log_odds_ratio"]["point"]
    assert lo_sat > lo_mid


def test_log_odds_stays_finite_at_a_rate_of_exactly_one(tmp_path):
    """Haldane is the reason the DPO arms, which reach 1.000 on quadrant C,
    have a finite interval at all."""
    judge = _judge(tmp_path, {"M2": {"A": 1.0, "B": 0.5, "C": 1.0, "D": 0.5}})
    out = discrimination_report("M2", "wg_refusal", judge, b=200)
    cell = out["ratios"]["A_over_D"]["log_odds_ratio"]
    assert np.isfinite(cell["point"])
    assert np.isfinite(cell["ci_low"]) and np.isfinite(cell["ci_high"])
    assert out["haldane_correction"] == HALDANE


def test_log_odds_uses_the_haldane_corrected_value(tmp_path):
    n = _n_per_quadrant()
    rates = {"M2": {"A": 1.0, "B": 0.5, "C": 1.0, "D": 0.5}}
    judge = _judge(tmp_path, rates)
    out = discrimination_report("M2", "wg_refusal", judge, b=50)
    a_succ, a_n = n["A"], n["A"]
    d_succ, d_n = int(round(0.5 * n["D"])), n["D"]
    expected = np.log(
        ((a_succ + HALDANE) / (a_n - a_succ + HALDANE))
        / ((d_succ + HALDANE) / (d_n - d_succ + HALDANE))
    )
    assert out["ratios"]["A_over_D"]["log_odds_ratio"]["point"] == pytest.approx(expected)


def test_a_thin_denominator_is_flagged_unreliable(tmp_path):
    """Zero withholding prompts in the benign quadrant gives a risk ratio of
    0.0 with a zero-width interval, which reads as a significant REVERSE
    effect unless it is flagged."""
    judge = _judge(tmp_path, {"M2": {"A": 0.5, "B": 0.0, "C": 0.5, "D": 0.0}})
    out = discrimination_report("M2", "wg_refusal", judge, b=100)
    cell = out["ratios"]["A_over_D"]
    assert cell["risk_ratio_reliable"] is False
    assert cell["successes"]["D"] == 0
    assert cell["min_successes_required"] == MIN_SUCCESSES_FOR_RATIO


def test_a_thick_denominator_is_not_flagged(tmp_path):
    judge = _judge(tmp_path, {"M2": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}})
    out = discrimination_report("M2", "wg_refusal", judge, b=50)
    assert out["ratios"]["A_over_D"]["risk_ratio_reliable"] is True


def test_an_indiscriminate_model_scores_one_and_zero(tmp_path):
    """Equal rates everywhere: risk ratio 1, log odds ratio 0, whatever the
    level. That is the invariance the scale exists for."""
    for level in (0.25, 0.75):
        judge = _judge(tmp_path, {"M2": {q: level for q in "ABCD"}}, f"j{level}.json")
        out = discrimination_report("M2", "wg_refusal", judge, b=100)
        cell = out["ratios"]["A_over_D"]
        assert cell["risk_ratio"]["point"] == pytest.approx(1.0)
        assert cell["log_odds_ratio"]["point"] == pytest.approx(0.0, abs=1e-9)


def test_a_continuous_metric_is_refused(tmp_path):
    judge = _judge(tmp_path, {"M2": {q: 0.5 for q in "ABCD"}})
    with pytest.raises(ValueError, match="continuous"):
        discrimination_report("M2", "sr_score", judge, b=10)


# ---- movement -----------------------------------------------------------

def test_movement_is_the_mean_absolute_per_quadrant_change(tmp_path):
    rates = {
        "M1": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
        "M2": {"A": 0.5, "B": 0.25, "C": 0.5, "D": 0.0},
    }
    judge = _judge(tmp_path, rates)
    out = movement_report("M1", "M2", "wg_refusal", judge)
    assert out["delta_by_quadrant"]["A"] == pytest.approx(0.5)
    assert out["delta_by_quadrant"]["D"] == pytest.approx(0.0)
    assert out["mean_absolute_movement"] == pytest.approx(
        np.mean([0.5, 0.25, 0.5, 0.0])
    )


def test_movement_counts_a_decrease_as_movement(tmp_path):
    """Absolute value, so an arm that lowers withholding has not 'done
    nothing'."""
    rates = {
        "M1": {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5},
        "M2": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
    }
    judge = _judge(tmp_path, rates)
    out = movement_report("M1", "M2", "wg_refusal", judge)
    assert out["mean_absolute_movement"] == pytest.approx(0.5)
    assert all(v < 0 for v in out["delta_by_quadrant"].values())


def test_movement_drops_a_prompt_unusable_at_either_stage(tmp_path):
    rates = {"M1": {q: 0.5 for q in "ABCD"}, "M2": {q: 0.5 for q in "ABCD"}}
    payload = judge_payload(rates, scorers=("regex", "wildguard", "strong_reject"))
    for rec in payload["records"]:
        if rec["stage"] == "M2_behavior" and rec["quadrant"] == "A":
            rec["wildguard"] = {"response_refusal": None, "judge_status": "failed"}
    path = tmp_path / "holes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    out = movement_report("M1", "M2", "wg_refusal", path)
    assert "A" not in out["delta_by_quadrant"]
    assert out["n_shared"] == sum(
        n for q, n in _n_per_quadrant().items() if q != "A"
    )
