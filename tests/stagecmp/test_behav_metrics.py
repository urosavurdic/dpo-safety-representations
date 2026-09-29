"""Behavioural metrics, and the two reporting rules they must not break."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.crossbranch.analyze import distribution, total_variation
from src.stagecmp.behav_metrics import (
    label_of,
    mcnemar,
    paired_difference,
    per_prompt,
    stage_profile,
    tv_and_agreement,
    value_of,
)
from tests.stagecmp.conftest import judge_payload


def test_label_precedence_matches_the_frozen_classifier():
    """degenerate > refusal > soft_deflection > comply."""
    assert label_of({"regex": {"refused": True, "degenerate": True,
                               "soft_deflection": True}}) == "degenerate"
    assert label_of({"regex": {"refused": True, "degenerate": False,
                               "soft_deflection": True}}) == "refusal"
    assert label_of({"regex": {"refused": False, "degenerate": False,
                               "soft_deflection": True}}) == "soft_deflection"
    assert label_of({"regex": {"refused": False, "degenerate": False,
                               "soft_deflection": False}}) == "comply"
    assert label_of({"regex": {"refused": None}}) is None


def test_unscored_rows_are_omitted_not_counted_as_zero(judge_file):
    """The failure mode that previously produced a wrong claim: an unscored
    scorer must vanish from the denominator, never read as a zero."""
    assert per_prompt("M3_direct", "sr_score", judge_file) == {}
    assert value_of({"strong_reject": {"score": None}}, "sr_score") is None


def test_stage_profile_rates_are_hand_checkable(judge_file):
    p = stage_profile("M3_direct", judge_file)["per_quadrant"]
    assert p["A"]["withhold"]["rate"] == pytest.approx(1.0)
    assert p["D"]["withhold"]["rate"] == pytest.approx(1 / 3)
    assert p["A"]["n"] == 24


def test_mcnemar_counts_discordant_pairs_only(judge_file):
    """M1 withholds on nothing, M3_direct on all of quadrant A, so every pair
    is discordant in one direction."""
    r = mcnemar("M1", "M3_direct", "withhold", "A", judge_file)
    assert r["b"] == 0
    assert r["c"] == 24
    assert r["n_discordant"] == 24
    assert r["p_exact"] < 1e-6
    assert "DISCORDANT COUNT" in r["b_def"]


def test_mcnemar_is_symmetric_in_its_counts(judge_file):
    fwd = mcnemar("M1", "M3_direct", "withhold", "A", judge_file)
    rev = mcnemar("M3_direct", "M1", "withhold", "A", judge_file)
    assert (fwd["b"], fwd["c"]) == (rev["c"], rev["b"])
    assert fwd["p_exact"] == pytest.approx(rev["p_exact"])


def test_mcnemar_refuses_a_continuous_metric(judge_file):
    with pytest.raises(ValueError, match="binary"):
        mcnemar("M1", "M3_direct", "sr_score", "A", judge_file)


def test_tv_matches_the_crossbranch_implementation(judge_file):
    r = tv_and_agreement("M1", "M3_direct", "A", judge_file)
    p = distribution(["comply"] * 24)
    q = distribution(["refusal"] * 24)
    assert r["total_variation"] == pytest.approx(total_variation(p, q))


def test_tv_never_travels_without_its_two_companions(judge_file):
    """TV moves when the degenerate rate moves, and can be zero at chance
    agreement, so the distributions and the per-prompt agreement must be
    present in the same payload."""
    r = tv_and_agreement("M2", "M3_direct", "A", judge_file)
    assert "four_way_pre" in r and "four_way_post" in r
    assert "per_prompt_agreement" in r
    assert "aggregate" in r["reading"]


def test_tv_zero_with_chance_agreement_is_constructible(tmp_path, records):
    """The case the companion metric exists for: identical distributions,
    disagreement on every prompt."""
    rates = {"X": {"A": 0.5}, "Y": {"A": 0.5}}
    payload = judge_payload(rates, records)
    # flip Y's assignment so the same rate lands on the opposite prompts
    for rec in payload["records"]:
        if rec["condition"] == "Y_behavior" and rec["quadrant"] == "A":
            rec["regex"]["refused"] = not rec["regex"]["refused"]
    path = tmp_path / "flip.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    r = tv_and_agreement("X", "Y", "A", path)
    assert r["total_variation"] == pytest.approx(0.0)
    assert r["per_prompt_agreement"] == pytest.approx(0.0)


def test_paired_difference_drops_incomplete_units_whole(tmp_path, records):
    """A prompt unusable at either stage is dropped entirely, the CF1 rule."""
    rates = {"P": {"A": 0.5}, "Q": {"A": 0.5}}
    payload = judge_payload(rates, records)
    dropped = 0
    for rec in payload["records"]:
        if rec["condition"] == "Q_behavior" and rec["quadrant"] == "A" and dropped < 4:
            rec["regex"]["refused"] = None
            dropped += 1
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    r = paired_difference("P", "Q", "withhold", "A", path, b=100)
    assert r["dropped_incomplete"]["shared"] == 20
    assert r["n_effective"] == 20
