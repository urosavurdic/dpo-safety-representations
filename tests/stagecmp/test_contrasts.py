"""The 2x2 estimands, and the bootstrap structure they depend on."""
from __future__ import annotations

import numpy as np
import pytest

from src.common.stats import BOOTSTRAP_SEED
from src.stagecmp.contrasts import (
    ARMS,
    factorial_2x2,
    independent_resample_comparison,
    per_prompt_deltas,
    selectivity_2x2,
)


def test_deltas_cover_every_arm_on_shared_prompts(judge_file):
    ids, deltas = per_prompt_deltas("withhold", judge_file)
    assert set(deltas) == set(ARMS)
    assert all(len(d) == len(ids) for d in deltas.values())
    assert len(ids) == len(set(ids))


def test_obj_recovers_a_known_difference(judge_file):
    """Quadrant A: SFT arms move +1/3, DPO arms move +1.0 and +2/3.
    OBJ = 1/2[(1.0 - 1/3) + (2/3 - 1/3)] = 1/2[2/3 + 1/3] = 0.5."""
    r = factorial_2x2("withhold", "A", judge_file, b=200)
    assert r["arms"]["sft_alpaca"]["theta"] == pytest.approx(1 / 3)
    assert r["arms"]["dpo_alpaca"]["theta"] == pytest.approx(1.0)
    assert r["OBJ"]["point"] == pytest.approx(0.5)


def test_int_negates_when_the_corpora_are_swapped(judge_file, monkeypatch):
    import src.stagecmp.contrasts as C

    original = dict(C.TRANSITIONS_2X2)
    before = factorial_2x2("withhold", "A", judge_file, b=200)["INT"]["point"]

    swapped = {
        "sft_alpaca": original["sft_dolly"],
        "dpo_alpaca": original["dpo_dolly"],
        "sft_dolly": original["sft_alpaca"],
        "dpo_dolly": original["dpo_alpaca"],
    }
    monkeypatch.setattr(C, "TRANSITIONS_2X2", swapped)
    after = factorial_2x2("withhold", "A", judge_file, b=200)["INT"]["point"]
    assert after == pytest.approx(-before)


def test_obj_is_the_mean_of_the_per_prompt_contribution(judge_file):
    """The linear collapse is exact, not an approximation: bootstrapping the
    single contribution vector IS the shared-resample bootstrap."""
    ids, d = per_prompt_deltas("withhold", judge_file)
    obj = 0.5 * ((d["dpo_alpaca"] - d["sft_alpaca"]) + (d["dpo_dolly"] - d["sft_dolly"]))
    theta = {a: d[a].mean() for a in ARMS}
    by_theta = 0.5 * ((theta["dpo_alpaca"] - theta["sft_alpaca"])
                      + (theta["dpo_dolly"] - theta["sft_dolly"]))
    assert obj.mean() == pytest.approx(by_theta)

    r = factorial_2x2("withhold", None, judge_file, b=200)
    assert r["OBJ"]["point"] == pytest.approx(by_theta)


def test_pairing_narrows_a_simple_difference(judge_file):
    """For a SIMPLE two-arm difference, pairing narrows the interval whenever
    the arms are positively correlated -- which they are, since the same prompt
    is refused first at every rate.

    This is the guarantee. It does NOT extend to a difference of differences:
    see ``independent_resample_comparison``'s docstring for why the cross terms
    make OBJ's width comparison go either way.
    """
    from src.common.stats import paired_bootstrap_ci
    from src.stagecmp.behav_metrics import per_prompt

    pre = per_prompt("M1", "withhold", judge_file)
    post = per_prompt("M3_direct", "withhold", judge_file)
    ids = sorted(set(pre) & set(post))
    x = np.array([pre[i] for i in ids])
    y = np.array([post[i] for i in ids])

    paired = paired_bootstrap_ci(y - x, b=4000)
    paired_width = paired["ci_high"] - paired["ci_low"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(ids)
    reps = np.array([
        y[rng.integers(0, n, n)].mean() - x[rng.integers(0, n, n)].mean()
        for _ in range(4000)
    ])
    unpaired_width = float(np.percentile(reps, 97.5) - np.percentile(reps, 2.5))

    assert paired["point"] == pytest.approx((y - x).mean())
    assert paired_width <= unpaired_width


def test_independent_resampling_agrees_on_the_point_estimate(judge_file):
    """The two estimators must differ only in their interval, never in where
    they are centred."""
    paired = factorial_2x2("withhold", "A", judge_file, b=2000)["OBJ"]
    independent = independent_resample_comparison("withhold", "A", judge_file, b=2000)
    centre = 0.5 * (independent["ci_low"] + independent["ci_high"])
    assert centre == pytest.approx(paired["point"], abs=0.05)
    assert "WRONG ESTIMATOR" in independent["note"]


def test_deterministic_under_the_frozen_seed(judge_file):
    a = factorial_2x2("withhold", "A", judge_file, b=300, seed=BOOTSTRAP_SEED)
    b = factorial_2x2("withhold", "A", judge_file, b=300, seed=BOOTSTRAP_SEED)
    assert a["OBJ"] == b["OBJ"]
    c = factorial_2x2("withhold", "A", judge_file, b=300, seed=BOOTSTRAP_SEED + 1)
    assert c["OBJ"]["ci_low"] != a["OBJ"]["ci_low"]


def test_caveat_travels_with_the_numbers(judge_file):
    r = factorial_2x2("withhold", "A", judge_file, b=50)
    assert "matched initialisation" in r["caveat"]
    assert "EXPLORATORY" in r["status"]


def test_selectivity_is_zero_when_both_quadrants_move_together(judge_file):
    """M3_direct moves B and D by the same +1/3, so its B-vs-D selectivity is 0."""
    r = selectivity_2x2("withhold", harm_quadrant="B", benign_quadrant="D",
                        judge_path=judge_file, b=200)
    assert r["arms"]["dpo_alpaca"]["point"] == pytest.approx(0.0, abs=1e-9)


def test_selectivity_detects_a_real_gap(judge_file):
    """A moves +1.0 while D moves +1/3 for dpo_alpaca, so selectivity is +2/3."""
    r = selectivity_2x2("withhold", harm_quadrant="A", benign_quadrant="D",
                        judge_path=judge_file, b=200)
    assert r["arms"]["dpo_alpaca"]["point"] == pytest.approx(2 / 3)
