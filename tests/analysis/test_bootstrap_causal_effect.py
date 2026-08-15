from src.analysis.bootstrap_causal_effect import build_paired_outcomes, bootstrap_effect_ci


def test_build_paired_outcomes_pairs_matched_prompts_and_excludes_wrong_quadrant():
    rows = [
        {"prompt": "p1", "quadrant": "C", "stage": "M3_baseline", "response": "It is not advisable to do that."},
        {"prompt": "p1", "quadrant": "C", "stage": "M3_ablated", "response": "Sure, here's how."},
        {"prompt": "p2", "quadrant": "A", "stage": "M3_baseline", "response": "Sure, here's how."},
    ]
    pairs = build_paired_outcomes(rows, quadrant="C", category="soft_deflection")
    assert pairs == [(True, False)]


def test_bootstrap_effect_ci_recovers_known_effect():
    # 10 pairs: baseline always True, other always False -> exactly 100% reduction
    pairs = [(True, False)] * 10
    result = bootstrap_effect_ci(pairs, n_bootstrap=100, seed=0)
    assert result["n_bootstrap_replicates"] == 100
    abs_e = result["absolute_effect"]
    assert abs(abs_e["mean"] - 1.0) < 1e-6
    assert abs(abs_e["ci_low"] - 1.0) < 1e-6
    assert abs(abs_e["ci_high"] - 1.0) < 1e-6
    rel_e = result["relative_effect"]
    assert abs(rel_e["mean"] - 1.0) < 1e-6
    assert rel_e["n_defined_replicates"] == 100


def test_bootstrap_effect_ci_never_resamples_conditions_independently():
    # If baseline/other were resampled independently, a 50/50 mix would often
    # NOT reproduce the exact 0.5 absolute effect every single replicate -
    # paired resampling must preserve the per-prompt link.
    pairs = [(True, False), (True, True), (False, False), (False, True)] * 5
    result = bootstrap_effect_ci(pairs, n_bootstrap=200, seed=3)
    # Sanity: mean absolute effect should be near the true value (0.25: 5/20 True->False net)
    assert -0.1 < result["absolute_effect"]["mean"] < 0.4


def test_bootstrap_effect_ci_handles_undefined_relative_effect():
    # baseline always False -> baseline_rate is 0 in every resample ->
    # relative effect is undefined (0/0) in every replicate.
    pairs = [(False, False), (False, True)] * 5
    result = bootstrap_effect_ci(pairs, n_bootstrap=50, seed=1)
    assert result["relative_effect"]["mean"] is None
    assert result["relative_effect"]["n_defined_replicates"] == 0
    # absolute effect is still well-defined
    assert result["absolute_effect"]["mean"] is not None


def test_bootstrap_effect_ci_deterministic_given_fixed_seed():
    pairs = [(True, False), (False, True), (True, True)] * 4
    r1 = bootstrap_effect_ci(pairs, n_bootstrap=50, seed=9)
    r2 = bootstrap_effect_ci(pairs, n_bootstrap=50, seed=9)
    assert r1 == r2