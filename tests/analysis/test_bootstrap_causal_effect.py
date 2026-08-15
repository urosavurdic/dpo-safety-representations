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
    mean_effect, lo, hi = bootstrap_effect_ci(pairs, n_bootstrap=100, seed=0)
    assert abs(mean_effect - 1.0) < 1e-6
    assert abs(lo - 1.0) < 1e-6
    assert abs(hi - 1.0) < 1e-6