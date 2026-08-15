import numpy as np

from src.interpretability.bootstrap_direction_stability import (
    bootstrap_directions,
    summarize_stability,
    summarize_stability_full,
)
from src.analysis.eval_refusal_direction import diff_in_means_direction


def _toy_data():
    pooled = np.array([
        [[5.0, 0.0]], [[5.0, 0.0]],   # A x2, identical
        [[-5.0, 0.0]], [[-5.0, 0.0]], # D x2, identical
    ])
    quadrants = np.array(["A", "A", "D", "D"])
    return pooled, quadrants


def test_bootstrap_directions_shape():
    pooled, quadrants = _toy_data()
    boots = bootstrap_directions(pooled, quadrants, n_bootstrap=10, seed=1)
    assert boots.shape == (10, 1, 2)


def test_stable_direction_gives_near_perfect_bootstrap_agreement():
    # All A rows identical, all D rows identical -> every resample gives
    # EXACTLY the same direction; cosine sim should be ~1.0 always.
    pooled, quadrants = _toy_data()
    original = diff_in_means_direction(pooled, quadrants)
    boots = bootstrap_directions(pooled, quadrants, n_bootstrap=20, seed=2)
    mean_sim, std_sim = summarize_stability(boots, original, layer=0)
    assert abs(mean_sim - 1.0) < 1e-6
    assert std_sim < 1e-6


def test_summarize_stability_full_reports_median_and_percentiles():
    """PROJECT_CONTEXT.md's bootstrap spec asks for mean/median/std/2.5%/97.5%,
    not just mean+std."""
    pooled, quadrants = _toy_data()
    original = diff_in_means_direction(pooled, quadrants)
    boots = bootstrap_directions(pooled, quadrants, n_bootstrap=20, seed=2)
    stats = summarize_stability_full(boots, original, layer=0)
    assert set(stats.keys()) == {"mean", "median", "std", "ci_low_2.5pct", "ci_high_97.5pct"}
    assert abs(stats["mean"] - 1.0) < 1e-6
    assert abs(stats["median"] - 1.0) < 1e-6
    assert stats["ci_low_2.5pct"] <= stats["mean"] <= stats["ci_high_97.5pct"]


def test_bootstrap_deterministic_given_fixed_seed():
    pooled, quadrants = _toy_data()
    boots_a = bootstrap_directions(pooled, quadrants, n_bootstrap=15, seed=7)
    boots_b = bootstrap_directions(pooled, quadrants, n_bootstrap=15, seed=7)
    assert np.allclose(boots_a, boots_b)