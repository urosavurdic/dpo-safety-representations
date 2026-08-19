import json

import numpy as np

from src.interpretability.bottleneck_layer import (
    bootstrap_bottleneck_layers,
    cohens_d,
    find_bottleneck_layer,
    per_layer_separability,
    summarize_bottleneck_bootstrap,
)


def test_cohens_d_zero_for_identical_distributions():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert cohens_d(x, y) == 0.0


def test_cohens_d_positive_when_x_greater_than_y():
    x = np.array([5.0, 6.0, 7.0])
    y = np.array([1.0, 2.0, 3.0])
    d = cohens_d(x, y)
    assert d > 0
    # Well-separated, low-variance groups -> large effect size
    assert d > 1.0


def test_cohens_d_handles_zero_variance_without_nan():
    # All-identical values in both groups (zero pooled variance) - must
    # return 0.0, not NaN/inf from a division by zero.
    x = np.array([3.0, 3.0, 3.0])
    y = np.array([3.0, 3.0, 3.0])
    assert cohens_d(x, y) == 0.0


def test_cohens_d_handles_tiny_groups_without_crashing():
    assert cohens_d(np.array([1.0]), np.array([2.0])) == 0.0  # n<2, degenerate


def _toy_pooled_and_quadrants(n_layers=3, hidden_dim=4):
    """3 prompts per quadrant, n_layers x hidden_dim each. Layer 1 is
    constructed to have the cleanest A-vs-D separation (bottleneck), layers
    0 and 2 deliberately noisier/overlapping."""
    rng = np.random.default_rng(0)
    quadrants = np.array(["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3)

    pooled = rng.normal(scale=0.1, size=(12, n_layers, hidden_dim))
    # Layer 1: push A and C strongly positive on dim 0, B and D strongly
    # negative -> should be the argmax bottleneck for BOTH metrics.
    pooled[quadrants == "A", 1, 0] += 5.0
    pooled[quadrants == "C", 1, 0] += 5.0
    pooled[quadrants == "B", 1, 0] -= 5.0
    pooled[quadrants == "D", 1, 0] -= 5.0
    return pooled, quadrants


def test_per_layer_separability_shape():
    pooled, quadrants = _toy_pooled_and_quadrants()
    d_a_vs_d, d_harm_vs_surface = per_layer_separability(pooled, quadrants)
    assert d_a_vs_d.shape == (3,)
    assert d_harm_vs_surface.shape == (3,)


def test_bottleneck_layer_identifies_the_constructed_separable_layer():
    pooled, quadrants = _toy_pooled_and_quadrants()
    d_a_vs_d, d_harm_vs_surface = per_layer_separability(pooled, quadrants)

    layer_a_d, effect_a_d = find_bottleneck_layer(d_a_vs_d)
    layer_hs, effect_hs = find_bottleneck_layer(d_harm_vs_surface)

    assert layer_a_d == 1
    assert layer_hs == 1
    assert abs(effect_a_d) > 2.0  # constructed as a huge, obvious effect
    assert abs(effect_hs) > 2.0


def test_find_bottleneck_layer_uses_magnitude_not_raw_value():
    # A layer with a large NEGATIVE effect should still be found over a
    # small positive one - argmax of |effect|, not effect itself.
    effect_sizes = np.array([0.1, -3.0, 0.5])
    idx, val = find_bottleneck_layer(effect_sizes)
    assert idx == 1
    assert val == -3.0


def test_bootstrap_bottleneck_layers_shape():
    pooled, quadrants = _toy_pooled_and_quadrants()
    a_d, hs = bootstrap_bottleneck_layers(pooled, quadrants, n_bootstrap=25, seed=0)
    assert a_d.shape == (25,)
    assert hs.shape == (25,)


def test_bootstrap_bottleneck_layers_concentrates_on_the_constructed_layer():
    # Layer 1 has a huge, obvious effect (see _toy_pooled_and_quadrants) -
    # near every resample should still pick it as the winner.
    pooled, quadrants = _toy_pooled_and_quadrants()
    a_d, hs = bootstrap_bottleneck_layers(pooled, quadrants, n_bootstrap=100, seed=0)
    assert (a_d == 1).mean() > 0.9
    assert (hs == 1).mean() > 0.9


def test_bootstrap_bottleneck_layers_deterministic_given_fixed_seed():
    pooled, quadrants = _toy_pooled_and_quadrants()
    a_d1, hs1 = bootstrap_bottleneck_layers(pooled, quadrants, n_bootstrap=20, seed=3)
    a_d2, hs2 = bootstrap_bottleneck_layers(pooled, quadrants, n_bootstrap=20, seed=3)
    np.testing.assert_array_equal(a_d1, a_d2)
    np.testing.assert_array_equal(hs1, hs2)


def test_summarize_bottleneck_bootstrap_sharp_peak():
    # 95/100 resamples pick layer 2 -> mode should be layer 2 with high mode_frac.
    layer_samples = np.array([2] * 95 + [1] * 3 + [3] * 2)
    summary = summarize_bottleneck_bootstrap(layer_samples, n_layers=5)
    assert summary["mode_layer"] == 2
    assert summary["mode_frac"] == 0.95
    assert summary["ci_low_2.5pct"] <= 2 <= summary["ci_high_97.5pct"]
    assert summary["layer_counts"] == {1: 3, 2: 95, 3: 2}


def test_summarize_bottleneck_bootstrap_wide_spread_gives_low_mode_frac():
    # Argmax noise case: winner is roughly uniform across several layers -
    # mode_frac should be low, signalling the argmax on the full sample
    # isn't a reliable peak.
    rng = np.random.default_rng(0)
    layer_samples = rng.integers(0, 5, size=1000)
    summary = summarize_bottleneck_bootstrap(layer_samples, n_layers=5)
    assert summary["mode_frac"] < 0.3
