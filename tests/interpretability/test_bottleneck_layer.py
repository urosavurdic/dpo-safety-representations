import json

import numpy as np

from src.interpretability.bottleneck_layer import (
    cohens_d,
    find_bottleneck_layer,
    per_layer_separability,
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
