import numpy as np

from src.eval_refusal_direction import (
    cosine_similarity_per_layer,
    diff_in_means_direction,
    project_onto_direction,
)


def _toy_data():
    # 4 prompts, 2 layers, 3-dim hidden. A and D are trivially separable
    # along dim 0 so the expected direction is exactly known.
    pooled = np.array([
        [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],   # A
        [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],   # A
        [[-5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]], # D
        [[-5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]], # D
    ])
    quadrants = np.array(["A", "A", "D", "D"])
    return pooled, quadrants


def test_diff_in_means_direction_is_unit_norm_and_points_at_A():
    pooled, quadrants = _toy_data()
    direction = diff_in_means_direction(pooled, quadrants)
    assert direction.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(direction, axis=-1), 1.0, atol=1e-6)
    # A is at +x, D at -x -> direction should point along +x
    np.testing.assert_allclose(direction[:, 0], 1.0, atol=1e-6)


def test_cosine_similarity_identical_directions_is_one():
    pooled, quadrants = _toy_data()
    d = diff_in_means_direction(pooled, quadrants)
    sim = cosine_similarity_per_layer(d, d)
    np.testing.assert_allclose(sim, 1.0, atol=1e-6)


def test_project_onto_direction_shape_and_sign():
    pooled, quadrants = _toy_data()
    direction = diff_in_means_direction(pooled, quadrants)
    proj = project_onto_direction(pooled, direction)
    assert proj.shape == (4, 2)
    # A-quadrant rows should project positive, D-quadrant negative
    assert (proj[:2] > 0).all()
    assert (proj[2:] < 0).all()