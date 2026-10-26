"""Known-answer checks for the contrast-geometry and update-size estimators."""
import numpy as np
import pytest

from src.stagecmp import run_geometry as rr
from src.stagecmp.behaviour_levels import wilson


def test_decompose_identity_and_known_rotation():
    c_pre = np.array([3.0, 0.0, 0.0])
    # post = 2 x old direction + 2 x orthogonal: R = sqrt(8)/3
    c_post = np.array([2.0, 2.0, 0.0])
    d = rr._decompose(c_pre, c_post)
    assert d["along"] == pytest.approx(2 / 3)
    assert d["orth"] == pytest.approx(2 / 3)
    assert d["along"] ** 2 + d["orth"] ** 2 == pytest.approx(d["R"] ** 2)
    assert d["old_direction_share"] == pytest.approx(0.5)


def test_decompose_pure_amplification_has_no_orthogonal_part():
    c_pre = np.array([1.0, -2.0, 0.5])
    d = rr._decompose(c_pre, 1.7 * c_pre)
    assert d["R"] == pytest.approx(1.7)
    assert d["along"] == pytest.approx(1.7)
    assert d["orth"] == pytest.approx(0.0, abs=1e-6)


def test_decompose_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    pre, post = rng.normal(size=(5, 8)), rng.normal(size=(5, 8))
    batch = rr._decompose(pre, post)
    for i in range(5):
        one = rr._decompose(pre[i], post[i])
        for k in one:
            assert batch[k][i] == pytest.approx(one[k])


def test_count_matrix_rows_are_resamples():
    counts = rr._count_matrix(np.random.default_rng(1), 7, 50)
    assert counts.shape == (50, 7)
    assert np.all(counts.sum(axis=1) == 7)
    assert np.all(counts >= 0)


def test_count_matrix_mean_equals_resampled_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(9, 3))
    draws = rng.integers(0, 9, size=(4, 9))
    counts = np.zeros((4, 9))
    np.add.at(counts, (np.arange(4)[:, None], draws), 1.0)
    for k in range(4):
        assert np.allclose(counts[k] @ x / 9, x[draws[k]].mean(0))


def test_wilson_interval_contains_rate_and_handles_edges():
    w = wilson(30, 100)
    assert w["ci_low"] < 0.3 < w["ci_high"]
    zero = wilson(0, 50)
    assert zero["rate"] == 0 and zero["ci_low"] == pytest.approx(0, abs=1e-12) and zero["ci_high"] > 0
    assert wilson(0, 0)["rate"] is None


def test_permutation_null_detects_class_specific_change(tmp_path, monkeypatch):
    """A drift that moves only the harmful rows must beat the label null; a
    drift common to all rows must not move the contrast at all."""
    rng = np.random.default_rng(3)
    n_a, n_d, h = 40, 40, 16
    pre = rng.normal(size=(n_a + n_d, 1, h))
    pre[:n_a, 0, 0] += 3.0
    shift = np.zeros(h)
    shift[1] = 5.0
    post_specific = pre.copy()
    post_specific[:n_a, 0] += shift
    post_common = pre + shift
    for name, arr in {"P": pre, "S": post_specific, "C": post_common}.items():
        np.save(tmp_path / f"{name}_final.npy", arr)
        np.save(tmp_path / f"{name}_pooled.npy", arr)
    monkeypatch.setattr(rr, "ACT_DIR", tmp_path)
    monkeypatch.setattr(rr, "LAYERS", (0,))
    monkeypatch.setattr(rr, "ARMS", {"specific": ("P", "S"), "common": ("P", "C")})
    groups = {"A_est": np.arange(n_a), "D_est": np.arange(n_a, n_a + n_d)}
    out = rr.update_size(groups, n_perm=500, seed=0)["final_token"]["0"]
    assert out["specific"]["contrast_change"] > out["specific"]["null_q99"]
    assert out["specific"]["p_perm"] < 0.01
    assert out["common"]["contrast_change"] == pytest.approx(0.0, abs=1e-9)
