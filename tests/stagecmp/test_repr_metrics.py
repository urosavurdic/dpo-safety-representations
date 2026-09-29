"""Representational metrics: the invariances that make them interpretable."""
from __future__ import annotations

import numpy as np
import pytest

from src.stagecmp.repr_metrics import SATURATION_PAIRS, linear_cka, pair_report


def test_cka_of_a_matrix_with_itself_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 7))
    assert linear_cka(x, x) == pytest.approx(1.0)


def test_cka_is_invariant_to_rotation():
    """The property that makes CKA comparable across checkpoints at all."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-9)


def test_cka_is_invariant_to_isotropic_scaling_and_translation():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 6))
    assert linear_cka(x, 7.5 * x) == pytest.approx(1.0, abs=1e-9)
    assert linear_cka(x, x + 3.0) == pytest.approx(1.0, abs=1e-9)


def test_cka_of_independent_gaussians_is_small():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(400, 5))
    b = rng.normal(size=(400, 5))
    assert linear_cka(a, b) < 0.25


def test_cka_is_symmetric():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(30, 5)), rng.normal(size=(30, 5))
    assert linear_cka(a, b) == pytest.approx(linear_cka(b, a))


def test_pair_report_has_every_layer_and_the_documented_caveats(act_dir, directions_dir):
    r = pair_report("M1", "M2", act_dir, directions_dir, "final_token")
    assert len(r["per_layer"]) == 6                     # toy fixture layer count
    assert {row["layer"] for row in r["per_layer"]} == set(range(6))
    assert "ASYMMETRIC" in r["reading"]
    # the direction-cosine pooling mismatch must travel with the numbers
    assert "pooling-INDEPENDENT" in r["direction_cosine_is_pooling_independent"] or \
           "MEAN-POOLED" in r["direction_cosine_is_pooling_independent"]


def test_identical_stages_give_no_drift_and_unit_contrast_ratio(act_dir, directions_dir):
    """A stage compared with itself is the degenerate case every metric must pass."""
    r = pair_report("M1", "M1", act_dir, directions_dir, "final_token")
    for row in r["per_layer"]:
        assert row["drift_mean"] == pytest.approx(0.0, abs=1e-9)
        assert row["relative_drift_mean"] == pytest.approx(0.0, abs=1e-9)
        assert row["cka"] == pytest.approx(1.0, abs=1e-6)
        assert row["contrast_norm"]["ratio"] == pytest.approx(1.0, abs=1e-9)
        assert row["selectivity"]["A_minus_D"] == pytest.approx(0.0, abs=1e-9)


def test_selectivity_is_the_quadrant_drift_difference(act_dir, directions_dir):
    r = pair_report("M1", "M2", act_dir, directions_dir, "final_token")
    for row in r["per_layer"]:
        dq = row["drift_by_quadrant"]
        assert row["selectivity"]["A_minus_D"] == pytest.approx(dq["A"] - dq["D"])
        assert row["selectivity"]["C_minus_B"] == pytest.approx(dq["C"] - dq["B"])


def test_rho_ad_perp_is_asymmetric(act_dir, directions_dir):
    """It is measured against the PRE stage's subspace, so the pair order matters.
    Reporting it in a symmetric matrix would be wrong."""
    fwd = pair_report("M1", "M2", act_dir, directions_dir, "final_token")
    rev = pair_report("M2", "M1", act_dir, directions_dir, "final_token")
    f = [r["rho_AD_perp"]["rho_AD_perp"] for r in fwd["per_layer"]]
    b = [r["rho_AD_perp"]["rho_AD_perp"] for r in rev["per_layer"]]
    assert any(abs(x - y) > 1e-9 for x, y in zip(f, b) if x is not None and y is not None)


def test_shape_mismatch_between_stages_is_refused(act_dir, directions_dir, tmp_path):
    import numpy as np
    import shutil

    bad = tmp_path / "bad_acts"
    shutil.copytree(act_dir, bad)
    arr = np.load(bad / "M2_final.npy")
    np.save(bad / "M2_final.npy", arr[:, :3, :])       # drop layers
    with pytest.raises(ValueError, match="shape mismatch"):
        pair_report("M1", "M2", bad, directions_dir, "final_token")


def test_saturation_pairs_span_the_expected_range():
    """The check is only meaningful if the pairs bracket large and small change."""
    assert ("M0", "M1") in SATURATION_PAIRS          # instruction tuning: large
    assert ("M2", "M3") in SATURATION_PAIRS          # mediated DPO: small
    assert ("M1", "M2") in SATURATION_PAIRS          # safety SFT
    assert ("M1", "M3_direct") in SATURATION_PAIRS   # direct DPO
