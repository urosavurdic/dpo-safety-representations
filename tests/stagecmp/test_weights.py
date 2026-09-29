"""Weight-space algebra, on toy factors. Never touches a real adapter."""
from __future__ import annotations

import numpy as np
import pytest

from src.stagecmp.weights import (
    SCALING,
    LowRankTerm,
    Update,
    chain_updates,
    column_basis,
    compare_updates,
    cosine,
    difference,
    effective_rank,
    frobenius_norm,
    inner,
    parse_adapter,
    principal_angles_deg,
    random_subspace_angles,
    singular_values,
)

OUT, IN, R = 12, 9, 3


def term(seed, sign=1.0):
    rng = np.random.default_rng(seed)
    return LowRankTerm(sign=sign, b=rng.normal(size=(OUT, R)), a=rng.normal(size=(R, IN)))


def test_scaling_matches_the_project_config():
    assert SCALING == 2.0          # alpha 128 / r 64


def test_norm_from_grams_matches_the_dense_norm():
    u = Update([term(0), term(1)])
    assert frobenius_norm(u) == pytest.approx(np.linalg.norm(u.dense(), "fro"))


def test_inner_product_from_grams_matches_the_dense_one():
    u, v = Update([term(2)]), Update([term(3), term(4)])
    dense = float(np.sum(u.dense() * v.dense()))
    assert inner(u, v) == pytest.approx(dense)


def test_cosine_of_an_update_with_itself_is_one():
    u = Update([term(5), term(6)])
    assert cosine(u, u) == pytest.approx(1.0)


def test_cosine_is_sign_aware():
    u = Update([term(7)])
    minus = Update([LowRankTerm(sign=-1.0, b=u.terms[0].b, a=u.terms[0].a)])
    assert cosine(u, minus) == pytest.approx(-1.0)


def test_chain_differencing_equals_the_direct_difference():
    """dW(X->Y) built from signed terms must equal W_Y - W_X computed densely."""
    adapters = {}
    for i, repo in enumerate([
        "urosavurdic/qwen2.5-1.5b-m1-helpful",
        "urosavurdic/qwen2.5-1.5b-m2-safety",
        "urosavurdic/qwen2.5-1.5b-m3-dpo",
        "urosavurdic/qwen2.5-1.5b-m3-direct-dpo",
    ]):
        rng = np.random.default_rng(100 + i)
        adapters[repo] = {(0, "o_proj"): {"B": rng.normal(size=(OUT, R)),
                                          "A": rng.normal(size=(R, IN))}}

    w_m3 = sum(SCALING * a[(0, "o_proj")]["B"] @ a[(0, "o_proj")]["A"]
               for r, a in adapters.items()
               if r in ("urosavurdic/qwen2.5-1.5b-m1-helpful",
                        "urosavurdic/qwen2.5-1.5b-m2-safety",
                        "urosavurdic/qwen2.5-1.5b-m3-dpo"))
    w_direct = sum(SCALING * a[(0, "o_proj")]["B"] @ a[(0, "o_proj")]["A"]
                   for r, a in adapters.items()
                   if r in ("urosavurdic/qwen2.5-1.5b-m1-helpful",
                            "urosavurdic/qwen2.5-1.5b-m3-direct-dpo"))

    d = difference("M3_direct", "M3", adapters)[(0, "o_proj")]
    assert d.dense() == pytest.approx(w_m3 - w_direct)


def test_difference_with_itself_is_zero():
    adapters = {}
    for i, repo in enumerate(["urosavurdic/qwen2.5-1.5b-m1-helpful",
                              "urosavurdic/qwen2.5-1.5b-m2-safety"]):
        rng = np.random.default_rng(200 + i)
        adapters[repo] = {(0, "o_proj"): {"B": rng.normal(size=(OUT, R)),
                                          "A": rng.normal(size=(R, IN))}}
    d = difference("M2", "M2", adapters)[(0, "o_proj")]
    assert frobenius_norm(d) == pytest.approx(0.0, abs=1e-10)


def test_chain_updates_accumulates_the_whole_chain():
    adapters = {}
    for i, repo in enumerate(["urosavurdic/qwen2.5-1.5b-m1-helpful",
                              "urosavurdic/qwen2.5-1.5b-m2-safety",
                              "urosavurdic/qwen2.5-1.5b-m3-dpo"]):
        rng = np.random.default_rng(300 + i)
        adapters[repo] = {(0, "o_proj"): {"B": rng.normal(size=(OUT, R)),
                                          "A": rng.normal(size=(R, IN))}}
    assert len(chain_updates("M3", adapters)[(0, "o_proj")].terms) == 3
    assert len(chain_updates("M1", adapters)[(0, "o_proj")].terms) == 1


def test_singular_values_and_basis_match_a_dense_svd():
    u = Update([term(8), term(9)])
    dense_sv = np.linalg.svd(u.dense(), compute_uv=False)
    sv = singular_values(u)
    k = min(len(sv), len(dense_sv))
    assert sv[:k] == pytest.approx(dense_sv[:k])

    basis = column_basis(u)
    assert basis.T @ basis == pytest.approx(np.eye(basis.shape[1]), abs=1e-9)
    # every column of dW must lie in the recovered basis
    resid = u.dense() - basis @ (basis.T @ u.dense())
    assert np.abs(resid).max() == pytest.approx(0.0, abs=1e-8)


def test_principal_angles_of_a_subspace_with_itself_are_zero():
    u = Update([term(10)])
    basis = column_basis(u)
    ang = principal_angles_deg(basis, basis)
    # arccos is ill-conditioned near 1: arccos(1 - eps) ~ sqrt(2 eps), so a
    # singular value off by 1e-16 becomes ~1e-8 rad ~ 1e-6 deg. That is the
    # floating-point floor for this quantity, not a defect.
    assert ang["max_deg"] == pytest.approx(0.0, abs=1e-4)


def test_known_orthogonal_subspaces_give_ninety_degrees():
    a = np.eye(6)[:, :2]
    b = np.eye(6)[:, 2:4]
    assert principal_angles_deg(a, b)["mean_deg"] == pytest.approx(90.0)


def test_effective_rank_is_one_for_a_rank_one_update():
    b = np.ones((OUT, 1))
    a = np.ones((1, IN))
    u = Update([LowRankTerm(sign=1.0, b=b, a=a)])
    assert effective_rank(singular_values(u)) == pytest.approx(1.0)


def test_random_control_shows_unrelated_subspaces_are_nearly_orthogonal():
    """The reason a large principal angle is not by itself a finding."""
    ctl = random_subspace_angles(hidden=256, rank=16, n_samples=10, seed=1)
    assert ctl["mean_deg"] > 60.0
    assert ctl["p2.5"] <= ctl["mean_deg"] <= ctl["p97.5"]


def test_compare_updates_always_reports_the_control():
    u, v = Update([term(11)]), Update([term(12)])
    r = compare_updates(u, v)
    assert "random_subspace_control" in r
    assert "not like-for-like" in r["reading"]


def test_parse_adapter_rejects_a_missing_factor():
    good = {"base_model.model.layers.0.self_attn.o_proj.lora_A.weight": np.zeros((R, IN)),
            "base_model.model.layers.0.self_attn.o_proj.lora_B.weight": np.zeros((OUT, R))}
    assert set(parse_adapter(good)) == {(0, "o_proj")}
    with pytest.raises(KeyError, match="missing an A or B"):
        parse_adapter({k: v for k, v in good.items() if "lora_A" in k})
