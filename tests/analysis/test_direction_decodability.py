"""Toy tests for src/analysis/direction_decodability.py (WP-Decode, CF3 §4.4).

Synthetic activations with a planted category signal in a subspace ORTHOGONAL
to the A-D direction, so residualizing out A-D leaves the category decodable.
"""
import numpy as np
import pytest

from src.analysis import direction_decodability as dd


CATS = list(dd.CF3_CATEGORIES)


def _metadata(n_per_cat=10):
    meta = []
    for ci, cat in enumerate(CATS):
        for k in range(n_per_cat):
            meta.append({
                "quadrant": "A", "project_category": cat,
                "source_id": f"A_{cat}_{k}", "prompt": f"a {cat} {k}",
                "record_id": f"A_{cat}_{k}",
            })
        for k in range(n_per_cat):
            meta.append({
                "quadrant": "C", "project_category": cat,
                "pair_id": f"C_{cat}_{k}", "prompt": f"c {cat} {k}",
                "record_id": f"C_{cat}_{k}",
            })
    return meta


def _activations(meta, n_layers=29, hidden=16, seed=0):
    rng = np.random.default_rng(seed)
    n = len(meta)
    pooled = rng.standard_normal((n, n_layers, hidden)) * 0.1
    # A-D direction lives on axis 0; category signal on axes 1..4 (orthogonal)
    for i, row in enumerate(meta):
        pooled[i, dd.CF3_LAYER, 0] += 5.0 if row["quadrant"] == "A" else -5.0
        cat_axis = 1 + CATS.index(row["project_category"])
        pooled[i, dd.CF3_LAYER, cat_axis] += 3.0
    return pooled


def _directions(n_layers=29, hidden=16):
    d = np.zeros((n_layers, hidden))
    d[:, 0] = 1.0  # A-D direction = axis 0 at every layer
    return {"M2": d.copy(), "M3": d.copy()}


def test_select_ac_rows_filters_to_valid_categories():
    meta = _metadata(3) + [{"quadrant": "A", "project_category": "homonyms"}]
    idx, labels, groups, quad = dd.select_ac_rows(meta)
    assert set(labels.tolist()) == set(CATS)
    assert len(idx) == 24  # 4 cats x 3 x 2 quadrants; the homonyms row dropped


def test_ac_compat_raises_on_mismatch():
    meta = _metadata(3)
    # drop one category from C only
    meta = [m for m in meta if not (m["quadrant"] == "C" and m["project_category"] == CATS[0])]
    _idx, labels, _g, quad = dd.select_ac_rows(meta)
    with pytest.raises(dd.ACCompatError):
        dd.check_ac_category_compat(labels, quad)


def test_residualize_removes_the_direction_component():
    rng = np.random.default_rng(1)
    d = rng.standard_normal(8); d /= np.linalg.norm(d)
    h = rng.standard_normal((5, 8))
    out = dd.residualize(h, d)
    np.testing.assert_allclose(out @ d, 0.0, atol=1e-9)


def test_cf3_runs_and_reports_the_frozen_shape():
    meta = _metadata(12)
    pooled = _activations(meta)
    rep = dd.cf3(pooled, pooled + 0.0, meta, _directions(), n_boot=200)
    assert rep["endpoint"] == "CF3"
    assert rep["status"].startswith("confirmatory_secondary") or "downgraded" in rep["status"]
    assert set(rep["categories"]) == set(CATS)
    # category signal is orthogonal to the residualized-out A-D axis -> decodable
    assert rep["M2"]["macro_f1"] > 0.8
    assert rep["bootstrap_group_diff"]["unit"].startswith("independent groups")
    # identical activations for M2 and M3 -> difference ~ 0
    assert abs(rep["cf3_macroF1_M3_minus_M2"]) < 1e-6


def test_cf3_downgrades_when_ac_incompatible():
    meta = _metadata(6)
    meta = [m for m in meta if not (m["quadrant"] == "C" and m["project_category"] == CATS[1])]
    pooled = _activations(meta)
    rep = dd.cf3(pooled, pooled, meta, _directions(), n_boot=50)
    assert rep["status"] == "downgraded_within_source_exploratory"
    assert "downgrade_reason" in rep


def test_group_key_schema():
    assert dd.group_key({"quadrant": "C", "pair_id": "P1"}) == "P1"
    assert dd.group_key({"quadrant": "A", "source_id": "S1"}) == "S1"
    k = dd.group_key({"quadrant": "A", "prompt": "some prompt"})
    assert k.startswith("fam_")
