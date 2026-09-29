"""Depth attribution is an accounting identity, so it is tested as one."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.stagecmp.attribution import layer_writes, profile, split_available


def test_writes_sum_to_the_endpoint_projection():
    """SUM_l (h_l - h_{l-1}) . d == (h_L - h_0) . d, exactly."""
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(7, 9, 5))
    d = rng.normal(size=5)
    unit = d / np.linalg.norm(d)
    w = layer_writes(acts, d)
    endpoint = (acts[:, -1] - acts[:, 0]) @ unit
    assert w.sum(axis=1) == pytest.approx(endpoint)


def test_a_constructed_write_is_recovered():
    """A layer that adds exactly c*d shows up as c."""
    n_layers, hidden = 6, 4
    d = np.zeros(hidden)
    d[1] = 1.0
    acts = np.zeros((3, n_layers, hidden))
    for l in range(1, n_layers):
        # every block passes the stream through unchanged, except block 3,
        # which adds 2.5*d. Later blocks must CARRY that forward, or the
        # difference to the next layer registers as a cancelling write.
        acts[:, l] = acts[:, l - 1] + (2.5 * d if l == 3 else 0.0)
    w = layer_writes(acts, d)
    assert w[:, 2] == pytest.approx(2.5)       # write INTO layer 3 sits at index 2
    assert np.delete(w, 2, axis=1) == pytest.approx(0.0)


def test_orthogonal_writes_contribute_nothing():
    n_layers, hidden = 5, 4
    d = np.array([1.0, 0, 0, 0])
    orth = np.array([0, 1.0, 0, 0])
    acts = np.zeros((2, n_layers, hidden))
    for l in range(1, n_layers):
        acts[:, l] = acts[:, l - 1] + 3.0 * orth
    assert layer_writes(acts, d) == pytest.approx(0.0)


def test_cancelling_writes_leave_no_endpoint_trace():
    """A write a later layer undoes contributes nothing, which is why the
    cumulative curve is reported beside the per-layer one."""
    d = np.array([1.0, 0, 0])
    acts = np.zeros((2, 4, 3))
    acts[:, 1] = acts[:, 0] + 5.0 * d
    acts[:, 2] = acts[:, 1] - 5.0 * d
    acts[:, 3] = acts[:, 2]
    w = layer_writes(acts, d)
    assert w[:, 0] == pytest.approx(5.0)
    assert w[:, 1] == pytest.approx(-5.0)
    assert w.sum(axis=1) == pytest.approx(0.0)


def test_per_layer_axis_uses_each_layers_own_direction():
    """With a (n_layers, hidden) direction, layer l's write is read with the
    axis belonging to the state after block l."""
    n, n_layers, hidden = 2, 4, 3
    acts = np.zeros((n, n_layers, hidden))
    acts[:, 1] = acts[:, 0] + np.array([1.0, 0, 0])
    acts[:, 2] = acts[:, 1] + np.array([0, 1.0, 0])
    acts[:, 3] = acts[:, 2] + np.array([0, 0, 1.0])
    direction = np.eye(hidden + 1)[:n_layers, :hidden]  # layer l -> axis l
    direction = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    w = layer_writes(acts, direction)
    assert w[:, 0] == pytest.approx(1.0)
    assert w[:, 1] == pytest.approx(1.0)
    assert w[:, 2] == pytest.approx(1.0)


def test_profile_reports_quadrants_and_the_ad_gap(act_dir, directions_dir):
    p = profile("M2", act_dir, directions_dir, "final_token")
    assert set(p["per_quadrant"]) == {"A", "B", "C", "D"}
    assert "ad_gap" in p
    gap = p["ad_gap"]
    a = np.array(p["per_quadrant"]["A"]["write"])
    dd = np.array(p["per_quadrant"]["D"]["write"])
    assert gap["write"] == pytest.approx(a - dd)
    assert gap["cumulative"][-1] == pytest.approx((a - dd).sum())
    assert "EXPLORATORY" in p["status"]


def test_fixed_reference_changes_the_axis(act_dir, directions_dir):
    own = profile("M2", act_dir, directions_dir, "final_token")
    ref = profile("M2", act_dir, directions_dir, "final_token", reference_stage="M1")
    assert own["axis_stage"] == "M2"
    assert ref["axis_stage"] == "M1"


def test_sublayer_split_is_reported_absent_until_it_is_extracted(act_dir):
    assert split_available("M2", act_dir) is False


def test_missing_fixture_raises_rather_than_reaching_for_real_data(tmp_path, directions_dir):
    """A default that silently reads results/activations would make this test
    pass against production data (CONTRIBUTING.md)."""
    with pytest.raises(Exception):
        profile("M2", tmp_path / "nope", directions_dir, "final_token")
