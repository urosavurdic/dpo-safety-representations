"""Delta assembly and control vectors, on toy activations.

The adoption tests matter as much as the arithmetic: the whole point of the
adopt path is that it REFUSES stale or mismatched arrays rather than quietly
producing a delta from the wrong row set.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.crossbranch import delta as D
from src.v2_io import binding, identity_snapshot, write_json_lf

N_LAYERS, HIDDEN = 5, 4
LAYER = 2
BENCH_SHA, SPLIT_SHA = "bench" * 8, "split" * 8


def make_rows():
    spec = [("A", "direction_estimation"), ("A", "held_out_behavioral"),
            ("B", None), ("B", None), ("C", None),
            ("D", "direction_estimation"), ("D", "held_out_behavioral")]
    return [
        {"record_id": f"r{i}", "prompt": f"p{i}", "quadrant": q, "split": s,
         "source": "toy"}
        for i, (q, s) in enumerate(spec)
    ]


def make_ctx(tmp_path, rows):
    act = tmp_path / "activations"
    act.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(
        rows=rows,
        benchmark_sha=BENCH_SHA,
        split_sha=SPLIT_SHA,
        paths=SimpleNamespace(activations=act),
        snapshot=identity_snapshot(rows),
    )
    ctx.bind = lambda: binding("bench.jsonl", BENCH_SHA, "split.json", SPLIT_SHA)
    return ctx


def write_stage(ctx, stage, arr, *, bound=True, metadata=None, rows=None):
    act = ctx.paths.activations
    np.save(act / f"{stage}_final.npy", arr)
    np.save(act / f"{stage}_pooled.npy", arr)
    md = metadata if metadata is not None else identity_snapshot(rows or ctx.rows)
    write_json_lf(act / f"{stage}_metadata.json", md)
    if bound:
        write_json_lf(
            act / f"{stage}_metadata_binding.json",
            binding("bench.jsonl", BENCH_SHA, "split.json", SPLIT_SHA),
        )


def toy(n, seed):
    return np.random.default_rng(seed).standard_normal((n, N_LAYERS, HIDDEN))


# ---------------------------------------------------------------------------


def test_assemble_delta_is_exactly_post_minus_pre(tmp_path):
    rows = make_rows()
    ctx = make_ctx(tmp_path, rows)
    pre, post = toy(len(rows), 1), toy(len(rows), 2)
    write_stage(ctx, "PRE", pre)
    write_stage(ctx, "POST", post)

    out = D.assemble_delta(ctx, "PRE", "POST", LAYER)
    np.testing.assert_allclose(
        out["delta"], post[:, LAYER, :] - pre[:, LAYER, :], rtol=0, atol=0
    )
    assert list(out["record_ids"]) == [r["record_id"] for r in rows]
    np.testing.assert_allclose(out["norms"], np.linalg.norm(out["delta"], axis=1))


def test_assemble_preserves_benchmark_row_order(tmp_path):
    rows = make_rows()
    ctx = make_ctx(tmp_path, rows)
    write_stage(ctx, "PRE", toy(len(rows), 1))
    write_stage(ctx, "POST", toy(len(rows), 2))
    out = D.assemble_delta(ctx, "PRE", "POST", LAYER)
    assert list(out["quadrants"]) == [r["quadrant"] for r in rows]


def test_layer_out_of_range_raises(tmp_path):
    rows = make_rows()
    ctx = make_ctx(tmp_path, rows)
    write_stage(ctx, "PRE", toy(len(rows), 1))
    write_stage(ctx, "POST", toy(len(rows), 2))
    with pytest.raises(IndexError):
        D.assemble_delta(ctx, "PRE", "POST", N_LAYERS + 3)


# ---- adoption: must refuse rather than degrade ----------------------------


def test_adopt_accepts_a_bound_stage(tmp_path):
    rows = make_rows()
    ctx = make_ctx(tmp_path, rows)
    write_stage(ctx, "S", toy(len(rows), 3))
    sidecar = D.adopt_activation(ctx, "S", tmp_path / "cb")
    data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    assert data["adoption"] == "v2_binding"
    assert data["benchmark_sha256"] == BENCH_SHA


def test_adopt_accepts_legacy_metadata_that_matches_and_writes_its_own_sidecar(tmp_path):
    rows = make_rows()
    ctx = make_ctx(tmp_path, rows)
    write_stage(ctx, "S", toy(len(rows), 3), bound=False)
    out = tmp_path / "cb"
    sidecar = D.adopt_activation(ctx, "S", out)
    data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    assert data["adoption"] == "legacy_metadata_verified"
    # never writes into the activations directory
    assert not list(ctx.paths.activations.glob("*crossbranch*"))
    assert Path(sidecar).parent == out


def test_adopt_refuses_a_stale_row_set(tmp_path):
    """The 370-vs-654 case: right filenames, wrong rows."""
    rows = make_rows()
    ctx = make_ctx(tmp_path, rows)
    stale = rows[:3]
    write_stage(ctx, "S", toy(3, 4), bound=False,
                metadata=identity_snapshot(stale))
    with pytest.raises(RuntimeError, match="does not match the frozen benchmark"):
        D.adopt_activation(ctx, "S", tmp_path / "cb")


def test_adopt_refuses_when_arrays_are_absent(tmp_path):
    ctx = make_ctx(tmp_path, make_rows())
    with pytest.raises(FileNotFoundError, match="never extracts"):
        D.adopt_activation(ctx, "MISSING", tmp_path / "cb")


# ---- controls ------------------------------------------------------------


def test_shuffle_within_quadrant_never_crosses_a_quadrant():
    quads = np.array(list("AABBCDD"), dtype=object)
    perm = D.shuffle_within_quadrant(quads, np.random.default_rng(0))
    assert sorted(perm) == list(range(len(quads)))
    assert all(quads[i] == quads[perm[i]] for i in range(len(quads)))


def test_shuffle_within_quadrant_is_deterministic_and_seed_sensitive():
    quads = np.array(list("AAAABBBB"), dtype=object)
    a = D.shuffle_within_quadrant(quads, np.random.default_rng(0))
    b = D.shuffle_within_quadrant(quads, np.random.default_rng(0))
    c = D.shuffle_within_quadrant(quads, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_normmatched_random_matches_each_row_norm():
    d = np.random.default_rng(5).standard_normal((10, 6))
    r = D.normmatched_random(d, np.random.default_rng(0))
    np.testing.assert_allclose(
        np.linalg.norm(r, axis=1), np.linalg.norm(d, axis=1), rtol=1e-9
    )


def test_normmatched_random_is_a_different_direction_than_the_delta():
    d = np.random.default_rng(5).standard_normal((10, 6))
    r = D.normmatched_random(d, np.random.default_rng(0))
    assert not np.allclose(d, r)
    cos = (d * r).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(r, axis=1))
    # Matched in magnitude, uninformative in direction: that is the whole
    # point of this control.
    assert np.abs(cos).mean() < 0.8


def test_normmatched_random_leaves_zero_rows_at_zero():
    d = np.zeros((3, 4))
    d[1] = [1.0, 0, 0, 0]
    r = D.normmatched_random(d, np.random.default_rng(0))
    np.testing.assert_allclose(r[0], 0)
    np.testing.assert_allclose(r[2], 0)
    assert np.linalg.norm(r[1]) == pytest.approx(1.0)


def test_apply_permutation_moves_the_right_rows():
    d = np.arange(12, dtype=float).reshape(4, 3)
    perm = np.array([3, 2, 1, 0])
    np.testing.assert_array_equal(D.apply_permutation(d, perm), d[perm])


# ---- dose diagnostic -----------------------------------------------------


def test_dose_ratio_report_computes_median_p95_max_per_quadrant():
    delta = np.array([[3.0, 4.0], [6.0, 8.0], [1.0, 0.0]])   # norms 5, 10, 1
    pre = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]])     # norms 1, 2, 1
    quads = np.array(["A", "A", "B"], dtype=object)
    rep = D.dose_ratio_report(delta, pre, quads)
    assert rep["A"]["median"] == pytest.approx(5.0)   # ratios 5 and 5
    assert rep["B"]["max"] == pytest.approx(1.0)
    assert rep["_note"].startswith("descriptive only")


# ---- artifact I/O --------------------------------------------------------


def test_npz_roundtrip_and_delta_map(tmp_path):
    vecs = np.random.default_rng(0).standard_normal((4, 3))
    ids = ["a", "b", "c", "d"]
    path = D.save_delta_npz(tmp_path / "x.npz", vecs, ids)
    m = D.load_delta_map(path)
    assert list(m) == ids
    np.testing.assert_allclose(m["c"], vecs[2].astype(np.float32), rtol=1e-6)


def test_load_delta_map_rejects_duplicate_ids(tmp_path):
    vecs = np.zeros((2, 3))
    path = D.save_delta_npz(tmp_path / "d.npz", vecs, ["same", "same"])
    with pytest.raises(RuntimeError, match="duplicate record_ids"):
        D.load_delta_map(path)


def test_seed_spawn_order_is_documented_and_stable():
    assert D.CROSSBRANCH_SEED == 20260904
    assert D.SPAWN_ORDER == ("shuffle_within_quadrant", "normmatched_random")
