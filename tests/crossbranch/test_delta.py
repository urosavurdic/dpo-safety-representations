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

from src.crossbranch import delta as D
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
    """SPAWN_ORDER is append-only: the P0 streams must stay first and in
    order, so already-generated Stage-1 artifacts remain reproducible. New
    stages append (see test_spawn_order_is_append_only_so_p0_streams_never_change)."""
    assert D.CROSSBRANCH_SEED == 20260904
    assert D.SPAWN_ORDER[:2] == ("shuffle_within_quadrant", "normmatched_random")
    assert D.SPAWN_ORDER == (
        "shuffle_within_quadrant",
        "normmatched_random",
        "normmatched_random_source",
        "shuffle_global",
    )


# ---- Stage-2 vectors -------------------------------------------------------


def test_spawn_order_is_append_only_so_p0_streams_never_change():
    """Reproducibility guard. Stage-1 artifacts are already generated and a
    result validated against them; appending Stage-2 names to SPAWN_ORDER
    must leave the first two streams byte-identical."""
    assert D.SPAWN_ORDER[:2] == ("shuffle_within_quadrant", "normmatched_random")
    two = np.random.default_rng(D.CROSSBRANCH_SEED).spawn(2)
    full = np.random.default_rng(D.CROSSBRANCH_SEED).spawn(len(D.SPAWN_ORDER))
    for i in range(2):
        a = np.random.default_rng(D.CROSSBRANCH_SEED).spawn(2)[i]
        b = np.random.default_rng(D.CROSSBRANCH_SEED).spawn(len(D.SPAWN_ORDER))[i]
        np.testing.assert_array_equal(a.standard_normal(5), b.standard_normal(5))
    assert len(two) == 2 and len(full) == len(D.SPAWN_ORDER)


def test_dosematch_gives_each_row_the_target_norm():
    src = np.array([[3.0, 4.0], [1.0, 0.0]])      # norms 5, 1
    tgt = np.array([[0.0, 2.0], [6.0, 8.0]])      # norms 2, 10
    out = D.dosematch_to(src, tgt)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [2.0, 10.0], rtol=1e-9)


def test_dosematch_preserves_direction_only_changes_length():
    src = np.array([[3.0, 4.0]])
    tgt = np.array([[10.0, 0.0]])
    out = D.dosematch_to(src, tgt)
    cos = float((src[0] @ out[0]) / (np.linalg.norm(src[0]) * np.linalg.norm(out[0])))
    assert cos == pytest.approx(1.0)


def test_dosematch_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        D.dosematch_to(np.zeros((2, 3)), np.zeros((3, 3)))


def test_direction_dose_scalar_uses_only_the_calibration_split():
    delta = np.array([
        [10.0, 0.0],   # A, direction_estimation  -> counted (norm 10)
        [100.0, 0.0],  # A, held_out_behavioral   -> excluded
        [20.0, 0.0],   # D, direction_estimation  -> counted (norm 20)
        [500.0, 0.0],  # B, no split              -> excluded
    ])
    quads = np.array(["A", "A", "D", "B"], dtype=object)
    splits = np.array(
        ["direction_estimation", "held_out_behavioral", "direction_estimation", None],
        dtype=object,
    )
    assert D.direction_dose_scalar(delta, quads, splits) == pytest.approx(15.0)


def test_direction_dose_scalar_raises_without_calibration_rows():
    delta = np.array([[1.0, 0.0]])
    with pytest.raises(RuntimeError, match="no direction_estimation rows"):
        D.direction_dose_scalar(
            delta, np.array(["B"], dtype=object), np.array([None], dtype=object)
        )


def test_load_direction_vector_rejects_a_non_unit_direction(tmp_path):
    bad = np.zeros((29, 4))
    bad[24] = [5.0, 0.0, 0.0, 0.0]     # norm 5, not ~1
    np.save(tmp_path / "S_direction_654.npy", bad)
    with pytest.raises(RuntimeError, match="direction norm"):
        D.load_direction_vector("S", 24, tmp_path)


def test_load_direction_vector_prefers_v2_over_legacy(tmp_path):
    v2, legacy = np.zeros((29, 4)), np.zeros((29, 4))
    v2[24] = [1.0, 0.0, 0.0, 0.0]
    legacy[24] = [0.0, 1.0, 0.0, 0.0]
    np.save(tmp_path / "S_direction_654.npy", v2)
    np.save(tmp_path / "S_direction.npy", legacy)
    np.testing.assert_allclose(D.load_direction_vector("S", 24, tmp_path), [1, 0, 0, 0])


def test_load_direction_vector_falls_back_to_legacy_when_v2_absent(tmp_path):
    legacy = np.zeros((29, 4))
    legacy[24] = [0.0, 1.0, 0.0, 0.0]
    np.save(tmp_path / "S_direction.npy", legacy)
    np.testing.assert_allclose(D.load_direction_vector("S", 24, tmp_path), [0, 1, 0, 0])


def test_load_direction_vector_raises_when_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="no direction for"):
        D.load_direction_vector("MISSING", 24, tmp_path)


def test_constant_delta_array_is_the_same_vector_on_every_row():
    d = np.array([0.6, 0.8])          # unit norm
    out = D.constant_delta_array(d, 5.0, 3)
    assert out.shape == (3, 2)
    for row in out:
        np.testing.assert_allclose(row, [3.0, 4.0])
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [5.0, 5.0, 5.0])


# ---- direction decomposition (approved 2026-09-10) ----------------------

def test_decompose_dosematched_components_are_orthogonal_and_full_magnitude():
    import numpy as np
    from src.crossbranch.delta import decompose_dosematched

    rng = np.random.default_rng(0)
    delta = rng.normal(size=(50, 16))
    d = rng.normal(size=16)
    par, perp = decompose_dosematched(delta, d)

    du = d / np.linalg.norm(d)
    # parallel really is along d, perp really is orthogonal to d
    assert np.allclose(perp @ du, 0, atol=1e-9)
    par_cross = par - (par @ du)[:, None] * du[None, :]
    assert np.allclose(par_cross, 0, atol=1e-9)
    # both rescaled to the row's original ||delta||
    full = np.linalg.norm(delta, axis=1)
    assert np.allclose(np.linalg.norm(par, axis=1), full, rtol=1e-6)
    assert np.allclose(np.linalg.norm(perp, axis=1), full, rtol=1e-6)


def test_decompose_dosematched_zero_and_degenerate_rows_stay_zero():
    import numpy as np
    from src.crossbranch.delta import decompose_dosematched

    d = np.array([1.0, 0.0, 0.0])
    delta = np.array([
        [0.0, 0.0, 0.0],   # zero row -> both components zero
        [3.0, 0.0, 0.0],   # exactly along d -> perp is zero, parallel = full
        [0.0, 4.0, 0.0],   # exactly orthogonal -> parallel is zero, perp = full
    ])
    par, perp = decompose_dosematched(delta, d)
    assert np.allclose(par[0], 0) and np.allclose(perp[0], 0)
    assert np.allclose(perp[1], 0)
    assert np.isclose(np.linalg.norm(par[1]), 3.0)
    assert np.allclose(par[2], 0)
    assert np.isclose(np.linalg.norm(perp[2]), 4.0)


def test_decomposition_conditions_are_registered_and_optional():
    from src.crossbranch.branches import get, DEFERRED_CONDITIONS

    for name in ("xfer_delta_source_parallel", "xfer_delta_source_perp"):
        c = get(name)
        assert c.kind == "vector" and c.stage_gate == "optional"
        assert c.checkpoint == "target_pre"
    assert DEFERRED_CONDITIONS == ()


# --------------------------------------------------------------------------- #
# norm-matched global shuffle (added 2026-09-12)
# --------------------------------------------------------------------------- #
def test_rescale_to_row_norms_matches_reference_norms_exactly():
    """Each row keeps its own direction but takes the reference row's norm."""
    from src.crossbranch.delta import rescale_to_row_norms

    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
    reference = np.array([[10.0, 0.0], [0.0, 7.0], [1.0, 1.0]])
    out = rescale_to_row_norms(vectors, reference)

    want = np.linalg.norm(reference, axis=1)
    got = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(got, want, atol=1e-9)

    # directions are untouched (unit vectors unchanged)
    for before, after in zip(vectors, out):
        np.testing.assert_allclose(
            before / np.linalg.norm(before), after / np.linalg.norm(after), atol=1e-9
        )


def test_rescale_to_row_norms_keeps_zero_reference_rows_at_zero():
    """The identity arm injects nothing on a zero-delta row, so no control may."""
    from src.crossbranch.delta import rescale_to_row_norms

    out = rescale_to_row_norms(
        np.array([[5.0, 5.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [3.0, 4.0]]),
    )
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(out[1]), 5.0, atol=1e-9)


def test_rescale_to_row_norms_rejects_shape_mismatch():
    from src.crossbranch.delta import rescale_to_row_norms

    with pytest.raises(ValueError, match="shape mismatch"):
        rescale_to_row_norms(np.zeros((3, 2)), np.zeros((4, 2)))


def test_normmatched_global_shuffle_is_a_permutation_at_identity_dose():
    """The load-bearing property: rows carry a DIFFERENT prompt's delta
    direction, at the SAME magnitude the identity arm would have injected."""
    from src.crossbranch.delta import (
        apply_permutation, rescale_to_row_norms, shuffle_global,
    )

    rng = np.random.default_rng(0)
    src = rng.standard_normal((12, 5)) * np.array([[1.0], [9.0]] * 6)
    perm = shuffle_global(len(src), np.random.default_rng(1))
    shuffled = apply_permutation(src, perm)
    matched = rescale_to_row_norms(shuffled, src)

    # dose is identity's, row by row
    np.testing.assert_allclose(
        np.linalg.norm(matched, axis=1), np.linalg.norm(src, axis=1), atol=1e-9
    )
    # but the direction came from the permuted row, not the row itself
    moved = [i for i in range(len(src)) if perm[i] != i]
    assert moved, "permutation left every row in place; seed choice is degenerate"
    for i in moved:
        cos_self = float(
            src[i] @ matched[i]
            / (np.linalg.norm(src[i]) * np.linalg.norm(matched[i]))
        )
        cos_donor = float(
            src[perm[i]] @ matched[i]
            / (np.linalg.norm(src[perm[i]]) * np.linalg.norm(matched[i]))
        )
        assert cos_donor == pytest.approx(1.0, abs=1e-9)
        assert cos_self < 0.999
