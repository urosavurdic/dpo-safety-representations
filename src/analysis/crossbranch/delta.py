"""Delta assembly and control-vector construction (CPU, torch-free).

Builds, for every benchmark row, the per-prompt activation delta

    Delta(x) = h_post(x, L, t_final) - h_pre(x, L, t_final)

and the control vectors the Stage-1 gate consumes. Reads activations
read-only; never writes into results/activations/.

Scope note: this module builds only what Stage 1 executes, plus the
within-quadrant shuffle (built and tested now, first consumed in Stage 2).
The dose-matched, constant-direction and Procrustes vectors are Stage 2 and
are deliberately absent -- building untested artifacts ahead of the
conditions that consume them is how they drift out of sync.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.v2_pipeline import (
    activation_paths,
    activations_bound,
    build_context,
    load_bound_activation,
)
from src.analysis.crossbranch.branches import (
    BRANCHES,
    INJECT_LAYER,
    resolve,
    stages_needed,
)
from src.v2_io import identity_snapshot, load_json, sha256_file, write_json_lf

# One documented seed; children are spawned in a fixed order so each control
# has an independent stream that is still reproducible from this one number.
CROSSBRANCH_SEED = 20260904
# APPEND-ONLY. numpy's SeedSequence spawns child i deterministically from the
# parent seed and index i, so adding names to the END leaves every earlier
# stream byte-identical -- verified, and pinned by a regression test. Never
# reorder or insert: that would silently change already-generated Stage-1
# artifacts and break reproducibility of a result we have already validated.
SPAWN_ORDER = (
    "shuffle_within_quadrant",      # P0
    "normmatched_random",           # P0  (matched to ||delta_target||)
    "normmatched_random_source",    # Stage 2 (matched to ||delta_source||)
    "shuffle_global",               # Stage 2 (optional condition)
)

CROSSBRANCH_DIR = Path("results/crossbranch")
DELTAS_DIR = CROSSBRANCH_DIR / "deltas"


# ---------------------------------------------------------------------------
# Activation access
# ---------------------------------------------------------------------------


def adopt_activation(ctx, stage: str, out_dir: Path = DELTAS_DIR) -> Path:
    """Bind a stage's already-extracted activations, read-only.

    Preferred path: the stage carries a v2 ``*_metadata_binding.json`` and
    ``activations_bound`` passes, in which case that sidecar is authoritative
    and we record a pointer to it.

    Fallback: a stage extracted by the legacy script has ``_metadata.json``
    but no binding sidecar. We verify its metadata equals
    ``identity_snapshot(ctx.rows)`` and its row count matches, then write OUR
    OWN provenance sidecar under results/crossbranch/ -- never into
    results/activations/, which this package treats as read-only.

    Raises when neither holds. This function never extracts activations and
    never falls back to a stale array; a mismatch is a blocker, by design.
    """
    final_path, pooled_path, metadata_path, binding_path = activation_paths(
        ctx, stage
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out_dir / f"adopted_{stage}_activation_binding.json"

    if activations_bound(ctx, stage):
        write_json_lf(
            sidecar,
            {
                **ctx.bind(),
                "stage": stage,
                "adoption": "v2_binding",
                "v2_binding_path": binding_path.as_posix(),
                "final_path": final_path.as_posix(),
                "n_rows": len(ctx.rows),
            },
        )
        return sidecar

    if not final_path.exists():
        raise FileNotFoundError(
            f"{stage}: {final_path} is missing. This extension never extracts "
            "activations. Run the frozen v2 extract stage "
            "(python -m src.analysis.v2_pipeline run) first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(f"{stage}: {metadata_path} is missing.")

    metadata = load_json(metadata_path)
    expected = ctx.snapshot
    if metadata != expected:
        raise RuntimeError(
            f"{stage}: {metadata_path} does not match the frozen benchmark "
            f"({len(metadata)} rows saved vs {len(expected)} expected). "
            "Refusing to adopt. Legacy 370-era activations are never used; "
            "run the frozen v2 extract stage for this checkpoint."
        )

    n_saved = int(np.load(final_path, mmap_mode="r").shape[0])
    if n_saved != len(expected):
        raise RuntimeError(
            f"{stage}: {final_path} has {n_saved} rows, expected "
            f"{len(expected)}."
        )

    write_json_lf(
        sidecar,
        {
            **ctx.bind(),
            "stage": stage,
            "adoption": "legacy_metadata_verified",
            "final_path": final_path.as_posix(),
            "final_sha256": sha256_file(final_path),
            "metadata_path": metadata_path.as_posix(),
            "n_rows": n_saved,
        },
    )
    return sidecar


def stage_layer_matrix(ctx, stage: str, layer: int = INJECT_LAYER) -> np.ndarray:
    """(n_rows, hidden) float64 slice of a stage's `_final` activations."""
    if activations_bound(ctx, stage):
        final, _pooled, _meta = load_bound_activation(ctx, stage)
    else:
        adopt_activation(ctx, stage)  # verifies or raises
        final_path, *_ = activation_paths(ctx, stage)
        final = np.load(final_path, mmap_mode="r")

    if final.ndim != 3:
        raise RuntimeError(
            f"{stage}: expected (rows, layers, hidden); got {final.shape}"
        )
    if not 0 <= layer < final.shape[1]:
        raise IndexError(
            f"layer {layer} out of range for {stage} with "
            f"{final.shape[1]} hidden_states entries"
        )
    return np.asarray(final[:, layer, :], dtype=np.float64)


# ---------------------------------------------------------------------------
# Delta assembly
# ---------------------------------------------------------------------------


def assemble_delta(
    ctx, pre_stage: str, post_stage: str, layer: int = INJECT_LAYER
) -> dict:
    """Delta = post - pre at `layer`, in benchmark row order."""
    pre = stage_layer_matrix(ctx, pre_stage, layer)
    post = stage_layer_matrix(ctx, post_stage, layer)
    if pre.shape != post.shape:
        raise RuntimeError(
            f"shape mismatch {pre_stage}{pre.shape} vs {post_stage}{post.shape}"
        )

    delta = post - pre
    return {
        "delta": delta,
        "pre": pre,
        "record_ids": np.array([r["record_id"] for r in ctx.rows], dtype=object),
        "quadrants": np.array([r.get("quadrant") for r in ctx.rows], dtype=object),
        "splits": np.array([r.get("split") for r in ctx.rows], dtype=object),
        "norms": np.linalg.norm(delta, axis=1),
        "pre_stage": pre_stage,
        "post_stage": post_stage,
        "layer": layer,
    }


def assemble_branch_delta(ctx, branch: str, layer: int = INJECT_LAYER) -> dict:
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}")
    b = BRANCHES[branch]
    out = assemble_delta(ctx, b["pre"], b["post"], layer)
    out["branch"] = branch
    return out


# ---------------------------------------------------------------------------
# Control vectors
# ---------------------------------------------------------------------------


def _rngs() -> dict[str, np.random.Generator]:
    children = np.random.default_rng(CROSSBRANCH_SEED).spawn(len(SPAWN_ORDER))
    return dict(zip(SPAWN_ORDER, children))


def shuffle_within_quadrant(
    quadrants: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Permutation that reassigns deltas only within each quadrant block.

    Destroys the prompt<->delta correspondence while preserving each
    quadrant's delta distribution exactly -- so a condition that still works
    under this shuffle is not using prompt-specific information.
    """
    quadrants = np.asarray(quadrants, dtype=object)
    perm = np.arange(len(quadrants))
    for q in sorted({q for q in quadrants.tolist() if q is not None}):
        idx = np.flatnonzero(quadrants == q)
        perm[idx] = rng.permutation(idx)
    return perm


def shuffle_global(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n)


def rescale_to_row_norms(
    vectors: np.ndarray, reference: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Rescale each row of ``vectors`` to the norm of the SAME row of
    ``reference``, leaving its direction untouched.

    Used to build the norm-matched global shuffle. A bare global permutation
    changes two things at once: which prompt class the delta came from, AND
    the per-row injected magnitude -- quadrants do not share a delta-norm
    distribution (quadrant C's deltas are the largest in this benchmark), so
    drawing from the global pool systematically dilutes what lands in C.
    Rescaling each permuted row back to the norm the identity arm would have
    injected there holds dose fixed, so the remaining difference is
    attributable to class membership rather than magnitude.

    A row whose reference norm is zero stays zero, matching
    ``normmatched_random``'s convention: the identity arm injects nothing
    there, so neither may a control.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if vectors.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: vectors {vectors.shape} vs reference {reference.shape}"
        )
    cur = np.linalg.norm(vectors, axis=1, keepdims=True)
    want = np.linalg.norm(reference, axis=1, keepdims=True)
    return vectors * (want / (cur + eps))


def normmatched_random(delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-row isotropic random vector with ||r(x)|| == ||delta(x)||.

    Rows whose delta is exactly zero stay zero -- rescaling a zero-norm row
    to a random direction would inject a perturbation the real condition
    never applies.
    """
    delta = np.asarray(delta, dtype=np.float64)
    target = np.linalg.norm(delta, axis=1, keepdims=True)
    g = rng.standard_normal(delta.shape)
    gn = np.linalg.norm(g, axis=1, keepdims=True)
    gn[gn == 0.0] = 1.0
    return g / gn * target


def apply_permutation(delta: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Row i now carries the delta that belonged to row perm[i]."""
    return np.asarray(delta)[np.asarray(perm)]


# ---------------------------------------------------------------------------
# Stage-2 vectors (built only when Stage 2 is actually being prepared)
# ---------------------------------------------------------------------------


def dosematch_to(
    source_delta: np.ndarray, target_delta: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Rescale each source row to the TARGET branch's per-row delta norm.

    Separates "is this delta reusable" from "is a perturbation of this size
    enough". NOTE for the write-up: this arm consumes ||delta_target(x)||,
    i.e. the target branch's own post-DPO activation change for that same
    prompt, so it is a STRICTLY STRONGER oracle than the plain identity arm
    and must be labelled as such wherever it is reported.
    """
    source_delta = np.asarray(source_delta, dtype=np.float64)
    target_delta = np.asarray(target_delta, dtype=np.float64)
    if source_delta.shape != target_delta.shape:
        raise ValueError(
            f"shape mismatch: source {source_delta.shape} vs target {target_delta.shape}"
        )
    src_norm = np.linalg.norm(source_delta, axis=1, keepdims=True)
    tgt_norm = np.linalg.norm(target_delta, axis=1, keepdims=True)
    return source_delta * (tgt_norm / (src_norm + eps))


def direction_dose_scalar(
    delta: np.ndarray, quadrants: np.ndarray, splits: np.ndarray
) -> float:
    """median ||delta(x)|| over the direction_estimation half of A and D.

    Calibration-only by construction: held-out behavioural rows never enter
    it, so the dose given to a direction arm is not tuned on the rows the
    intervention is later judged on.
    """
    quadrants = np.asarray(quadrants, dtype=object)
    splits = np.asarray(splits, dtype=object)
    mask = np.array(
        [
            q in ("A", "D") and s == "direction_estimation"
            for q, s in zip(quadrants.tolist(), splits.tolist())
        ]
    )
    if not mask.any():
        raise RuntimeError(
            "no direction_estimation rows in quadrants A/D -- cannot calibrate "
            "a direction dose without them"
        )
    return float(np.median(np.linalg.norm(np.asarray(delta)[mask], axis=1)))


def load_direction_vector(
    stage: str, layer: int = INJECT_LAYER, directions_dir="results/refusal_direction"
) -> np.ndarray:
    """One layer's unit-norm A-D direction for `stage`.

    Prefers the v2 artifact; falls back to the legacy name only if the v2 one
    is absent. Raises rather than silently using a direction whose norm is not
    ~1, which would quietly change the meaning of the dose scalar.
    """
    directions_dir = Path(directions_dir)
    v2 = directions_dir / f"{stage}_v2_direction.npy"
    legacy = directions_dir / f"{stage}_direction.npy"
    path = v2 if v2.exists() else legacy
    if not path.exists():
        raise FileNotFoundError(
            f"no direction for {stage}: looked for {v2} then {legacy}"
        )
    arr = np.load(path)
    if arr.ndim != 2 or not 0 <= layer < arr.shape[0]:
        raise RuntimeError(f"{path}: expected (layers, hidden); got {arr.shape}")
    d = np.asarray(arr[layer], dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise RuntimeError(
            f"{path} layer {layer}: direction norm {norm:.6f}, expected ~1.0"
        )
    return d


def constant_delta_array(direction: np.ndarray, scale: float, n_rows: int) -> np.ndarray:
    """The same `scale * direction` vector for every row.

    This is what makes the direction arms comparable to the delta arms: they
    go through the identical per-row injector at the identical site and
    timing, differing only in WHICH vector is injected.
    """
    d = np.asarray(direction, dtype=np.float64)
    return np.tile(scale * d, (n_rows, 1))


def decompose_dosematched(
    delta: np.ndarray, direction: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Split each row of ``delta`` into its component along ``direction`` and
    the residual, then rescale EACH back to that row's original ``||delta(x)||``.

    Returns ``(parallel_dosematched, perp_dosematched)``. After rescaling both
    injected arms carry the SAME per-row magnitude as the identity arm, so a
    null on one component cannot be read as "this arm just injected less" --
    the only thing that differs between parallel, perp and identity is the
    direction of the injected vector. Rows whose delta is exactly zero, and
    rows whose one component is degenerate (e.g. delta is exactly parallel to
    ``direction`` so the perp part is ~0), stay zero for that component and
    are reported by ``assemble_decomposition``.

    ``direction`` is the same unit A-D refusal direction the ``dir_source``
    arm uses, so ``parallel`` is "the refusal-direction part of the DPO delta,
    at full delta magnitude" and ``perp`` is "everything else in the DPO
    delta, at full delta magnitude".
    """
    delta = np.asarray(delta, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64).ravel()
    d_norm = np.linalg.norm(d)
    if d_norm <= eps:
        raise ValueError("direction is the zero vector")
    d = d / d_norm
    coeff = delta @ d                                  # (n,)
    par = coeff[:, None] * d[None, :]                  # (n, h)
    perp = delta - par
    full = np.linalg.norm(delta, axis=1, keepdims=True)

    def _dose(comp: np.ndarray) -> np.ndarray:
        cn = np.linalg.norm(comp, axis=1, keepdims=True)
        # A component whose norm is a negligible fraction of the row's full
        # delta is numerically degenerate (delta is ~parallel or ~orthogonal
        # to d). Zero it rather than amplify float noise back to full
        # magnitude -- the threshold is relative so it holds at any scale.
        ok = cn > np.maximum(1e-9 * full, eps)
        scale = np.where(ok, full / np.where(ok, cn, 1.0), 0.0)
        return comp * scale

    return _dose(par), _dose(perp)


# ---------------------------------------------------------------------------
# Dose diagnostic (descriptive only -- never a gate criterion)
# ---------------------------------------------------------------------------


def dose_ratio_report(
    delta: np.ndarray, pre: np.ndarray, quadrants: np.ndarray
) -> dict:
    """||Delta(x)|| / ||h_pre(x)|| by quadrant: median, p95, max.

    How large the injected perturbation is relative to the residual stream it
    is added to, in natural units. Descriptive only: it carries no threshold
    and never enters the gate. Its use is as the cheapest early warning that
    a coefficient of 2.0 will drive generation collapse.
    """
    dn = np.linalg.norm(np.asarray(delta, dtype=np.float64), axis=1)
    hn = np.linalg.norm(np.asarray(pre, dtype=np.float64), axis=1)
    safe = np.where(hn == 0.0, np.nan, hn)
    ratio = dn / safe

    quadrants = np.asarray(quadrants, dtype=object)
    out: dict[str, dict] = {}
    for q in sorted({q for q in quadrants.tolist() if q is not None}):
        vals = ratio[quadrants == q]
        vals = vals[~np.isnan(vals)]
        out[q] = {
            "n": int(vals.size),
            "median": float(np.median(vals)) if vals.size else None,
            "p95": float(np.percentile(vals, 95)) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
        }
    out["_note"] = "descriptive only; not a gate criterion"
    return out


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def save_delta_npz(path: Path, vectors: np.ndarray, record_ids, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectors": np.asarray(vectors, dtype=np.float32),
        "record_ids": np.asarray(record_ids, dtype=object),
        "norms": np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1),
    }
    for k, v in extra.items():
        payload[k] = np.asarray(v)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **payload)
    tmp.replace(path)
    return path


def load_delta_npz(path: Path) -> dict:
    with np.load(Path(path), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def load_delta_map(path: Path) -> dict[str, np.ndarray]:
    """record_id -> (hidden,) float32 vector, as the injector consumes it."""
    data = load_delta_npz(path)
    ids = [str(r) for r in data["record_ids"].tolist()]
    vecs = data["vectors"]
    if len(ids) != vecs.shape[0]:
        raise RuntimeError(f"{path}: {len(ids)} ids vs {vecs.shape[0]} vectors")
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{path}: duplicate record_ids")
    return {rid: vecs[i] for i, rid in enumerate(ids)}


def artifact_path(key: str, layer: int = INJECT_LAYER, out_dir=DELTAS_DIR) -> Path:
    return Path(out_dir) / f"{key}_L{layer}.npz"


def assert_direction_matches_dir(
    out_dir, source_branch: str, target_branch: str, *, force: bool = False
) -> None:
    """Refuse to assemble one direction's vectors over another's.

    Artifact filenames are DIRECTION-NEUTRAL by design (``delta_source_L24.npz``
    means "the source branch's delta", not "branch A's delta"), because the
    condition vocabulary is direction-neutral too. That is fine as long as each
    direction gets its own ``--out-dir``. It is a silent data-corruption hazard
    the moment it does not: running the reciprocal (B->A) assembly into the
    directory already holding the A->B vectors would overwrite delta_source
    with Delta_B while every consuming filename, condition name and test stays
    identical -- the run would succeed and the numbers would be wrong.

    So: if the target directory already carries a binding recording a DIFFERENT
    (source, target) pair, raise and name the fix. ``force`` is the deliberate
    override for genuinely re-assembling the same directory on purpose.
    """
    binding_path = Path(out_dir) / "crossbranch_deltas_binding.json"
    if force or not binding_path.exists():
        return
    try:
        roles = load_json(binding_path).get("roles") or {}
    except Exception:  # unreadable/partial binding: let assembly proceed
        return
    prev = (roles.get("source_branch"), roles.get("target_branch"))
    if prev == (None, None) or prev == (source_branch, target_branch):
        return
    raise SystemExit(
        f"{binding_path} records direction {prev[0]}->{prev[1]}, but this run "
        f"assembles {source_branch}->{target_branch}. Artifact filenames are "
        "direction-neutral, so continuing would overwrite the "
        f"{prev[0]}->{prev[1]} vectors in place and every downstream condition "
        "would silently consume the wrong deltas.\n"
        f"  Use a separate directory, e.g. --out-dir "
        f"results/crossbranch/deltas_{source_branch}to{target_branch}\n"
        "  (or pass force=True / --force-direction if you really mean to "
        "re-assemble this directory)."
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def assemble_p0(
    ctx,
    source_branch: str = "A",
    target_branch: str = "B",
    layer: int = INJECT_LAYER,
    out_dir: Path = DELTAS_DIR,
    force_direction: bool = False,
) -> dict:
    """Build every artifact the Stage-1 gate consumes, plus diagnostics.

    Written: delta_target, delta_source, normmatched_random_target,
    delta_source_shuf_wq (built + tested now, first consumed in Stage 2).
    """
    roles = resolve(source_branch, target_branch)
    out_dir = Path(out_dir)
    assert_direction_matches_dir(
        out_dir, source_branch, target_branch, force=force_direction
    )

    for stage in stages_needed(source_branch, target_branch):
        adopt_activation(ctx, stage, out_dir)

    rng = _rngs()
    tgt = assemble_branch_delta(ctx, target_branch, layer)
    src = assemble_branch_delta(ctx, source_branch, layer)

    written: dict[str, str] = {}

    written["delta_target"] = str(
        save_delta_npz(
            artifact_path("delta_target", layer, out_dir),
            tgt["delta"], tgt["record_ids"],
        )
    )
    written["delta_source"] = str(
        save_delta_npz(
            artifact_path("delta_source", layer, out_dir),
            src["delta"], src["record_ids"],
        )
    )
    written["normmatched_random_target"] = str(
        save_delta_npz(
            artifact_path("normmatched_random_target", layer, out_dir),
            normmatched_random(tgt["delta"], rng["normmatched_random"]),
            tgt["record_ids"],
        )
    )
    perm = shuffle_within_quadrant(src["quadrants"], rng["shuffle_within_quadrant"])
    written["delta_source_shuf_wq"] = str(
        save_delta_npz(
            artifact_path("delta_source_shuf_wq", layer, out_dir),
            apply_permutation(src["delta"], perm), src["record_ids"], perm=perm,
        )
    )

    diagnostics = {
        "dose_ratio_target": dose_ratio_report(
            tgt["delta"], tgt["pre"], tgt["quadrants"]
        ),
        "dose_ratio_source": dose_ratio_report(
            src["delta"], src["pre"], src["quadrants"]
        ),
    }

    provenance = {
        **ctx.bind(),
        "layer": layer,
        "position": "final",
        "seed": CROSSBRANCH_SEED,
        "spawn_order": list(SPAWN_ORDER),
        "roles": roles,
        "n_rows": len(ctx.rows),
        "artifacts": written,
        "artifact_sha256": {
            k: sha256_file(v) for k, v in written.items()
        },
        "diagnostics": diagnostics,
    }
    write_json_lf(out_dir / "crossbranch_deltas_binding.json", provenance)
    return provenance


def assemble_stage2(
    ctx,
    source_branch: str = "A",
    target_branch: str = "B",
    layer: int = INJECT_LAYER,
    out_dir: Path = DELTAS_DIR,
    direction_source_stage: str | None = None,
    direction_target_stage: str | None = None,
) -> dict:
    """Build the vectors Stage 2 consumes, on top of the P0 ones.

    Writes: delta_source_dosematched, normmatched_random_source,
    dir_source_const, dir_target_const, delta_source_shuf_global (optional
    condition). Requires assemble_p0 to have run first -- Stage 2 reuses
    delta_source / delta_target rather than recomputing them, so the two
    stages can never disagree about what the deltas are.
    """
    roles = resolve(source_branch, target_branch)
    out_dir = Path(out_dir)

    src_path = artifact_path("delta_source", layer, out_dir)
    tgt_path = artifact_path("delta_target", layer, out_dir)
    for path in (src_path, tgt_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run the P0 assembly first "
                "(python -m src.analysis.crossbranch.delta)"
            )
    src = load_delta_npz(src_path)
    tgt = load_delta_npz(tgt_path)
    src_vecs = np.asarray(src["vectors"], dtype=np.float64)
    tgt_vecs = np.asarray(tgt["vectors"], dtype=np.float64)
    record_ids = src["record_ids"]
    if [str(r) for r in record_ids.tolist()] != [str(r) for r in tgt["record_ids"].tolist()]:
        raise RuntimeError("delta_source and delta_target disagree on record_ids")

    quadrants = np.array([r.get("quadrant") for r in ctx.rows], dtype=object)
    splits = np.array([r.get("split") for r in ctx.rows], dtype=object)

    # Direction stages default to each branch's POST-DPO checkpoint.
    d_src_stage = direction_source_stage or roles["source_post"]
    d_tgt_stage = direction_target_stage or roles["target_post"]
    d_src = load_direction_vector(d_src_stage, layer)
    d_tgt = load_direction_vector(d_tgt_stage, layer)

    # One scalar per branch, each dosed to its OWN branch's delta scale, so
    # the concept arm is comparable to the delta arm it is contrasted with.
    s_src = direction_dose_scalar(src_vecs, quadrants, splits)
    s_tgt = direction_dose_scalar(tgt_vecs, quadrants, splits)

    rng = _rngs()
    written: dict[str, str] = {}

    written["delta_source_dosematched"] = str(
        save_delta_npz(
            artifact_path("delta_source_dosematched", layer, out_dir),
            dosematch_to(src_vecs, tgt_vecs), record_ids,
        )
    )
    written["normmatched_random_source"] = str(
        save_delta_npz(
            artifact_path("normmatched_random_source", layer, out_dir),
            normmatched_random(src_vecs, rng["normmatched_random_source"]), record_ids,
        )
    )
    written["dir_source_const"] = str(
        save_delta_npz(
            artifact_path("dir_source_const", layer, out_dir),
            constant_delta_array(d_src, s_src, len(record_ids)), record_ids,
        )
    )
    written["dir_target_const"] = str(
        save_delta_npz(
            artifact_path("dir_target_const", layer, out_dir),
            constant_delta_array(d_tgt, s_tgt, len(record_ids)), record_ids,
        )
    )
    perm_global = shuffle_global(len(record_ids), rng["shuffle_global"])
    globally_shuffled = apply_permutation(src_vecs, perm_global)
    written["delta_source_shuf_global"] = str(
        save_delta_npz(
            artifact_path("delta_source_shuf_global", layer, out_dir),
            globally_shuffled, record_ids, perm=perm_global,
        )
    )
    # Same permutation, but each row rescaled back to the norm the identity
    # arm injects there -- isolates class membership from class-correlated
    # dose (see rescale_to_row_norms).
    written["delta_source_shuf_global_normmatched"] = str(
        save_delta_npz(
            artifact_path("delta_source_shuf_global_normmatched", layer, out_dir),
            rescale_to_row_norms(globally_shuffled, src_vecs),
            record_ids, perm=perm_global,
        )
    )

    provenance = {
        **ctx.bind(),
        "stage": "stage2",
        "layer": layer,
        "position": "final",
        "seed": CROSSBRANCH_SEED,
        "spawn_order": list(SPAWN_ORDER),
        "roles": roles,
        "direction_source_stage": d_src_stage,
        "direction_target_stage": d_tgt_stage,
        "dose_scalars": {"s_source": s_src, "s_target": s_tgt},
        "dose_scalar_rule": (
            "median ||delta|| over quadrant A/D rows with "
            "split == direction_estimation; calibration-only, never held-out"
        ),
        "n_rows": len(record_ids),
        "artifacts": written,
        "artifact_sha256": {k: sha256_file(v) for k, v in written.items()},
        "oracle_note": (
            "delta_source_dosematched consumes ||delta_target(x)||, the target "
            "branch's own post-DPO change for that prompt -- a strictly "
            "stronger oracle than the plain identity arm. Label it as such."
        ),
    }
    write_json_lf(out_dir / "crossbranch_stage2_deltas_binding.json", provenance)
    return provenance


def assemble_decomposition(
    ctx,
    source_branch: str = "A",
    target_branch: str = "B",
    layer: int = INJECT_LAYER,
    out_dir: Path = DELTAS_DIR,
    direction_source_stage: str | None = None,
) -> dict:
    """Build the dose-matched parallel / perpendicular decomposition of the
    source delta along the source branch's A-D refusal direction.

    Writes ``delta_source_parallel`` and ``delta_source_perp``, each rescaled
    per row back to ``||delta_source(x)||`` so the two arms inject the same
    per-row magnitude as the identity arm and differ from it (and from each
    other) only in direction. This is the pair the frozen plan's deferred
    note requires -- injecting the perp component alone would confound
    "removed the refusal-direction part" with "injected a smaller vector".

    Requires the P0 assembly (reuses ``delta_source``). Deterministic: no RNG,
    so no SeedSequence stream is consumed and ``SPAWN_ORDER`` is untouched.
    """
    roles = resolve(source_branch, target_branch)
    out_dir = Path(out_dir)
    src_path = artifact_path("delta_source", layer, out_dir)
    if not src_path.exists():
        raise FileNotFoundError(
            f"{src_path} missing -- run the P0 assembly first "
            "(python -m src.analysis.crossbranch.delta)"
        )
    src = load_delta_npz(src_path)
    src_vecs = np.asarray(src["vectors"], dtype=np.float64)
    record_ids = src["record_ids"]

    d_src_stage = direction_source_stage or roles["source_post"]
    d_src = load_direction_vector(d_src_stage, layer)
    cos_delta_dir = float(
        np.median(
            np.abs(src_vecs @ d_src)
            / (np.linalg.norm(src_vecs, axis=1) + 1e-12)
        )
    )

    par, perp = decompose_dosematched(src_vecs, d_src)
    zero_par = int((np.linalg.norm(par, axis=1) < 1e-9).sum())
    zero_perp = int((np.linalg.norm(perp, axis=1) < 1e-9).sum())

    written = {
        "delta_source_parallel": str(
            save_delta_npz(
                artifact_path("delta_source_parallel", layer, out_dir),
                par, record_ids,
            )
        ),
        "delta_source_perp": str(
            save_delta_npz(
                artifact_path("delta_source_perp", layer, out_dir),
                perp, record_ids,
            )
        ),
    }

    provenance = {
        **ctx.bind(),
        "stage": "decomposition",
        "layer": layer,
        "position": "final",
        "roles": roles,
        "direction_source_stage": d_src_stage,
        "rule": (
            "delta_source split along the source A-D refusal direction into "
            "parallel + perpendicular, each rescaled per row to "
            "||delta_source(x)|| so both arms match the identity arm's per-row "
            "magnitude and differ only in direction"
        ),
        "median_abs_cos(delta_source, d_source)": cos_delta_dir,
        "n_rows": len(record_ids),
        "zero_rows": {"parallel": zero_par, "perp": zero_perp},
        "artifacts": written,
        "artifact_sha256": {k: sha256_file(v) for k, v in written.items()},
    }
    write_json_lf(out_dir / "crossbranch_decomposition_deltas_binding.json", provenance)
    return provenance


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble crossbranch delta artifacts.")
    p.add_argument("--eval-set", default=None)
    p.add_argument("--benchmark-sha256", default=None)
    p.add_argument("--split-manifest", default="logs/direction_split_manifest.json")
    p.add_argument("--source-branch", default="A", choices=sorted(BRANCHES))
    p.add_argument("--target-branch", default="B", choices=sorted(BRANCHES))
    p.add_argument("--layer", type=int, default=INJECT_LAYER)
    p.add_argument("--out-dir", default=str(DELTAS_DIR))
    p.add_argument(
        "--stage2",
        action="store_true",
        help="Additionally build the Stage-2 vectors (dose-matched source "
             "delta, source-matched random, the two constant direction "
             "vectors, global shuffle). Requires the P0 assembly first.",
    )
    p.add_argument(
        "--decomposition",
        action="store_true",
        help="Additionally build the dose-matched parallel/perpendicular split "
             "of delta_source along the source A-D refusal direction "
             "(delta_source_parallel, delta_source_perp). Requires the P0 "
             "assembly first.",
    )
    p.add_argument("--direction-source-stage", default=None)
    p.add_argument("--direction-target-stage", default=None)
    p.add_argument(
        "--force-direction",
        action="store_true",
        help="Re-assemble into a directory whose binding records a DIFFERENT "
             "source->target pair. Off by default: artifact filenames are "
             "direction-neutral, so this would overwrite the other direction's "
             "vectors in place and every downstream condition would silently "
             "consume the wrong deltas. Give each direction its own --out-dir "
             "instead.",
    )
    args = p.parse_args()

    ctx = build_context(args)
    prov = assemble_p0(
        ctx, args.source_branch, args.target_branch, args.layer, Path(args.out_dir),
        force_direction=args.force_direction,
    )
    print(f"Assembled {len(prov['artifacts'])} P0 artifacts into {args.out_dir}")
    for key, path in prov["artifacts"].items():
        print(f"  {key}: {path}")
    print("\nDose ratio ||Delta||/||h|| (descriptive only):")
    for name in ("dose_ratio_target", "dose_ratio_source"):
        print(f"  {name}:")
        for q, s in prov["diagnostics"][name].items():
            if q.startswith("_"):
                continue
            print(
                f"    {q}: median={s['median']:.4f} p95={s['p95']:.4f} "
                f"max={s['max']:.4f} (n={s['n']})"
            )

    if args.stage2:
        s2 = assemble_stage2(
            ctx, args.source_branch, args.target_branch, args.layer,
            Path(args.out_dir),
            direction_source_stage=args.direction_source_stage,
            direction_target_stage=args.direction_target_stage,
        )
        print(f"\nAssembled {len(s2['artifacts'])} Stage-2 artifacts")
        for key, path in s2["artifacts"].items():
            print(f"  {key}: {path}")
        print(
            f"\nDirection doses (calibration split only): "
            f"s_source={s2['dose_scalars']['s_source']:.4f} "
            f"(from {s2['direction_source_stage']}), "
            f"s_target={s2['dose_scalars']['s_target']:.4f} "
            f"(from {s2['direction_target_stage']})"
        )

    if args.decomposition:
        dc = assemble_decomposition(
            ctx, args.source_branch, args.target_branch, args.layer,
            Path(args.out_dir),
            direction_source_stage=args.direction_source_stage,
        )
        print(f"\nAssembled {len(dc['artifacts'])} decomposition artifacts")
        for key, path in dc["artifacts"].items():
            print(f"  {key}: {path}")
        print(
            f"  median |cos(delta_source, d_source)| = "
            f"{dc['median_abs_cos(delta_source, d_source)']:.4f}   "
            f"zero rows: parallel={dc['zero_rows']['parallel']} "
            f"perp={dc['zero_rows']['perp']}"
        )


if __name__ == "__main__":
    main()
