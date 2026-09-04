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
SPAWN_ORDER = ("shuffle_within_quadrant", "normmatched_random")

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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def assemble_p0(
    ctx,
    source_branch: str = "A",
    target_branch: str = "B",
    layer: int = INJECT_LAYER,
    out_dir: Path = DELTAS_DIR,
) -> dict:
    """Build every artifact the Stage-1 gate consumes, plus diagnostics.

    Written: delta_target, delta_source, normmatched_random_target,
    delta_source_shuf_wq (built + tested now, first consumed in Stage 2).
    """
    roles = resolve(source_branch, target_branch)
    out_dir = Path(out_dir)

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


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble crossbranch delta artifacts.")
    p.add_argument("--eval-set", default=None)
    p.add_argument("--benchmark-sha256", default=None)
    p.add_argument("--split-manifest", default="logs/direction_split_manifest.json")
    p.add_argument("--source-branch", default="A", choices=sorted(BRANCHES))
    p.add_argument("--target-branch", default="B", choices=sorted(BRANCHES))
    p.add_argument("--layer", type=int, default=INJECT_LAYER)
    p.add_argument("--out-dir", default=str(DELTAS_DIR))
    args = p.parse_args()

    ctx = build_context(args)
    prov = assemble_p0(
        ctx, args.source_branch, args.target_branch, args.layer, Path(args.out_dir)
    )
    print(f"Assembled {len(prov['artifacts'])} artifacts into {args.out_dir}")
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


if __name__ == "__main__":
    main()
