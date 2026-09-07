"""Final-token repair for the FLLMPT paper (audit RED-1).

The v2 intervention pipeline (``v2_pipeline.stage_direction`` /
``stage_direction_crossfit``) builds the A-D difference-in-means direction from
``*_pooled.npy`` -- the **mean of the last 5 non-padding tokens**
(``POOL_WINDOW = 5``). The preregistered analysis plan
(``docs/audit/analysis_plan.md`` section 4) specifies the **final non-padding
prompt token**, from ``*_final.npy``:

    d_final^{s,l} = unit( mean(final[A_est])^{s,l} - mean(final[D_est])^{s,l} )

computed per layer, on the ``direction_estimation`` split only, RAW (no
centering), normalised only after the mean difference.

This module builds an **explicitly versioned** final-token direction and its
matched random-ablation control, and (CPU-only) re-runs CF3 on the final-token
directions. It NEVER reads, writes, relabels or reuses the pooled direction
files or the pooled causal outputs. Everything it produces lives under
``results/final_token_repair/``.

CPU-only, no ``torch`` import. Reuses the torch-free helpers in
``src.analysis.control_directions`` and ``src.analysis.direction_decodability``.

    python -m src.analysis.final_token_repair --stages M2 M3 M2_alt M3_alt --recompute-cf3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.analysis.control_directions import (
    ABLATION_LAYERS,
    CONTROL_SEED,
    build_stage_control_record,
    diff_in_means_direction,
)

# --- explicit, non-collidable pooling names -------------------------------- #
# The value is the activation-array filename suffix. "final_token" is the
# preregistered choice; "mean_last5" is the name for the CURRENT pooled
# implementation. These names cannot be confused with the old ``_v2_direction``
# files, which carry no pooling tag at all.
POOLING_TO_SUFFIX = {"final_token": "final", "mean_last5": "pooled"}
POOL_WINDOW_FOR = {"final_token": None, "mean_last5": 5}

# The cross-fit partition is frozen by the pooled run. We re-derive it here
# (torch-free copy of v2_pipeline.crossfit_folds) and a test asserts it matches
# both v2_pipeline's function AND the committed xfit5 bindings byte-for-byte.
CROSSFIT_SEED = 20260907
DIRECTION_SPLIT = "direction_estimation"
HELD_OUT_SPLIT = "held_out_behavioral"
FROZEN_BENCHMARK_SHA256 = (
    "e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b"
)
SPLIT_MANIFEST_CANONICAL_SHA256 = (
    "880381606de7aa2ffbdb8f7c75303cf4937167ed1a2e1b417afeb33761fcf8f1"
)

REPAIR_ROOT = Path("results/final_token_repair")
DIRECTIONS_DIR = REPAIR_ROOT / "directions"
BINDINGS_DIR = REPAIR_ROOT / "bindings"
SUMMARIES_DIR = REPAIR_ROOT / "summaries"

STAGE_MODEL_HINT = {
    "M3": "urosavurdic/qwen2.5-1.5b-m3-dpo (chain on Qwen/Qwen2.5-1.5B)",
    "M3_alt": "urosavurdic/qwen2.5-1.5b-m3-alt-dpo",
    "M3_direct": "urosavurdic/qwen2.5-1.5b-m3-direct-dpo",
    "M3_direct_alt": "urosavurdic/qwen2.5-1.5b-m3-direct-alt-dpo",
    "M2": "urosavurdic/qwen2.5-1.5b-m2-safety",
    "M2_alt": "urosavurdic/qwen2.5-1.5b-m2-alt-safety",
}


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - git absent
        return "unknown"


def crossfit_folds_local(record_ids: Sequence, k: int, seed: int = CROSSFIT_SEED):
    """Torch-free copy of ``v2_pipeline.crossfit_folds`` (verified equal by a
    test). Deterministic K-way partition; depends only on the id SET + seed."""
    ids = sorted({str(r) for r in record_ids if r is not None})
    if k < 2:
        raise ValueError(f"cross-fit needs K >= 2, got {k}")
    if len(ids) < k:
        raise ValueError(f"cannot build {k} folds from {len(ids)} distinct ids")
    order = np.random.default_rng(seed).permutation(len(ids))
    folds: list[list[str]] = [[] for _ in range(k)]
    for position, idx in enumerate(order):
        folds[position % k].append(ids[int(idx)])
    return [sorted(f) for f in folds]


# --------------------------------------------------------------------------- #
# activation + metadata loading
# --------------------------------------------------------------------------- #
@dataclass
class StageArrays:
    stage: str
    pooling: str
    array_path: Path
    array_sha256: str
    arr: np.ndarray  # (n_rows, n_layers, hidden)
    quadrants: np.ndarray
    splits: np.ndarray
    record_ids: list
    n_rows: int


def load_stage(act_dir: Path, stage: str, pooling: str) -> StageArrays:
    if pooling not in POOLING_TO_SUFFIX:
        raise ValueError(
            f"pooling must be one of {sorted(POOLING_TO_SUFFIX)}, got {pooling!r}"
        )
    suffix = POOLING_TO_SUFFIX[pooling]
    apath = act_dir / f"{stage}_{suffix}.npy"
    mpath = act_dir / f"{stage}_metadata.json"
    if not apath.exists() or not mpath.exists():
        raise FileNotFoundError(
            f"{stage}: need {apath} and {mpath}. "
            f"(M3_direct / M3_direct_alt local copies are 370-row pre-v2 - their "
            f"654-row _final.npy must come from the Drive activations_bundle.)"
        )
    arr = np.load(apath)
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    if arr.shape[0] != len(meta):
        raise RuntimeError(f"{stage}: array rows {arr.shape[0]} != metadata {len(meta)}")
    quad = np.array([r.get("quadrant") for r in meta])
    spl = np.array([r.get("split") for r in meta])
    if not (spl == DIRECTION_SPLIT).any():
        raise RuntimeError(
            f"{stage}: metadata has no '{DIRECTION_SPLIT}' rows -- this is a "
            f"pre-v2 (370-row) activation set; a fresh 654-row extraction is "
            f"required for the final-token repair."
        )
    return StageArrays(
        stage=stage,
        pooling=pooling,
        array_path=apath,
        array_sha256=sha256_file(apath),
        arr=arr,
        quadrants=quad,
        splits=spl,
        record_ids=[r.get("record_id") for r in meta],
        n_rows=len(meta),
    )


# --------------------------------------------------------------------------- #
# direction construction
# --------------------------------------------------------------------------- #
def full_direction(sa: StageArrays) -> np.ndarray:
    """d = unit(mean(arr[A_est]) - mean(arr[D_est])) per layer, no centering."""
    return diff_in_means_direction(
        sa.arr, sa.quadrants, "A", "D",
        splits=sa.splits, restrict_split=DIRECTION_SPLIT,
    )


def fold_directions(sa: StageArrays, k: int, seed: int = CROSSFIT_SEED):
    """Per-fold direction = unit(mean(arr[A_est \\ fold_j]) - mean(arr[D_est])).

    Only quadrant-A estimation rows are folded out (a quad-A test row
    contributes to nothing but mean(A_est)); the D centroid keeps all 120.
    Returns a list of dicts: {fold, test_record_ids, direction, n_A, n_D}.
    """
    a_est_mask = (sa.quadrants == "A") & (sa.splits == DIRECTION_SPLIT)
    d_est_mask = (sa.quadrants == "D") & (sa.splits == DIRECTION_SPLIT)
    a_ids = [str(rid) for rid, keep in zip(sa.record_ids, a_est_mask) if keep]
    folds = crossfit_folds_local(a_ids, k, seed)
    flat = [rid for f in folds for rid in f]
    if len(flat) != len(set(flat)) or set(flat) != set(a_ids):
        raise RuntimeError(f"{sa.stage}: fold partition is not a clean cover of A_est")

    id_arr = np.array([str(r) for r in sa.record_ids])
    md = np.asarray(sa.arr[d_est_mask]).mean(axis=0)
    out = []
    for j, fold_ids in enumerate(folds):
        drop = np.isin(id_arr, np.array(fold_ids, dtype=id_arr.dtype))
        keep_a = a_est_mask & ~drop
        if not keep_a.any():
            raise RuntimeError(f"{sa.stage} fold {j}: no A rows left for the direction")
        ma = np.asarray(sa.arr[keep_a]).mean(axis=0)
        delta = ma - md
        norms = np.linalg.norm(delta, axis=-1, keepdims=True)
        direction = (delta / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)
        out.append({
            "fold": j,
            "test_record_ids": sorted(fold_ids),
            "direction": direction,
            "n_A": int(keep_a.sum()),
            "n_D": int(d_est_mask.sum()),
        })
    return out


def cos_per_layer(a: np.ndarray, b: np.ndarray) -> list:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = np.where((na == 0) | (nb == 0), 1.0, na * nb)
    return (np.sum(a * b, axis=-1) / denom).tolist()


# --------------------------------------------------------------------------- #
# bindings
# --------------------------------------------------------------------------- #
def make_binding(
    *,
    stage: str,
    pooling: str,
    kind: str,
    direction_path: Path,
    sa: StageArrays,
    n_A: int,
    n_D: int,
    split_name: str,
    fold: int | None = None,
    k: int | None = None,
    cross_fit_seed: int | None = None,
    coefficient=None,
) -> dict:
    return {
        "analysis": f"final_token_repair::{kind}",
        "stage": stage,
        "pooling": pooling,
        "pool_window": POOL_WINDOW_FOR[pooling],
        "direction_construction": (
            "unit(mean(final[A_direction_estimation]) - "
            "mean(final[D_direction_estimation])); RAW, no centering; "
            "normalised per layer AFTER the mean difference"
        ),
        "activation_source_path": sa.array_path.as_posix(),
        "activation_source_sha256": sa.array_sha256,
        "direction_path": Path(direction_path).as_posix(),
        "direction_sha256": sha256_file(direction_path),
        "direction_shape": list(np.load(direction_path).shape),
        "benchmark_sha256": FROZEN_BENCHMARK_SHA256,
        "split_manifest_sha256": SPLIT_MANIFEST_CANONICAL_SHA256,
        "direction_split_seed": 45,
        "cross_fit_seed": cross_fit_seed,
        "cross_fit_k": k,
        "cross_fit_fold": fold,
        "split_name": split_name,
        "n_A_rows": n_A,
        "n_D_rows": n_D,
        "layers": list(ABLATION_LAYERS),
        "coefficient": coefficient,
        "control_seed": CONTROL_SEED,
        "model_checkpoint_hint": STAGE_MODEL_HINT.get(stage),
        "code_commit": code_commit(),
        "preregistered": pooling == "final_token",
        "deviation_note": (
            "REPAIR of audit RED-1: the committed v2 pipeline built this "
            "direction from _pooled (mean of last 5 tokens). analysis_plan.md 4 "
            "fixes the canonical direction on _final (final prompt token). This "
            "artifact is the preregistered final-token direction; it does NOT "
            "overwrite results/refusal_direction/{stage}_v2_direction.npy (the "
            "pooled one, kept for the disclosed deviation)."
            if pooling == "final_token" else
            "mean_last5 (mean of the last 5 non-padding tokens) - the CURRENT "
            "pooled implementation, reproduced here for explicit comparison only. "
            "NOT the preregistered direction (analysis_plan.md 4 fixes _final)."
        ),
    }


# --------------------------------------------------------------------------- #
# build directions for one stage
# --------------------------------------------------------------------------- #
def build_stage(
    act_dir: Path, stage: str, pooling: str, k: int
) -> dict:
    DIRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    BINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    sa = load_stage(act_dir, stage, pooling)

    n_a_est = int(((sa.quadrants == "A") & (sa.splits == DIRECTION_SPLIT)).sum())
    n_d_est = int(((sa.quadrants == "D") & (sa.splits == DIRECTION_SPLIT)).sum())

    # 1. full direction
    d_full = full_direction(sa).astype(np.float32)
    tag = "final_token" if pooling == "final_token" else "mean_last5"
    full_path = DIRECTIONS_DIR / f"{stage}_{tag}_L0-28.npy"
    np.save(full_path, d_full)
    full_binding = make_binding(
        stage=stage, pooling=pooling, kind="full_direction",
        direction_path=full_path, sa=sa, n_A=n_a_est, n_D=n_d_est,
        split_name=DIRECTION_SPLIT,
    )
    (BINDINGS_DIR / f"{stage}_{tag}_L0-28_binding.json").write_text(
        json.dumps(full_binding, indent=2), encoding="utf-8"
    )

    # 2. cross-fit fold directions (same partition as the pooled xfit5 run)
    fold_report = []
    fds = fold_directions(sa, k, CROSSFIT_SEED)
    for fd in fds:
        fp = DIRECTIONS_DIR / f"{stage}_{tag}_xfit{k}_fold{fd['fold']}.npy"
        np.save(fp, fd["direction"])
        fb = make_binding(
            stage=stage, pooling=pooling, kind="crossfit_fold_direction",
            direction_path=fp, sa=sa, n_A=fd["n_A"], n_D=fd["n_D"],
            split_name=DIRECTION_SPLIT, fold=fd["fold"], k=k,
            cross_fit_seed=CROSSFIT_SEED,
        )
        fb["test_record_ids"] = fd["test_record_ids"]
        cpl = cos_per_layer(fd["direction"], d_full)
        fb["cos_with_full_direction_per_layer"] = cpl
        fb["cos_with_full_direction_at_L24"] = cpl[24] if len(cpl) > 24 else None
        (BINDINGS_DIR / f"{stage}_{tag}_xfit{k}_fold{fd['fold']}_binding.json").write_text(
            json.dumps(fb, indent=2), encoding="utf-8"
        )
        fold_report.append({
            "fold": fd["fold"], "n_test_rows": len(fd["test_record_ids"]),
            "n_A": fd["n_A"], "n_D": fd["n_D"],
            "cos_with_full_direction_at_L24": fb["cos_with_full_direction_at_L24"],
        })

    # 3. matched random-ablation control (final activations + final direction ->
    #    fully self-consistent, unlike the committed pooled-direction/final-gamma
    #    hybrid). Uses control_directions.build_stage_control_record, which
    #    computes d_AD itself from the array it is given -- here the FINAL array.
    splits_str = np.array([s if s is not None else "" for s in sa.splits])
    control = build_stage_control_record(
        sa.arr, sa.quadrants, splits_str,
        record_ids=sa.record_ids, seed=CONTROL_SEED,
    )
    # sanity: the control's own d_AD must equal our d_full (both final, same split)
    _L = 24 if d_full.shape[0] > 24 else d_full.shape[0] - 1
    _dc = full_direction(sa)
    ctrl_cos = float(
        np.dot(d_full[_L], _dc[_L])
        / (np.linalg.norm(d_full[_L]) * np.linalg.norm(_dc[_L]))
    )
    control[f"_final_token_direction_matches_control_dAD_cos_L{_L}"] = round(ctrl_cos, 8)
    (BINDINGS_DIR / f"{stage}_{tag}_control.json").write_text(
        json.dumps(control, indent=2), encoding="utf-8"
    )

    # comparison vs the committed pooled v2 direction, if present
    pooled_v2 = Path("results/refusal_direction") / f"{stage}_v2_direction.npy"
    vs_pooled = None
    if pooled_v2.exists():
        vp = np.load(pooled_v2)
        vs_pooled = {
            "pooled_v2_path": pooled_v2.as_posix(),
            "pooled_v2_sha256": sha256_file(pooled_v2),
            "cos_final_token_vs_pooled_v2_per_layer": cos_per_layer(d_full, vp),
        }

    return {
        "stage": stage,
        "pooling": pooling,
        "n_A_direction_estimation": n_a_est,
        "n_D_direction_estimation": n_d_est,
        "n_A_held_out": int(((sa.quadrants == "A") & (sa.splits == HELD_OUT_SPLIT)).sum()),
        "n_D_held_out": int(((sa.quadrants == "D") & (sa.splits == HELD_OUT_SPLIT)).sum()),
        "full_direction_path": full_path.as_posix(),
        "full_direction_sha256": sha256_file(full_path),
        "cross_fit_folds": fold_report,
        "control_summary": {
            "n_calibration_rows": control["ablation_control"]["n_calibration_rows"],
            "gamma_per_layer": {
                l: control["ablation_control"]["per_layer"][l]["gamma"]
                for l in control["ablation_control"]["per_layer"]
            },
        },
        "vs_committed_pooled_v2_direction": vs_pooled,
    }


# --------------------------------------------------------------------------- #
# CF3 on the final-token directions (fully CPU-reproducible)
# --------------------------------------------------------------------------- #
def recompute_cf3(
    act_dir: Path,
    benchmark: str,
    m2_direction_path: Path,
    m3_direction_path: Path,
    out_path: Path,
    n_boot: int = 10000,
) -> dict:
    """Re-run CF3 (analysis_plan.md 4.4) with FINAL-TOKEN M2/M3 directions and
    the 654-row _final.npy activations. No GPU, no judge -- CF3 is a CPU
    LogisticRegression probe on residualised activations."""
    from src.analysis import direction_decodability as dd

    meta = json.loads((act_dir / "M2_metadata.json").read_text(encoding="utf-8"))
    bench_by_id = {}
    for line in Path(benchmark).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("record_id"):
            bench_by_id[r["record_id"]] = r
    for row in meta:
        b = bench_by_id.get(row.get("record_id"), {})
        for kkey in ("project_category", "source_category", "source_id", "pair_id"):
            row.setdefault(kkey, b.get(kkey))

    m2 = np.load(act_dir / "M2_final.npy")
    m3 = np.load(act_dir / "M3_final.npy")
    d_m2 = np.load(m2_direction_path)
    d_m3 = np.load(m3_direction_path)

    report = dd.cf3(m2, m3, meta, {"M2": d_m2, "M3": d_m3}, n_boot=n_boot)
    report["pooling"] = "final_token"
    report["provenance"] = {
        "residualization": "each stage residualised with ITS OWN final-token d_AD at layer 28",
        "M2_direction_path": Path(m2_direction_path).as_posix(),
        "M2_direction_sha256": sha256_file(m2_direction_path),
        "M3_direction_path": Path(m3_direction_path).as_posix(),
        "M3_direction_sha256": sha256_file(m3_direction_path),
        "M2_activation_path": (act_dir / "M2_final.npy").as_posix(),
        "M2_activation_sha256": sha256_file(act_dir / "M2_final.npy"),
        "M3_activation_path": (act_dir / "M3_final.npy").as_posix(),
        "M3_activation_sha256": sha256_file(act_dir / "M3_final.npy"),
        "benchmark_sha256": FROZEN_BENCHMARK_SHA256,
        "code_commit": code_commit(),
        "note": (
            "REPAIR of audit RED-2: the committed direction_decodability_cf3.json "
            "could not be reproduced from the repo (no committed M2_v2_direction.npy). "
            "This run uses the explicit final-token M2/M3 directions built by "
            "final_token_repair and is fully reproducible on CPU."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act-dir", default="results/activations")
    ap.add_argument("--stages", nargs="+", default=["M2", "M3", "M2_alt", "M3_alt"])
    ap.add_argument("--pooling", choices=sorted(POOLING_TO_SUFFIX),
                    default="final_token",
                    help="final_token (preregistered) or mean_last5 (the current "
                         "pooled implementation, for explicit comparison).")
    ap.add_argument("--k", type=int, default=5, help="cross-fit K (frozen at 5).")
    ap.add_argument("--recompute-cf3", action="store_true",
                    help="also re-run CF3 on the final-token M2/M3 directions "
                         "(requires M2 and M3 in --stages).")
    ap.add_argument("--benchmark",
                    default="data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl")
    ap.add_argument("--out-dir", default=str(SUMMARIES_DIR))
    args = ap.parse_args()

    act_dir = Path(args.act_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"pooling": args.pooling, "k": args.k, "code_commit": code_commit(),
              "stages": {}, "skipped": {}}
    for stage in args.stages:
        try:
            report["stages"][stage] = build_stage(act_dir, stage, args.pooling, args.k)
            print(f"  {stage}: final-token direction + {args.k} fold directions + control written")
        except FileNotFoundError as exc:
            report["skipped"][stage] = str(exc)
            print(f"  {stage}: SKIPPED - {exc}")

    (out_dir / "final_token_directions.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"-> {out_dir / 'final_token_directions.json'}")

    if args.recompute_cf3:
        tag = "final_token" if args.pooling == "final_token" else "mean_last5"
        m2p = DIRECTIONS_DIR / f"M2_{tag}_L0-28.npy"
        m3p = DIRECTIONS_DIR / f"M3_{tag}_L0-28.npy"
        if not (m2p.exists() and m3p.exists()):
            raise SystemExit(
                "--recompute-cf3 needs M2 and M3 final-token directions; add "
                "them to --stages first."
            )
        cf3 = recompute_cf3(act_dir, args.benchmark, m2p, m3p,
                            out_dir / "final_token_cf3.json")
        d = cf3.get("cf3_macroF1_M3_minus_M2")
        ci = cf3.get("bootstrap_group_diff", {})
        print(f"  CF3 (final-token): M2 F1={cf3.get('M2', {}).get('macro_f1'):.6f}  "
              f"M3 F1={cf3.get('M3', {}).get('macro_f1'):.6f}  cf3={d:+.6f}  "
              f"CI=[{ci.get('ci_low'):+.5f}, {ci.get('ci_high'):+.5f}]")
        print(f"-> {out_dir / 'final_token_cf3.json'}")


if __name__ == "__main__":
    main()
