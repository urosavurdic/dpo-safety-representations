"""Representation and update-size artifacts for the matched-start comparison.

    python -m src.stagecmp.run_geometry [--b 10000]

Writes into ``results/stagecmp/``, each with a ``*_binding.json`` sidecar:

``contrast_geometry.json``
    Per pooling, layer and arm: the contrast-norm ratio R, the cosine between
    the post and the start contrast, and the split of the post contrast into a
    component **along** the start direction and one **orthogonal** to it (both
    in units of the start norm, so along^2 + orth^2 = R^2). Recipe contrasts
    (DPO minus SFT, pooled over branches, per branch, and the branch gap) for R,
    the orthogonal part and the cosine. One joint bootstrap over the estimation
    prompts, all checkpoints recomputed on the same resampled rows; nominal 95%
    and Bonferroni 98.75% (four layers) percentile intervals. Also the
    point-estimate depth profile at every layer, start norms, and per-layer
    RMS-rescaled R.

``update_size.json``
    How far each arm moved, on three scales: merged LoRA weight change
    (Frobenius, exact from the rank-64 factors), activation drift and CKA. Plus
    the label-permutation null for the contrast change: the per-prompt
    activation changes are reassigned at random between the harmful and benign
    estimation rows. If an arm moved every prompt alike, however far, its
    contrast change would sit inside this null.

``separability.json``
    Scale-free and out-of-sample views, each checkpoint on its own direction
    (estimated on the estimation rows only): d' on the held-out harmful and
    benign rows, AUROC of reduced-cue harmful (C) against alarming benign (B),
    and where B and C sit between the benign (0) and harmful (1) estimation
    means.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from src.common.stats import BOOTSTRAP_SEED, percentile_ci
from src.stagecmp import STATUS_EXPLORATORY
from src.stagecmp.repr_metrics import linear_cka
from src.v2_io import sha256_file, write_json_lf

ACT_DIR = Path("results/activations")
OUT_DIR = Path("results/stagecmp")
LAYERS = (16, 20, 24, 28)
POOLINGS = {"final_token": "final", "mean_last5": "pooled"}
#: the four matched-start arms, then the same DPO configuration started after SFT
ARMS = {
    "sft_alpaca": ("M1", "M2"),
    "dpo_alpaca": ("M1", "M3_direct"),
    "sft_dolly": ("M1_alt", "M2_alt"),
    "dpo_dolly": ("M1_alt", "M3_direct_alt"),
    "dpo_after_sft_alpaca": ("M2", "M3"),
    "dpo_after_sft_dolly": ("M2_alt", "M3_alt"),
}
STAGES = sorted({s for pair in ARMS.values() for s in pair})
BONFERRONI = 1 - 0.05 / len(LAYERS)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _meta():
    meta = json.loads((ACT_DIR / "M1_metadata.json").read_text(encoding="utf-8"))
    quad = np.array([r["quadrant"] for r in meta])
    split = np.array([r.get("split") or "" for r in meta])
    groups = {
        "A_est": np.where((quad == "A") & (split == "direction_estimation"))[0],
        "D_est": np.where((quad == "D") & (split == "direction_estimation"))[0],
        "A_ho": np.where((quad == "A") & (split == "held_out_behavioral"))[0],
        "D_ho": np.where((quad == "D") & (split == "held_out_behavioral"))[0],
        "B": np.where(quad == "B")[0],
        "C": np.where(quad == "C")[0],
    }
    return meta, groups


def _assert_row_order(meta) -> None:
    ids = [r["record_id"] for r in meta]
    for stage in STAGES:
        other = json.loads((ACT_DIR / f"{stage}_metadata.json").read_text(encoding="utf-8"))
        if [r["record_id"] for r in other] != ids:
            raise RuntimeError(f"{stage}: row order differs from M1; joint resampling is invalid")


def _layer(stage: str, suffix: str, layer: int) -> np.ndarray:
    arr = np.load(ACT_DIR / f"{stage}_{suffix}.npy", mmap_mode="r")
    return np.asarray(arr[:, layer], dtype=np.float64)


def _count_matrix(rng, n_group: int, b: int) -> np.ndarray:
    """``(b, n_group)`` bootstrap counts; row k is replicate k's resample."""
    draws = rng.integers(0, n_group, size=(b, n_group))
    counts = np.zeros((b, n_group))
    np.add.at(counts, (np.arange(b)[:, None], draws), 1.0)
    return counts


def _ci(reps: np.ndarray, point: float) -> dict:
    out = {"point": float(point), **percentile_ci(reps, 0.95)}
    adj = percentile_ci(reps, BONFERRONI)
    out["adj_low"], out["adj_high"] = adj["ci_low"], adj["ci_high"]
    return out


def _decompose(c_pre: np.ndarray, c_post: np.ndarray) -> dict:
    """R, cosine, along and orthogonal parts; vectorised over a leading axis."""
    n_pre = np.linalg.norm(c_pre, axis=-1)
    n_post = np.linalg.norm(c_post, axis=-1)
    cos = (c_pre * c_post).sum(-1) / (n_pre * n_post)
    ratio = n_post / n_pre
    along = ratio * cos
    orth = ratio * np.sqrt(np.clip(1 - cos ** 2, 0, None))
    return {"R": ratio, "cosine": cos, "along": along, "orth": orth,
            "old_direction_share": cos ** 2}


# ----------------------------------------------------------- geometry --

def geometry(groups, b: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nA, nD = len(groups["A_est"]), len(groups["D_est"])
    cA, cD = _count_matrix(rng, nA, b), _count_matrix(rng, nD, b)
    out = {}
    for pooling, suffix in POOLINGS.items():
        per_layer = {}
        for layer in LAYERS:
            point_c, boot_c = {}, {}
            for stage in STAGES:
                x = _layer(stage, suffix, layer)
                xa, xd = x[groups["A_est"]], x[groups["D_est"]]
                point_c[stage] = xa.mean(0) - xd.mean(0)
                boot_c[stage] = cA @ xa / nA - cD @ xd / nD
            cell = {"start_norm": {s: float(np.linalg.norm(point_c[s])) for s in ("M1", "M1_alt")}}
            arms, arm_reps = {}, {}
            for arm, (pre, post) in ARMS.items():
                pt = _decompose(point_c[pre], point_c[post])
                bs = _decompose(boot_c[pre], boot_c[post])
                arm_reps[arm] = bs
                arms[arm] = {k: _ci(bs[k], pt[k]) for k in pt}
            cell["arms"] = arms
            recipe = {}
            for key in ("R", "orth", "along", "cosine"):
                d_a = arm_reps["dpo_alpaca"][key] - arm_reps["sft_alpaca"][key]
                d_d = arm_reps["dpo_dolly"][key] - arm_reps["sft_dolly"][key]
                p_a = arms["dpo_alpaca"][key]["point"] - arms["sft_alpaca"][key]["point"]
                p_d = arms["dpo_dolly"][key]["point"] - arms["sft_dolly"][key]["point"]
                recipe[key] = {
                    "pooled": _ci((d_a + d_d) / 2, (p_a + p_d) / 2),
                    "alpaca": _ci(d_a, p_a),
                    "dolly": _ci(d_d, p_d),
                    "branch_gap": _ci(d_a - d_d, p_a - p_d),
                }
            cell["recipe_dpo_minus_sft"] = recipe
            per_layer[str(layer)] = cell
            print(f"  geometry {pooling} L{layer}: done")
        out[pooling] = per_layer
    return out


def depth_profile() -> dict:
    """Point estimates at every layer (descriptive; no interval)."""
    meta, groups = _meta()
    out = {}
    for pooling, suffix in POOLINGS.items():
        arrays = {s: np.load(ACT_DIR / f"{s}_{suffix}.npy", mmap_mode="r") for s in STAGES}
        n_layers = arrays["M1"].shape[1]
        prof = {arm: {k: [] for k in ("R", "cosine", "along", "orth")} for arm in ARMS}
        for layer in range(n_layers):
            c = {}
            for s in STAGES:
                x = np.asarray(arrays[s][:, layer], dtype=np.float64)
                c[s] = x[groups["A_est"]].mean(0) - x[groups["D_est"]].mean(0)
            for arm, (pre, post) in ARMS.items():
                d = _decompose(c[pre], c[post])
                for k in prof[arm]:
                    prof[arm][k].append(float(d[k]))
        out[pooling] = prof
    return out


def rescaled_ratio(groups) -> dict:
    """R divided by the RMS row-norm ratio, per layer: the global-rescaling check."""
    out = {}
    for layer in LAYERS:
        cell = {}
        for arm, (pre, post) in list(ARMS.items())[:4]:
            xp, xq = _layer(pre, "final", layer), _layer(post, "final", layer)
            rms_pre = float(np.sqrt((np.linalg.norm(xp, axis=1) ** 2).mean()))
            rms_post = float(np.sqrt((np.linalg.norm(xq, axis=1) ** 2).mean()))
            c_pre = xp[groups["A_est"]].mean(0) - xp[groups["D_est"]].mean(0)
            c_post = xq[groups["A_est"]].mean(0) - xq[groups["D_est"]].mean(0)
            ratio = float(np.linalg.norm(c_post) / np.linalg.norm(c_pre))
            cell[arm] = {"rms_pre": rms_pre, "rms_post": rms_post,
                         "rms_ratio": rms_post / rms_pre, "contrast_ratio": ratio,
                         "contrast_over_rms": ratio / (rms_post / rms_pre)}
        out[str(layer)] = cell
    return out


# -------------------------------------------------------- update size --

def update_size(groups, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    est = np.concatenate([groups["A_est"], groups["D_est"]])
    nA = len(groups["A_est"])
    labels = np.zeros((n_perm, len(est)), dtype=bool)
    for k in range(n_perm):
        labels[k, rng.permutation(len(est))[:nA]] = True
    weights_a = labels / nA
    weights_d = (~labels) / (len(est) - nA)
    out = {}
    for pooling, suffix in POOLINGS.items():
        per_layer = {}
        for layer in LAYERS:
            cell = {}
            for arm, (pre, post) in ARMS.items():
                xp, xq = _layer(pre, suffix, layer), _layer(post, suffix, layer)
                drift = xq - xp
                start_norm = np.linalg.norm(xp[groups["A_est"]].mean(0) - xp[groups["D_est"]].mean(0))
                obs = np.linalg.norm(drift[groups["A_est"]].mean(0)
                                     - drift[groups["D_est"]].mean(0)) / start_norm
                de = drift[est]
                null = np.linalg.norm(weights_a @ de - weights_d @ de, axis=1) / start_norm
                row_drift = np.linalg.norm(drift, axis=1)
                cell[arm] = {
                    "contrast_change": float(obs),
                    "null_mean": float(null.mean()),
                    "null_q95": float(np.quantile(null, 0.95)),
                    "null_q99": float(np.quantile(null, 0.99)),
                    "ratio_to_null_q95": float(obs / np.quantile(null, 0.95)),
                    "p_perm": float((1 + (null >= obs).sum()) / (1 + n_perm)),
                    "drift_mean": float(row_drift.mean()),
                    "relative_drift_mean": float((row_drift / np.linalg.norm(xp, axis=1)).mean()),
                    "drift_A_over_D": float(row_drift[groups["A_est"]].mean()
                                            / row_drift[groups["D_est"]].mean()),
                    "cka": float(linear_cka(xp, xq)),
                }
            per_layer[str(layer)] = cell
            print(f"  update size {pooling} L{layer}: done")
        out[pooling] = per_layer
    return out


def weight_change() -> dict:
    """Exact merged-weight change per arm from the cached LoRA factors."""
    from src.seedrep.extract import update_norms
    from src.stagecmp.weights import difference, load_adapter, parse_adapter
    from src.training.model import STAGE_ADAPTER_CHAINS

    repos = sorted({r for pre, post in ARMS.values()
                    for s in (pre, post) for r in STAGE_ADAPTER_CHAINS[s]})
    adapters = {r: parse_adapter(load_adapter(r)) for r in repos}
    return {arm: update_norms(difference(pre, post, adapters)) for arm, (pre, post) in ARMS.items()}


# -------------------------------------------------------- separability --

def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def separability(groups, b: int, seed: int) -> dict:
    stages = ["M1", "M2", "M3_direct", "M1_alt", "M2_alt", "M3_direct_alt"]
    order = ["A_est", "D_est", "A_ho", "D_ho", "B", "C"]
    rng = np.random.default_rng(seed + 2)
    draws = [{g: rng.choice(groups[g], size=len(groups[g]), replace=True) for g in order}
             for _ in range(b)]

    def stats(x, idx):
        c = x[idx["A_est"]].mean(0) - x[idx["D_est"]].mean(0)
        u = c / np.linalg.norm(c)
        m_d = x[idx["D_est"]].mean(0) @ u
        span = x[idx["A_est"]].mean(0) @ u - m_d
        pa, pd = x[idx["A_ho"]] @ u, x[idx["D_ho"]] @ u
        pb, pc = x[idx["B"]] @ u, x[idx["C"]] @ u
        pooled_sd = np.sqrt((pa.var(ddof=1) + pd.var(ddof=1)) / 2)
        return {"d_prime_heldout": (pa.mean() - pd.mean()) / pooled_sd,
                "auroc_C_vs_B": _auroc(pc, pb),
                "z_B": (pb.mean() - m_d) / span, "z_C": (pc.mean() - m_d) / span}

    out = {}
    for pooling, suffix in POOLINGS.items():
        per_layer = {}
        for layer in LAYERS:
            pts, reps = {}, {}
            for s in stages:
                x = _layer(s, suffix, layer)
                pts[s] = stats(x, groups)
                rows = [stats(x, d) for d in draws]
                reps[s] = {k: np.array([r[k] for r in rows]) for k in pts[s]}
            cell = {s: {k: _ci(reps[s][k], pts[s][k]) for k in pts[s]} for s in stages}
            for branch, (start, sft, dpo) in {"alpaca": ("M1", "M2", "M3_direct"),
                                              "dolly": ("M1_alt", "M2_alt", "M3_direct_alt")}.items():
                cell[f"dpo_minus_sft_{branch}"] = {
                    k: _ci(reps[dpo][k] - reps[sft][k], pts[dpo][k] - pts[sft][k]) for k in pts[dpo]}
                cell[f"dpo_minus_start_{branch}"] = {
                    k: _ci(reps[dpo][k] - reps[start][k], pts[dpo][k] - pts[start][k]) for k in pts[dpo]}
            per_layer[str(layer)] = cell
            print(f"  separability {pooling} L{layer}: done")
        out[pooling] = per_layer
    return out


# --------------------------------------------------------------- main --

def _write(name: str, payload: dict, inputs: list[Path]) -> None:
    target = OUT_DIR / f"{name}.json"
    write_json_lf(target, {"status": STATUS_EXPLORATORY, "produced_utc": _now(), **payload})
    write_json_lf(target.with_name(f"{name}_binding.json"), {
        "status": STATUS_EXPLORATORY, "produced_utc": _now(),
        "sha256": {str(p): sha256_file(p) for p in inputs},
    })
    print(f"wrote {target}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=10_000)
    ap.add_argument("--b-separability", type=int, default=2_000)
    ap.add_argument("--only", nargs="*", default=None,
                    help="geometry update_size separability (default: all)")
    args = ap.parse_args()
    meta, groups = _meta()
    _assert_row_order(meta)
    inputs = [ACT_DIR / f"{s}_{suf}.npy" for s in STAGES for suf in POOLINGS.values()]
    want = set(args.only or ["geometry", "update_size", "separability"])
    config = {"b": args.b, "seed": BOOTSTRAP_SEED, "layers": list(LAYERS),
              "arms": {k: list(v) for k, v in ARMS.items()},
              "group_sizes": {g: int(len(v)) for g, v in groups.items()}}

    if "geometry" in want:
        _write("contrast_geometry", {
            "config": config,
            "by_pooling": geometry(groups, args.b, BOOTSTRAP_SEED),
            "depth_profile": depth_profile(),
            "rms_rescaled_final_token": rescaled_ratio(groups),
        }, inputs)
    if "update_size" in want:
        _write("update_size", {
            "config": {**config, "n_permutations": args.b},
            "weight_change": weight_change(),
            "by_pooling": update_size(groups, args.b, BOOTSTRAP_SEED),
        }, inputs)
    if "separability" in want:
        _write("separability", {
            "config": {**config, "b": args.b_separability},
            "by_pooling": separability(groups, args.b_separability, BOOTSTRAP_SEED),
        }, inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
