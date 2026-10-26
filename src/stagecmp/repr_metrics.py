"""Representational distance between two checkpoints on the identical 654 rows.

Which metric answers which question, and which are safe to compare across
layers:

===========================  ====================================  ============
metric                       question                              comparable?
===========================  ====================================  ============
linear CKA                   same relational geometry?             yes, bounded
cos(d_pre, d_post)           did the A-D axis rotate?              yes, bounded
principal angles             did the top-variance subspace turn?   yes, symmetric
rho_AD_perp                  is the update NEW structure?          yes, ASYMMETRIC
contrast norm ratio          did the contrast grow?                yes, a ratio
drift ||h_post - h_pre||     how far did each prompt move?         NO - see below
relative drift               move relative to the stream           yes
quadrant selectivity         is the movement safety-relevant?      yes
===========================  ====================================  ============

Two cautions the report carries in-file:

* **raw drift is not comparable across layers or stages.** Residual-stream norms
  grow with depth, so a bigger number deeper in the network means very little on
  its own. Use the relative form, or the A-minus-D selectivity, where the common
  growth cancels.
* **CKA and cosine are bounded and saturate.** Published work finds activation
  cosine staying above 0.96 through SFT while sparse-autoencoder latents for the
  same layers fall to 0.557, and linear CKA above 0.998 across RL stages. A
  matrix of near-ceiling CKA says nothing on its own. ``saturation_check`` runs
  first, on a handful of pairs spanning the expected range, and reports whether
  these metrics have enough spread to be worth computing everywhere.
"""
from __future__ import annotations

import numpy as np

from src.analysis.subspace_geometry import (
    centered_ad_subspace,
    contrast,
    orthogonal_update_fraction,
    principal_angles_deg,
)
from src.common.activations import l2_normalize
from src.common.quadrants import QUADRANTS
from src.common.stats import (
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    bootstrap_nonlinear_contrast,
)
from src.stagecmp import STATUS_EXPLORATORY
from src.stagecmp.loaders import load_stage_arrays, stage_direction
from src.stagecmp.pairs import HEADLINE_PAIRS, pair_key

DIRECTION_SPLIT = "direction_estimation"
R_PRIMARY = 5
#: layers carrying bootstrap intervals. Fixed in advance and deliberately
#: small: this is the grid the 36-pair matrix already uses, so it is not
#: chosen from the results, and four intervals per pair does not invite the
#: multiplicity misreading that 29 would.
CI_LAYERS = (16, 20, 24, 28)
#: pairs spanning the expected range, for the saturation pre-check
SATURATION_PAIRS = (
    ("M0", "M1"),            # instruction tuning: expected large
    ("M2", "M3"),            # mediated DPO: expected small
    ("M1", "M2"),            # safety SFT
    ("M1", "M3_direct"),     # direct DPO
)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA on column-centred matrices, ``(n_samples, n_features)``.

    Invariant to rotation, isotropic scaling and translation; dominated by the
    leading principal components, which is exactly why it can sit near 1 while a
    decision boundary has moved. Never report it alone.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = x.T @ y
    num = float(np.sum(xty * xty))
    xtx = x.T @ x
    yty = y.T @ y
    den = float(np.sqrt(np.sum(xtx * xtx) * np.sum(yty * yty)))
    return num / den if den > 1e-30 else float("nan")


def _masks(meta):
    quads = np.array([r["quadrant"] for r in meta])
    splits = np.array([r.get("split") or "" for r in meta])
    est = splits == DIRECTION_SPLIT
    return quads, (quads == "A") & est, (quads == "D") & est


def pair_report(
    pre: str,
    post: str,
    act_dir,
    directions_dir,
    pooling: str = "final_token",
    layers=None,
    with_subspace: bool = True,
) -> dict:
    """Every per-layer representational quantity for one ordered pair."""
    a_pre, meta = load_stage_arrays(pre, pooling, act_dir)
    a_post, _ = load_stage_arrays(post, pooling, act_dir)
    if a_pre.shape != a_post.shape:
        raise ValueError(f"{pre}/{post} shape mismatch: {a_pre.shape} vs {a_post.shape}")
    quads, a_mask, d_mask = _masks(meta)

    d_pre = stage_direction(pre, directions_dir)
    d_post = stage_direction(post, directions_dir)

    n_layers = a_pre.shape[1]
    layers = list(layers) if layers is not None else list(range(n_layers))

    diffs = a_post - a_pre                       # (n, L, hidden)
    per_layer = []
    for l in layers:
        xp, xq = a_pre[:, l], a_post[:, l]
        dl = diffs[:, l]

        drift = np.linalg.norm(dl, axis=1)
        norm_pre = np.linalg.norm(xp, axis=1)
        rel = drift / np.where(norm_pre > 1e-12, norm_pre, np.nan)

        c_pre = contrast(xp, a_mask, d_mask)
        c_post = contrast(xq, a_mask, d_mask)
        n_pre = float(np.linalg.norm(c_pre))
        n_post = float(np.linalg.norm(c_post))

        row = {
            "layer": int(l),
            "cka": linear_cka(xp, xq),
            "direction_cosine": float(l2_normalize(d_pre[l]) @ l2_normalize(d_post[l])),
            "contrast_norm": {"pre": n_pre, "post": n_post,
                              "ratio": n_post / n_pre if n_pre > 1e-12 else float("nan")},
            "drift_mean": float(drift.mean()),
            "relative_drift_mean": float(np.nanmean(rel)),
            "drift_by_quadrant": {q: float(drift[quads == q].mean()) for q in QUADRANTS},
            "relative_drift_by_quadrant": {
                q: float(np.nanmean(rel[quads == q])) for q in QUADRANTS},
        }
        dq = row["drift_by_quadrant"]
        row["selectivity"] = {
            "A_minus_D": dq["A"] - dq["D"],
            "C_minus_B": dq["C"] - dq["B"],
        }

        if with_subspace:
            u_pre, sv_pre = centered_ad_subspace(xp, a_mask, d_mask, r=R_PRIMARY)
            u_post, _ = centered_ad_subspace(xq, a_mask, d_mask, r=R_PRIMARY)
            row["rho_AD_perp"] = orthogonal_update_fraction(c_pre, c_post, u_pre)
            row["principal_angles"] = principal_angles_deg(u_pre, u_post)

        per_layer.append(row)

    return {
        "status": STATUS_EXPLORATORY,
        "pre": pre,
        "post": post,
        "pooling": pooling,
        "n_rows": int(a_pre.shape[0]),
        "per_layer": per_layer,
        "reading": (
            "rho_AD_perp is ASYMMETRIC: it is measured against the PRE stage's "
            "own subspace, so swapping pre and post gives a different number. "
            "Raw drift grows with depth and is not comparable across layers; "
            "read relative_drift or the selectivity gaps instead."
        ),
        "direction_cosine_is_pooling_independent": (
            "resolve_direction_path() takes no pooling argument and returns "
            "{stage}_direction_654.npy, the MEAN-POOLED direction, because the "
            "final-token directions exist for only 4 of the 9 stages. So "
            "direction_cosine is identical under both pooling settings and is "
            "NOT matched to the activations analysed here. Every other quantity "
            "in this report is computed from the activations directly and does "
            "honour the pooling. To make direction_cosine pooling-matched, "
            "generate final-token directions for the remaining five stages "
            "first (CPU, no model load)."
        ),
    }


def ratio_confidence_intervals(
    pre: str,
    post: str,
    act_dir,
    pooling: str = "final_token",
    layers=CI_LAYERS,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap intervals for the SCALE-FREE quantities, at ``CI_LAYERS``.

    Three ratios, each immune to the objection that a difference between two
    transitions is really a difference in how much the transition moved
    anything at all:

    * ``contrast_norm_ratio`` -- ||c_post|| / ||c_pre||, on the A/D
      direction-estimation rows, exactly as ``pair_report`` computes it.
    * ``drift_ratio_A_over_D`` and ``drift_ratio_C_over_B`` -- mean per-prompt
      displacement on one quadrant over another. A transition that moves every
      prompt equally scores 1.0 whatever its magnitude, so these separate
      anisotropic movement from a large isotropic one.

    Row indices are resampled, not values, so the pre and post checkpoints are
    always read on the same resampled prompts -- see
    ``contrasts.bootstrap_nonlinear_contrast``.
    """
    a_pre, meta = load_stage_arrays(pre, pooling, act_dir)
    a_post, _ = load_stage_arrays(post, pooling, act_dir)
    if a_pre.shape != a_post.shape:
        raise ValueError(f"{pre}/{post} shape mismatch: {a_pre.shape} vs {a_post.shape}")
    quads, a_mask, d_mask = _masks(meta)
    a_est, d_est = np.flatnonzero(a_mask), np.flatnonzero(d_mask)

    # The grid is fixed for the real 29-layer stack; toy fixtures are shallower,
    # so drop layers that do not exist rather than indexing past the end.
    n_layers = a_pre.shape[1]
    layers = [int(l) for l in layers if 0 <= int(l) < n_layers]

    out = {}
    for l in layers:
        xp, xq = a_pre[:, l], a_post[:, l]
        drift = np.linalg.norm(a_post[:, l] - a_pre[:, l], axis=1)

        def contrast_ratio(g, xp=xp, xq=xq):
            pre_norm = float(np.linalg.norm(xp[g[0]].mean(0) - xp[g[1]].mean(0)))
            if pre_norm <= 1e-12:
                return float("nan")
            return float(np.linalg.norm(xq[g[0]].mean(0) - xq[g[1]].mean(0)) / pre_norm)

        cells = {
            "contrast_norm_ratio": bootstrap_nonlinear_contrast(
                [a_est, d_est], contrast_ratio, b=b, seed=seed),
        }
        for name, (q_num, q_den) in (("drift_ratio_A_over_D", ("A", "D")),
                                     ("drift_ratio_C_over_B", ("C", "B"))):
            def drift_ratio(g, drift=drift):
                den = float(drift[g[1]].mean())
                if den <= 1e-12:
                    return float("nan")
                return float(drift[g[0]].mean() / den)

            cells[name] = bootstrap_nonlinear_contrast(
                [np.flatnonzero(quads == q_num), np.flatnonzero(quads == q_den)],
                drift_ratio, b=b, seed=seed)
        out[str(int(l))] = cells

    return {
        "status": STATUS_EXPLORATORY,
        "pre": pre,
        "post": post,
        "pooling": pooling,
        "layers": [int(l) for l in layers],
        "by_layer": out,
        "reading": (
            "These are RATIOS, so a transition that moves everything equally "
            "scores 1.0 regardless of magnitude. That is the point: it "
            "separates 'the harmful/benign structure changed' from 'this "
            "transition simply moved the model further'. Intervals are "
            "percentile, from a shared resample of the two row groups."
        ),
    }


def saturation_check(
    act_dir,
    directions_dir,
    pooling: str = "final_token",
    layers=(16, 20, 24, 28),
    threshold: float = 0.02,
) -> dict:
    """Do the bounded metrics have enough spread to be worth computing everywhere?

    Runs CKA and direction cosine on pairs chosen to span the expected range. If
    the spread across those pairs is below ``threshold``, the metric is at its
    ceiling and carries no information; it is then reported as a control rather
    than built into a full matrix.
    """
    out = {"status": STATUS_EXPLORATORY, "pooling": pooling,
           "threshold": threshold, "layers": list(layers), "pairs": {}}
    cka_vals = {l: [] for l in layers}
    cos_vals = {l: [] for l in layers}

    for pre, post in SATURATION_PAIRS:
        rep = pair_report(pre, post, act_dir, directions_dir, pooling,
                          layers=layers, with_subspace=False)
        cells = {}
        for row in rep["per_layer"]:
            l = row["layer"]
            cells[l] = {"cka": row["cka"],
                        "direction_cosine": row["direction_cosine"],
                        "contrast_norm_ratio": row["contrast_norm"]["ratio"]}
            cka_vals[l].append(row["cka"])
            cos_vals[l].append(row["direction_cosine"])
        out["pairs"][pair_key(pre, post)] = cells

    verdict = {}
    for l in layers:
        c, k = np.array(cos_vals[l]), np.array(cka_vals[l])
        verdict[int(l)] = {
            "cka_min": float(k.min()), "cka_max": float(k.max()),
            "cka_spread": float(k.max() - k.min()),
            "cosine_min": float(c.min()), "cosine_max": float(c.max()),
            "cosine_spread": float(c.max() - c.min()),
        }
    max_cka = max(v["cka_spread"] for v in verdict.values())
    max_cos = max(v["cosine_spread"] for v in verdict.values())
    out["verdict"] = {
        "per_layer": verdict,
        "max_cka_spread": max_cka,
        "max_cosine_spread": max_cos,
        "cka_has_range": bool(max_cka >= threshold),
        "cosine_has_range": bool(max_cos >= threshold),
        "reading": (
            "If cka_has_range is false, linear CKA is at its ceiling on these "
            "checkpoints and a full 9x9 CKA matrix would be decorative. That is "
            "itself reportable: global geometry preserved while the "
            "safety-specific contrast changes is the inherited-not-created "
            "claim, provided it sits beside metrics that demonstrably move."
        ),
    }
    return out


def headline_pairs(
    act_dir,
    directions_dir,
    pooling: str = "final_token",
    pairs=HEADLINE_PAIRS,
    with_ci: bool = True,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Full-depth report for the eight pre-named pairs.

    Point estimates at every layer; bootstrap intervals for the scale-free
    ratios at ``CI_LAYERS`` only. The ratios are what any cross-transition
    comparison has to rest on, so they are the quantities that get intervals.
    """
    out = {}
    for pre, post in pairs:
        rep = pair_report(pre, post, act_dir, directions_dir, pooling)
        if with_ci:
            rep["ratio_cis"] = ratio_confidence_intervals(
                pre, post, act_dir, pooling, b=b, seed=seed)
        out[pair_key(pre, post)] = rep
    return {
        "status": STATUS_EXPLORATORY,
        "pooling": pooling,
        "ci_layers": list(CI_LAYERS) if with_ci else None,
        "pairs": out,
    }
