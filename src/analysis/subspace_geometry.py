"""H1/H2 subspace geometry (WP-Geom), per docs/audit/analysis_plan.md §4.

CPU-only, operates on ``_final`` activation arrays + metadata. Primary H1/H2
update quantity is the *un-normalised* A-D contrast difference
``Delta_AD^l = c_M3^l - c_M2^l`` with ``c_s^l = m_A^{s,l} - m_D^{s,l}`` on the
``direction_estimation`` split. The centered A/D subspace ``U_s`` (used ONLY
here) is the top-r right singular vectors of the equal-group-weighted centered
A_est u D_est activations - the top-*variance* subspace of those activations,
NOT by construction a subspace of the contrast.

Reported quantities:
  * ``rho_AD_perp^l`` = ||Delta_AD_perp||^2 / ||Delta_AD||^2  (orthogonal update
    fraction, primary; Delta_AD_perp = Delta_AD - U_M2 U_M2^T Delta_AD)
  * principal angles between U_M2 and U_M3 (mean, max, degrees)
  * participation ratio / effective rank trajectory
  * global-drift diagnostic ``mu_M3 - mu_M2`` and its own orthogonal fraction
    (SECONDARY - not H1/H2 evidence)
  * leading-vs-orthogonal explained variance of d inside U_s
"""
from __future__ import annotations

import numpy as np

R_PRIMARY = 5
R_SENSITIVITY = 10
DIRECTION_SPLIT = "direction_estimation"


def _ad_estimation_masks(quadrants, splits):
    est = np.array([str(s) == DIRECTION_SPLIT for s in splits])
    return (quadrants == "A") & est, (quadrants == "D") & est


def group_means(pooled_layer, a_mask, d_mask):
    return pooled_layer[a_mask].mean(0), pooled_layer[d_mask].mean(0)


def contrast(pooled_layer, a_mask, d_mask):
    """c^l = m_A - m_D (un-normalised)."""
    m_a, m_d = group_means(pooled_layer, a_mask, d_mask)
    return m_a - m_d


def midpoint(pooled_layer, a_mask, d_mask):
    """mu^l = 1/2 (m_A + m_D) - the global-drift diagnostic reference."""
    m_a, m_d = group_means(pooled_layer, a_mask, d_mask)
    return 0.5 * (m_a + m_d)


def centered_ad_subspace(pooled_layer, a_mask, d_mask, r=R_PRIMARY):
    """U (hidden, r) + singular values. Equal group weights: rows of A_est get
    1/(2 n_A), rows of D_est get 1/(2 n_D); centre by mu = 1/2 (m_A + m_D)."""
    m_a, m_d = group_means(pooled_layer, a_mask, d_mask)
    mu = 0.5 * (m_a + m_d)
    n_a, n_d = int(a_mask.sum()), int(d_mask.sum())
    rows, weights = [], []
    for mask, n in ((a_mask, n_a), (d_mask, n_d)):
        w = 1.0 / (2 * n)
        for h in pooled_layer[mask]:
            rows.append(h - mu)
            weights.append(np.sqrt(w))
    x_tilde = np.asarray(rows) * np.asarray(weights)[:, None]
    # SVD: right singular vectors are the subspace basis
    _u, sv, vt = np.linalg.svd(x_tilde, full_matrices=False)
    r = min(r, vt.shape[0])
    return vt[:r].T, sv


def project_out(subspace_U, vec):
    """vec - U U^T vec."""
    return vec - subspace_U @ (subspace_U.T @ vec)


def orthogonal_update_fraction(c_m2, c_m3, subspace_U_m2):
    """rho_AD_perp = ||Delta_perp||^2 / ||Delta||^2, Delta = c_M3 - c_M2,
    Delta_perp = Delta - U_M2 U_M2^T Delta."""
    delta = c_m3 - c_m2
    denom = float(delta @ delta)
    if denom <= 1e-24:
        return {"rho_AD_perp": None, "delta_norm": float(np.sqrt(denom))}
    delta_perp = project_out(subspace_U_m2, delta)
    return {
        "rho_AD_perp": float((delta_perp @ delta_perp) / denom),
        "delta_norm": float(np.sqrt(denom)),
        "delta_perp_norm": float(np.linalg.norm(delta_perp)),
    }


def principal_angles_deg(subspace_a, subspace_b):
    from scipy.linalg import subspace_angles

    angles = np.degrees(subspace_angles(subspace_a, subspace_b))
    return {"mean_deg": float(angles.mean()), "max_deg": float(angles.max()),
            "per_component_deg": [float(a) for a in angles]}


def participation_ratio(singular_values):
    s2 = np.asarray(singular_values, float) ** 2
    if s2.sum() <= 0:
        return None
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def effective_rank(singular_values):
    s2 = np.asarray(singular_values, float) ** 2
    total = s2.sum()
    if total <= 0:
        return None
    p = s2 / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def leading_vs_orthogonal_variance(unit_d, subspace_U, singular_values):
    """Component of the unit contrast direction inside U_s, and the
    explained-variance fraction along that in-subspace component vs the rest
    of the r-subspace."""
    coeffs = subspace_U.T @ unit_d                    # (r,)
    in_sub = subspace_U @ coeffs
    in_norm = float(np.linalg.norm(in_sub))
    s2 = np.asarray(singular_values, float)[: subspace_U.shape[1]] ** 2
    if s2.sum() <= 0 or in_norm <= 1e-12:
        return {"d_in_subspace_norm": in_norm, "leading_var_fraction": None}
    q1 = in_sub / in_norm
    var_along = float(np.sum((s2 * (subspace_U.T @ q1) ** 2)))
    return {
        "d_in_subspace_norm": in_norm,
        "d_out_of_subspace_norm": float(np.linalg.norm(unit_d - in_sub)),
        "leading_var_fraction": var_along / float(s2.sum()),
    }


def layer_report(pooled_m2, pooled_m3, quadrants, splits, layer, r=R_PRIMARY):
    """Full per-layer H1/H2 geometry for the M2->M3 transition."""
    a2, d2 = _ad_estimation_masks(quadrants, splits)
    # M2 and M3 score the identical fixed benchmark -> same masks
    c_m2 = contrast(pooled_m2[:, layer], a2, d2)
    c_m3 = contrast(pooled_m3[:, layer], a2, d2)
    u_m2, sv_m2 = centered_ad_subspace(pooled_m2[:, layer], a2, d2, r=r)
    u_m3, sv_m3 = centered_ad_subspace(pooled_m3[:, layer], a2, d2, r=r)

    mu_m2 = midpoint(pooled_m2[:, layer], a2, d2)
    mu_m3 = midpoint(pooled_m3[:, layer], a2, d2)
    drift = mu_m3 - mu_m2
    drift_perp = project_out(u_m2, drift)
    drift_denom = float(drift @ drift)

    unit_d_m3 = c_m3 / (np.linalg.norm(c_m3) or 1.0)

    return {
        "layer": int(layer),
        "r": int(r),
        "orthogonal_update": orthogonal_update_fraction(c_m2, c_m3, u_m2),
        "principal_angles_M2_vs_M3": principal_angles_deg(u_m2, u_m3),
        "participation_ratio": {"M2": participation_ratio(sv_m2),
                                "M3": participation_ratio(sv_m3)},
        "effective_rank": {"M2": effective_rank(sv_m2), "M3": effective_rank(sv_m3)},
        "leading_vs_orthogonal_variance_M3": leading_vs_orthogonal_variance(
            unit_d_m3, u_m3, sv_m3
        ),
        "global_drift_diagnostic": {
            "note": "SECONDARY - not H1/H2 evidence (analysis_plan.md §4)",
            "drift_norm": float(np.sqrt(max(drift_denom, 0.0))),
            "drift_orthogonal_fraction": (
                None if drift_denom <= 1e-24
                else float((drift_perp @ drift_perp) / drift_denom)
            ),
        },
        "contrast_norm": {"M2": float(np.linalg.norm(c_m2)),
                          "M3": float(np.linalg.norm(c_m3))},
    }


def bootstrap_rho_ad_perp(
    pooled_m2, pooled_m3, quadrants, splits, layer, *, r=R_PRIMARY,
    n_boot=10000, seed=20260904,
):
    """Prompt-level bootstrap CI (percentile) on rho_AD_perp by resampling
    A_est and D_est jointly by prompt."""
    a2, d2 = _ad_estimation_masks(quadrants, splits)
    a_idx = np.flatnonzero(a2)
    d_idx = np.flatnonzero(d2)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        a_s = rng.choice(a_idx, size=len(a_idx), replace=True)
        d_s = rng.choice(d_idx, size=len(d_idx), replace=True)
        m2l, m3l = pooled_m2[:, layer], pooled_m3[:, layer]
        c_m2 = m2l[a_s].mean(0) - m2l[d_s].mean(0)
        c_m3 = m3l[a_s].mean(0) - m3l[d_s].mean(0)
        # subspace re-estimated on the resample (M2)
        full_a = np.zeros(len(quadrants), bool); full_a[a_s] = True
        full_d = np.zeros(len(quadrants), bool); full_d[d_s] = True
        try:
            u_m2, _ = centered_ad_subspace(m2l, full_a, full_d, r=r)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        res = orthogonal_update_fraction(c_m2, c_m3, u_m2)
        if res["rho_AD_perp"] is not None:
            vals.append(res["rho_AD_perp"])
    vals = np.asarray(vals)
    return {
        "n_boot": int(len(vals)),
        "seed": seed,
        "interval": "percentile",
        "mean": float(vals.mean()) if len(vals) else None,
        "ci_low": float(np.percentile(vals, 2.5)) if len(vals) else None,
        "ci_high": float(np.percentile(vals, 97.5)) if len(vals) else None,
    }


def main():  # pragma: no cover - [exec:T4], needs regenerated 654-row activations
    """Run the M2->M3 subspace geometry across all layers, r=5 (primary) and
    r=10 (sensitivity). Needs ``results/activations/{M2,M3}_final.npy``."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--out", default="results/interpretability/subspace_geometry.json")
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    args = parser.parse_args()

    act = Path(args.act_dir)
    def _load(stage):
        arr = np.load(act / f"{stage}_final.npy")
        meta = json.loads((act / f"{stage}_metadata.json").read_text(encoding="utf-8"))
        return arr, np.array([r["quadrant"] for r in meta]), np.array([r.get("split") or "" for r in meta])

    m2, quads, splits = _load("M2")
    m3, _q, _s = _load("M3")
    layers = args.layers or list(range(m2.shape[1]))
    out = {"r_primary": R_PRIMARY, "r_sensitivity": R_SENSITIVITY, "per_layer": {}}
    for r in (R_PRIMARY, R_SENSITIVITY):
        out["per_layer"][f"r{r}"] = [
            layer_report(m2, m3, quads, splits, l, r=r) for l in layers
        ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
