"""Does d_AD track harmfulness, or surface cues? (2x2 factorial audit)

The benchmark is a 2x2: harmful x surface-cue-strength.

           | cue-strong        | cue-reduced
  harmful  | A (HarmBench)     | C (reduced-cue rewrites)
  benign   | B (XSTest)        | D (Alpaca/Dolly/OASST1)

The preregistered direction is the DIAGONAL of that square,
``d_AD = mu_A - mu_D``, so it confounds the two factors. Algebraically it
decomposes exactly:

    d_H  = 1/2[(mu_A - mu_B) + (mu_C - mu_D)]     harmfulness main effect
    d_S  = 1/2[(mu_A - mu_C) + (mu_B - mu_D)]     surface-cue main effect
    d_HS = 1/2[(mu_A - mu_C) - (mu_B - mu_D)]     interaction
    d_AD = d_H + d_S                              (exact; asserted below)

If d_AD aligns mostly with d_H it is a harmfulness contrast; if mostly with
d_S it is largely a cue/register detector that happens to correlate with
harm on this benchmark. This is a REPRESENTATION AUDIT ONLY - it does not
replace d_AD, which stays the preregistered causal handle (analysis_plan.md
fixes it before generation). Nothing here is used to pick a new direction.

Cell-size note: A=150 B=250 C=104 D=150 are unequal, so cell means are
equally weighted (a balanced factorial estimand). Counts are reported.

CPU only. Needs results/activations/{stage}_final.npy + {stage}_metadata.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACT_DIR = Path("results/activations")
STAGES = ["M0", "M1", "M2", "M3", "M3_direct", "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt"]
LAYER = 24

# quadrant -> (harmful, cue_strong)
FACTORS = {"A": (1, 1), "B": (0, 1), "C": (1, 0), "D": (0, 0)}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def cohens_d(x, y):
    """Pooled-SD effect size between two 1-D samples of projections."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    s = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / s) if s > 0 else None


def load_stage(stage):
    meta = json.loads((ACT_DIR / f"{stage}_metadata.json").read_text(
        encoding="utf-8", errors="replace"))
    arr = np.load(ACT_DIR / f"{stage}_final.npy")
    return meta, arr


def audit_stage(stage, ad_rows="est", layer=LAYER):
    """ad_rows: 'est' mirrors how d_AD is really built (A/D restricted to the
    direction_estimation split; B/C have no split so are used whole - an
    asymmetry worth stating). 'all' treats every cell the same."""
    meta, arr = load_stage(stage)
    q = np.array([r["quadrant"] for r in meta])
    sp = np.array([r.get("split") for r in meta])

    idx = {}
    for cell in "ABCD":
        m = q == cell
        if ad_rows == "est" and cell in ("A", "D"):
            m &= sp == "direction_estimation"
        idx[cell] = np.where(m)[0]
    if any(len(v) == 0 for v in idx.values()):
        return {"stage": stage, "error": f"empty cell(s): "
                f"{ {k: len(v) for k, v in idx.items()} }"}

    mu = {c: arr[idx[c]].mean(axis=0) for c in "ABCD"}      # each (n_layers, hidden)
    L = layer
    A, B, C, D = (mu[c][L] for c in "ABCD")

    d_AD = A - D
    d_H = 0.5 * ((A - B) + (C - D))
    d_S = 0.5 * ((A - C) + (B - D))
    d_HS = 0.5 * ((A - C) - (B - D))

    decomposition_residual = float(np.linalg.norm(d_H + d_S - d_AD))

    # how much of ||d_AD||^2 each main effect contributes (they are not
    # orthogonal, so the cross term is reported rather than hidden)
    nH2, nS2 = float(d_H @ d_H), float(d_S @ d_S)
    cross = float(2 * d_H @ d_S)

    # projections of every prompt (all rows, not just the estimation half)
    proj = {}
    for name, d in (("d_AD", d_AD), ("d_H", d_H), ("d_S", d_S)):
        u = _unit(d)
        p = arr[:, L, :] @ u
        proj[name] = {c: p[np.where(q == c)[0]] for c in "ABCD"}

    def sep(name, pos, neg):
        p = proj[name]
        return cohens_d(np.concatenate([p[c] for c in pos]),
                        np.concatenate([p[c] for c in neg]))

    return {
        "stage": stage,
        "layer": L,
        "ad_rows": ad_rows,
        "cell_counts": {c: int(len(idx[c])) for c in "ABCD"},
        "decomposition_residual_should_be_0": decomposition_residual,
        "alignment_of_preregistered_d_AD": {
            "cos_with_d_H_harmfulness": _cos(d_AD, d_H),
            "cos_with_d_S_surface_cue": _cos(d_AD, d_S),
            "cos_with_d_HS_interaction": _cos(d_AD, d_HS),
        },
        "magnitudes": {
            "norm_d_AD": float(np.linalg.norm(d_AD)),
            "norm_d_H": float(np.linalg.norm(d_H)),
            "norm_d_S": float(np.linalg.norm(d_S)),
            "norm_d_HS": float(np.linalg.norm(d_HS)),
            "cos_d_H_vs_d_S": _cos(d_H, d_S),
            "sq_norm_share": {"d_H": nH2, "d_S": nS2, "cross_term": cross},
        },
        "separation_cohens_d": {
            # does each direction separate the factor it is supposed to?
            "d_AD__harmful(AC)_vs_benign(BD)": sep("d_AD", "AC", "BD"),
            "d_AD__cuestrong(AB)_vs_cuereduced(CD)": sep("d_AD", "AB", "CD"),
            "d_H__harmful(AC)_vs_benign(BD)": sep("d_H", "AC", "BD"),
            "d_H__cuestrong(AB)_vs_cuereduced(CD)": sep("d_H", "AB", "CD"),
            "d_S__harmful(AC)_vs_benign(BD)": sep("d_S", "AC", "BD"),
            "d_S__cuestrong(AB)_vs_cuereduced(CD)": sep("d_S", "AB", "CD"),
        },
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--ad-rows", default="est", choices=["est", "all"])
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--out", default="results/interpretability/factorial_direction_audit.json")
    args = ap.parse_args()

    report = {"layer": args.layer, "ad_rows": args.ad_rows,
              "factor_map": {k: {"harmful": v[0], "cue_strong": v[1]}
                             for k, v in FACTORS.items()},
              "per_stage": {}}
    print(f"{'stage':15s} {'cos(dAD,dH)':>12s} {'cos(dAD,dS)':>12s} "
          f"{'|dH|':>8s} {'|dS|':>8s} {'dAD sep harm':>13s} {'dAD sep cue':>12s}")
    for st in args.stages:
        try:
            r = audit_stage(st, ad_rows=args.ad_rows, layer=args.layer)
        except FileNotFoundError as e:
            r = {"stage": st, "error": f"missing activation file: {e}"}
        report["per_stage"][st] = r
        if "error" in r:
            print(f"{st:15s} SKIP  {r['error']}")
            continue
        a = r["alignment_of_preregistered_d_AD"]
        m = r["magnitudes"]
        s = r["separation_cohens_d"]
        print(f"{st:15s} {a['cos_with_d_H_harmfulness']:12.4f} "
              f"{a['cos_with_d_S_surface_cue']:12.4f} "
              f"{m['norm_d_H']:8.2f} {m['norm_d_S']:8.2f} "
              f"{s['d_AD__harmful(AC)_vs_benign(BD)']:13.3f} "
              f"{s['d_AD__cuestrong(AB)_vs_cuereduced(CD)']:12.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
