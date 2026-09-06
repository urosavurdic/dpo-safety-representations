"""Is the A-D direction driven by D's training-adjacent sub-sources?

Quadrant D = Alpaca 50 + Dolly-15k 50 + OASST1 50. M1 (and M1_alt) were
SFT'd on Alpaca (Dolly for the alt branch), so 2/3 of D shares its
distribution with training data. The direction d = mean(A_est) - mean(D_est)
could therefore pick up "familiar with the SFT distribution" rather than
"benign". Quadrant A is HarmBench only - no training overlap - so the
confound is entirely on the D side.

This script re-estimates the direction with D restricted to each single
sub-source and reports how far the direction moves. If cos(d_full, d_OASST1)
is ~1.0, the training-adjacent halves of D are not steering the direction.

CPU only. Needs results/activations/{stage}_final.npy + {stage}_metadata.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACT_DIR = Path("results/activations")
STAGES = ["M0", "M1", "M2", "M3", "M3_direct", "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt"]
INDEP_SOURCE = "OASST1"        # the one D sub-source with no training overlap
LAYER = 24                     # the intervention layer; report this one explicitly


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_stage(stage):
    meta = json.loads((ACT_DIR / f"{stage}_metadata.json").read_text(encoding="utf-8", errors="replace"))
    arr = np.load(ACT_DIR / f"{stage}_final.npy")  # (n, n_layers, hidden)
    return meta, arr


def direction(arr, a_idx, d_idx):
    """(n_layers, hidden) unit A-D contrast from the given row indices."""
    ma = arr[a_idx].mean(axis=0)     # (n_layers, hidden)
    md = arr[d_idx].mean(axis=0)
    diff = ma - md
    return np.stack([_unit(diff[L]) for L in range(diff.shape[0])])


def cos_per_layer(d1, d2):
    return np.array([float(np.dot(d1[L], d2[L])) for L in range(d1.shape[0])])


def z_c(arr, d, c_idx, a_idx, d_idx, layer):
    """C's normalised position between D (0) and A (1) along d at `layer`."""
    proj = arr[:, layer, :] @ d[layer]
    pA, pD, pC = proj[a_idx].mean(), proj[d_idx].mean(), proj[c_idx].mean()
    gap = pA - pD
    return float((pC - pD) / gap) if abs(gap) > 1e-6 else None


def analyse_stage(stage):
    meta, arr = load_stage(stage)
    q = np.array([r["quadrant"] for r in meta])
    sp = np.array([r.get("split") for r in meta])
    src = np.array([r.get("source") for r in meta])

    a_est = np.where((q == "A") & (sp == "direction_estimation"))[0]
    d_est = np.where((q == "D") & (sp == "direction_estimation"))[0]
    c_all = np.where(q == "C")[0]
    if len(a_est) == 0 or len(d_est) == 0:
        return {"stage": stage, "error": "no direction_estimation A/D rows in metadata"}

    d_full = direction(arr, a_est, d_est)
    per_source = {}
    for s in sorted(set(src[d_est].tolist())):
        d_sub_idx = d_est[src[d_est] == s]
        d_s = direction(arr, a_est, d_sub_idx)
        cpl = cos_per_layer(d_full, d_s)
        per_source[s] = {
            "n_D_rows": int(len(d_sub_idx)),
            "cos_vs_full_mean_L1_28": float(cpl[1:].mean()),
            "cos_vs_full_at_L24": float(cpl[LAYER]),
            "z_C_at_L24": z_c(arr, d_s, c_all, a_est, d_est, LAYER),
        }

    indep = per_source.get(INDEP_SOURCE, {})
    return {
        "stage": stage,
        "n_A_est": int(len(a_est)),
        "n_D_est": int(len(d_est)),
        "z_C_full_at_L24": z_c(arr, d_full, c_all, a_est, d_est, LAYER),
        "per_source": per_source,
        "headline_cos_full_vs_OASST1_at_L24": indep.get("cos_vs_full_at_L24"),
        "headline_z_C_shift_full_to_OASST1": (
            None if indep.get("z_C_at_L24") is None
            else round(indep["z_C_at_L24"] - z_c(arr, d_full, c_all, a_est, d_est, LAYER), 4)
        ),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--out", default="results/interpretability/direction_source_robustness.json")
    args = ap.parse_args()

    report = {"layer": LAYER, "independent_source": INDEP_SOURCE, "per_stage": {}}
    for st in args.stages:
        try:
            r = analyse_stage(st)
        except FileNotFoundError as e:
            r = {"stage": st, "error": f"missing activation file: {e}"}
        report["per_stage"][st] = r
        if "error" in r:
            print(f"  {st:14s} SKIP  {r['error']}")
        else:
            print(f"  {st:14s} cos(d_full, d_OASST1)@L24 = {r['headline_cos_full_vs_OASST1_at_L24']:.4f}   "
                  f"z_C shift full->OASST1 = {r['headline_z_C_shift_full_to_OASST1']:+.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
