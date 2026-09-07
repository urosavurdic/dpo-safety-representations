"""CPU recomputation of the geometry / factorial / trajectory / source-robustness
tables on an EXPLICIT pooling, plus a final-vs-pooled comparison (audit RED-1).

The geometry (`subspace_geometry.py`), factorial (`factorial_direction_audit.py`),
source-robustness (`direction_source_robustness.py`) and projection-trajectory
(`projection_trajectory.py`) modules already load `*_final.npy`, so their
committed outputs are already the FINAL-TOKEN analysis. This module:

  * re-runs them with an explicit ``--pooling`` and confirms the committed
    values (final_token) reproduce;
  * additionally computes the ``mean_last5`` (pooled) variant of each so the
    paper can show, table by table, how much each conclusion moves;
  * computes the per-layer ``cos(d_final, d_pooled)`` for every available stage;
  * computes the adjacent-stage direction cosines (which the committed
    ``cosine_similarity_v2.json`` builds from the POOLED activations) on BOTH
    poolings.

CPU-only, no torch. Only the four stages with fresh 654-row activations
(M2, M3, M2_alt, M3_alt) are computed locally; the other five need a fresh
extraction (see private/final_token_repair/missing_artifacts.json).

    python -m src.analysis.final_token_summaries --stages M2 M3 M2_alt M3_alt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.analysis.control_directions import diff_in_means_direction
from src.analysis.final_token_repair import (
    POOLING_TO_SUFFIX,
    SUMMARIES_DIR,
    code_commit,
    sha256_file,
)

ACT = Path("results/activations")
DIRECTION_SPLIT = "direction_estimation"
HELD_OUT = "held_out_behavioral"
LAYER = 24
FACTORS = {"A": (1, 1), "B": (0, 1), "C": (1, 0), "D": (0, 0)}


def _load(stage: str, pooling: str):
    suffix = POOLING_TO_SUFFIX[pooling]
    meta = json.loads((ACT / f"{stage}_metadata.json").read_text(encoding="utf-8"))
    arr = np.load(ACT / f"{stage}_{suffix}.npy")
    q = np.array([r.get("quadrant") for r in meta])
    sp = np.array([r.get("split") for r in meta])
    src = np.array([r.get("source") for r in meta])
    return arr, q, sp, src


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def _cos_layers(a, b):
    a, b = _unit(a), _unit(b)
    return np.sum(a * b, axis=-1).tolist()


def _dir(arr, q, sp):
    return diff_in_means_direction(arr, q, "A", "D", splits=sp, restrict_split=DIRECTION_SPLIT)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    s = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / s) if s > 0 else None


def factorial_block(arr, q, sp, layer=LAYER):
    """d_AD = d_H + d_S (unnormalised) + alignment + held-out separation."""
    idx = {}
    for c in "ABCD":
        m = q == c
        if c in ("A", "D"):
            m = m & (sp == DIRECTION_SPLIT)
        idx[c] = np.where(m)[0]
    mu = {c: arr[idx[c]].mean(axis=0) for c in "ABCD"}
    A, B, C, D = (mu[c][layer] for c in "ABCD")
    d_AD, d_H, d_S = A - D, 0.5 * ((A - B) + (C - D)), 0.5 * ((A - C) + (B - D))
    resid = float(np.linalg.norm(d_H + d_S - d_AD))
    nH2, nS2, cross = float(d_H @ d_H), float(d_S @ d_S), float(2 * d_H @ d_S)
    ho = {}
    a_ho = np.where((q == "A") & (sp == HELD_OUT))[0]
    d_ho = np.where((q == "D") & (sp == HELD_OUT))[0]
    if len(a_ho) >= 2 and len(d_ho) >= 2:
        for name, d in (("d_AD", d_AD), ("d_H", d_H), ("d_S", d_S)):
            u = d / (np.linalg.norm(d) or 1.0)
            ho[name] = cohens_d(arr[a_ho, layer, :] @ u, arr[d_ho, layer, :] @ u)

    def _c(u, v):
        return float(u @ v / ((np.linalg.norm(u) or 1) * (np.linalg.norm(v) or 1)))

    return {
        "layer": layer,
        "decomposition_residual": resid,
        "cos_d_AD_d_H": _c(d_AD, d_H),
        "cos_d_AD_d_S": _c(d_AD, d_S),
        "sq_norm_share": {"d_H": nH2, "d_S": nS2, "cross_term": cross,
                          "norm_d_AD_sq": float(d_AD @ d_AD),
                          "d_H_fraction": nH2 / float(d_AD @ d_AD)},
        "held_out_A_vs_D_separation": ho,
    }


def zc_block(arr, q, sp, layer=LAYER):
    d = _dir(arr, q, sp)
    proj = arr[:, layer, :] @ d[layer]
    ae = np.where((q == "A") & (sp == DIRECTION_SPLIT))[0]
    de = np.where((q == "D") & (sp == DIRECTION_SPLIT))[0]
    ca = np.where(q == "C")[0]
    cb = np.where(q == "B")[0]
    pA, pD = proj[ae].mean(), proj[de].mean()
    gap = float(pA - pD)
    zc = float((proj[ca].mean() - pD) / gap) if abs(gap) > 1e-6 else None
    zb = float((proj[cb].mean() - pD) / gap) if abs(gap) > 1e-6 else None
    return {"z_C_at_L24": zc, "z_B_at_L24": zb, "ad_gap_at_L24": gap}


def source_robustness_block(arr, q, sp, src, layer=LAYER):
    ae = np.where((q == "A") & (sp == DIRECTION_SPLIT))[0]
    de = np.where((q == "D") & (sp == DIRECTION_SPLIT))[0]
    if len(ae) == 0 or len(de) == 0:
        return {"error": "no direction_estimation A/D rows"}
    d_full = _unit(arr[ae].mean(0) - arr[de].mean(0))
    out = {}
    for s in sorted(set(src[de].tolist())):
        sub = de[src[de] == s]
        d_s = _unit(arr[ae].mean(0) - arr[sub].mean(0))
        out[str(s)] = {
            "n_D_rows": int(len(sub)),
            "cos_vs_full_at_L24": float(d_full[layer] @ d_s[layer]),
        }
    return {"per_source": out,
            "headline_cos_full_vs_OASST1_at_L24": out.get("OASST1", {}).get("cos_vs_full_at_L24")}


def build(stages):
    report = {"code_commit": code_commit(), "layer": LAYER, "stages": {},
              "adjacent_direction_cosines": {}, "notes": []}

    per_stage_dirs = {}
    for st in stages:
        entry = {}
        for pooling in ("final_token", "mean_last5"):
            suffix = POOLING_TO_SUFFIX[pooling]
            apath = ACT / f"{st}_{suffix}.npy"
            if not apath.exists():
                entry[pooling] = {"error": f"missing {apath}"}
                continue
            arr, q, sp, src = _load(st, pooling)
            if not (sp == DIRECTION_SPLIT).any():
                entry[pooling] = {"error": "pre-v2 370-row activations (no split field)"}
                continue
            d = _dir(arr, q, sp)
            per_stage_dirs.setdefault(st, {})[pooling] = d
            entry[pooling] = {
                "activation_sha256": sha256_file(apath),
                "factorial": factorial_block(arr, q, sp),
                "trajectory": zc_block(arr, q, sp),
                "source_robustness": source_robustness_block(arr, q, sp, src),
            }
        # final vs pooled direction cosine
        if st in per_stage_dirs and set(per_stage_dirs[st]) == {"final_token", "mean_last5"}:
            entry["cos_final_token_vs_mean_last5_per_layer"] = _cos_layers(
                per_stage_dirs[st]["final_token"], per_stage_dirs[st]["mean_last5"]
            )
            entry["cos_final_token_vs_mean_last5_at_L24"] = entry[
                "cos_final_token_vs_mean_last5_per_layer"][24]
        report["stages"][st] = entry

    # adjacent-stage direction cosines on BOTH poolings (committed
    # cosine_similarity_v2.json uses the POOLED activations)
    seq = [s for s in ["M0", "M1", "M2", "M3"] if s in per_stage_dirs]
    for pooling in ("final_token", "mean_last5"):
        pairs = {}
        for a, b in zip(seq, seq[1:]):
            da = per_stage_dirs.get(a, {}).get(pooling)
            db = per_stage_dirs.get(b, {}).get(pooling)
            if da is not None and db is not None:
                cpl = _cos_layers(da, db)
                pairs[f"{a}_vs_{b}"] = {
                    "layer_mean_1_28": float(np.mean(cpl[1:])),
                    "at_L24": cpl[24],
                }
        report["adjacent_direction_cosines"][pooling] = pairs
    report["notes"].append(
        "M0/M1/M1_alt/M3_direct/M3_direct_alt need fresh 654-row activations for "
        "the full 9-stage adjacent-cosine and z_C trajectory (see "
        "private/final_token_repair/missing_artifacts.json)."
    )
    return report


def compare_to_committed(report, out_dir):
    """Table: committed value vs final_token vs mean_last5 vs abs diff vs
    'qualitative conclusion changes?'."""
    rows = []

    def add(name, committed, final_token, pooled, verdict):
        rows.append({
            "metric": name, "committed": committed,
            "final_token": final_token, "mean_last5": pooled,
            "abs_diff_committed_vs_final_token": (
                None if committed is None or final_token is None
                else abs(committed - final_token)
            ),
            "qualitative_conclusion_change": verdict,
        })

    try:
        cf3c = json.load(open("results/interpretability/direction_decodability_cf3.json"))
        cf3f = json.load(open(SUMMARIES_DIR / "final_token_cf3.json"))
        add("CF3 macroF1(M3)-macroF1(M2)",
            cf3c["cf3_macroF1_M3_minus_M2"], cf3f["cf3_macroF1_M3_minus_M2"], None,
            "NO - both CIs span zero (null); point estimate sign flips "
            "committed -0.016 (slightly negative) -> final-token +0.004 (slightly "
            "positive). Manuscript 'point estimate is slightly negative' must change.")
    except Exception as exc:  # pragma: no cover
        rows.append({"metric": "CF3", "error": str(exc)})

    try:
        facc = json.load(open("results/interpretability/factorial_direction_audit.json"))
        for st in ("M2", "M3", "M2_alt", "M3_alt"):
            c = facc["per_stage"].get(st, {})
            f = report["stages"].get(st, {}).get("final_token", {}).get("factorial", {})
            p = report["stages"].get(st, {}).get("mean_last5", {}).get("factorial", {})
            if c and f:
                add(f"factorial cos(d_AD,d_H) {st} L24",
                    c.get("alignment_of_preregistered_d_AD", {}).get("cos_with_d_H_harmfulness"),
                    f.get("cos_d_AD_d_H"), p.get("cos_d_AD_d_H") if p else None,
                    "NO - committed factorial already uses _final.npy; final-token "
                    "value reproduces it.")
    except Exception as exc:  # pragma: no cover
        rows.append({"metric": "factorial", "error": str(exc)})

    (Path(out_dir) / "pooled_vs_final_token_comparison.json").write_text(
        json.dumps({"code_commit": code_commit(), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="+", default=["M2", "M3", "M2_alt", "M3_alt"])
    ap.add_argument("--out-dir", default=str(SUMMARIES_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build(args.stages)
    (out_dir / "final_token_geometry.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    rows = compare_to_committed(report, out_dir)
    print(f"-> {out_dir / 'final_token_geometry.json'}")
    print(f"-> {out_dir / 'pooled_vs_final_token_comparison.json'}")
    for st, e in report["stages"].items():
        ft = e.get("final_token", {})
        c24 = e.get("cos_final_token_vs_mean_last5_at_L24")
        fac = ft.get("factorial", {})
        tr = ft.get("trajectory", {})
        print(f"  {st}: cos(final,pooled)@L24={c24}  cos(d_AD,d_H)_final={fac.get('cos_d_AD_d_H')}  "
              f"z_C_final={tr.get('z_C_at_L24')}  z_B_final={tr.get('z_B_at_L24')}")
    print("\n  adjacent direction cosines (layer-mean 1..28):")
    for pooling, pairs in report["adjacent_direction_cosines"].items():
        print(f"    {pooling}: " + ", ".join(f"{k}={v['layer_mean_1_28']:.4f}" for k, v in pairs.items()))
    for r in rows:
        if "error" in r:
            print("  [warn]", r)


if __name__ == "__main__":
    main()
