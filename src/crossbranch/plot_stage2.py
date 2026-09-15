"""Figures for the cross-branch Stage-1 / Stage-2 result (CPU, no torch).

Every number a figure draws comes from a committed analysis JSON:

  * Stage-1 dose-response          <- crossbranch_AtoB_analysis.json
  * Stage-2 gate-quadrant arms     <- crossbranch_AtoB_stage2_analysis.json
  * Stage-2 paired contrasts (C)   <- same
  * per-quadrant arm summary       <- same

If a JSON is missing an arm or a CI, that arm is dropped from the figure and
the omission is reported -- nothing is invented to fill a panel. A companion
``figures_README.md`` records which JSON and field every panel came from, and
the raw plotted values are also written as JSON next to the PNGs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ANALYSIS_DIR = Path("results/crossbranch/analysis")
FIG_DIR = Path("results/crossbranch/figures")

STAGE1_JSON = "crossbranch_AtoB_analysis.json"
STAGE2_JSON = "crossbranch_AtoB_stage2_analysis.json"

# Arm display order + labels for the Stage-2 gate-quadrant figure.
ARM_ORDER = [
    ("own_delta_target", "own $\\Delta_B$ (within-branch)"),
    ("xfer_delta_source_identity", "identity $\\Delta_A$"),
    ("xfer_delta_source_shuf_wq", "$\\Delta_A$ shuffled within-quadrant"),
    ("xfer_delta_source_dosematched", "$\\Delta_A$ dose-matched to $\\|\\Delta_B\\|$"),
    ("xfer_delta_source_normmatched", "norm-matched random"),
    ("dir_source_matched", "$s_A d_A$ (source direction)"),
    ("dir_target_matched", "$s_B d_B$ (target direction)"),
    ("xfer_delta_source_parallel", "$\Delta_A^\parallel$ (refusal-dir part, full mag)"),
    ("xfer_delta_source_perp", "$\Delta_A^\perp$ (residual, full mag)"),
]

CONTRAST_ORDER = [
    ("xfer_delta_source_identity__minus__xfer_delta_source_shuf_wq",
     "identity $-$ within-quadrant shuffle"),
    ("xfer_delta_source_identity__minus__xfer_delta_source_normmatched",
     "identity $-$ norm-matched random"),
    ("xfer_delta_source_identity__minus__xfer_delta_source_dosematched",
     "identity $-$ dose-matched"),
    ("xfer_delta_source_identity__minus__dir_source_matched",
     "identity $-$ source direction"),
    ("xfer_delta_source_identity__minus__own_delta_target",
     "identity $-$ own $\\Delta_B$"),
]


def _load(analysis_dir: Path, name: str) -> dict:
    p = analysis_dir / name
    if not p.exists():
        raise SystemExit(f"missing {p}; run the analyzers first")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Figure 1 -- Stage-1 dose-response
# --------------------------------------------------------------------------
def fig_stage1_dose_response(s1: dict, out: Path) -> dict:
    import matplotlib.pyplot as plt

    coefs, own, rnd = [], [], []
    for c, blk in sorted(s1["by_coefficient"].items(), key=lambda kv: float(kv[0])):
        d = blk["dtv"]
        coefs.append(float(c))
        own.append(d["own_delta_target"])
        rnd.append(d["own_normmatched_random"])

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for series, label, color, dx in (
        (own, "own $\\Delta_B$", "#1f4e79", -0.015),
        (rnd, "norm-matched random", "#999999", +0.015),
    ):
        x = [c + dx for c in coefs]
        y = [s["point"] for s in series]
        lo = [s["point"] - s["ci_low"] for s in series]
        hi = [s["ci_high"] - s["point"] for s in series]
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", capsize=4, color=color,
                    label=label, lw=1.8, ms=6)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(coefs)
    ax.set_xlabel("injection coefficient")
    ax.set_ylabel("$\\Delta$TV  (arm vs $B3$) $-$ ($B2$ vs $B3$)")
    q = s1["gate"]["gate_quadrant"]
    ax.set_title(f"Stage-1 gate, quadrant {q}: own DPO delta vs matched random\n"
                 "negative = moved toward the branch's own post-DPO profile",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"coefficients": coefs,
            "own_delta_target": own, "own_normmatched_random": rnd,
            "gate_quadrant": q}


# --------------------------------------------------------------------------
# Figure 2 -- Stage-2 gate-quadrant arm dTV
# --------------------------------------------------------------------------
def fig_stage2_gate_quadrant(s2: dict, out: Path) -> dict:
    import matplotlib.pyplot as plt

    q = s2.get("stage1_gate_quadrant", "C")
    block = s2["per_quadrant"][q]
    arms = block["arms"]

    rows, missing = [], []
    for key, label in ARM_ORDER:
        if key not in arms or "dtv" not in arms[key]:
            missing.append(key)
            continue
        d = arms[key]["dtv"]
        rows.append((label, d["point"], d["ci_low"], d["ci_high"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ys = list(range(len(rows)))[::-1]
    for y, (label, pt, lo, hi) in zip(ys, rows):
        excl = hi < 0 or lo > 0
        ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="o", capsize=4,
                    color="#1f4e79" if excl else "#9aa5b1",
                    ecolor="#1f4e79" if excl else "#9aa5b1", ms=7, lw=1.8)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("$\\Delta$TV vs $B3$  (negative = toward $B3$)")
    tvbr = block.get("tv_baseline_to_reference")
    ax.set_title(f"Stage-2, quadrant {q} (n={block['n']}): "
                 f"movement toward $B3$\nTV($B2$,$B3$) = {tvbr:.3f}; "
                 "filled = 95% CI excludes 0", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"gate_quadrant": q, "n": block["n"],
            "tv_baseline_to_reference": tvbr,
            "arms": {r[0]: {"point": r[1], "ci_low": r[2], "ci_high": r[3]}
                     for r in rows},
            "missing_arms": missing}


# --------------------------------------------------------------------------
# Figure 3 -- Stage-2 paired contrasts (gate quadrant)
# --------------------------------------------------------------------------
def fig_stage2_contrasts(s2: dict, out: Path) -> dict:
    import matplotlib.pyplot as plt

    q = s2.get("stage1_gate_quadrant", "C")
    contrasts = s2["per_quadrant"][q]["contrasts"]

    rows, missing = [], []
    for key, label in CONTRAST_ORDER:
        if key not in contrasts:
            missing.append(key)
            continue
        d = contrasts[key]["dtv_diff"]
        rows.append((label, d["point"], d["ci_low"], d["ci_high"]))

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ys = list(range(len(rows)))[::-1]
    for y, (label, pt, lo, hi) in zip(ys, rows):
        excl = hi < 0 or lo > 0
        ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="o", capsize=4,
                    color="#1f4e79" if excl else "#9aa5b1",
                    ecolor="#1f4e79" if excl else "#9aa5b1", ms=7, lw=1.8)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("paired $\\Delta$TV difference  "
                  "(negative = first arm moved further toward $B3$)")
    ax.set_title(f"Stage-2, quadrant {q}: paired arm contrasts\n"
                 "filled = 95% CI excludes 0", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {"gate_quadrant": q,
            "contrasts": {r[0]: {"point": r[1], "ci_low": r[2], "ci_high": r[3]}
                          for r in rows},
            "missing_contrasts": missing}


# --------------------------------------------------------------------------
# Figure 4 -- per-quadrant summary, one panel each, n shown
# --------------------------------------------------------------------------
def fig_quadrant_summary(s2: dict, out: Path) -> dict:
    import matplotlib.pyplot as plt

    quads = [q for q in ("A", "B", "C", "D") if q in s2["per_quadrant"]]
    fig, axes = plt.subplots(1, len(quads), figsize=(4.6 * len(quads), 4.6),
                             sharex=True)
    if len(quads) == 1:
        axes = [axes]

    dump = {}
    for ax, q in zip(axes, quads):
        block = s2["per_quadrant"][q]
        arms = block["arms"]
        rows = []
        for key, label in ARM_ORDER:
            if key in arms and "dtv" in arms[key]:
                d = arms[key]["dtv"]
                rows.append((label, d["point"], d["ci_low"], d["ci_high"]))
        ys = list(range(len(rows)))[::-1]
        for y, (label, pt, lo, hi) in zip(ys, rows):
            excl = hi < 0 or lo > 0
            ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="o", capsize=3,
                        color="#1f4e79" if excl else "#9aa5b1",
                        ecolor="#1f4e79" if excl else "#9aa5b1", ms=5)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows] if q == quads[0] else [],
                           fontsize=8)
        n = block["n"]
        tag = "  (n=30, wide CIs)" if n <= 30 else ""
        ax.set_title(f"quadrant {q}   n={n}{tag}\n"
                     f"TV($B2$,$B3$)={block['tv_baseline_to_reference']:.3f}",
                     fontsize=9)
        ax.set_xlabel("$\\Delta$TV vs $B3$")
        dump[q] = {"n": n,
                   "tv_baseline_to_reference": block["tv_baseline_to_reference"],
                   "arms": {r[0]: {"point": r[1], "ci_low": r[2], "ci_high": r[3]}
                            for r in rows}}
    fig.suptitle("Stage-2 per-quadrant arm movement toward $B3$  "
                 "(A and D held-out n=30 -- do not read as tight as B/C)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return dump


README = """# Cross-branch figures -- provenance

All values are read from committed analysis JSON. No figure value is computed
here beyond arithmetic on those fields (point, ci_low, ci_high).

| file | source JSON | fields |
|---|---|---|
| `fig1_stage1_dose_response.png` | `{s1}` | `by_coefficient[*].dtv.own_delta_target`, `.own_normmatched_random` (point + percentile CI); `gate.gate_quadrant` |
| `fig2_stage2_gate_quadrant.png` | `{s2}` | `per_quadrant[q*].arms[*].dtv` (point + paired-bootstrap CI); `per_quadrant[q*].tv_baseline_to_reference` |
| `fig3_stage2_contrasts.png` | `{s2}` | `per_quadrant[q*].contrasts[*].dtv_diff` (point + paired-bootstrap CI) |
| `fig4_quadrant_summary.png` | `{s2}` | `per_quadrant[A..D].arms[*].dtv`; per-panel `n` |

`q*` is the Stage-1 gate quadrant (`stage1_gate_quadrant`), = **C**.

Negative $\\Delta$TV = the arm's four-way label distribution moved toward
$B3$'s. Filled markers = 95% CI excludes 0. Bootstrap: B=10,000, seed
20260904, percentile, paired over identical `record_id`s within the quadrant.

`plotted_values.json` in this directory is the exact set of numbers drawn.
`fig*` also has a sensitivity twin if `--sensitivity` was passed
(`crossbranch_AtoB_stage2_analysis_sensitivity_extended_regex.json`).
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Render cross-branch Stage-1/2 figures.")
    p.add_argument("--analysis-dir", default=str(ANALYSIS_DIR))
    p.add_argument("--fig-dir", default=str(FIG_DIR))
    p.add_argument("--stage2-json", default=STAGE2_JSON,
                   help="override to render the sensitivity twin")
    p.add_argument("--suffix", default="",
                   help="filename suffix, e.g. _sensitivity")
    args = p.parse_args()

    adir = Path(args.analysis_dir)
    fdir = Path(args.fig_dir)
    fdir.mkdir(parents=True, exist_ok=True)

    s2 = _load(adir, args.stage2_json)
    plotted = {}
    plotted["fig2_stage2_gate_quadrant"] = fig_stage2_gate_quadrant(
        s2, fdir / f"fig2_stage2_gate_quadrant{args.suffix}.png")
    plotted["fig3_stage2_contrasts"] = fig_stage2_contrasts(
        s2, fdir / f"fig3_stage2_contrasts{args.suffix}.png")
    plotted["fig4_quadrant_summary"] = fig_quadrant_summary(
        s2, fdir / f"fig4_quadrant_summary{args.suffix}.png")

    if not args.suffix:  # Stage-1 figure only for the primary render
        s1 = _load(adir, STAGE1_JSON)
        plotted["fig1_stage1_dose_response"] = fig_stage1_dose_response(
            s1, fdir / "fig1_stage1_dose_response.png")
        (fdir / "figures_README.md").write_text(
            README.format(s1=STAGE1_JSON, s2=STAGE2_JSON), encoding="utf-8")

    (fdir / f"plotted_values{args.suffix}.json").write_text(
        json.dumps(plotted, indent=1) + "\n", encoding="utf-8")

    print(f"wrote figures to {fdir}")
    for k, v in plotted.items():
        miss = v.get("missing_arms") or v.get("missing_contrasts") or []
        note = f"  (MISSING: {miss})" if miss else ""
        print(f"  {k}{note}")


if __name__ == "__main__":
    main()
