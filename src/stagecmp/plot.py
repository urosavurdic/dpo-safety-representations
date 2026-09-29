"""The WP-4 figure: safety SFT versus DPO, across depth.

Design decisions and why, following the data-viz procedure:

**Form.** The data's job is change-over-depth for four named transitions, so a
line chart, not bars. Two panels, one axis each - never a dual y-axis.

**Colour by job.** Identity is *categorical*, but the four arms are a 2x2, so
colour carries only the OBJECTIVE (2 hues) and linestyle carries the CORPUS
(solid Alpaca, dashed Dolly). That is a composite encoding: it halves the hue
count and, more importantly, makes colour non-load-bearing, so the figure
survives the greyscale printing an IEEE proceedings may apply. Hues are slots 1
and 2 of the reference categorical theme, in fixed order.

**Validation.** `scripts/validate_palette.js` could not be run here - node is not
installed on this machine. Rather than eyeball it, the design avoids depending on
the result: two hues, each also distinguished by linestyle and marker. For the
record, the reference palette documents slots 1-2 at worst adjacent CVD
Delta E 9.1 (light, target >= 8) and normal-vision Delta E 19.6 (floor >= 15).

**Marks.** 2px lines, markers at ~8px, recessive grid and axes, text in ink
tokens rather than series colour, legend plus direct labels on all four series
(<= 4 series, so both).

No hover layer: this renders to PDF/PNG for print, where the interaction rules
do not apply.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("results/stagecmp/figures")

# reference categorical theme, slots 1 and 2 (light mode)
C_SFT = "#2a78d6"     # blue
C_DPO = "#eb6834"     # orange
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e2e0da"

ARMS = [
    ("M1__to__M2", "safety SFT, Alpaca", C_SFT, "-", "o"),
    ("M1_alt__to__M2_alt", "safety SFT, Dolly", C_SFT, "--", "s"),
    ("M1__to__M3_direct", "DPO, Alpaca", C_DPO, "-", "o"),
    ("M1_alt__to__M3_direct_alt", "DPO, Dolly", C_DPO, "--", "s"),
]


def _series(payload, key, extract):
    rows = payload["headline_pairs"][key]["per_layer"]
    xs = [r["layer"] for r in rows]
    ys = [extract(r) for r in rows]
    return xs, ys


def build(pooling: str = "final_token", out_dir=OUT_DIR) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = Path(f"results/stagecmp/representational_{pooling}.json")
    payload = json.loads(src.read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.size": 8,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)

    panels = [
        (axes[0], lambda r: r["contrast_norm"]["ratio"],
         "Contrast norm ratio  $\\|c_{post}\\|/\\|c_{pre}\\|$",
         "(a)  Does the A–D contrast grow?", 1.0),
        (axes[1], lambda r: r["selectivity"]["A_minus_D"],
         "Drift selectivity  A $-$ D",
         "(b)  Is the movement safety-relevant?", 0.0),
    ]

    for ax, extract, ylabel, title, ref in panels:
        ax.axhline(ref, color=INK_MUTED, lw=0.8, ls=":", zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for key, label, colour, ls, marker in ARMS:
            xs, ys = _series(payload, key, extract)
            ax.plot(xs, ys, color=colour, ls=ls, lw=2.0, marker=marker,
                    markersize=2.8, markevery=4, zorder=3, label=label,
                    solid_capstyle="round")
            # direct label at the right edge (<= 4 series: legend AND labels)
            ax.annotate(label.split(",")[1].strip(), xy=(xs[-1], ys[-1]),
                        xytext=(3, 0), textcoords="offset points",
                        color=INK_SOFT, fontsize=6, va="center")
        ax.set_xlabel("layer")
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=6)
        ax.set_xlim(0, 28)
        ax.set_xticks([0, 8, 16, 24, 28])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.09), labelcolor=INK_SOFT)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"sft_vs_dpo_depth_{pooling}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(f"{stem}.png")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooling", default="final_token",
                    choices=["final_token", "mean_last5"])
    args = ap.parse_args()
    print("wrote", build(args.pooling))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
