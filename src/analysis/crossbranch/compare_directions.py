"""Reciprocal analysis: put A->B and B->A side by side (CPU, torch-free).

The transfer experiment is inherently directional. Running only Alpaca->Dolly
leaves a real alternative open: that whatever moved the target is a property of
the *target* branch (Dolly's pre-DPO model being easy to push toward
soft-deflection) rather than a property of the transferred delta. The
reciprocal direction is the control for that -- if the same pattern appears
with the roles swapped, it is a property of the delta/site; if it appears in
only one direction, it is a property of that branch.

This module does no new statistics. It reads the two Stage-2 analysis JSONs
(produced by analyze_stage2 with --source-branch/--target-branch swapped) and
reports, per quadrant, whether each direction's readout agrees on the three
questions that carry the interpretation:

  moves      -- does the identity arm's dTV CI exclude 0 (moved toward the
                target's own post-DPO profile)?
  structured -- does identity beat a per-row norm-matched random vector
                (contrast CI excludes 0)? distinguishes a real directional
                effect from a generic perturbation of the same size.
  prompt_conditioned
             -- does identity beat the WITHIN-QUADRANT SHUFFLED delta
                (contrast CI excludes 0)? this is the one that separates
                "the delta's per-prompt content transfers" from "a shared
                per-quadrant component transfers".

Each is a CI-based verdict with three values: "yes", "no", or "underpowered"
(the CI covers 0 AND is wide enough that the quadrant cannot resolve the
question -- reported separately from a genuine null so a 30-row quadrant is
never read as evidence of absence).

The agreement summary is descriptive. It is NOT a hypothesis test on the
difference between directions: the two directions inject different vectors
into different checkpoints, so their statistics are independent runs, not a
paired contrast, and no combined p-value is computed or implied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.crossbranch.analyze_stage2 import (
    DIR_SOURCE,
    IDENTITY,
    NORMMATCHED,
    SHUF_WQ,
)
from src.analysis.crossbranch.branches import direction_tag

# A CI that covers zero but is at least this wide (in TV units) is reported as
# "underpowered" rather than "no". Chosen to be the resolution of a ~10-row
# quadrant: with n=30 held-out A/D rows, one row moving between categories
# shifts TV by 1/30 = 0.033, and a CI spanning more than three such steps
# cannot distinguish a null from a moderate effect. Descriptive threshold for
# READING the result; it changes no statistic and gates no decision.
UNDERPOWERED_CI_WIDTH = 0.10

QUESTIONS = {
    "moves": ("arm", IDENTITY, "dtv"),
    "structured": ("contrast", f"{IDENTITY}__minus__{NORMMATCHED}", "dtv_diff"),
    "prompt_conditioned": ("contrast", f"{IDENTITY}__minus__{SHUF_WQ}", "dtv_diff"),
    "beats_concept_direction": (
        "contrast", f"{IDENTITY}__minus__{DIR_SOURCE}", "dtv_diff"
    ),
}


def verdict(stat: dict | None, *, width: float = UNDERPOWERED_CI_WIDTH) -> dict:
    """CI-based three-way verdict on 'is this reliably negative?'.

    Negative is the direction of interest throughout: dTV negative = moved
    toward the target's post-DPO profile; a negative contrast = the primary
    arm moved further than its control.
    """
    if stat is None:
        return {"verdict": "absent", "point": None, "ci": None}
    lo, hi, pt = stat["ci_low"], stat["ci_high"], stat["point"]
    if hi < 0:
        v = "yes"
    elif lo > 0:
        v = "no_opposite"
    elif (hi - lo) >= width:
        v = "underpowered"
    else:
        v = "no"
    return {
        "verdict": v,
        "point": pt,
        "ci": [lo, hi],
        "ci_width": hi - lo,
    }


def _stat(block: dict, kind: str, key: str, field: str):
    section = block.get("arms" if kind == "arm" else "contrasts", {})
    entry = section.get(key)
    return entry.get(field) if entry else None


def read_direction(block: dict) -> dict:
    """The four question verdicts for one quadrant of one direction."""
    return {
        name: verdict(_stat(block, kind, key, field))
        for name, (kind, key, field) in QUESTIONS.items()
    }


def compare(analysis_a: dict, analysis_b: dict) -> dict:
    """Side-by-side readout of two Stage-2 analyses (one per direction)."""
    tag_a = analysis_a.get("direction_tag", "first")
    tag_b = analysis_b.get("direction_tag", "second")

    quadrants = [
        q for q in ("A", "B", "C", "D")
        if q in analysis_a.get("per_quadrant", {})
        and q in analysis_b.get("per_quadrant", {})
    ]

    per_quadrant = {}
    for q in quadrants:
        blk_a = analysis_a["per_quadrant"][q]
        blk_b = analysis_b["per_quadrant"][q]
        va, vb = read_direction(blk_a), read_direction(blk_b)
        agreement = {
            name: {
                tag_a: va[name]["verdict"],
                tag_b: vb[name]["verdict"],
                "agree": va[name]["verdict"] == vb[name]["verdict"],
                "both_conclusive": va[name]["verdict"] not in ("underpowered", "absent")
                and vb[name]["verdict"] not in ("underpowered", "absent"),
            }
            for name in QUESTIONS
        }
        per_quadrant[q] = {
            "n": {tag_a: blk_a.get("n"), tag_b: blk_b.get("n")},
            "tv_baseline_to_reference": {
                tag_a: blk_a.get("tv_baseline_to_reference"),
                tag_b: blk_b.get("tv_baseline_to_reference"),
            },
            tag_a: va,
            tag_b: vb,
            "agreement": agreement,
        }

    return {
        "directions": [tag_a, tag_b],
        "coef": {tag_a: analysis_a.get("coef"), tag_b: analysis_b.get("coef")},
        "underpowered_ci_width": UNDERPOWERED_CI_WIDTH,
        "per_quadrant": per_quadrant,
        "notes": [
            "Descriptive side-by-side only. The two directions inject different "
            "vectors into different checkpoints, so their statistics are "
            "independent runs, not a paired contrast: no combined p-value is "
            "computed and none should be inferred from agreement.",
            "'prompt_conditioned' = identity beats the WITHIN-QUADRANT SHUFFLED "
            "delta. Shuffle preserves the per-quadrant delta distribution and "
            "destroys the prompt<->delta pairing, so a 'no' here means the "
            "movement is carried by a shared per-quadrant component rather than "
            "by the delta's per-prompt content.",
            "'underpowered' is reported separately from 'no' so a 30-row "
            "held-out quadrant is never read as evidence of absence.",
            "Agreement across directions argues the pattern is a property of "
            "the delta and the injection site; disagreement argues it is a "
            "property of one branch's own pre-DPO model.",
        ],
    }


def _fmt(v: dict) -> str:
    if v["point"] is None:
        return "absent"
    lo, hi = v["ci"]
    return f"{v['verdict']:>13s} {v['point']:+.3f} [{lo:+.3f},{hi:+.3f}]"


def print_comparison(result: dict) -> None:
    a, b = result["directions"]
    print(f"Reciprocal comparison: {a} vs {b}")
    for q, blk in result["per_quadrant"].items():
        print(
            f"\nquadrant {q}   n: {a}={blk['n'][a]} {b}={blk['n'][b]}   "
            f"TV(pre,post): {a}={blk['tv_baseline_to_reference'][a]:.3f} "
            f"{b}={blk['tv_baseline_to_reference'][b]:.3f}"
        )
        for name in QUESTIONS:
            ag = blk["agreement"][name]
            mark = "  ==" if ag["agree"] else "  !="
            print(f"  {name:24s}{mark}")
            print(f"      {a}: {_fmt(blk[a][name])}")
            print(f"      {b}: {_fmt(blk[b][name])}")


def load_analysis(path: Path, tag: str | None = None) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if tag:
        data["direction_tag"] = tag
    return data


def main() -> None:
    p = argparse.ArgumentParser(
        description="Side-by-side reciprocal (A->B vs B->A) Stage-2 readout."
    )
    p.add_argument("--analysis-dir", default="results/crossbranch/analysis")
    p.add_argument("--first", default="AtoB")
    p.add_argument("--second", default="BtoA")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    d = Path(args.analysis_dir)
    paths = {
        t: d / f"crossbranch_{t}_stage2_analysis.json"
        for t in (args.first, args.second)
    }
    missing = [str(v) for v in paths.values() if not v.exists()]
    if missing:
        raise SystemExit(
            "Missing Stage-2 analysis for one direction:\n  "
            + "\n  ".join(missing)
            + "\nRun analyze_stage2 for both directions first, e.g.\n"
            "  python -m src.analysis.crossbranch.analyze_stage2 "
            "--source-branch B --target-branch A"
        )

    result = compare(
        load_analysis(paths[args.first], args.first),
        load_analysis(paths[args.second], args.second),
    )
    print_comparison(result)

    out = Path(args.out) if args.out else d / "crossbranch_reciprocal_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
