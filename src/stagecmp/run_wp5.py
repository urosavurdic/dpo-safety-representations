"""Driver for the judged-behaviour artifacts, beyond the matrix and the pairs.

    python -m src.stagecmp.run_wp5

Writes into ``results/stagecmp/``, each with a ``*_binding.json`` sidecar:

``stage_comparison_sr_score.json``
    The 2x2 factorial under StrongREJECT. The WildGuard factorial already had
    an artifact; this one did not, so the two-judge claim rested on numbers
    that existed only in a terminal.

``scale_robust_behaviour.json``
    The answer to the objection that an additive gap between quadrants grows
    with how far a transition moved anything at all. Per arm: mean absolute
    movement, and the post-checkpoint's harmful-vs-benign discrimination on
    both the risk-ratio and the log-odds scale, with intervals.

``diagnostics.json``
    Degenerate-output rate per stage -- the reason M0 carries no behavioural
    claim -- and the judge validation: WildGuard's prompt-harm label against
    the benchmark's own design labels, and the two judges against each other.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from src.common.quadrants import QUADRANTS
from src.common.stats import BOOTSTRAP_B, BOOTSTRAP_SEED
from src.stagecmp import STATUS_EXPLORATORY
from src.stagecmp.agreement import auroc, cohens_kappa, spearman
from src.stagecmp.behav_metrics import discrimination_report, movement_report
from src.stagecmp.contrasts import factorial_2x2, selectivity_2x2
from src.stagecmp.loaders import (
    JUDGE_DEFAULT,
    behavior_rows,
    require_scorer,
    scorer_coverage,
)
from src.stagecmp.pairs import TRANSITIONS_2X2
from src.v2_io import sha256_file, write_json_lf

OUT_DIR = Path("results/stagecmp")
#: The WP-5 judging pass: all nine stages, all 654 rows, both LLM judges.
#: ``loaders.JUDGE_DEFAULT`` still points at the PRE-WP-5 file, which carries
#: StrongREJECT and WildGuard on quadrant C of M2 and M3 only. Reading it by
#: accident does not raise -- it silently yields empty quadrants -- so this
#: module names the complete file explicitly and verifies coverage before
#: computing anything.
JUDGE_COMPLETE = Path(
    "results/behavioral_judges/behavioral_judges_20260930T001320Z.json"
)
ALL_STAGES = ("M0", "M1", "M2", "M3", "M3_direct",
              "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt")
#: a stage whose generations are degenerate this often carries no behavioural
#: claim. Fixed here rather than chosen after seeing which stage fails it.
DEGENERATE_LIMIT = 0.10
#: the harmful quadrants, by benchmark design
HARMFUL_QUADRANTS = ("A", "C")
#: a judge may miss this fraction of a stage's rows to malformed
#: generations before the run is refused. One row of M1_alt is
#: malformed under WildGuard; anything near this bound is not.
MAX_UNSCORED_FRACTION = 0.01


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _binding(target: Path, judge_path, extra=None) -> None:
    write_json_lf(
        target.with_name(target.stem + "_binding.json"),
        {
            "status": STATUS_EXPLORATORY,
            "produced_utc": _now(),
            "inputs": {"judge_file": str(judge_path), **(extra or {})},
            "sha256": {"judge_file": sha256_file(judge_path)},
        },
    )


# --------------------------------------------------------------- R3: SR ----- #

def _terms(factorial: dict) -> dict:
    """``factorial_2x2`` returns OBJ/CORP/INT at the top level alongside its
    metadata. Lift just the three terms plus the arm means, so the artifact
    matches the shape the WildGuard one already has."""
    out = {term: factorial[term] for term in ("OBJ", "CORP", "INT")}
    out["arms"] = factorial.get("arms")
    out["n_effective"] = factorial.get("n_effective")
    return out


def sr_factorial(judge_path=JUDGE_DEFAULT, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED) -> dict:
    """The 2x2 under StrongREJECT, which scores harmful assistance.

    Sign convention matters and is easy to misread: ``sr_score`` RISES with
    harmful assistance, so a NEGATIVE objective term means DPO reduced harmful
    assistance more than safety SFT did. The WildGuard artifact measures
    refusal, which moves the other way; agreement between them is agreement in
    substance, not in sign.
    """
    return {
        "status": STATUS_EXPLORATORY,
        "produced_utc": _now(),
        "metric": "sr_score",
        "scorer": "strong_reject",
        "sign_convention": (
            "sr_score rises with harmful assistance. OBJ < 0 means DPO reduced "
            "harmful assistance MORE than safety SFT. The wg_refusal artifact "
            "measures refusal and therefore carries the opposite sign."
        ),
        "contrasts_2x2": {
            # same shape as stage_comparison_wg_refusal.json, so memo.py and
            # make_numbers.py read both artifacts through one code path
            "by_quadrant": {
                q: _terms(factorial_2x2(metric="sr_score", quadrant=q,
                                        judge_path=judge_path, b=b, seed=seed))
                for q in QUADRANTS
            },
            "selectivity": {
                f"{h}_minus_{n}": selectivity_2x2(
                    metric="sr_score", harm_quadrant=h, benign_quadrant=n,
                    judge_path=judge_path, b=b, seed=seed)
                for h, n in (("A", "D"), ("C", "B"))
            },
        },
    }


# --------------------------------------------- R2: the scale-robust view ---- #

def scale_robust_behaviour(
    metric: str = "wg_refusal",
    judge_path=JUDGE_DEFAULT,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Magnitude and discrimination, separated.

    The additive quadrant contrasts confound the two: an arm that moves every
    quadrant a long way shows a large A-minus-D gap without discriminating any
    better. ``mean_absolute_movement`` is the magnitude term; the risk ratio
    and log odds ratio are magnitude-invariant.

    The two ratio scales disagree in an instructive way and both are reported.
    The risk ratio is unusable where the benign rate rests on a handful of
    prompts -- it is the interval, not the point estimate, that shows this.
    """
    arms, checkpoints = {}, {}
    for arm, (pre, post, *_rest) in TRANSITIONS_2X2.items():
        arms[arm] = {
            "pre": pre,
            "post": post,
            "movement": movement_report(pre, post, metric, judge_path),
        }
        if post not in checkpoints:
            checkpoints[post] = discrimination_report(
                post, metric, judge_path, b=b, seed=seed)
        if pre not in checkpoints:
            checkpoints[pre] = discrimination_report(
                pre, metric, judge_path, b=b, seed=seed)

    return {
        "status": STATUS_EXPLORATORY,
        "produced_utc": _now(),
        "metric": metric,
        "arms": arms,
        "discrimination_by_checkpoint": checkpoints,
        "reading": (
            "Compare mean_absolute_movement across arms before reading any "
            "quadrant gap: the objective difference is first of all a "
            "difference in how far the transition moved the model. For "
            "discrimination use the log odds ratio, not the risk ratio -- "
            "where a benign rate rests on a few prompts the risk ratio is "
            "unstable and its interval says so, while the Haldane-corrected "
            "log odds ratio stays finite at a rate of exactly 0 or 1."
        ),
    }


# ------------------------------------------------- R5: validity checks ------ #

def diagnostics(judge_path=JUDGE_DEFAULT) -> dict:
    """Degeneracy per stage, and what the judges are worth on THIS benchmark.

    Published F1 is measured on another distribution, so it is cited but not
    relied on. Two checks are computable here:

    * **construct** -- WildGuard's prompt-harm label never enters the
      analysis, so comparing it with the benchmark's design labels validates
      the judge and the benchmark at once;
    * **convergent** -- two judges trained independently, on different base
      models, scoring the same responses.
    """
    degeneracy = {}
    for stage in ALL_STAGES:
        rows = behavior_rows(stage, judge_path)
        flags = []
        for row in rows.values():
            rx = row.get("regex") or {}
            flags.append(bool(rx.get("degenerate")) or bool(rx.get("malformed")))
        rate = float(np.mean(flags)) if flags else None
        degeneracy[stage] = {
            "n": len(flags),
            "degenerate_rate": rate,
            "usable_for_behavioural_claims": bool(
                rate is not None and rate <= DEGENERATE_LIMIT),
        }

    quad, wg_prompt_harm, wg_refusal, wg_harm, sr = [], [], [], [], []
    for stage in ALL_STAGES:
        for row in behavior_rows(stage, judge_path).values():
            wg = row.get("wildguard") or {}
            s = (row.get("strong_reject") or {}).get("score")
            if wg.get("malformed") or wg.get("response_refusal") is None or s is None:
                continue
            quad.append(row["quadrant"])
            wg_prompt_harm.append(bool(wg.get("prompt_harm")))
            wg_refusal.append(bool(wg.get("response_refusal")))
            wg_harm.append(bool(wg.get("response_harm")))
            sr.append(float(s))

    quad = np.array(quad)
    wg_prompt_harm = np.array(wg_prompt_harm)
    wg_refusal = np.array(wg_refusal)
    wg_harm = np.array(wg_harm)
    sr = np.array(sr)
    design = np.isin(quad, HARMFUL_QUADRANTS)

    return {
        "status": STATUS_EXPLORATORY,
        "produced_utc": _now(),
        "degeneracy": {
            "limit": DEGENERATE_LIMIT,
            "rule": (
                "A stage above the limit carries NO behavioural claim: a "
                "refusal label on degenerate text does not mean the model "
                "refused. The limit was fixed before inspecting which stage "
                "fails it."
            ),
            "by_stage": degeneracy,
            "excluded": [s for s, d in degeneracy.items()
                         if not d["usable_for_behavioural_claims"]],
        },
        "judge_validation": {
            "n_complete_case": int(sr.size),
            "construct_wg_prompt_harm_vs_design": {
                "kappa": float(cohens_kappa(wg_prompt_harm, design)),
                "accuracy": float((wg_prompt_harm == design).mean()),
                "prompt_harm_rate_by_quadrant": {
                    q: float(wg_prompt_harm[quad == q].mean()) for q in QUADRANTS
                },
                "reading": (
                    "prompt_harm is not used anywhere in the analysis, so this "
                    "is an independent check that the judge reads this "
                    "benchmark's intent axis as designed -- and, equally, that "
                    "the intent axis is real."
                ),
            },
            "convergent_wg_vs_strongreject": {
                "refusal_auroc_against_sr": float(auroc(-sr, wg_refusal)),
                "refusal_spearman_against_sr": float(spearman(sr, wg_refusal.astype(float))),
                "kappa_refusal_vs_sr_below": {
                    f"{t:g}": float(cohens_kappa(wg_refusal, sr < t))
                    for t in (0.1, 0.2, 0.3, 0.5)
                },
                "harm_auroc_against_sr": float(auroc(sr, wg_harm)),
                "mean_sr_when_wg_says_refused": float(sr[wg_refusal].mean()),
                "mean_sr_when_wg_says_complied": float(sr[~wg_refusal].mean()),
                "reading": (
                    "The two judges agree well on REFUSAL and much less well on "
                    "graded HARM. They are therefore not interchangeable, and "
                    "refusal is the metric the behavioural analysis leads with. "
                    "Any kappa here needs a threshold on a continuous score, "
                    "which is why AUROC is the headline and the kappas are "
                    "shown across several thresholds rather than one."
                ),
            },
            "published_properties": {
                "wildguard_f1_refusal": 0.886,
                "wildguard_f1_prompt_harm": 0.889,
                "strongreject_spearman_vs_human": 0.84,
                "reading": (
                    "Author-reported, on the authors' own test sets. Cited for "
                    "provenance; the checks above are what apply here."
                ),
            },
        },
    }


def assert_complete_coverage(judge_path) -> dict:
    """Fail before computing if either LLM judge is materially short of rows.

    The failure this prevents is not hypothetical: with the pre-WP-5 judge
    file, ``factorial_2x2`` on a quadrant other than C raises deep inside an
    index expression, and on a different code path it would simply have
    produced an empty-quadrant mean.

    ``require_scorer`` demands every row, which is the right rule for a
    preregistered producer but wrong here: a malformed generation is a
    legitimate complete-case exclusion and one such row exists (M1_alt,
    WildGuard). So the gate is a stated tolerance, and the exact shortfall is
    recorded in the artifact rather than waved through.
    """
    coverage = scorer_coverage(ALL_STAGES, judge_path)
    short = {}
    for scorer in ("strong_reject", "wildguard"):
        for stage, counts in coverage.items():
            n = counts["n_rows"]
            missing = n - counts[scorer]
            if n and missing / n > MAX_UNSCORED_FRACTION:
                short[f"{scorer}::{stage}"] = f"{counts[scorer]}/{n}"
    if short:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(short.items()))
        raise SystemExit(
            f"refusing to run: judge coverage below the "
            f"{MAX_UNSCORED_FRACTION:.0%} tolerance in {judge_path}\n"
            f"  {detail}\n  the complete pass is {JUDGE_COMPLETE}"
        )
    return coverage


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default=str(JUDGE_COMPLETE),
                    help="judge file; must cover all nine stages")
    ap.add_argument("--b", type=int, default=BOOTSTRAP_B)
    args = ap.parse_args()

    coverage = assert_complete_coverage(args.judge)
    print(f"coverage verified on {args.judge}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {
        "stage_comparison_sr_score": sr_factorial(args.judge, b=args.b),
        "scale_robust_behaviour": scale_robust_behaviour(
            "wg_refusal", args.judge, b=args.b),
        "diagnostics": diagnostics(args.judge),
    }
    payloads["diagnostics"]["scorer_coverage"] = coverage
    for name, payload in payloads.items():
        target = OUT_DIR / f"{name}.json"
        write_json_lf(target, payload)
        _binding(target, args.judge)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
