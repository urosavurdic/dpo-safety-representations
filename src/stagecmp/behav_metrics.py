"""Behavioural distance between checkpoints on the identical 654 prompts.

Scorer-parametric from the start. The frozen regex classifier is complete for
all nine stages, so the full matrix is available with no GPU. StrongREJECT and
WildGuard currently cover only quadrant-C rows of M2 and M3 (see
``loaders.scorer_coverage``); once a judging pass fills them in, the same
functions produce the same tables under those scorers with no rewrite.

Two disciplines carried over from ``src/crossbranch/analyze.py``, both of which
exist because the obvious shortcut is wrong:

* total variation is never reported without the four-way distribution beside
  it, because a rise in the degenerate rate moves TV exactly as much as a rise
  in refusal does;
* TV is never reported without per-prompt agreement beside it, because two
  conditions can have identical label DISTRIBUTIONS while disagreeing on every
  individual prompt.
"""
from __future__ import annotations

import numpy as np

from src.analysis.mcnemar_direction_specificity import mcnemar_exact
from src.common.quadrants import CATEGORIES, QUADRANTS
from src.common.stats import (
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    bootstrap_nonlinear_contrast,
    paired_bootstrap_ci,
    rate_with_ci,
)
from src.crossbranch.analyze import distribution, total_variation
from src.stagecmp.loaders import JUDGE_DEFAULT, behavior_rows

#: per-prompt scalars this module can extract, and which scorer each needs
METRICS = {
    "withhold": "regex",        # refusal OR soft deflection
    "refusal": "regex",         # refusal only
    "degenerate": "regex",
    "sr_score": "strong_reject",
    "wg_harm": "wildguard",
    "wg_refusal": "wildguard",
}
BINARY_METRICS = {"withhold", "refusal", "degenerate", "wg_harm", "wg_refusal"}


def label_of(row: dict) -> str | None:
    """Four-way regex label, in the frozen precedence order from
    ``common/refusal_classifier.classify_completion``:
    degenerate > refusal > soft_deflection > comply."""
    rx = row.get("regex") or {}
    if rx.get("refused") is None:
        return None
    if rx.get("degenerate"):
        return "degenerate"
    if rx.get("refused"):
        return "refusal"
    if rx.get("soft_deflection"):
        return "soft_deflection"
    return "comply"


def value_of(row: dict, metric: str) -> float | None:
    """One per-prompt scalar, or ``None`` when that scorer did not run."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; known: {sorted(METRICS)}")
    if METRICS[metric] == "regex":
        label = label_of(row)
        if label is None:
            return None
        if metric == "withhold":
            return float(label in ("refusal", "soft_deflection"))
        if metric == "refusal":
            return float(label == "refusal")
        return float(label == "degenerate")
    if metric == "sr_score":
        return (row.get("strong_reject") or {}).get("score")
    wg = row.get("wildguard") or {}
    return wg.get("response_harm" if metric == "wg_harm" else "response_refusal")


def per_prompt(stage: str, metric: str, judge_path=JUDGE_DEFAULT) -> dict[str, float]:
    """``record_id -> value``, omitting prompts this scorer did not cover."""
    out = {}
    for rid, row in behavior_rows(stage, judge_path).items():
        v = value_of(row, metric)
        if v is not None:
            out[rid] = float(v)
    return out


def quadrant_of(stage: str, judge_path=JUDGE_DEFAULT) -> dict[str, str]:
    return {rid: row["quadrant"] for rid, row in behavior_rows(stage, judge_path).items()}


# ------------------------------------------------------------ per stage ----- #

def stage_profile(stage: str, judge_path=JUDGE_DEFAULT) -> dict:
    """Four-way label distribution and rates per quadrant for one stage."""
    rows = behavior_rows(stage, judge_path)
    by_quadrant = {q: [] for q in QUADRANTS}
    for row in rows.values():
        label = label_of(row)
        if label is not None:
            by_quadrant[row["quadrant"]].append(label)

    out = {"stage": stage, "n_rows": len(rows), "per_quadrant": {}}
    for q, labels in by_quadrant.items():
        n = len(labels)
        counts = {c: labels.count(c) for c in CATEGORIES}
        withhold = counts["refusal"] + counts["soft_deflection"]
        out["per_quadrant"][q] = {
            "n": n,
            "four_way_counts": counts,
            "four_way": {c: (counts[c] / n if n else None) for c in CATEGORIES},
            "withhold": rate_with_ci(withhold, n),
            "refusal": rate_with_ci(counts["refusal"], n),
            "degenerate": rate_with_ci(counts["degenerate"], n),
        }
    return out


# --------------------------------------------------------------- pairwise ---- #

def _shared(stage_x, stage_y, metric, judge_path, quadrant=None):
    """Complete units only: a prompt unusable at EITHER stage is dropped whole,
    the same rule CF1 uses."""
    vx = per_prompt(stage_x, metric, judge_path)
    vy = per_prompt(stage_y, metric, judge_path)
    quad = quadrant_of(stage_x, judge_path)
    ids = sorted(set(vx) & set(vy))
    if quadrant is not None:
        ids = [i for i in ids if quad.get(i) == quadrant]
    return ids, np.array([vx[i] for i in ids]), np.array([vy[i] for i in ids])


def paired_difference(
    stage_x: str,
    stage_y: str,
    metric: str = "withhold",
    quadrant: str | None = None,
    judge_path=JUDGE_DEFAULT,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired bootstrap CI on ``mean(y - x)`` over complete prompt units."""
    ids, x, y = _shared(stage_x, stage_y, metric, judge_path, quadrant)
    n_x = len(per_prompt(stage_x, metric, judge_path))
    n_y = len(per_prompt(stage_y, metric, judge_path))
    result = paired_bootstrap_ci(y - x, b=b, seed=seed)
    result.update(
        {
            "pre": stage_x,
            "post": stage_y,
            "metric": metric,
            "quadrant": quadrant or "all",
            "mean_pre": float(x.mean()) if len(x) else None,
            "mean_post": float(y.mean()) if len(y) else None,
            "dropped_incomplete": {"pre_rows": n_x, "post_rows": n_y, "shared": len(ids)},
        }
    )
    return result


def mcnemar(
    stage_x: str,
    stage_y: str,
    metric: str = "withhold",
    quadrant: str | None = None,
    judge_path=JUDGE_DEFAULT,
) -> dict:
    """Exact-binomial McNemar on a paired binary outcome, keyed on record_id.

    ``b`` and ``c`` are DISCORDANT-PAIR COUNTS here. In the bootstrap artifacts
    of this repo the key ``b`` means the bootstrap replicate count instead --
    see results/README.md. Same letter, unrelated meanings.
    """
    if metric not in BINARY_METRICS:
        raise ValueError(f"McNemar needs a binary metric; {metric!r} is not one")
    ids, x, y = _shared(stage_x, stage_y, metric, judge_path, quadrant)
    xb, yb = x.astype(bool), y.astype(bool)
    b = int(np.sum(xb & ~yb))   # yes at x, no at y
    c = int(np.sum(~xb & yb))   # no at x, yes at y
    return {
        "pre": stage_x,
        "post": stage_y,
        "metric": metric,
        "quadrant": quadrant or "all",
        "n_pairs": len(ids),
        "b_def": f"{stage_x}=1 and {stage_y}=0 (DISCORDANT COUNT, not bootstrap B)",
        "c_def": f"{stage_x}=0 and {stage_y}=1",
        "b": b,
        "c": c,
        "n_discordant": b + c,
        "p_exact": mcnemar_exact(b, c),
    }


def tv_and_agreement(
    stage_x: str,
    stage_y: str,
    quadrant: str | None = None,
    judge_path=JUDGE_DEFAULT,
) -> dict:
    """Four-way TV distance, the two distributions it came from, and per-prompt
    agreement. All three together, never TV alone."""
    rows_x = behavior_rows(stage_x, judge_path)
    rows_y = behavior_rows(stage_y, judge_path)
    ids = sorted(set(rows_x) & set(rows_y))
    if quadrant is not None:
        ids = [i for i in ids if rows_x[i]["quadrant"] == quadrant]
    lx = [label_of(rows_x[i]) for i in ids]
    ly = [label_of(rows_y[i]) for i in ids]
    keep = [k for k, (a, b_) in enumerate(zip(lx, ly)) if a and b_]
    lx = [lx[k] for k in keep]
    ly = [ly[k] for k in keep]
    if not lx:
        return {"pre": stage_x, "post": stage_y, "quadrant": quadrant or "all", "n": 0}
    p, q = distribution(lx), distribution(ly)
    agree = float(np.mean([a == b_ for a, b_ in zip(lx, ly)]))
    return {
        "pre": stage_x,
        "post": stage_y,
        "quadrant": quadrant or "all",
        "n": len(lx),
        "total_variation": total_variation(p, q),
        "four_way_pre": dict(zip(CATEGORIES, [float(v) for v in p])),
        "four_way_post": dict(zip(CATEGORIES, [float(v) for v in q])),
        "per_prompt_agreement": agree,
        "reading": (
            "TV is an aggregate distance: it moves when the degenerate rate "
            "moves, and it can be near zero while per-prompt agreement is at "
            "chance. Read all three fields together."
        ),
    }


# ------------------------------------------------- scale-robust framing ----- #

#: Haldane-Anscombe correction. Applied because the DPO arms reach a refusal
#: rate of 1.000 on quadrant C, where the untransformed odds are infinite.
HALDANE = 0.5
#: minimum withholding prompts in BOTH quadrants before a risk ratio is
#: quotable. Fixed in advance; the log odds ratio carries no such limit
#: because the Haldane correction keeps it finite at a rate of 0 or 1.
MIN_SUCCESSES_FOR_RATIO = 10


def _odds(successes: int, n: int, correction: float = HALDANE) -> float:
    return (successes + correction) / (n - successes + correction)


def discrimination_report(
    stage: str,
    metric: str = "wg_refusal",
    judge_path=JUDGE_DEFAULT,
    pairs=(("A", "D"), ("C", "B")),
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Ratio-scale discrimination for ONE checkpoint, as a scale-robust check.

    An additive gap between two quadrants grows with how much the checkpoint
    withholds overall, so a transition that moves every quadrant a long way
    will show a large gap without discriminating any better. Two quantities
    here are invariant to that:

    * **risk ratio** ``p_harmful / p_benign`` -- 1.0 for a checkpoint that
      withholds indiscriminately, whatever its level.
    * **log odds ratio**, Haldane-corrected, which stays finite at a rate of
      exactly 0 or 1 and is the right scale when rates approach a bound.

    Both are reported so it can be stated plainly what the additive
    contrasts do **not** establish: that one objective discriminates better
    than the other. Read them beside ``rate_by_quadrant``, because a ratio
    between two tiny rates is precise-looking and nearly uninformative.
    """
    vals = per_prompt(stage, metric, judge_path)
    quads = quadrant_of(stage, judge_path)
    if metric not in BINARY_METRICS:
        raise ValueError(f"{metric!r} is continuous; risk ratios need a binary metric")

    rids = sorted(vals)
    y = np.array([vals[r] for r in rids], dtype=float)
    q = np.array([quads[r] for r in rids])

    rates = {}
    for quadrant in QUADRANTS:
        m = q == quadrant
        rates[quadrant] = rate_with_ci(int(y[m].sum()), int(m.sum()))

    out = {"stage": stage, "metric": metric, "rate_by_quadrant": rates,
           "haldane_correction": HALDANE, "ratios": {}}

    for q_num, q_den in pairs:
        i_num = np.flatnonzero(q == q_num)
        i_den = np.flatnonzero(q == q_den)

        def risk_ratio(g):
            den = float(y[g[1]].mean())
            if den <= 0.0:
                return float("nan")
            return float(y[g[0]].mean() / den)

        def log_odds_ratio(g):
            o_num = _odds(int(y[g[0]].sum()), len(g[0]))
            o_den = _odds(int(y[g[1]].sum()), len(g[1]))
            return float(np.log(o_num / o_den))

        successes = {q_num: int(y[i_num].sum()), q_den: int(y[i_den].sum())}
        out["ratios"][f"{q_num}_over_{q_den}"] = {
            "risk_ratio": bootstrap_nonlinear_contrast(
                [i_num, i_den], risk_ratio, b=b, seed=seed),
            "log_odds_ratio": bootstrap_nonlinear_contrast(
                [i_num, i_den], log_odds_ratio, b=b, seed=seed),
            "successes": successes,
            # A risk ratio cannot exceed 1 / p_denominator, so once the
            # numerator rate approaches 1 the statistic is pinned against a
            # ceiling set by the OTHER quadrant and stops being able to express
            # the difference. Recorded so the comparison can be read honestly:
            # a point estimate near its own ceiling is a floor on the effect,
            # not a measurement of it. The log odds ratio is unbounded.
            "risk_ratio_ceiling": (
                float(len(i_den) / max(successes[q_den], 1))
                if successes[q_den] else None
            ),
            # A ratio whose numerator or denominator rests on a handful of
            # prompts is not quotable however tight its interval looks: with a
            # zero numerator it reads as a significant *reverse* effect, and
            # with a near-zero denominator it explodes. The log odds ratio is
            # Haldane-corrected and stays usable, which is why it is the one
            # reported first.
            "risk_ratio_reliable": bool(min(successes.values()) >= MIN_SUCCESSES_FOR_RATIO),
            "min_successes_required": MIN_SUCCESSES_FOR_RATIO,
        }

    out["reading"] = (
        "A risk ratio near 1 means the checkpoint does not distinguish the two "
        "quadrants at all. These are properties of ONE checkpoint, not of a "
        "transition, so they answer 'does this model discriminate' rather than "
        "'did this training step change discrimination'. When the denominator "
        "rate is only a few prompts the ratio is unstable however tight the "
        "interval looks -- check rate_by_quadrant before quoting it."
    )
    return out


def movement_report(
    pre: str,
    post: str,
    metric: str = "wg_refusal",
    judge_path=JUDGE_DEFAULT,
) -> dict:
    """How far a transition moved the metric, averaged over quadrants.

    This is the magnitude term the additive contrasts confound with
    discrimination. Reporting it beside them is what shows the objective
    difference is one of magnitude.
    """
    rids, shared_pre, shared_post = _shared(pre, post, metric, judge_path)
    quads = quadrant_of(pre, judge_path)
    per_quadrant, moves = {}, []
    for quadrant in QUADRANTS:
        idx = [k for k, r in enumerate(rids) if quads.get(r) == quadrant]
        if not idx:
            continue
        delta = float(np.mean([shared_post[k] - shared_pre[k] for k in idx]))
        per_quadrant[quadrant] = delta
        moves.append(abs(delta))
    return {
        "pre": pre,
        "post": post,
        "metric": metric,
        "delta_by_quadrant": per_quadrant,
        "mean_absolute_movement": float(np.mean(moves)) if moves else None,
        "n_shared": len(rids),
    }
