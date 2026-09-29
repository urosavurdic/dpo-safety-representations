"""The 2x2 estimands: objective, corpus, and their interaction.

For a per-prompt scalar ``f``, a transition ``T = (pre -> post)`` and a
population ``P``::

    Delta_T(x) = f_post(x) - f_pre(x)          paired by record_id
    theta_T    = mean over x in P of Delta_T(x)

    OBJ  = 1/2[th_DPOa - th_SFTa] + 1/2[th_DPOd - th_SFTd]
    CORP = 1/2[th_SFTa - th_SFTd] + 1/2[th_DPOa - th_DPOd]
    INT  =    [th_DPOa - th_SFTa] -    [th_DPOd - th_SFTd]

``OBJ > 0`` means DPO moves the quantity more than safety SFT does, from the
same checkpoint on the same data.

All four transitions are evaluated on the SAME 654 record_ids, so each estimand
is linear in the four per-prompt difference columns and collapses to a single
per-prompt contribution. Bootstrapping that one vector IS the shared-resample
bootstrap -- linearity commutes with averaging -- and it inherits the frozen
seed, B and percentile interval for free. Resampling the transitions
independently would destroy the pairing across eight checkpoints and inflate the
interval; ``tests/stagecmp/test_contrasts.py`` asserts both facts.

Same algebra as ``confirmatory_behavioral_endpoints.compute_crossfit_branch_contrasts``
applies to the history/corpus factors. Different factor, identical shape.
"""
from __future__ import annotations

import numpy as np

from src.common.stats import BOOTSTRAP_B, BOOTSTRAP_SEED, paired_bootstrap_ci
from src.stagecmp import STATUS_EXPLORATORY
from src.stagecmp.behav_metrics import per_prompt, quadrant_of
from src.stagecmp.loaders import JUDGE_DEFAULT
from src.stagecmp.pairs import CONTRAST_CAVEAT, TRANSITIONS_2X2

ARMS = ("sft_alpaca", "dpo_alpaca", "sft_dolly", "dpo_dolly")


def per_prompt_deltas(
    metric: str = "withhold",
    judge_path=JUDGE_DEFAULT,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """``(record_ids, {arm: Delta_T})`` over prompts complete in every arm."""
    values = {}
    for arm in ARMS:
        pre, post, _obj, _corp = TRANSITIONS_2X2[arm]
        values[arm] = (per_prompt(pre, metric, judge_path),
                       per_prompt(post, metric, judge_path))

    shared = None
    for pre_vals, post_vals in values.values():
        ids = set(pre_vals) & set(post_vals)
        shared = ids if shared is None else (shared & ids)
    ids = sorted(shared or [])

    deltas = {
        arm: np.array([post_vals[i] - pre_vals[i] for i in ids])
        for arm, (pre_vals, post_vals) in values.items()
    }
    return ids, deltas


def factorial_2x2(
    metric: str = "withhold",
    quadrant: str | None = None,
    judge_path=JUDGE_DEFAULT,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """OBJ, CORP and INT with paired bootstrap CIs, plus the four arm means."""
    ids, deltas = per_prompt_deltas(metric, judge_path)
    if quadrant is not None:
        quads = quadrant_of(TRANSITIONS_2X2["sft_alpaca"][0], judge_path)
        keep = np.array([k for k, i in enumerate(ids) if quads.get(i) == quadrant])
        ids = [ids[k] for k in keep]
        deltas = {arm: d[keep] for arm, d in deltas.items()}

    if not ids:
        return {"status": STATUS_EXPLORATORY, "metric": metric,
                "quadrant": quadrant or "all", "n_effective": 0}

    d_sa, d_da = deltas["sft_alpaca"], deltas["dpo_alpaca"]
    d_sd, d_dd = deltas["sft_dolly"], deltas["dpo_dolly"]

    obj = 0.5 * ((d_da - d_sa) + (d_dd - d_sd))
    corp = 0.5 * ((d_sa - d_sd) + (d_da - d_dd))
    inter = (d_da - d_sa) - (d_dd - d_sd)

    out = {
        "status": STATUS_EXPLORATORY,
        "metric": metric,
        "quadrant": quadrant or "all",
        "n_effective": len(ids),
        "bootstrap": {"seed": seed, "b": b, "interval": "percentile"},
        "caveat": CONTRAST_CAVEAT,
        "arms": {
            arm: {
                "pre": TRANSITIONS_2X2[arm][0],
                "post": TRANSITIONS_2X2[arm][1],
                "objective": TRANSITIONS_2X2[arm][2],
                "corpus": TRANSITIONS_2X2[arm][3],
                "theta": float(deltas[arm].mean()),
                **{k: v for k, v in paired_bootstrap_ci(deltas[arm], b=b, seed=seed).items()
                   if k in ("ci_low", "ci_high")},
            }
            for arm in ARMS
        },
        "OBJ": paired_bootstrap_ci(obj, b=b, seed=seed),
        "CORP": paired_bootstrap_ci(corp, b=b, seed=seed),
        "INT": paired_bootstrap_ci(inter, b=b, seed=seed),
        "sign_convention": {
            "OBJ": "positive => DPO moves the metric more than safety SFT, from the same start on the same data",
            "CORP": "positive => Alpaca arms move more than Dolly arms",
            "INT": "positive => the objective gap is larger in the Alpaca column than the Dolly column",
        },
    }
    out["within_corpus_gaps"] = {
        "alpaca_dpo_minus_sft": paired_bootstrap_ci(d_da - d_sa, b=b, seed=seed),
        "dolly_dpo_minus_sft": paired_bootstrap_ci(d_dd - d_sd, b=b, seed=seed),
    }
    return out


def selectivity_2x2(
    metric: str = "withhold",
    harm_quadrant: str = "A",
    benign_quadrant: str = "D",
    judge_path=JUDGE_DEFAULT,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Does a transition make the model more DISCRIMINATING, or just more refusing?

    ``selectivity_T = mean Delta_T over harmful - mean Delta_T over benign``.
    A transition that raises withholding equally on both gains nothing.

    Unlike OBJ/CORP/INT this is a difference between two DIFFERENT prompt sets,
    so it does not collapse to one per-prompt contribution. It uses a shared
    resample of the two groups per replicate (``joint_resample_indices``
    pattern) with one rng across all four arms, so the pairing across the eight
    checkpoints survives.
    """
    ids, deltas = per_prompt_deltas(metric, judge_path)
    quads = quadrant_of(TRANSITIONS_2X2["sft_alpaca"][0], judge_path)
    h_idx = np.array([k for k, i in enumerate(ids) if quads.get(i) == harm_quadrant])
    b_idx = np.array([k for k, i in enumerate(ids) if quads.get(i) == benign_quadrant])
    if h_idx.size == 0 or b_idx.size == 0:
        return {"status": STATUS_EXPLORATORY, "n_harm": int(h_idx.size),
                "n_benign": int(b_idx.size)}

    def sel(d, hi, bi):
        return float(d[hi].mean() - d[bi].mean())

    point = {arm: sel(deltas[arm], h_idx, b_idx) for arm in ARMS}
    rng = np.random.default_rng(seed)
    reps = {arm: np.empty(b) for arm in ARMS}
    obj_reps = np.empty(b)
    for r in range(b):
        hs = rng.integers(0, h_idx.size, size=h_idx.size)
        bs = rng.integers(0, b_idx.size, size=b_idx.size)
        hh, bb = h_idx[hs], b_idx[bs]
        vals = {arm: sel(deltas[arm], hh, bb) for arm in ARMS}
        for arm in ARMS:
            reps[arm][r] = vals[arm]
        obj_reps[r] = 0.5 * ((vals["dpo_alpaca"] - vals["sft_alpaca"])
                             + (vals["dpo_dolly"] - vals["sft_dolly"]))

    def ci(x):
        return {"point": None, "ci_low": float(np.percentile(x, 2.5)),
                "ci_high": float(np.percentile(x, 97.5))}

    out = {
        "status": STATUS_EXPLORATORY,
        "metric": metric,
        "definition": f"mean Delta over {harm_quadrant} minus mean Delta over {benign_quadrant}",
        "n_harm": int(h_idx.size),
        "n_benign": int(b_idx.size),
        "bootstrap": {"seed": seed, "b": b, "interval": "percentile",
                      "note": "two-group shared resample, one rng across arms"},
        "caveat": CONTRAST_CAVEAT,
        "arms": {arm: {**ci(reps[arm]), "point": point[arm]} for arm in ARMS},
        "OBJ_selectivity": {**ci(obj_reps),
                            "point": 0.5 * ((point["dpo_alpaca"] - point["sft_alpaca"])
                                            + (point["dpo_dolly"] - point["sft_dolly"]))},
        "reading": (
            "Positive arm selectivity means the transition raised withholding on "
            "harmful prompts more than on benign ones. A transition can have a "
            "large OBJ on both quadrants and near-zero selectivity, which would "
            "mean it raised refusal indiscriminately."
        ),
    }
    return out


def independent_resample_comparison(
    metric: str = "withhold",
    quadrant: str | None = None,
    judge_path=JUDGE_DEFAULT,
    b: int = 2000,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Diagnostic only: OBJ under INDEPENDENT per-arm resampling.

    This is the WRONG estimator: the four arms are evaluated on the same
    prompts, so treating their means as independent misstates the sampling
    distribution.

    Note what is and is not claimed. For a SIMPLE difference the paired
    bootstrap is narrower whenever the two arms are positively correlated, which
    they are. For a DIFFERENCE OF DIFFERENCES such as OBJ that does not follow:

        Var(A - B + C - D) = sum of variances
                             - 2Cov(A,B) + 2Cov(A,C) - 2Cov(A,D)
                             - 2Cov(B,C) + 2Cov(B,D) - 2Cov(C,D)

    the cross terms carry mixed signs, so the paired interval can be narrower or
    wider than the independent one depending on the correlation structure.
    Pairing is used here because it is CORRECT, not because it is tighter.
    """
    ids, deltas = per_prompt_deltas(metric, judge_path)
    if quadrant is not None:
        quads = quadrant_of(TRANSITIONS_2X2["sft_alpaca"][0], judge_path)
        keep = np.array([k for k, i in enumerate(ids) if quads.get(i) == quadrant])
        deltas = {arm: d[keep] for arm, d in deltas.items()}
    n = len(next(iter(deltas.values())))
    rng = np.random.default_rng(seed)
    reps = np.empty(b)
    for i in range(b):
        draws = {arm: d[rng.integers(0, n, size=n)].mean() for arm, d in deltas.items()}
        reps[i] = 0.5 * ((draws["dpo_alpaca"] - draws["sft_alpaca"])
                         + (draws["dpo_dolly"] - draws["sft_dolly"]))
    return {
        "note": "WRONG ESTIMATOR, kept as a diagnostic: independent per-arm resampling",
        "ci_low": float(np.percentile(reps, 2.5)),
        "ci_high": float(np.percentile(reps, 97.5)),
        "width": float(np.percentile(reps, 97.5) - np.percentile(reps, 2.5)),
        "b": b,
        "seed": seed,
    }
