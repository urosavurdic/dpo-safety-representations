"""Shared statistics helpers (WP-Stat).

Wilson CIs for complete-case rates, plus the frozen bootstrap constants and
percentile-CI helpers the confirmatory endpoints (CF1, CF2, §4.5 trajectory,
WP-Geom, WP-Decode, WP-Stat) all use. See docs/audit/analysis_plan.md §2 and
§9 (frozen constants).
"""
from __future__ import annotations

import numpy as np
from statsmodels.stats.proportion import proportion_confint

# Frozen bootstrap constants (analysis_plan.md §9).
BOOTSTRAP_SEED = 20260904
BOOTSTRAP_B = 10_000
BOOTSTRAP_INTERVAL = "percentile"


def rate_with_ci(successes: int, total: int, confidence: float = 0.95):
    if total == 0:
        return {"rate": None, "ci_low": None, "ci_high": None, "n": 0}
    rate = successes / total
    ci_low, ci_high = proportion_confint(
        successes, total, alpha=1 - confidence, method="wilson"
    )
    return {"rate": rate, "ci_low": ci_low, "ci_high": ci_high, "n": total}


def percentile_ci(samples, confidence: float = 0.95) -> dict:
    """Percentile CI + point summary for a 1-D array of bootstrap replicates."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return {"mean": None, "median": None, "ci_low": None, "ci_high": None, "n": 0}
    lo = (1 - confidence) / 2 * 100
    hi = (1 + confidence) / 2 * 100
    return {
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "ci_low": float(np.percentile(samples, lo)),
        "ci_high": float(np.percentile(samples, hi)),
        "n": int(samples.size),
    }


def paired_bootstrap_ci(
    per_unit_values,
    *,
    statistic=np.mean,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """Prompt-level paired bootstrap CI on ``statistic`` of ``per_unit_values``.

    ``per_unit_values`` is one value per COMPLETE prompt unit (e.g. per-prompt
    ``SR_M3 - SR_M2`` for CF1, or per-prompt ``(SR_ablatedAD - SR_baseline) -
    (SR_ablatedRandom - SR_baseline)`` for CF2). Resamples complete units with
    replacement; percentile interval; deterministic under ``seed``.
    """
    values = np.asarray(per_unit_values, dtype=float)
    values = values[~np.isnan(values)]
    n = values.size
    if n == 0:
        return {"n_effective": 0, "point": None, "mean": None, "ci_low": None,
                "ci_high": None, "b": b, "seed": seed,
                "interval": BOOTSTRAP_INTERVAL}
    rng = np.random.default_rng(seed)
    reps = np.empty(b)
    for i in range(b):
        reps[i] = statistic(values[rng.integers(0, n, size=n)])
    lo = (1 - confidence) / 2 * 100
    hi = (1 + confidence) / 2 * 100
    return {
        "n_effective": int(n),
        "point": float(statistic(values)),
        "mean": float(reps.mean()),
        "ci_low": float(np.percentile(reps, lo)),
        "ci_high": float(np.percentile(reps, hi)),
        "b": b,
        "seed": seed,
        "interval": BOOTSTRAP_INTERVAL,
    }


def joint_resample_indices(group_index_arrays, seed: int = BOOTSTRAP_SEED, rng=None):
    """One bootstrap draw: resample each group's positions with replacement.
    ``group_index_arrays`` is a list of 1-D index arrays (e.g. [A_idx, D_idx]).
    Returns the concatenated resampled indices and a matching group-label
    array (0,1,... in the order given). Use a shared ``rng`` across a loop for
    a reproducible stream from a single ``seed``."""
    rng = rng or np.random.default_rng(seed)
    parts, labels = [], []
    for label, idx in enumerate(group_index_arrays):
        sampled = rng.choice(idx, size=len(idx), replace=True)
        parts.append(sampled)
        labels.append(np.full(len(sampled), label))
    return np.concatenate(parts), np.concatenate(labels)
