"""Matched C-pair representation deltas (WP-Adjunct), analysis_plan.md §"Adjunct".

Every C1-eligible candidate carries its exact StrongREJECT ``source_prompt``.
Comparing WITHIN each ``(source_overt, candidate_reduced_cue)`` pair - same
underlying request, only the wording changed - and aggregating the paired
differences is a materially stronger design than any cross-benchmark A-vs-C
comparison, because it holds "which specific request is this" constant as a
blocking factor.

**Honest limit (state it plainly):** this controls for "which request", NOT
for "wording and nothing else". The rewrite bundles cue-word removal with
other incidental changes (length, register, operational detail). So the paired
comparison isolates "the wording change as actually made" - a bundle - not a
single orthogonal factor. Smaller confound than the cross-benchmark case, not
zero.

CPU-only. Consumes ``_final`` activations for the paired companion set
(``source_overt`` + ``candidate`` rows sharing a ``pair_id``).
"""
from __future__ import annotations

import numpy as np

HONEST_LIMIT = (
    "Controls for WHICH request, not for wording-and-nothing-else: the rewrite "
    "bundles cue removal with length/register/detail changes. Isolates 'the "
    "wording change as made', a bundle - smaller confound than cross-benchmark, "
    "not zero (analysis_plan.md §Adjunct)."
)


def _pairs(metadata):
    by_pair = {}
    for i, row in enumerate(metadata):
        pid = row.get("pair_id")
        variant = row.get("judged_prompt_variant") or row.get("variant")
        if pid is None or variant is None:
            continue
        by_pair.setdefault(pid, {})[variant] = i
    return {pid: v for pid, v in by_pair.items()
            if "source_overt" in v and "candidate_reduced_cue" in v}


def matched_pair_deltas(pooled_final, metadata, direction, *, layer):
    """Per pair: projection onto d_AD, and L2 activation distance, at ``layer``,
    for the overt vs reduced-cue variant. Positive ``proj_delta`` => the
    reduced-cue variant sits FURTHER toward A along d_AD than its overt source."""
    pairs = _pairs(metadata)
    rows = []
    for pid, idx in sorted(pairs.items()):
        h_overt = pooled_final[idx["source_overt"], layer]
        h_cand = pooled_final[idx["candidate_reduced_cue"], layer]
        p_overt = float(h_overt @ direction[layer])
        p_cand = float(h_cand @ direction[layer])
        rows.append({
            "pair_id": pid,
            "proj_overt": p_overt,
            "proj_candidate": p_cand,
            "proj_delta_candidate_minus_overt": p_cand - p_overt,
            "l2_activation_distance": float(np.linalg.norm(h_cand - h_overt)),
        })
    return rows


def aggregate_paired(rows, *, seed=20260904, n_boot=10000):
    deltas = np.array([r["proj_delta_candidate_minus_overt"] for r in rows], float)
    dists = np.array([r["l2_activation_distance"] for r in rows], float)
    if deltas.size == 0:
        return {"n_pairs": 0, "honest_limit": HONEST_LIMIT}
    rng = np.random.default_rng(seed)
    boot = np.array([deltas[rng.integers(0, deltas.size, deltas.size)].mean()
                     for _ in range(n_boot)])
    return {
        "n_pairs": int(deltas.size),
        "mean_proj_delta": float(deltas.mean()),
        "proj_delta_ci_low": float(np.percentile(boot, 2.5)),
        "proj_delta_ci_high": float(np.percentile(boot, 97.5)),
        "mean_l2_activation_distance": float(dists.mean()),
        "seed": seed, "interval": "percentile",
        "honest_limit": HONEST_LIMIT,
    }
