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


ADJUNCT_EXTRACT_HINT = (
    "companion 'source_overt' activations are absent. Build + extract them:\n"
    "  python -m src.analysis.build_c_source_overt_adjunct\n"
    "  python -m src.analysis.v2_pipeline extract --stage M3 \\\n"
    "    --latest-pointer data/frozen_v2/adjunct_c_source_overt.LATEST_BENCHMARK.json \\\n"
    "    --split-manifest data/frozen_v2/adjunct_c_source_overt.split_manifest.json \\\n"
    "    --namespace c_source_overt"
)


def _load_meta(path):
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired_deltas_from_two_arrays(cand_arr, cand_ids, overt_arr, overt_ids, direction, *, layer):
    """cand_ids[i] / overt_ids[j] are C record_ids (overt with any
    '__source_overt' suffix already stripped). Pairs by shared record_id."""
    overt_ix = {rid: j for j, rid in enumerate(overt_ids)}
    rows = []
    for i, rid in enumerate(cand_ids):
        j = overt_ix.get(rid)
        if j is None:
            continue
        h_cand = cand_arr[i, layer]
        h_overt = overt_arr[j, layer]
        rows.append({
            "record_id": rid,
            "proj_overt": float(h_overt @ direction[layer]),
            "proj_candidate": float(h_cand @ direction[layer]),
            "proj_delta_candidate_minus_overt": float((h_cand - h_overt) @ direction[layer]),
            "l2_activation_distance": float(np.linalg.norm(h_cand - h_overt)),
        })
    return rows


def main():  # pragma: no cover - CLI over on-disk activations; math tested above
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="M3")
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--companion-act-dir", default="results/companions/c_source_overt/activations")
    parser.add_argument("--direction", default=None,
                        help="{stage}_direction_final.npy (default: results/refusal_direction/)")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--out", default="results/summaries/matched_pair_representation.json")
    args = parser.parse_args()

    cand_final = Path(args.act_dir) / f"{args.stage}_final.npy"
    overt_final = Path(args.companion_act_dir) / f"{args.stage}_final.npy"
    if not overt_final.exists():
        print(ADJUNCT_EXTRACT_HINT)
        return
    direction_path = Path(args.direction) if args.direction else \
        Path("results/refusal_direction") / f"{args.stage}_direction_final.npy"

    cand_arr = np.load(cand_final)
    cand_meta = _load_meta(Path(args.act_dir) / f"{args.stage}_metadata.json")
    overt_arr = np.load(overt_final)
    overt_meta = _load_meta(Path(args.companion_act_dir) / f"{args.stage}_metadata.json")
    direction = np.load(direction_path)

    cand_ids = [
        r.get("record_id") for r in cand_meta if r.get("quadrant") == "C"
    ]
    cand_rows_idx = [i for i, r in enumerate(cand_meta) if r.get("quadrant") == "C"]
    cand_arr = cand_arr[cand_rows_idx]

    overt_ids = [str(r.get("record_id", "")).replace("__source_overt", "") for r in overt_meta]

    rows = paired_deltas_from_two_arrays(
        cand_arr, cand_ids, overt_arr, overt_ids, direction, layer=args.layer,
    )
    agg = aggregate_paired(rows)
    report = {
        "stage": args.stage, "layer": args.layer,
        "direction_file": str(direction_path),
        "per_pair": rows,
        "aggregate": agg,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if agg["n_pairs"]:
        print(f"{agg['n_pairs']} matched pairs; mean proj delta (candidate - overt) = "
              f"{agg['mean_proj_delta']:+.4f} 95% CI "
              f"[{agg['proj_delta_ci_low']:+.4f}, {agg['proj_delta_ci_high']:+.4f}]")
    print(f"-> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
