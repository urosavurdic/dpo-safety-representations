"""Stage-2 cross-branch transfer analysis (CPU, torch-free).

Unlike Stage 1 there is **no single pass/fail gate**. This reports, per
quadrant (all four, separately), a set of arm comparisons the reader
interprets together:

    dTV_q(arm)  = TV(arm_q, B3_q) - TV(B2_q, B3_q)        negative = toward B3
    target_label_match(arm) = mean_i 1[label_arm(i) == label_B3(i)]   per quadrant
    refusal_shift(arm)      = refusal_rate(arm_q) - refusal_rate(B2_q)      (A, C)
    overrefusal_shift(arm)  = (refusal+soft_deflection)_q(arm) - same(B2)   (B, D)

plus PAIRED contrasts (first arm minus second), each with a 95% bootstrap CI:

    identity - shuf_wq       prompt-conditioned transfer vs a shared per-quadrant
                             offset. shuffle keeps the per-quadrant delta
                             *distribution* and destroys the prompt<->delta
                             pairing, so a difference CI that excludes 0
                             (identity more negative) is evidence the pairing
                             carries signal; a CI covering 0 means the movement
                             is a shared offset, not prompt-specific.
    identity - normmatched   vs a generic perturbation of the same per-row norm.
    identity - dosematched   magnitude control. dosematched is a STRICTLY
                             STRONGER oracle: it consumes ||Delta_B(x)||, the
                             target branch's own post-DPO change norm.
    identity - dir_source    the full delta vs branch A's generic A-D direction.
    dir_target - dir_source  within-branch concept vs transferred concept.
    identity - own_delta      (only if own_delta_target raw is present) cross-
                             branch identity vs the branch's own delta.

All statistics use a PAIRED bootstrap over identical record_ids WITHIN the
quadrant: one shared resampled index set per replicate, every arm plus B2 and
B3 recomputed on it. seed 20260904, B=10000, percentile -- the frozen
convention (src.eval_stats). Resampling arms independently would destroy the
pairing and inflate every difference interval.

Oracle disclosure, per arm: identity consumes the SOURCE branch's post-DPO
activation for that prompt only (weakest oracle); dosematched additionally
consumes ||Delta_B(x)|| and is a strictly stronger oracle; the direction arms
consume only calibration-split medians. No arm uses any behavioural label or
outcome to fit a vector.

Claim scope: results here speak only to whether THIS DPO-induced activation
delta transfers across THESE two upstream paths, at layer 24, final prompt
position, additive. Not "DPO is path-specific" in general.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.summarize_causal_ablation import classify_completion
from src.analysis.crossbranch.analyze import (
    QUADRANTS,
    assert_shared_rows,
    condition_key_from_filename,
    degeneracy_rate,
    distribution,
    four_way_rates,
    label_map,
    quadrant_map,
    total_variation,
)
from src.analysis.crossbranch.branches import direction_tag
from src.eval_stats import BOOTSTRAP_B, BOOTSTRAP_SEED
from src.v2_binding_guard import add_binding_cli_args, load_guarded_raw
from src.v2_io import write_json_lf

BASELINE = "baseline_target"     # B2, pre-DPO target
REFERENCE = "reference_target"   # B3, post-DPO target

IDENTITY = "xfer_delta_source_identity"
SHUF_WQ = "xfer_delta_source_shuf_wq"
NORMMATCHED = "xfer_delta_source_normmatched"
DOSEMATCHED = "xfer_delta_source_dosematched"
DIR_SOURCE = "dir_source_matched"
DIR_TARGET = "dir_target_matched"
OWN = "own_delta_target"         # optional within-branch context arm
PARALLEL = "xfer_delta_source_parallel"   # optional decomposition arm
PERP = "xfer_delta_source_perp"           # optional decomposition arm
SHUF_GLOBAL = "xfer_delta_source_shuf_global"  # optional: shuffle ACROSS quadrants
# Same global permutation, rescaled per row to the identity arm's own norm.
# The bare global shuffle moves prompt class and injected magnitude together
# (quadrants do not share a delta-norm distribution), so only this arm gives a
# dose-controlled test of class conditioning.
SHUF_GLOBAL_NM = "xfer_delta_source_shuf_global_normmatched"

STAGE2_ARMS = (IDENTITY, SHUF_WQ, NORMMATCHED, DOSEMATCHED, DIR_SOURCE, DIR_TARGET)
# Optional arms folded in only when their raw file is present.
OPTIONAL_ARMS = (OWN, PARALLEL, PERP, SHUF_GLOBAL, SHUF_GLOBAL_NM)

# (a, b) -> report the paired difference a_metric - b_metric with a CI.
CONTRASTS = (
    (IDENTITY, SHUF_WQ),
    (IDENTITY, NORMMATCHED),
    (IDENTITY, DOSEMATCHED),
    (IDENTITY, DIR_SOURCE),
    (DIR_TARGET, DIR_SOURCE),
)

REFUSAL = {"refusal"}
OVERREFUSAL = {"refusal", "soft_deflection"}

_METRIC_KEYS = ("dtv", "target_label_match", "refusal_shift", "overrefusal_shift")

NOTES = [
    "No single pass/fail gate. Read each quadrant's arms and contrasts together.",
    "dTV negative = the arm's four-way label distribution moved toward B3's. TV "
    "can fall for safety-irrelevant reasons (e.g. matching B3's degenerate "
    "rate), so always read dTV next to four_way_rates and the degeneracy rates.",
    "The Stage-1 gate quadrant is C (largest B2->B3 behavioural shift). A and D "
    "have n=30 held-out rows each, so their CIs are wide; B (250) and C (104) "
    "carry more weight.",
    "identity__minus__shuf_wq is the key discriminator. shuffle keeps the "
    "per-quadrant delta distribution but destroys the prompt<->delta pairing: a "
    "difference CI that excludes 0 with identity more negative is evidence for "
    "prompt-conditioned transfer; a CI covering 0 means the movement is a shared "
    "per-quadrant offset, not prompt-specific.",
    "target_label_match is per-prompt agreement with B3; it separates "
    "'reproduces B3's marginal distribution' (which TV alone can do) from "
    "'reproduces B3 on the same prompts'. baseline_target_label_match is the "
    "B2-vs-B3 floor for the same quadrant.",
    "Oracle framing per arm: identity uses the SOURCE branch's post-DPO "
    "activation for that prompt only (weakest); dosematched additionally uses "
    "||Delta_B(x)|| and is a strictly stronger oracle; the direction arms use "
    "only calibration-split medians. No arm uses a behavioural label to fit a "
    "vector.",
    "Claim scope: this speaks only to whether THIS DPO-induced activation delta "
    "transfers across THESE two upstream paths at layer 24, final prompt "
    "position, additive. Not 'DPO is path-specific' in general.",
]


def _rate(labels: list[str], keys: set[str]) -> float:
    return sum(1 for l in labels if l in keys) / len(labels) if labels else 0.0


def _dtv(arm_l: list[str], base_l: list[str], ref_l: list[str]) -> float:
    p_ref = distribution(ref_l)
    return total_variation(distribution(arm_l), p_ref) - total_variation(
        distribution(base_l), p_ref
    )


def _tlm(arm_l: list[str], ref_l: list[str]) -> float:
    return float(
        np.mean([1.0 if a == r else 0.0 for a, r in zip(arm_l, ref_l)])
    ) if arm_l else 0.0


def _point_metrics(base_l, ref_l, arm_l: dict[str, list[str]]) -> dict[str, dict]:
    out = {}
    for name, al in arm_l.items():
        out[name] = {
            "dtv": _dtv(al, base_l, ref_l),
            "target_label_match": _tlm(al, ref_l),
            "refusal_shift": _rate(al, REFUSAL) - _rate(base_l, REFUSAL),
            "overrefusal_shift": _rate(al, OVERREFUSAL) - _rate(base_l, OVERREFUSAL),
        }
    return out


def bootstrap_quadrant(
    arm_labels: dict[str, dict[str, str]],
    base: dict[str, str],
    ref: dict[str, str],
    ids: list[str],
    arms: list[str],
    contrasts: list[tuple[str, str]],
    *,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap for one quadrant's row set.

    One shared resampled index set per replicate; B2, B3 and every arm are
    recomputed on it, and every contrast is a per-replicate difference of
    those recomputed values (never two independently-resampled numbers
    subtracted).
    """
    n = len(ids)
    if n == 0:
        return {"n": 0}

    base_l = [base[i] for i in ids]
    ref_l = [ref[i] for i in ids]
    arm_l = {a: [arm_labels[a][i] for i in ids] for a in arms}

    point = _point_metrics(base_l, ref_l, arm_l)
    base_tlm_point = _tlm(base_l, ref_l)

    reps = {a: {k: np.empty(b) for k in _METRIC_KEYS} for a in arms}
    contrast_names = [f"{x}__minus__{y}" for x, y in contrasts]
    contrast_reps = {
        name: {"dtv_diff": np.empty(b), "target_label_match_diff": np.empty(b)}
        for name in contrast_names
    }
    base_tlm_reps = np.empty(b)

    rng = np.random.default_rng(seed)
    for k in range(b):
        pick = rng.integers(0, n, size=n)
        bl = [base_l[j] for j in pick]
        rl = [ref_l[j] for j in pick]
        al = {a: [arm_l[a][j] for j in pick] for a in arms}
        m = _point_metrics(bl, rl, al)
        for a in arms:
            for kk in _METRIC_KEYS:
                reps[a][kk][k] = m[a][kk]
        base_tlm_reps[k] = _tlm(bl, rl)
        for (x, y), name in zip(contrasts, contrast_names):
            contrast_reps[name]["dtv_diff"][k] = m[x]["dtv"] - m[y]["dtv"]
            contrast_reps[name]["target_label_match_diff"][k] = (
                m[x]["target_label_match"] - m[y]["target_label_match"]
            )

    lo_p = (1 - confidence) / 2 * 100
    hi_p = (1 + confidence) / 2 * 100

    def summ(arr, pt) -> dict:
        return {
            "point": float(pt),
            "ci_low": float(np.percentile(arr, lo_p)),
            "ci_high": float(np.percentile(arr, hi_p)),
            "b": b,
            "seed": seed,
            "interval": "percentile",
        }

    arms_out = {}
    for a in arms:
        arms_out[a] = {kk: summ(reps[a][kk], point[a][kk]) for kk in _METRIC_KEYS}
        arms_out[a]["degeneracy_rate"] = degeneracy_rate(arm_l[a])

    contrasts_out = {}
    for (x, y), name in zip(contrasts, contrast_names):
        contrasts_out[name] = {
            "dtv_diff": summ(
                contrast_reps[name]["dtv_diff"], point[x]["dtv"] - point[y]["dtv"]
            ),
            "target_label_match_diff": summ(
                contrast_reps[name]["target_label_match_diff"],
                point[x]["target_label_match"] - point[y]["target_label_match"],
            ),
        }

    return {
        "n": n,
        "tv_baseline_to_reference": total_variation(
            distribution(base_l), distribution(ref_l)
        ),
        "baseline_target_label_match": summ(base_tlm_reps, base_tlm_point),
        "baseline_degeneracy_rate": degeneracy_rate(base_l),
        "reference_degeneracy_rate": degeneracy_rate(ref_l),
        "arms": arms_out,
        "contrasts": contrasts_out,
    }


def analyze_stage2(
    raw_by_condition: dict[str, list[dict]],
    *,
    coef: float = 1.0,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    classify_fn=classify_completion,
) -> dict:
    """`raw_by_condition` keys: ``baseline_target``, ``reference_target`` (plain,
    from the model conditions) and each vector arm as ``f"{name}@{coef:g}"``
    (the form ``condition_key_from_filename`` produces).
    """
    labels = {k: label_map(v, classify_fn) for k, v in raw_by_condition.items()}
    if BASELINE not in labels or REFERENCE not in labels:
        raise RuntimeError(
            f"analyze_stage2 needs {BASELINE!r} and {REFERENCE!r}; got "
            f"{sorted(labels)}"
        )

    suffix = f"@{coef:g}"
    present = [a for a in STAGE2_ARMS if a + suffix in labels]
    if not present:
        raise RuntimeError(
            f"no Stage-2 arms found at coef {coef:g} (looked for "
            f"{[a + suffix for a in STAGE2_ARMS]}); keys present: {sorted(labels)}"
        )
    optional = [a for a in OPTIONAL_ARMS if a + suffix in labels]
    arms = list(present) + optional

    contrasts = [c for c in CONTRASTS if c[0] in arms and c[1] in arms]
    for extra in ((IDENTITY, OWN), (IDENTITY, PARALLEL), (IDENTITY, PERP),
                  (PARALLEL, PERP), (PARALLEL, DIR_SOURCE),
                  # The load-bearing pair for the class-level claim: the
                  # within-quadrant shuffle already matches identity, so the
                  # question is whether shuffling ACROSS quadrants finally
                  # breaks it. SHUF_WQ - SHUF_GLOBAL is that test -- but the
                  # bare version also changes per-row dose, so the *_NM pairs
                  # below are the ones that actually isolate class identity.
                  (IDENTITY, SHUF_GLOBAL), (SHUF_WQ, SHUF_GLOBAL),
                  (IDENTITY, SHUF_GLOBAL_NM), (SHUF_WQ, SHUF_GLOBAL_NM),
                  (SHUF_GLOBAL_NM, SHUF_GLOBAL)):
        if extra[0] in arms and extra[1] in arms and extra not in contrasts:
            contrasts.append(extra)

    used = {BASELINE: labels[BASELINE], REFERENCE: labels[REFERENCE]}
    for a in arms:
        used[a] = labels[a + suffix]
    ids = assert_shared_rows(used)
    quads = quadrant_map(next(iter(raw_by_condition.values())))

    arm_labels = {a: labels[a + suffix] for a in arms}

    per_quadrant = {}
    for q in QUADRANTS:
        q_ids = [i for i in ids if quads.get(i) == q]
        if not q_ids:
            continue
        block = bootstrap_quadrant(
            arm_labels, labels[BASELINE], labels[REFERENCE], q_ids, arms,
            contrasts, b=b, seed=seed,
        )
        block["primary_refusal_metric"] = (
            "refusal" if q in ("A", "C") else "refusal_or_soft_deflection"
        )
        rate_sources = {BASELINE: labels[BASELINE], REFERENCE: labels[REFERENCE]}
        rate_sources.update(arm_labels)
        block["four_way_rates"] = {
            name: four_way_rates([lab[i] for i in q_ids])
            for name, lab in rate_sources.items()
        }
        per_quadrant[q] = block

    return {
        "coef": coef,
        "n_rows_shared": len(ids),
        "arms_present": arms,
        "contrasts": [f"{x}__minus__{y}" for x, y in contrasts],
        "stage1_gate_quadrant": "C",
        "bootstrap": {"b": b, "seed": seed, "interval": "percentile", "paired": True},
        "per_quadrant": per_quadrant,
        "notes": NOTES,
    }


def _load_raw_dir(raw_dir: Path, tag: str, *, expect_sha, allow_unbound) -> dict:
    found: dict[str, list[dict]] = {}
    for path in sorted(raw_dir.glob(f"crossbranch_{tag}_*.json")):
        if path.name.endswith("_binding.json"):
            continue
        stem = path.stem[len(f"crossbranch_{tag}_"):]
        found[condition_key_from_filename(stem)] = load_guarded_raw(
            path, benchmark_sha256=expect_sha, allow_unbound=allow_unbound
        )
    return found


def _print_summary(result: dict) -> None:
    print(
        f"coef {result['coef']:g}  |  arms: {', '.join(result['arms_present'])}  |  "
        f"n_shared {result['n_rows_shared']}"
    )
    for q, blk in result["per_quadrant"].items():
        print(
            f"\nquadrant {q}  n={blk['n']}  TV(B2,B3)={blk['tv_baseline_to_reference']:.3f}  "
            f"primary={blk['primary_refusal_metric']}  "
            f"tlm_floor={blk['baseline_target_label_match']['point']:.3f}"
        )
        for a, m in blk["arms"].items():
            d, t = m["dtv"], m["target_label_match"]
            print(
                f"  {a:32s} dTV={d['point']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]  "
                f"tlm={t['point']:.3f} [{t['ci_low']:.3f},{t['ci_high']:.3f}]  "
                f"deg={m['degeneracy_rate']:.3f}"
            )
        for name, c in blk["contrasts"].items():
            dd = c["dtv_diff"]
            excludes_zero = dd["ci_high"] < 0 or dd["ci_low"] > 0
            print(
                f"  d {name:42s} dTV_diff={dd['point']:+.3f} "
                f"[{dd['ci_low']:+.3f},{dd['ci_high']:+.3f}]"
                f"{'  <- CI excludes 0' if excludes_zero else ''}"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-2 crossbranch transfer analysis.")
    p.add_argument("--raw-dir", default="results/crossbranch/raw")
    p.add_argument("--source-branch", default="A")
    p.add_argument("--target-branch", default="B")
    p.add_argument("--coef", type=float, default=1.0)
    p.add_argument("--out-dir", default="results/crossbranch/analysis")
    p.add_argument(
        "--out-name", default=None,
        help="override the output filename; default is "
             "crossbranch_<tag>_stage2_analysis.json at coef 1.0 and "
             "..._coef<c>.json otherwise, so a sweep does not clobber the "
             "canonical coef-1.0 file that plot_stage2/compare_directions read.",
    )
    add_binding_cli_args(p)
    args = p.parse_args()

    tag = direction_tag(args.source_branch, args.target_branch)
    found = _load_raw_dir(
        Path(args.raw_dir), tag,
        expect_sha=args.expect_benchmark_sha256, allow_unbound=args.allow_unbound,
    )
    if not found:
        raise SystemExit(
            f"No raw files matching crossbranch_{tag}_*.json in {args.raw_dir}"
        )

    result = analyze_stage2(found, coef=args.coef)
    result["direction_tag"] = tag
    result["source_branch"] = args.source_branch
    result["target_branch"] = args.target_branch
    if args.out_name:
        name = args.out_name
    elif abs(args.coef - 1.0) < 1e-9:
        name = f"crossbranch_{tag}_stage2_analysis.json"
    else:
        name = f"crossbranch_{tag}_stage2_analysis_coef{args.coef:g}.json"
    out = Path(args.out_dir) / name
    write_json_lf(out, result)
    _print_summary(result)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
