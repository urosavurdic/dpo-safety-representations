"""
3D-B pilot runner. Orchestrates one pilot run of src/data_pipeline/
lexical_outlierness.py (Methods 1/2, S6 percentiles, S7 agreement stats,
S8 source-balanced sensitivity, S9 confound diagnostics incl. category
sensitivity and length sensitivity) over the exact 209-row S1 population,
and writes:

  - logs/3d_b_lexical_outlierness_pilot.json  (full numeric results)
  - logs/3d_b_lexical_outlierness_pilot.md    (human-readable summary,
    with "observed evidence" / "predeclared criteria" / "researcher
    decision" kept explicitly separate per the 3D-B task brief - this
    script never makes or implies a GO/NO-GO call)
  - data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json (S4,
    frozen before scoring)

Run: python -m src.data_pipeline.lexical_outlierness_pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_pipeline.lexical_outlierness import (
    ALPHA,
    AMBIGUITY_NOTES,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    GROUPS_ARTIFACT_PATH,
    HIGH_TAIL_CUTOFF,
    LOW_TAIL_CUTOFF,
    OUT_JSON_PATH,
    OUT_MD_PATH,
    PERMUTATION_REPS,
    PERMUTATION_SEED,
    POPULATION_ARTIFACT_PATH,
    REPO_ROOT,
    REQUIRED_SOURCES,
    SHINGLE_LEN,
    SHINGLE_THRESHOLD,
    build_groups,
    calibrated_percentile,
    category_of,
    fold_selfinfo,
    fold_tfidf,
    load_population,
    normalize_text,
    score_selfinfo,
    score_tfidf,
    tokenize,
    word_ngrams_1_2,
)
from src.data_pipeline.build_c_source_authored_candidates import file_sha256

# ── formatting diagnostic regexes (S9 item 4) - exact patterns recorded
#    in the pilot JSON output for auditability (AMBIGUITY_NOTES #4) ────────
BULLET_MARKER_RE = re.compile(r"(?m)^\s*[-*\u2022]\s+")
NUMBERED_STEP_RE = re.compile(r"(?m)^\s*\d+[\.\)]\s+")
CODE_BLOCK_RE = re.compile(r"```")
SENTENCE_END_RE = re.compile(r"[.!?]+")

FORMATTING_DIAGNOSTIC_CONFIG = {
    "bullet_marker_regex": BULLET_MARKER_RE.pattern,
    "numbered_step_regex": NUMBERED_STEP_RE.pattern,
    "code_block_regex": CODE_BLOCK_RE.pattern,
    "multi_sentence_rule": "count of [.!?]+ matches >= 2, on raw prompt_text",
    "applied_to": "raw prompt_text (not S2-normalized - normalization "
    "collapses newlines, which line-anchored list/step markers depend on)",
}


def formatting_features(raw_text: str) -> dict:
    return {
        "has_bullet_marker": bool(BULLET_MARKER_RE.search(raw_text)),
        "has_numbered_step": bool(NUMBERED_STEP_RE.search(raw_text)),
        "has_code_block": bool(CODE_BLOCK_RE.search(raw_text)),
        "multi_sentence": len(SENTENCE_END_RE.findall(raw_text)) >= 2,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()
    except Exception:
        return "unknown"


def tail_of(p: Optional[float]) -> str:
    if p is None:
        return "unscored"
    if p >= HIGH_TAIL_CUTOFF:
        return "high"
    if p <= LOW_TAIL_CUTOFF:
        return "low"
    return "mid"


def overlap_and_jaccard(a: set, b: set) -> Tuple[Optional[float], float]:
    inter = len(a & b)
    union = len(a | b)
    denom_min = min(len(a), len(b))
    overlap = (inter / denom_min) if denom_min else None
    jac = (inter / union) if union else 0.0
    return overlap, jac


# ── fold-variant computation shared by primary / source-balanced /
#    category-balanced (all use the identical S2/S3/S6 mechanics; only
#    the reference weights differ, per S8's "weighting only touches the
#    reference statistics ... not group membership or vocabulary
#    membership") ────────────────────────────────────────────────────────
def compute_fold_variant(ref_feats, ref_tokens, weights):
    idf, mu, _ = fold_tfidf(ref_feats, weights=weights)
    counts, total, vsize = fold_selfinfo(ref_tokens, weights=weights)
    ref_scores_tfidf = [score_tfidf(f, idf, mu) for f in ref_feats]
    ref_scores_selfinfo = [score_selfinfo(t, counts, total, vsize) for t in ref_tokens]
    return {
        "idf": idf,
        "mu": mu,
        "counts": counts,
        "total": total,
        "vsize": vsize,
        "ref_scores_tfidf": ref_scores_tfidf,
        "ref_scores_selfinfo": ref_scores_selfinfo,
    }


def score_row_against_fold(feats_i, tokens_i, fold, weights_for_percentile):
    s_tfidf = score_tfidf(feats_i, fold["idf"], fold["mu"])
    s_selfinfo = score_selfinfo(tokens_i, fold["counts"], fold["total"], fold["vsize"])
    p_tfidf = calibrated_percentile(s_tfidf, fold["ref_scores_tfidf"], weights_for_percentile)
    p_selfinfo = calibrated_percentile(s_selfinfo, fold["ref_scores_selfinfo"], weights_for_percentile)
    return s_tfidf, s_selfinfo, p_tfidf, p_selfinfo


def spearman_safe(x: List[float], y: List[float]):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rho, _ = spearmanr(x, y)
    return None if np.isnan(rho) else float(rho)


def run_pilot() -> dict:
    rows = load_population()
    by_id = {r["record_id"]: r for r in rows}
    record_ids = [r["record_id"] for r in rows]

    # S4: groups, frozen BEFORE scoring.
    record_to_group, groups_artifact = build_groups(rows)
    GROUPS_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUPS_ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(groups_artifact, f, indent=2, sort_keys=True)
        f.write("\n")
    groups_artifact_sha256 = file_sha256(GROUPS_ARTIFACT_PATH)

    groups: Dict[str, List[str]] = defaultdict(list)
    for rid, gid in record_to_group.items():
        groups[gid].append(rid)

    # Precompute per-row primitives (independent of fold).
    normed = {rid: normalize_text(by_id[rid]["prompt_text"]) for rid in record_ids}
    tokens = {rid: tokenize(normed[rid]) for rid in record_ids}
    feats = {rid: word_ngrams_1_2(tokens[rid]) for rid in record_ids}
    source = {rid: by_id[rid]["source_dataset"] for rid in record_ids}
    category = {rid: category_of(by_id[rid]) for rid in record_ids}
    n_i = {rid: len(tokens[rid]) for rid in record_ids}

    raw_tfidf: Dict[str, float] = {}
    raw_selfinfo: Dict[str, float] = {}
    p_tfidf: Dict[str, Optional[float]] = {}
    p_selfinfo: Dict[str, Optional[float]] = {}
    flags: Dict[str, dict] = {rid: {} for rid in record_ids}
    oov_rate: Dict[str, Optional[float]] = {}

    p_tfidf_src_bal: Dict[str, Optional[float]] = {}
    p_selfinfo_src_bal: Dict[str, Optional[float]] = {}
    p_tfidf_cat_bal: Dict[str, Optional[float]] = {}
    p_selfinfo_cat_bal: Dict[str, Optional[float]] = {}
    source_balanced_halted_folds: List[str] = []
    fold_reference_counts: Dict[str, dict] = {}

    for gid, members in groups.items():
        ref_ids = [rid for rid in record_ids if rid not in members]
        ref_feats = [feats[rid] for rid in ref_ids]
        ref_tokens = [tokens[rid] for rid in ref_ids]

        if len(ref_ids) == 0:
            raise AssertionError(
                f"S2 halt condition: held-out group {gid} spans the entire "
                f"209-row pool - V_-g would be empty."
            )

        # ---- primary (unweighted, w_j=1) ----
        fold_primary = compute_fold_variant(ref_feats, ref_tokens, weights=None)
        for rid in members:
            s_t, s_s, p_t, p_s = score_row_against_fold(feats[rid], tokens[rid], fold_primary, None)
            raw_tfidf[rid], raw_selfinfo[rid] = s_t, s_s
            p_tfidf[rid], p_selfinfo[rid] = p_t, p_s
            if p_t is None:
                flags[rid]["insufficient_lexical_overlap"] = True
            if n_i[rid] == 0:
                flags[rid]["empty_after_normalization"] = True
            unseen = sum(1 for t in tokens[rid] if t not in fold_primary["counts"])
            oov_rate[rid] = (unseen / n_i[rid]) if n_i[rid] > 0 else None

        fold_reference_counts[gid] = {
            "n_reference_rows": len(ref_ids),
            "group_size": len(members),
        }

        # ---- S8 source-balanced ----
        ref_src_counts = Counter(source[rid] for rid in ref_ids)
        if any(ref_src_counts.get(s, 0) == 0 for s in REQUIRED_SOURCES):
            source_balanced_halted_folds.append(gid)
            for rid in members:
                p_tfidf_src_bal[rid] = None
                p_selfinfo_src_bal[rid] = None
                flags[rid]["source_balanced_undefined_reason"] = "fold_missing_required_source"
        else:
            w_src = [0.5 / ref_src_counts[source[rid]] for rid in ref_ids]
            fold_bal = compute_fold_variant(ref_feats, ref_tokens, weights=w_src)
            for rid in members:
                _, _, p_t, p_s = score_row_against_fold(feats[rid], tokens[rid], fold_bal, w_src)
                p_tfidf_src_bal[rid] = p_t
                p_selfinfo_src_bal[rid] = p_s

        # ---- S9 item 7 category-balanced ----
        ref_cat_counts = Counter(category[rid] for rid in ref_ids)
        K = len(ref_cat_counts)
        w_cat = [(1.0 / K) / ref_cat_counts[category[rid]] for rid in ref_ids]
        fold_cat = compute_fold_variant(ref_feats, ref_tokens, weights=w_cat)
        for rid in members:
            _, _, p_t, p_s = score_row_against_fold(feats[rid], tokens[rid], fold_cat, w_cat)
            p_tfidf_cat_bal[rid] = p_t
            p_selfinfo_cat_bal[rid] = p_s

    # ── S6 tails ─────────────────────────────────────────────────────────
    tail_tfidf = {rid: tail_of(p_tfidf[rid]) for rid in record_ids}
    tail_selfinfo = {rid: tail_of(p_selfinfo[rid]) for rid in record_ids}
    High_tfidf = {rid for rid, t in tail_tfidf.items() if t == "high"}
    Low_tfidf = {rid for rid, t in tail_tfidf.items() if t == "low"}
    High_si = {rid for rid, t in tail_selfinfo.items() if t == "high"}
    Low_si = {rid for rid, t in tail_selfinfo.items() if t == "low"}

    jointly_defined = [rid for rid in record_ids if p_tfidf[rid] is not None and p_selfinfo[rid] is not None]
    x_joint = [p_tfidf[rid] for rid in jointly_defined]
    y_joint = [p_selfinfo[rid] for rid in jointly_defined]

    # ── S7 agreement statistics ─────────────────────────────────────────
    spearman_obs = spearman_safe(x_joint, y_joint)
    high_overlap_obs, high_jaccard_obs = overlap_and_jaccard(High_tfidf, High_si)
    low_overlap_obs, low_jaccard_obs = overlap_and_jaccard(Low_tfidf, Low_si)

    # Permutation baseline: permute the tfidf-side tail-membership labels
    # across jointly-defined record ids, recompute vs the fixed selfinfo
    # tails, 10000 reps, fixed logged seed.
    rng_perm = np.random.default_rng(PERMUTATION_SEED)
    joint_arr = np.array(jointly_defined)
    tail_tfidf_joint = np.array([tail_tfidf[rid] for rid in jointly_defined])

    def null_stats():
        idx = rng_perm.permutation(len(joint_arr))
        perm_tail = tail_tfidf_joint[idx]
        perm_high = set(joint_arr[perm_tail == "high"])
        perm_low = set(joint_arr[perm_tail == "low"])
        ho, hj = overlap_and_jaccard(perm_high, High_si & set(joint_arr))
        lo, lj = overlap_and_jaccard(perm_low, Low_si & set(joint_arr))
        return ho, hj, lo, lj

    null_ho, null_hj, null_lo, null_lj = [], [], [], []
    for _ in range(PERMUTATION_REPS):
        ho, hj, lo, lj = null_stats()
        null_ho.append(ho if ho is not None else np.nan)
        null_hj.append(hj)
        null_lo.append(lo if lo is not None else np.nan)
        null_lj.append(lj)

    def null_summary(obs, null_vals):
        arr = np.array(null_vals, dtype=float)
        finite = arr[~np.isnan(arr)]
        if obs is None or len(finite) == 0:
            return {"observed": obs, "null_mean": None, "null_sd": None, "observed_percentile_in_null": None, "n_null_defined": int(len(finite))}
        pct = float(np.mean(finite <= obs))
        return {
            "observed": obs,
            "null_mean": float(np.mean(finite)),
            "null_sd": float(np.std(finite)),
            "observed_percentile_in_null": pct,
            "n_null_defined": int(len(finite)),
        }

    permutation_results = {
        "seed": PERMUTATION_SEED,
        "n_reps": PERMUTATION_REPS,
        "high_tail_overlap": null_summary(high_overlap_obs, null_ho),
        "high_tail_jaccard": null_summary(high_jaccard_obs, null_hj),
        "low_tail_overlap": null_summary(low_overlap_obs, null_lo),
        "low_tail_jaccard": null_summary(low_jaccard_obs, null_lj),
    }

    # Bootstrap: row-level resample w/ replacement over the 209-row pool,
    # multiplicity-aware Spearman + tail Jaccards, 10000 reps, fixed seed.
    rng_boot = np.random.default_rng(BOOTSTRAP_SEED)
    all_ids_arr = np.array(record_ids)
    n_pool = len(all_ids_arr)
    boot_spearman: List[float] = []
    boot_high_jaccard: List[float] = []
    boot_low_jaccard: List[float] = []
    undefined_spearman = 0
    undefined_high_j = 0
    undefined_low_j = 0

    for _ in range(BOOTSTRAP_REPS):
        idx = rng_boot.integers(0, n_pool, size=n_pool)
        sampled_ids = all_ids_arr[idx]
        mult = Counter(sampled_ids.tolist())

        exp_x, exp_y = [], []
        for rid, k in mult.items():
            if p_tfidf[rid] is not None and p_selfinfo[rid] is not None:
                exp_x.extend([p_tfidf[rid]] * k)
                exp_y.extend([p_selfinfo[rid]] * k)
        s = spearman_safe(exp_x, exp_y)
        if s is None:
            undefined_spearman += 1
        else:
            boot_spearman.append(s)

        for tail_a, tail_b, sink, undef_counter_name in (
            (High_tfidf, High_si, boot_high_jaccard, "high"),
            (Low_tfidf, Low_si, boot_low_jaccard, "low"),
        ):
            num = sum(k for rid, k in mult.items() if rid in tail_a and rid in tail_b)
            den = sum(k for rid, k in mult.items() if rid in tail_a or rid in tail_b)
            if den == 0:
                if undef_counter_name == "high":
                    undefined_high_j += 1
                else:
                    undefined_low_j += 1
            else:
                sink.append(num / den)

    def ci95(vals: List[float]):
        if not vals:
            return None
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"lo": float(lo), "hi": float(hi), "n_defined": len(vals)}

    bootstrap_results = {
        "seed": BOOTSTRAP_SEED,
        "n_reps": BOOTSTRAP_REPS,
        "spearman": {"ci95": ci95(boot_spearman), "n_undefined": undefined_spearman},
        "high_tail_jaccard": {"ci95": ci95(boot_high_jaccard), "n_undefined": undefined_high_j},
        "low_tail_jaccard": {"ci95": ci95(boot_low_jaccard), "n_undefined": undefined_low_j},
    }

    # ── S8 balanced sensitivity reporting ───────────────────────────────
    def balanced_report(primary, balanced):
        pairs = [(primary[rid], balanced[rid]) for rid in record_ids if primary[rid] is not None and balanced[rid] is not None]
        rho = spearman_safe([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= 2 else None
        confusion = sum(1 for rid in record_ids if primary[rid] is not None and balanced[rid] is not None and tail_of(primary[rid]) != tail_of(balanced[rid]))
        return {
            "n_paired": len(pairs),
            "spearman_primary_vs_balanced": rho,
            "tail_membership_changed_count": confusion,
        }

    source_balanced_sensitivity = {
        "halted_folds": source_balanced_halted_folds,
        "n_halted_folds": len(source_balanced_halted_folds),
        "tfidf": balanced_report(p_tfidf, p_tfidf_src_bal),
        "selfinfo": balanced_report(p_selfinfo, p_selfinfo_src_bal),
    }
    category_balanced_sensitivity = {
        "tfidf": balanced_report(p_tfidf, p_tfidf_cat_bal),
        "selfinfo": balanced_report(p_selfinfo, p_selfinfo_cat_bal),
    }

    # ── S9 confound diagnostics ──────────────────────────────────────────
    def tail_by_key_table(tail_map, key_map):
        table: Dict[str, Counter] = defaultdict(Counter)
        for rid in record_ids:
            table[tail_map[rid]][key_map[rid]] += 1
        return {tail: dict(counts) for tail, counts in table.items()}

    tail_by_source = {"tfidf": tail_by_key_table(tail_tfidf, source), "selfinfo": tail_by_key_table(tail_selfinfo, source)}
    tail_by_category = {"tfidf": tail_by_key_table(tail_tfidf, category), "selfinfo": tail_by_key_table(tail_selfinfo, category)}

    length_values = [n_i[rid] for rid in record_ids]
    length_by_tail = {
        m: {
            t: [n_i[rid] for rid in record_ids if tm[rid] == t]
            for t in ("high", "mid", "low", "unscored")
        }
        for m, tm in (("tfidf", tail_tfidf), ("selfinfo", tail_selfinfo))
    }

    def length_summary(vals: List[int]) -> dict:
        if not vals:
            return {"n": 0}
        arr = np.array(vals, dtype=float)
        return {"n": len(vals), "mean": float(arr.mean()), "median": float(np.median(arr)), "min": int(arr.min()), "max": int(arr.max()), "sd": float(arr.std())}

    length_summary_report = {
        "overall": length_summary(length_values),
        "by_tail": {m: {t: length_summary(v) for t, v in d.items()} for m, d in length_by_tail.items()},
        "spearman_length_vs_percentile": {
            "tfidf": spearman_safe([n_i[rid] for rid in record_ids if p_tfidf[rid] is not None], [p_tfidf[rid] for rid in record_ids if p_tfidf[rid] is not None]),
            "selfinfo": spearman_safe([n_i[rid] for rid in record_ids if p_selfinfo[rid] is not None], [p_selfinfo[rid] for rid in record_ids if p_selfinfo[rid] is not None]),
        },
    }

    fmt = {rid: formatting_features(by_id[rid]["prompt_text"]) for rid in record_ids}
    formatting_by_tail = {}
    for m, tm in (("tfidf", tail_tfidf), ("selfinfo", tail_selfinfo)):
        per_tail = {}
        for t in ("high", "mid", "low", "unscored"):
            members_t = [rid for rid in record_ids if tm[rid] == t]
            per_tail[t] = {
                "n": len(members_t),
                "bullet_marker_rate": (sum(fmt[rid]["has_bullet_marker"] for rid in members_t) / len(members_t)) if members_t else None,
                "numbered_step_rate": (sum(fmt[rid]["has_numbered_step"] for rid in members_t) / len(members_t)) if members_t else None,
                "code_block_rate": (sum(fmt[rid]["has_code_block"] for rid in members_t) / len(members_t)) if members_t else None,
                "multi_sentence_rate": (sum(fmt[rid]["multi_sentence"] for rid in members_t) / len(members_t)) if members_t else None,
            }
        formatting_by_tail[m] = per_tail

    def source_prediction_diagnostic(tail_map):
        scored = [rid for rid in record_ids if tail_map[rid] != "unscored"]
        if len(scored) < 4:
            return {"note": "too few scored rows"}
        y = np.array([1 if source[rid] == "StrongREJECT" else 0 for rid in scored])
        is_high = np.array([1.0 if tail_map[rid] == "high" else 0.0 for rid in scored])
        is_low = np.array([1.0 if tail_map[rid] == "low" else 0.0 for rid in scored])
        X = np.column_stack([is_high, is_low])
        if len(set(y.tolist())) < 2:
            return {"note": "single-class source label among scored rows - logistic regression undefined"}
        clf = LogisticRegression()
        clf.fit(X, y)
        pred = clf.predict(X)
        proba = clf.predict_proba(X)[:, 1]
        acc = float(accuracy_score(y, pred))
        try:
            auc = float(roc_auc_score(y, proba))
        except ValueError:
            auc = None
        majority_baseline = float(max(np.mean(y), 1 - np.mean(y)))
        return {"n": len(scored), "in_sample_accuracy": acc, "in_sample_auc": auc, "majority_class_baseline_accuracy": majority_baseline, "note": "in-sample fit/evaluate diagnostic - see AMBIGUITY_NOTES #2"}

    source_prediction = {"tfidf": source_prediction_diagnostic(tail_tfidf), "selfinfo": source_prediction_diagnostic(tail_selfinfo)}

    # length sensitivity (S9 item 8)
    def length_sensitivity(p_map):
        scored = [rid for rid in record_ids if p_map[rid] is not None]
        if len(scored) < 3:
            return {"note": "too few scored rows"}
        x = np.array([n_i[rid] for rid in scored], dtype=float).reshape(-1, 1)
        y = np.array([p_map[rid] for rid in scored], dtype=float)
        reg = LinearRegression().fit(x, y)
        pred = reg.predict(x)
        resid = y - pred
        # empirical percentile of residuals within the scored pool (S6-style rule) - AMBIGUITY_NOTES #3
        resid_pct = {}
        for i, rid in enumerate(scored):
            r = resid[i]
            lt = np.sum(resid < r)
            eq = np.sum(resid == r)
            resid_pct[rid] = (lt + 0.5 * eq) / len(resid)
        rho_resid_vs_orig = spearman_safe(list(resid_pct.values()), list(y))
        changed = 0
        for rid in scored:
            orig_tail = tail_of(p_map[rid])
            resid_tail = tail_of(resid_pct[rid])
            if orig_tail != resid_tail:
                changed += 1
        return {
            "n": len(scored),
            "regression_coef": float(reg.coef_[0]),
            "regression_intercept": float(reg.intercept_),
            "spearman_residual_percentile_vs_original_percentile": rho_resid_vs_orig,
            "tail_membership_changed_count": changed,
        }

    length_sensitivity_report = {"tfidf": length_sensitivity(p_tfidf), "selfinfo": length_sensitivity(p_selfinfo)}

    # OOV-rate diagnostic (S9 item 9)
    oov_spearman = {
        "tfidf": spearman_safe([oov_rate[rid] for rid in record_ids if oov_rate[rid] is not None and p_tfidf[rid] is not None], [p_tfidf[rid] for rid in record_ids if oov_rate[rid] is not None and p_tfidf[rid] is not None]),
        "selfinfo": spearman_safe([oov_rate[rid] for rid in record_ids if oov_rate[rid] is not None and p_selfinfo[rid] is not None], [p_selfinfo[rid] for rid in record_ids if oov_rate[rid] is not None and p_selfinfo[rid] is not None]),
    }

    category_uninformative = len(set(category.values())) <= 1 or all(
        len(set(category[rid] for rid in record_ids if source[rid] == s)) <= 1 for s in REQUIRED_SOURCES
    )

    unscored_tfidf = [rid for rid in record_ids if p_tfidf[rid] is None]
    unscored_selfinfo = [rid for rid in record_ids if p_selfinfo[rid] is None]

    result = {
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "input_population_artifact": str(POPULATION_ARTIFACT_PATH.relative_to(REPO_ROOT)),
            "input_population_artifact_sha256": file_sha256(POPULATION_ARTIFACT_PATH),
            "input_population_row_count": len(rows),
            "duplicate_template_grouping_artifact": str(GROUPS_ARTIFACT_PATH.relative_to(REPO_ROOT)),
            "duplicate_template_grouping_artifact_sha256": groups_artifact_sha256,
            "n_groups": len(groups),
            "group_size_distribution": dict(Counter(len(m) for m in groups.values())),
            "method_configuration": {
                "method1": "LOGO TF-IDF centroid distance, word n-grams (1,2), smoothed sklearn-style IDF, group-local vocab (policy B), stopwords retained",
                "method2": f"LOGO smoothed token self-information, unigrams only, alpha={ALPHA}, stopwords retained",
                "grouping": f"exact/normalized-dup seed + character-{SHINGLE_LEN}gram shingle Jaccard >= {SHINGLE_THRESHOLD} union-find",
                "percentile": "S6 fold-calibrated percentile, '<' + 0.5*'=' tie rule",
                "high_tail_cutoff": HIGH_TAIL_CUTOFF,
                "low_tail_cutoff": LOW_TAIL_CUTOFF,
            },
            "random_seeds": {"permutation_seed": PERMUTATION_SEED, "bootstrap_seed": BOOTSTRAP_SEED},
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
                "platform": platform.platform(),
            },
            "flagged_implementation_ambiguities": AMBIGUITY_NOTES,
        },
        "population": {
            "n": len(rows),
            "source_counts": dict(Counter(source.values())),
            "category_counts": dict(Counter(category.values())),
        },
        "scoring": {
            "unscored_tfidf": unscored_tfidf,
            "unscored_selfinfo": unscored_selfinfo,
            "n_unscored_tfidf": len(unscored_tfidf),
            "n_unscored_selfinfo": len(unscored_selfinfo),
            "tail_realized_counts": {
                "tfidf": dict(Counter(tail_tfidf.values())),
                "selfinfo": dict(Counter(tail_selfinfo.values())),
            },
            "row_level": {
                rid: {
                    "source": source[rid],
                    "category": category[rid],
                    "n_tokens": n_i[rid],
                    "raw_tfidf": (None if rid not in raw_tfidf or (isinstance(raw_tfidf.get(rid), float) and np.isnan(raw_tfidf[rid])) else raw_tfidf[rid]),
                    "raw_selfinfo": (None if rid not in raw_selfinfo or (isinstance(raw_selfinfo.get(rid), float) and np.isnan(raw_selfinfo[rid])) else raw_selfinfo[rid]),
                    "p_tfidf": p_tfidf[rid],
                    "p_selfinfo": p_selfinfo[rid],
                    "tail_tfidf": tail_tfidf[rid],
                    "tail_selfinfo": tail_selfinfo[rid],
                    "oov_rate": oov_rate.get(rid),
                    "flags": flags[rid],
                }
                for rid in record_ids
            },
        },
        "agreement_statistics": {
            "spearman_p_tfidf_p_selfinfo": spearman_obs,
            "n_jointly_defined": len(jointly_defined),
            "high_tail_overlap_coefficient": high_overlap_obs,
            "high_tail_jaccard": high_jaccard_obs,
            "low_tail_overlap_coefficient": low_overlap_obs,
            "low_tail_jaccard": low_jaccard_obs,
            "permutation_baseline": permutation_results,
            "bootstrap_uncertainty": bootstrap_results,
        },
        "source_balanced_sensitivity": source_balanced_sensitivity,
        "confound_diagnostics": {
            "tail_by_source": tail_by_source,
            "tail_by_category": tail_by_category,
            "category_field_policy": "harm_area only, missing/blank -> 'unknown', no fallback (S9)",
            "category_uninformative_or_fully_source_confounded": category_uninformative,
            "length_summary": length_summary_report,
            "formatting_diagnostic_config": FORMATTING_DIAGNOSTIC_CONFIG,
            "formatting_by_tail": formatting_by_tail,
            "source_prediction_diagnostic": source_prediction,
            "source_balanced_sensitivity": source_balanced_sensitivity,
            "category_balanced_sensitivity": category_balanced_sensitivity,
            "length_sensitivity": length_sensitivity_report,
            "oov_rate_spearman_vs_percentile": oov_spearman,
        },
    }
    return result


def evidence_characterization(result: dict) -> str:
    """Descriptive-only characterization per the 3D-B task brief ('clearly
    consistent' / 'mixed' / 'clearly inconsistent') - never a GO/NO-GO
    call, which the brief reserves for the researcher."""
    ag = result["agreement_statistics"]
    rho = ag["spearman_p_tfidf_p_selfinfo"]
    hj = ag["permutation_baseline"]["high_tail_jaccard"]
    lj = ag["permutation_baseline"]["low_tail_jaccard"]

    def clearly_above_null(stat_block):
        if stat_block["observed"] is None or stat_block["observed_percentile_in_null"] is None:
            return None
        return stat_block["observed_percentile_in_null"] >= 0.95

    above = [clearly_above_null(hj), clearly_above_null(lj)]
    if rho is None or any(a is None for a in above):
        return "indeterminate (insufficient jointly-defined/tail data to characterize)"
    if rho <= 0.1 or not any(above):
        return "clearly inconsistent with a stable within-harmful lexical-outlierness ranking"
    if rho >= 0.3 and all(above):
        return "clearly consistent with a stable within-harmful lexical-outlierness ranking (methods agreement only - see confound diagnostics separately for downstream-use suitability, per S11)"
    return "mixed"


def write_markdown(result: dict, evidence: str) -> str:
    ag = result["agreement_statistics"]
    prov = result["provenance"]
    cd = result["confound_diagnostics"]
    lines = []
    lines.append("# 3D-B — Within-Harmful Lexical Outlierness Pilot: Results\n")
    lines.append(f"Generated: {prov['generated_at_utc']} | commit: {prov['git_commit']}\n")
    lines.append("Implements logs/3d_a_lexical_outlierness_design.md exactly. Does not compute CUE.\n")

    lines.append("## 1. Observed evidence\n")
    lines.append(f"- Spearman(p_tfidf, p_selfinfo) over {ag['n_jointly_defined']} jointly-defined rows: **{ag['spearman_p_tfidf_p_selfinfo']}**")
    lines.append(f"- High-tail overlap coefficient: {ag['high_tail_overlap_coefficient']} | Jaccard: {ag['high_tail_jaccard']}")
    lines.append(f"- Low-tail overlap coefficient: {ag['low_tail_overlap_coefficient']} | Jaccard: {ag['low_tail_jaccard']}")
    pb = ag["permutation_baseline"]
    lines.append(f"- Permutation baseline (seed={pb['seed']}, reps={pb['n_reps']}):")
    for k in ("high_tail_overlap", "high_tail_jaccard", "low_tail_overlap", "low_tail_jaccard"):
        lines.append(f"  - {k}: {pb[k]}")
    bs = ag["bootstrap_uncertainty"]
    lines.append(f"- Bootstrap (seed={bs['seed']}, reps={bs['n_reps']}):")
    lines.append(f"  - Spearman 95% CI: {bs['spearman']['ci95']} (undefined reps: {bs['spearman']['n_undefined']})")
    lines.append(f"  - High-tail Jaccard 95% CI: {bs['high_tail_jaccard']['ci95']} (undefined reps: {bs['high_tail_jaccard']['n_undefined']})")
    lines.append(f"  - Low-tail Jaccard 95% CI: {bs['low_tail_jaccard']['ci95']} (undefined reps: {bs['low_tail_jaccard']['n_undefined']})\n")

    lines.append("### Confound diagnostics (do not hide)\n")
    lines.append(f"- Tail-by-source (tfidf): {cd['tail_by_source']['tfidf']}")
    lines.append(f"- Tail-by-source (selfinfo): {cd['tail_by_source']['selfinfo']}")
    lines.append(f"- Category field uninformative-or-fully-source-confounded: **{cd['category_uninformative_or_fully_source_confounded']}**")
    lines.append(f"- Spearman(length, percentile): tfidf={cd['length_summary']['spearman_length_vs_percentile']['tfidf']}, selfinfo={cd['length_summary']['spearman_length_vs_percentile']['selfinfo']}")
    lines.append(f"- Source-prediction diagnostic (tail membership alone): {cd['source_prediction_diagnostic']}")
    lines.append(f"- Source-balanced sensitivity: {cd['source_balanced_sensitivity']}")
    lines.append(f"- Category-balanced sensitivity: {cd['category_balanced_sensitivity']}")
    lines.append(f"- Length sensitivity: {cd['length_sensitivity']}")
    lines.append(f"- OOV-rate vs percentile (Spearman): {cd['oov_rate_spearman_vs_percentile']}\n")

    lines.append("## 2. Predeclared criteria\n")
    lines.append("None. logs/3d_a_lexical_outlierness_design.md S11 predeclares no numerical GO/NO-GO threshold. "
                  "The decision framework there is qualitative (clearly-disagree / agree-but-confounded / agree-and-acceptable), "
                  "and even that qualitative call is explicitly left to the researcher.\n")

    lines.append("## 3. Researcher decision\n")
    lines.append(f"**Not made by this task.** Descriptive evidence characterization only: **{evidence}**. "
                  "This script does not decide STOP/REDESIGN vs. proceed-to-S10-human-validation; per the 3D-B task brief "
                  "and S11 of the design doc, that call belongs to the researcher, informed by both the agreement statistics "
                  "above and the confound diagnostics (a method can pass agreement and still fail suitability for downstream "
                  "use if diagnostics show source/category/length/formatting dominate the ranking - both conclusions must "
                  "stay visibly separate, per S9).\n")

    lines.append("## 4. Flagged implementation ambiguities\n")
    for note in prov["flagged_implementation_ambiguities"]:
        lines.append(f"- **#{note['id']} ({note['location']})**: {note['note']}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run the 3D-B lexical outlierness pilot (single run).")
    parser.parse_args()
    result = run_pilot()
    evidence = evidence_characterization(result)
    result["evidence_characterization"] = evidence
    result["researcher_go_no_go_decision"] = "NOT_MADE_BY_THIS_TASK"

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")

    md = write_markdown(result, evidence)
    with open(OUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Wrote {OUT_JSON_PATH}")
    print(f"Wrote {OUT_MD_PATH}")
    print(f"Wrote {GROUPS_ARTIFACT_PATH}")
    print(f"n_groups={result['provenance']['n_groups']}")
    print(f"spearman={ag_spearman(result)}  evidence={evidence}")


def ag_spearman(result):
    return result["agreement_statistics"]["spearman_p_tfidf_p_selfinfo"]


if __name__ == "__main__":
    main()
