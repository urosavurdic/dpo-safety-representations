"""
Task C-B -- Paired delta analysis for the existing R104 (`c_paired`)
quadrant-C construction.

This module implements EXACTLY the locked analysis contract in
`logs/c_existing_construction_audit_spec.md` section 7 (C-A). It is an
IMPLEMENTATION of that contract, not a new design: every input path,
hash, feature definition, sign convention, seed, and statistical
procedure below is copied from that document, not invented here. Where
this module makes an implementation decision the C-A document leaves
open (see the two `IMPLEMENTATION DECISION` notes below), that decision
is stated explicitly rather than silently resolved.

What this module explicitly does NOT do (matches C-A section 9 and the
C-B task brief):
  - does not create, rewrite, or modify any prompt;
  - does not modify the frozen benchmark, the review queue CSV, or any
    other C-A-pinned input;
  - does not run model inference or GPU code (the reused CUE scorer is
    a CPU-only TF-IDF + logistic regression fit, not an LLM);
  - does not access the web;
  - does not build a B/D construction or a common-CUE/contrastive score;
  - does not assign a final KEEP / KEEP AS SECONDARY / INCONCLUSIVE /
    DROP label (C-A section 7.7: that is explicitly deferred, out of
    scope for this implementation step);
  - does not print raw prompt text, matched lexicon terms, classifier
    weights, or any prompt-level ranking to any committed output --
    only aggregate statistics and, where a row is excluded, its
    `record_id` (an opaque identifier, not prompt content).

Fail-closed: every path in C-A section 7.1 is hash-verified before use;
any mismatch aborts (raises SystemExit) rather than warning and
continuing, mirroring the existing pattern in
`src/data_pipeline/build_c_source_authored_candidates.py::verify_source_file`
(reused, not reimplemented) and `src/v2_io.py::resolve_benchmark`.

IMPLEMENTATION DECISION 1 (CUE reference fitting for R104's StrongREJECT
text): C-A section 7.5 says the existing CUE score
(`src.cue_scoring.TfidfLogRegCueScorer`) should be "reuse[d] unmodified,
scored per prompt," without specifying which H/D reference split to
fit it on. R104's source_prompt and candidate_prompt values are both
100% StrongREJECT-provenance text (C-A section 2). `src/cue_scoring.py`
already establishes a leave-one-source-out (LOSO) convention
specifically to avoid scoring a harmful source's own text using a
reference that includes that same source. This module follows that
established convention: it fits ONE TfidfLogRegCueScorer with
StrongREJECT held out of the H reference (H = HarmBench +
SimpleSafetyTests + XSTest, D = quadrant D), using
`src.cue_scoring.load_reference_texts_from_repo` unmodified, and scores
every R104 source/candidate prompt out-of-fold with that single fitted
scorer. This avoids the circularity of fitting on text that overlaps
the scored population's own source dataset. It does NOT change
`TfidfLogRegCueScorer` itself or `FROZEN_CUE_CONFIG`.

IMPLEMENTATION DECISION 2 (which features the section-7.4 low-coverage
rule applies to): section 7.4 names "the lexical-audit family only
(Fightin' Words, CUE, lexical-diversity features below)" as the set
excluded when `low_coverage_flag_*` is True, which is a coarser split
than section 7.5's four-way family tag (structural / formatting-
confound / lexical-audit / distributional-exploratory -- CUE is tagged
distributional-exploratory there, not lexical-audit). This module
applies the low-coverage exclusion to every frequency/lexicon-based
feature named or implied by section 7.4's own parenthetical list --
`fightin_words`, `fw_z_score`, `lexical_diversity`,
`lexical_risk_hit_count`, and `cue_tfidf_logreg_margin`
(`LOW_COVERAGE_SENSITIVE_FEATURES` below) -- and leaves every count/
structural/formatting-indicator feature unaffected, per section 7.4's
own stated reasoning ("low token-recognition coverage degrades a
frequency-based lexical score but not a length/format count").
`low_coverage_flag_source`/`low_coverage_flag_candidate` are False for
all 104 current rows, so this branch currently excludes nothing; it is
implemented so a future re-freeze does not silently bypass the rule.

Decision-framework note: section 7.7's KEEP / KEEP AS SECONDARY /
INCONCLUSIVE / DROP labels are explicitly "not applied by this
document" in C-A, and assigning them is explicitly out of scope for
this implementation task (final interpretation is a separate,
downstream step). This module computes every statistic section 7.7
says that labeling step would need, but leaves the label itself unset
(`decision_labels: null` in the JSON output, with a note).

Run (exact command, C-A section 7.9):
    python -m src.analysis.c_b_paired_delta_analysis \\
        --review-csv data/review/c_review_queue.csv \\
        --benchmark-latest data/frozen_v2/LATEST_BENCHMARK.json \\
        --gate-config logs/benchmark_gate_config.json \\
        --formatting-config-source logs/3d_b_lexical_outlierness_pilot.json \\
        --bootstrap-seed 20260901 --n-bootstrap 10000 \\
        --permutation-seed 20260902 --n-permutations 100000 \\
        --out-md logs/c_b_paired_delta_analysis.md \\
        --out-json logs/c_b_paired_delta_analysis.json

Every flag above is also this module's default, so running with no
arguments from the repository root reproduces the same command.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas
import scipy
import sklearn
from scipy import stats

from src.corpus_discrimination import word_tokenize, TOKENIZER_VERSION
from src.cue_scoring import (
    FROZEN_CUE_CONFIG,
    TfidfLogRegCueScorer,
    load_reference_texts_from_repo,
)
from src.data_pipeline.build_c_source_authored_candidates import (
    file_sha256,
    verify_source_file,
)
from src.diagnostics.score_lexical_risk_cues import score_prompt
from src.v2_io import load_json, load_jsonl, resolve_benchmark

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── C-A section 7.1: pinned input hashes (abort if any differs) ───────────
PINNED_INPUT_HASHES = {
    "data/review/c_review_queue.csv":
        "8f6dfba182e5d3595d9ac6292d13956dd1a027b18770da01f4ef510f236787bb",
    "data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl":
        "e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b",
    "data/frozen_v2/LATEST_BENCHMARK.json":
        "817885c1c50dcbb5babddaec05b938f0f47067151ababa3c669e893f38ea937a",
    "logs/benchmark_gate_config.json":
        "1ac73585f08a4d685996c96eecafdbfcc74478ab07f7f762a4a9de2b2568b743",
    "logs/3d_b_lexical_outlierness_pilot.json":
        "95b0b7771244f0c162627eb1aaeb92986b4e7ec9de737f4f38edaefec53ebce5",
    "src/corpus_discrimination.py":
        "1ca62c4f7c1f88398c2d22c60bc1f2f6be27be678b68e9675a8800bdb41a9bcc",
    "src/cue_scoring.py":
        "ea0aa39faee7f8358121cee460be3c3f45d840c555beaba8ee534da2033b7d1d",
    "src/diagnostics/score_lexical_risk_cues.py":
        "5fda0d1856814b0582c07cc50fb2a42acb3275601ccb0b17e6a602d32545b89a",
    "src/v2_io.py":
        "34ca0e74ac669061a6e0f9fd1758c87c034cbeebb3f4a5af7783aebdea524e72",
}

EXPECTED_REVIEW_ROWS = 104

# ── C-A section 7.6: fixed seeds/draw counts (predeclared) ────────────────
BOOTSTRAP_SEED = 20260901
N_BOOTSTRAP = 10_000
PERMUTATION_SEED = 20260902
N_PERMUTATIONS = 100_000
HOLM_ALPHA = 0.05

# ── C-A section 7.5: feature families ──────────────────────────────────
FAMILY_STRUCTURAL = "structural"
FAMILY_FORMATTING = "formatting_confound"
FAMILY_LEXICAL_AUDIT = "lexical_audit"
FAMILY_DISTRIBUTIONAL = "distributional_exploratory"

# Features affected by the section-7.4 asymmetric low-coverage exclusion
# rule -- see "IMPLEMENTATION DECISION 2" in the module docstring.
LOW_COVERAGE_SENSITIVE_FEATURES = {
    "fightin_words",
    "fw_z_score",
    "lexical_diversity",
    "lexical_risk_hit_count",
    "cue_tfidf_logreg_margin",
}

DEFAULT_REVIEW_CSV = "data/review/c_review_queue.csv"
DEFAULT_BENCHMARK_LATEST = "data/frozen_v2/LATEST_BENCHMARK.json"
DEFAULT_GATE_CONFIG = "logs/benchmark_gate_config.json"
DEFAULT_FORMATTING_CONFIG_SOURCE = "logs/3d_b_lexical_outlierness_pilot.json"
DEFAULT_OUT_MD = "logs/c_b_paired_delta_analysis.md"
DEFAULT_OUT_JSON = "logs/c_b_paired_delta_analysis.json"


class AuditFailClosed(SystemExit):
    """Raised (as SystemExit) when a fail-closed check fails. No output
    is written in this case."""


# ── text-derived feature primitives ───────────────────────────────────────
def sentence_count(text: str) -> int:
    """Count of [.!?]+ matches on raw (non-normalized) text -- identical
    rule to 3D-B's multi_sentence_rule, applied as a count rather than a
    >=2 boolean (C-A section 7.5)."""
    return len(re.findall(r"[.!?]+", text))


def mean_word_length(text: str) -> Optional[float]:
    tokens = word_tokenize(text)
    if not tokens:
        return None
    return sum(len(t) for t in tokens) / len(tokens)


def lexical_diversity(text: str) -> Optional[float]:
    """Type-token ratio. No existing implementation in this repository --
    C-A section 7.5 requires this be defined fresh using the same
    word_tokenize as every other lexical feature."""
    tokens = word_tokenize(text)
    if not tokens:
        return None
    return len(set(tokens)) / len(tokens)


def _regex_hit(text: str, pattern: str) -> int:
    return 1 if re.search(pattern, text) else 0


# ── review-queue loading and row-level parsing ─────────────────────────────
def load_review_rows(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def filter_valid_pairs(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Applies the currently-defined row-validity criteria (C-A section
    7.3): review_status == accept, researcher_harm_qc == yes,
    wrapper_or_context_concern == no. None of the 104 current rows are
    excluded by this (100% clean on all three per C-A section 2), but
    the filter is implemented generally so a future re-freeze cannot
    silently bypass it -- excluded rows are tracked with reasons, never
    dropped without a recorded reason (task brief: "never silently
    report expected count as analyzed count")."""
    valid, excluded = [], []
    for r in rows:
        reasons = []
        if r["review_status"] != "accept":
            reasons.append(f"review_status={r['review_status']!r} (expected 'accept')")
        if r["researcher_harm_qc"] != "yes":
            reasons.append(f"researcher_harm_qc={r['researcher_harm_qc']!r} (expected 'yes')")
        if r["wrapper_or_context_concern"] != "no":
            reasons.append(
                f"wrapper_or_context_concern={r['wrapper_or_context_concern']!r} (expected 'no')"
            )
        if reasons:
            excluded.append({"record_id": r["record_id"], "reasons": reasons})
        else:
            valid.append(r)
    return valid, excluded


def split_by_low_coverage(rows: List[dict]) -> Tuple[List[dict], List[str]]:
    """C-A section 7.4's asymmetric missing-data rule, applied to
    whichever row set a caller passes in (see LOW_COVERAGE_SENSITIVE_FEATURES
    and IMPLEMENTATION DECISION 2)."""
    kept, excluded_ids = [], []
    for r in rows:
        if parse_bool(r["low_coverage_flag_source"]) or parse_bool(r["low_coverage_flag_candidate"]):
            excluded_ids.append(r["record_id"])
        else:
            kept.append(r)
    return kept, excluded_ids


# ── C-A section 1: data/provenance integrity (evidence-hierarchy tier 1) ──
def verify_provenance_integrity(benchmark_c_rows: List[dict], review_rows: List[dict]) -> dict:
    """Cross-checks the frozen benchmark's quadrant-C rows against the
    review queue: same count, same record_id set, and the quadrant
    identity claims from C-A section 1 (100% c_paired, 100%
    StrongREJECT). Fails closed on any mismatch -- this is the highest
    tier of the task brief's evidence hierarchy ("data/provenance
    integrity") and is re-verified at run time rather than assumed from
    C-A's own (already-verified, but static) inventory."""
    errors = []
    if len(benchmark_c_rows) != EXPECTED_REVIEW_ROWS:
        errors.append(
            f"expected {EXPECTED_REVIEW_ROWS} frozen quadrant-C rows, "
            f"found {len(benchmark_c_rows)}"
        )
    bench_ids = {r.get("record_id") for r in benchmark_c_rows}
    review_ids = {r["record_id"] for r in review_rows}
    if bench_ids != review_ids:
        only_bench = len(bench_ids - review_ids)
        only_review = len(review_ids - bench_ids)
        errors.append(
            "record_id set mismatch between frozen benchmark quadrant C and "
            f"the review queue: {only_bench} record_id(s) only in the "
            f"benchmark, {only_review} only in the review queue"
        )
    n_non_c_paired = sum(1 for r in benchmark_c_rows if r.get("c_construction") != "c_paired")
    if n_non_c_paired:
        errors.append(f"{n_non_c_paired} frozen quadrant-C row(s) have c_construction != 'c_paired'")
    n_non_strongreject = sum(1 for r in benchmark_c_rows if r.get("source_dataset") != "StrongREJECT")
    if n_non_strongreject:
        errors.append(f"{n_non_strongreject} frozen quadrant-C row(s) have source_dataset != 'StrongREJECT'")
    if errors:
        raise AuditFailClosed("FAIL CLOSED: provenance integrity check failed:\n  " + "\n  ".join(errors))
    return {
        "frozen_quadrant_c_row_count": len(benchmark_c_rows),
        "review_queue_row_count": len(review_rows),
        "record_id_sets_match": True,
        "all_c_construction_c_paired": True,
        "all_source_dataset_strongreject": True,
    }


# ── CUE scorer (IMPLEMENTATION DECISION 1: LOSO, StrongREJECT held out) ───
def build_cue_scorer() -> Tuple[TfidfLogRegCueScorer, dict]:
    refs = load_reference_texts_from_repo()
    h_texts = list(refs["HarmBench"]) + list(refs["SimpleSafetyTests"]) + list(refs["XSTest"])
    d_texts = list(refs["quadrant_D"])
    scorer = TfidfLogRegCueScorer().fit(h_texts, d_texts)
    provenance = {
        "held_out_source": "StrongREJECT",
        "train_h_sources": ["HarmBench", "SimpleSafetyTests", "XSTest"],
        "train_d_pool": "quadrant_D",
        "n_h": len(h_texts),
        "n_d": len(d_texts),
        "reference_h_sha256": scorer.reference_h_sha256_,
        "reference_d_sha256": scorer.reference_d_sha256_,
        "config_version": FROZEN_CUE_CONFIG["config_version"],
        "note": (
            "StrongREJECT is held out of the H reference because every "
            "R104 source_prompt/candidate_prompt is StrongREJECT-provenance "
            "text (C-A section 2); scoring it with a reference that "
            "includes StrongREJECT would be circular. Mirrors the existing "
            "leave-one-source-out convention in src/cue_scoring.py."
        ),
        "construct_relevance_note": (
            "Not construct-relevant: 3f_a already disqualifies this score "
            "as C_cue ground truth, because it is fit directly on the "
            "harmful/benign label. Reported for completeness only -- a "
            "significant paired difference on this feature must not be "
            "described as evidence about C_cue."
        ),
    }
    return scorer, provenance


# ── feature registry (C-A section 7.5) ─────────────────────────────────────
def _feature_specs(formatting_config: dict) -> List[dict]:
    bullet_re = formatting_config["bullet_marker_regex"]
    numstep_re = formatting_config["numbered_step_regex"]
    codeblock_re = formatting_config["code_block_regex"]

    return [
        dict(
            name="word_count", family=FAMILY_STRUCTURAL,
            getter=lambda r, s, c, cue: (float(r["word_count_source"]), float(r["word_count_candidate"])),
        ),
        dict(
            name="character_count", family=FAMILY_STRUCTURAL,
            getter=lambda r, s, c, cue: (float(r["character_count_source"]), float(r["character_count_candidate"])),
        ),
        dict(
            name="sentence_count", family=FAMILY_STRUCTURAL,
            getter=lambda r, s, c, cue: (float(sentence_count(s)), float(sentence_count(c))),
        ),
        dict(
            name="mean_word_length", family=FAMILY_STRUCTURAL,
            getter=lambda r, s, c, cue: (mean_word_length(s), mean_word_length(c)),
        ),
        dict(
            name="has_bullet_marker", family=FAMILY_FORMATTING,
            getter=lambda r, s, c, cue: (float(_regex_hit(s, bullet_re)), float(_regex_hit(c, bullet_re))),
        ),
        dict(
            name="has_numbered_step", family=FAMILY_FORMATTING,
            getter=lambda r, s, c, cue: (float(_regex_hit(s, numstep_re)), float(_regex_hit(c, numstep_re))),
        ),
        dict(
            name="has_code_block", family=FAMILY_FORMATTING,
            getter=lambda r, s, c, cue: (float(_regex_hit(s, codeblock_re)), float(_regex_hit(c, codeblock_re))),
        ),
        dict(
            name="multi_sentence_flag", family=FAMILY_FORMATTING,
            getter=lambda r, s, c, cue: (
                float(1 if sentence_count(s) >= 2 else 0),
                float(1 if sentence_count(c) >= 2 else 0),
            ),
        ),
        dict(
            name="fightin_words", family=FAMILY_LEXICAL_AUDIT,
            getter=lambda r, s, c, cue: (float(r["fightin_words_source"]), float(r["fightin_words_candidate"])),
        ),
        dict(
            name="fw_z_score", family=FAMILY_LEXICAL_AUDIT,
            getter=lambda r, s, c, cue: (float(r["fw_z_score_source"]), float(r["fw_z_score_candidate"])),
        ),
        dict(
            name="lexical_diversity", family=FAMILY_LEXICAL_AUDIT,
            getter=lambda r, s, c, cue: (lexical_diversity(s), lexical_diversity(c)),
        ),
        dict(
            name="lexical_risk_hit_count", family=FAMILY_LEXICAL_AUDIT,
            getter=lambda r, s, c, cue: (float(score_prompt(s)[0]), float(score_prompt(c)[0])),
        ),
        dict(
            name="cue_tfidf_logreg_margin", family=FAMILY_DISTRIBUTIONAL,
            getter=lambda r, s, c, cue: (
                cue.score(s)["tfidf_logreg_score_margin"],
                cue.score(c)["tfidf_logreg_score_margin"],
            ),
        ),
    ]


# ── statistics (C-A section 7.6) ───────────────────────────────────────────
def describe(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None, "iqr": None}
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "n": n,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sd": float(arr.std(ddof=1)) if n > 1 else None,
        "iqr": float(q75 - q25),
    }


def cohens_dz(deltas: Sequence[float]) -> Optional[float]:
    arr = np.asarray(deltas, dtype=float)
    if len(arr) < 2:
        return None
    sd = arr.std(ddof=1)
    if sd == 0:
        return None
    return float(arr.mean() / sd)


def paired_bootstrap_ci(
    deltas: Sequence[float], seed: int, n_bootstrap: int, ci: float = 0.95
) -> Optional[dict]:
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = arr[idx].mean(axis=1)
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "ci_level": ci,
        "mean_ci_low": float(np.percentile(means, lo_pct)),
        "mean_ci_high": float(np.percentile(means, hi_pct)),
    }


def sign_flip_permutation_test(deltas: Sequence[float], seed: int, n_permutations: int) -> Optional[dict]:
    """Sign-flip permutation test on delta_i: randomly negate each pair's
    delta with probability 0.5, recompute the mean, repeat. Two-sided
    empirical p-value with a +1/+1 continuity correction (same
    convention already used by src/analysis/analyze_3d_h.py's
    permutation test)."""
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    observed = float(arr.mean())
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_permutations, n)).astype(np.float64) * 2 - 1
    permuted_means = (signs * arr).mean(axis=1)
    p_value = float((np.sum(np.abs(permuted_means) >= abs(observed)) + 1) / (n_permutations + 1))
    return {
        "test_statistic": "mean(delta_i) under random sign-flips",
        "observed_statistic": observed,
        "n_permutations": n_permutations,
        "seed": seed,
        "empirical_two_sided_p_value": p_value,
        "null_distribution_summary": {
            "mean": float(permuted_means.mean()),
            "sd": float(permuted_means.std(ddof=1)),
        },
    }


def sign_consistency(deltas: Sequence[float]) -> dict:
    arr = np.asarray(deltas, dtype=float)
    n_pos = int((arr > 0).sum())
    n_neg = int((arr < 0).sum())
    n_zero = int((arr == 0).sum())
    n_nonzero = n_pos + n_neg
    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
        "proportion_positive_of_nonzero": (n_pos / n_nonzero) if n_nonzero > 0 else None,
    }


def holm_bonferroni(p_values_named: Dict[str, Optional[float]], alpha: float = HOLM_ALPHA) -> Dict[str, dict]:
    """Standard Holm step-down procedure. Features with no p-value
    (e.g. valid_n == 0) are excluded from the correction and reported
    with adjusted_p_holm = None, per C-A section 7.6 item 5."""
    items = sorted(((k, v) for k, v in p_values_named.items() if v is not None), key=lambda kv: kv[1])
    m = len(items)
    results: Dict[str, dict] = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(max((m - i) * p, running_max), 1.0)
        running_max = adj
        results[name] = {
            "raw_p": p,
            "rank": i + 1,
            "adjusted_p_holm": adj,
            "reject_at_alpha": adj <= alpha,
        }
    for name, p in p_values_named.items():
        if p is None:
            results[name] = {"raw_p": None, "rank": None, "adjusted_p_holm": None, "reject_at_alpha": None}
    return results


def category_breakdown(records: List[dict], id_to_category: Dict[str, str]) -> Dict[str, dict]:
    """C-A section 7.6 item 6: descriptive-only comparison of
    project_category distribution between the delta's sign. No formal
    test (categories are unbalanced 6-41 rows across 4 levels within
    R104 alone -- C-A section 7.6)."""
    buckets: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        sign = "positive" if rec["delta"] > 0 else ("negative" if rec["delta"] < 0 else "zero")
        buckets[sign][id_to_category.get(rec["record_id"], "unknown")] += 1
    return {sign: dict(counts) for sign, counts in buckets.items()}


def length_sensitivity(records: List[dict], word_count_delta_by_id: Dict[str, float]) -> dict:
    """C-A section 7.6 item 8: stratified comparison at the word-count
    median (n=104 is treated as too small for a stable partial
    correlation, per the contract's own note), plus a Spearman
    correlation as a simple companion diagnostic. Descriptive
    "length sensitivity" / "attenuation after adjustment" language only
    -- no causal claim."""
    paired = [
        (rec["delta"], word_count_delta_by_id[rec["record_id"]])
        for rec in records
        if rec["record_id"] in word_count_delta_by_id
    ]
    if len(paired) < 3:
        return {"n": len(paired), "note": "insufficient n for a length-sensitivity check"}
    feat_delta = np.array([p[0] for p in paired], dtype=float)
    wc_delta = np.array([p[1] for p in paired], dtype=float)
    rho, p_val = stats.spearmanr(wc_delta, feat_delta)
    median_wc = float(np.median(wc_delta))
    below = wc_delta <= median_wc
    above = ~below
    return {
        "n": len(paired),
        "spearman_corr_with_word_count_delta": float(rho),
        "spearman_p_value": float(p_val),
        "median_word_count_delta_split": {
            "median_word_count_delta": median_wc,
            "at_or_below_median_stratum": {
                "n": int(below.sum()),
                "mean_feature_delta": float(feat_delta[below].mean()) if below.sum() > 0 else None,
            },
            "above_median_stratum": {
                "n": int(above.sum()),
                "mean_feature_delta": float(feat_delta[above].mean()) if above.sum() > 0 else None,
            },
        },
        "note": (
            "Descriptive length-sensitivity association only -- not a "
            "causal claim that length change 'caused' this feature's delta."
        ),
    }


def analyze_feature(
    records: List[dict], bootstrap_seed: int, n_bootstrap: int, permutation_seed: int, n_permutations: int
) -> dict:
    source_vals = [r["source"] for r in records]
    candidate_vals = [r["candidate"] for r in records]
    deltas = [r["delta"] for r in records]
    return {
        "valid_n": len(deltas),
        "source": describe(source_vals),
        "candidate": describe(candidate_vals),
        "delta": describe(deltas),
        "paired_effect_size_dz": cohens_dz(deltas),
        "bootstrap_ci": paired_bootstrap_ci(deltas, bootstrap_seed, n_bootstrap),
        "sign_flip_permutation_test": sign_flip_permutation_test(deltas, permutation_seed, n_permutations),
        "sign_consistency": sign_consistency(deltas) if deltas else None,
    }


# ── per-population orchestration ───────────────────────────────────────────
def analyze_population(
    pop_rows: List[dict],
    feature_specs: List[dict],
    cue_scorer: TfidfLogRegCueScorer,
    bootstrap_seed: int,
    n_bootstrap: int,
    permutation_seed: int,
    n_permutations: int,
) -> dict:
    rows_for_lexical, low_cov_excluded_ids = split_by_low_coverage(pop_rows)
    id_to_category = {r["record_id"]: (r.get("project_category") or "unknown") for r in pop_rows}
    word_count_delta_by_id = {
        r["record_id"]: float(r["word_count_candidate"]) - float(r["word_count_source"])
        for r in rows_for_lexical
    }

    features_out: Dict[str, dict] = {}
    p_values_for_holm: Dict[str, Optional[float]] = {}

    for spec in feature_specs:
        name = spec["name"]
        rows_here = rows_for_lexical if name in LOW_COVERAGE_SENSITIVE_FEATURES else pop_rows
        records, missing_ids = [], []
        for r in rows_here:
            src_val, cand_val = spec["getter"](r, r["source_prompt"], r["candidate_prompt"], cue_scorer)
            if src_val is None or cand_val is None:
                missing_ids.append(r["record_id"])
                continue
            records.append({"record_id": r["record_id"], "source": src_val, "candidate": cand_val, "delta": cand_val - src_val})

        feat_result = analyze_feature(records, bootstrap_seed, n_bootstrap, permutation_seed, n_permutations)
        feat_result["family"] = spec["family"]
        feat_result["directional_hypothesis"] = "None predeclared (two-sided test) -- C-A section 7.5"
        feat_result["low_coverage_sensitive"] = name in LOW_COVERAGE_SENSITIVE_FEATURES
        feat_result["rows_excluded_missing_value"] = missing_ids
        feat_result["category_sensitivity"] = category_breakdown(records, id_to_category)
        if name in LOW_COVERAGE_SENSITIVE_FEATURES:
            feat_result["length_sensitivity"] = length_sensitivity(records, word_count_delta_by_id)
        if name == "cue_tfidf_logreg_margin":
            feat_result["construct_relevance_note"] = (
                "Not construct-relevant: 3f_a already disqualifies this score "
                "as C_cue ground truth (fit directly on the harmful/benign "
                "label). Reported for completeness only; a significant "
                "paired difference here must not be described as evidence "
                "about C_cue."
            )

        features_out[name] = feat_result
        perm = feat_result["sign_flip_permutation_test"]
        p_values_for_holm[name] = perm["empirical_two_sided_p_value"] if perm else None

    return {
        "n_valid_pairs": len(pop_rows),
        "n_low_coverage_excluded_from_lexical_family": len(low_cov_excluded_ids),
        "low_coverage_excluded_record_ids": low_cov_excluded_ids,
        "features": features_out,
        "multiple_comparison_correction": {
            "method": "holm_bonferroni",
            "alpha": HOLM_ALPHA,
            "family_size": sum(1 for v in p_values_for_holm.values() if v is not None),
            "note": (
                "Applied separately within this population only (C-A "
                "section 7.6 item 5) -- not pooled across the three "
                "populations, which represent different predeclared "
                "evidence tiers rather than repeated measurements of the "
                "same question."
            ),
            "per_feature": holm_bonferroni(p_values_for_holm),
        },
        "source_sensitivity": {
            "applicable": False,
            "reason": "R104 is 100% StrongREJECT (C-A section 2) -- no within-R104 source contrast is computable.",
        },
        "decision_labels": None,
        "decision_labels_note": (
            "KEEP FOR HUMAN REVIEW / KEEP AS SECONDARY / INCONCLUSIVE / "
            "DROP labels (C-A section 7.7) are not assigned by this "
            "implementation -- final interpretation is out of scope for "
            "the C-B implementation task."
        ),
    }


# ── CLI / main ──────────────────────────────────────────────────────────
def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _display_path(p: Path) -> str:
    """Relative-to-repo-root path for display/config recording when
    possible (matches the rest of the repository's convention), falling
    back to the absolute path for outputs written outside the repository
    (e.g. by tests using a tmp_path fixture)."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--benchmark-latest", default=DEFAULT_BENCHMARK_LATEST)
    parser.add_argument("--gate-config", default=DEFAULT_GATE_CONFIG)
    parser.add_argument("--formatting-config-source", default=DEFAULT_FORMATTING_CONFIG_SOURCE)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--permutation-seed", type=int, default=PERMUTATION_SEED)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    return parser.parse_args(argv)


def get_code_version() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        )
    except Exception as exc:  # pragma: no cover - environment without git
        return {"generation_commit": None, "working_tree_dirty": None, "error": str(exc)}
    return {"generation_commit": commit, "working_tree_dirty": dirty}


def build_markdown(analysis: dict) -> str:
    lines = [
        "# C-B -- Paired Delta Analysis (R104 `c_paired` Quadrant-C Construction)",
        "",
        "Status: implementation of the locked contract in "
        "`logs/c_existing_construction_audit_spec.md` section 7. Descriptive "
        "and inferential statistics only -- no KEEP/DROP resource decision "
        "is made by this document (C-A section 7.7).",
        "",
        f"Generation commit: `{analysis['code_version'].get('generation_commit')}` "
        f"(working tree dirty: {analysis['code_version'].get('working_tree_dirty')})",
        "",
        "## Provenance integrity",
        "",
        f"- Frozen quadrant-C rows: {analysis['provenance_integrity']['frozen_quadrant_c_row_count']}",
        f"- Review-queue rows: {analysis['provenance_integrity']['review_queue_row_count']}",
        f"- record_id sets match: {analysis['provenance_integrity']['record_id_sets_match']}",
        "",
        "## Pair integrity",
        "",
        "| Population | expected_pairs | valid_pairs | excluded_pairs |",
        "|---|---|---|---|",
    ]
    for pop_name, pop_data in analysis["pair_integrity"]["populations"].items():
        lines.append(
            f"| {pop_name} | {pop_data['expected_pairs']} | {pop_data['valid_pairs']} | "
            f"{pop_data['excluded_pairs']} |"
        )
    lines += ["", "## Per-population, per-feature summary", ""]
    for pop_name, pop_result in analysis["results"].items():
        lines += [f"### {pop_name} (n={pop_result['n_valid_pairs']})", ""]
        lines += [
            "| Feature | Family | valid_n | mean(delta) | d_z | bootstrap 95% CI | perm p | Holm-adj p |",
            "|---|---|---|---|---|---|---|---|",
        ]
        holm = pop_result["multiple_comparison_correction"]["per_feature"]
        for feat_name, feat in pop_result["features"].items():
            d = feat["delta"]
            ci = feat["bootstrap_ci"]
            perm = feat["sign_flip_permutation_test"]
            ci_str = f"[{ci['mean_ci_low']:.4f}, {ci['mean_ci_high']:.4f}]" if ci else "n/a"
            dz = f"{feat['paired_effect_size_dz']:.4f}" if feat["paired_effect_size_dz"] is not None else "n/a"
            p_str = f"{perm['empirical_two_sided_p_value']:.4g}" if perm else "n/a"
            holm_p = holm.get(feat_name, {}).get("adjusted_p_holm")
            holm_str = f"{holm_p:.4g}" if holm_p is not None else "n/a"
            mean_d = f"{d['mean']:.4f}" if d["mean"] is not None else "n/a"
            lines.append(
                f"| {feat_name} | {feat['family']} | {feat['valid_n']} | {mean_d} | {dz} | "
                f"{ci_str} | {p_str} | {holm_str} |"
            )
        lines.append("")
        lines.append(f"Source sensitivity: {pop_result['source_sensitivity']['reason']}")
        lines.append("")
        lines.append(pop_result["decision_labels_note"])
        lines.append("")

    lines += [
        "## CUE score construct-relevance caveat",
        "",
        analysis["cue_scorer_provenance"]["construct_relevance_note"],
        "",
        "## Software versions (actual runtime; C-A section 7.8)",
        "",
        f"- Python: {analysis['software_versions']['python']}",
        f"- numpy: {analysis['software_versions']['numpy']}",
        f"- scipy: {analysis['software_versions']['scipy']}",
        f"- pandas: {analysis['software_versions']['pandas']}",
        f"- scikit-learn: {analysis['software_versions']['scikit_learn']}",
        "",
        "## Explicit non-actions (mirrors C-A section 9 / task brief scope)",
        "",
        "- Did not create, rewrite, or modify any prompt.",
        "- Did not modify any frozen input listed in section 7.1.",
        "- Did not run model inference or GPU code, and did not access the web.",
        "- Did not begin B/D construction or common-CUE/contrastive construction.",
        "- Did not assign a KEEP/KEEP-AS-SECONDARY/INCONCLUSIVE/DROP label.",
        "- Did not analyze R-AUTHORED: C-A section 3/6 states R-AUTHORED "
        "analysis has not started (100% review_status=pending) and section 7 "
        "does not include it in the locked contract.",
        "",
        "**Stop.**",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> dict:
    args = parse_args(argv)

    review_csv_path = _resolve(args.review_csv)
    benchmark_latest_path = _resolve(args.benchmark_latest)
    gate_config_path = _resolve(args.gate_config)
    formatting_config_source_path = _resolve(args.formatting_config_source)
    out_md_path = _resolve(args.out_md)
    out_json_path = _resolve(args.out_json)

    # 1. Fail-closed hash verification (C-A section 7.1).
    verified_hashes = {}
    verified_hashes["data/review/c_review_queue.csv"] = verify_source_file(
        review_csv_path, PINNED_INPUT_HASHES["data/review/c_review_queue.csv"], "c_review_queue.csv"
    )
    verified_hashes["data/frozen_v2/LATEST_BENCHMARK.json"] = verify_source_file(
        benchmark_latest_path,
        PINNED_INPUT_HASHES["data/frozen_v2/LATEST_BENCHMARK.json"],
        "LATEST_BENCHMARK.json",
    )
    verified_hashes["logs/benchmark_gate_config.json"] = verify_source_file(
        gate_config_path, PINNED_INPUT_HASHES["logs/benchmark_gate_config.json"], "benchmark_gate_config.json"
    )
    verified_hashes["logs/3d_b_lexical_outlierness_pilot.json"] = verify_source_file(
        formatting_config_source_path,
        PINNED_INPUT_HASHES["logs/3d_b_lexical_outlierness_pilot.json"],
        "3d_b_lexical_outlierness_pilot.json (formatting_diagnostic_config source)",
    )
    for rel in (
        "src/corpus_discrimination.py",
        "src/cue_scoring.py",
        "src/diagnostics/score_lexical_risk_cues.py",
        "src/v2_io.py",
    ):
        verified_hashes[rel] = verify_source_file(REPO_ROOT / rel, PINNED_INPUT_HASHES[rel], rel)

    benchmark_path, benchmark_sha = resolve_benchmark(latest_path=benchmark_latest_path)
    pinned_benchmark_sha = PINNED_INPUT_HASHES["data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl"]
    if benchmark_sha != pinned_benchmark_sha:
        raise AuditFailClosed(
            "FAIL CLOSED: frozen benchmark hash does not match the C-A "
            f"section 7.1 pinned value (expected {pinned_benchmark_sha}, "
            f"got {benchmark_sha}); a re-freeze may have occurred."
        )
    verified_hashes["data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl"] = benchmark_sha

    # 2. Load inputs.
    review_rows = load_review_rows(review_csv_path)
    if len(review_rows) != EXPECTED_REVIEW_ROWS:
        raise AuditFailClosed(
            f"FAIL CLOSED: expected exactly {EXPECTED_REVIEW_ROWS} review-queue "
            f"rows, got {len(review_rows)}"
        )
    pair_ids = [r["pair_id"] for r in review_rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise AuditFailClosed("FAIL CLOSED: duplicate pair_id values in review queue")

    benchmark_rows = load_jsonl(benchmark_path)
    benchmark_c_rows = [r for r in benchmark_rows if r.get("quadrant") == "C"]
    provenance_integrity = verify_provenance_integrity(benchmark_c_rows, review_rows)

    gate_config = load_json(gate_config_path)
    formatting_source = load_json(formatting_config_source_path)
    formatting_config = formatting_source["confound_diagnostics"]["formatting_diagnostic_config"]

    # Recorded for reproducibility only -- NOT part of the C-A section 7.1
    # fail-closed pinned list (that list does not include these two files),
    # but they are read internally by
    # src.cue_scoring.load_reference_texts_from_repo and their hashes are
    # recorded here so a re-run can be verified against the same content.
    additional_input_hashes = {
        "data/processed/controlled_eval.jsonl": file_sha256(
            REPO_ROOT / "data/processed/controlled_eval.jsonl"
        ),
        "data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl": file_sha256(
            REPO_ROOT / "data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl"
        ),
    }

    # 3. Row-validity filter and predeclared populations (C-A section 7.3).
    valid_rows, excluded_review_status = filter_valid_pairs(review_rows)
    populations = {
        "population_1_all_valid_accepted_pairs": valid_rows,
        "population_2_assistance_type_preserved_yes": [
            r for r in valid_rows if r["assistance_type_preserved"] == "yes"
        ],
        "population_3_assistance_type_preserved_partial": [
            r for r in valid_rows if r["assistance_type_preserved"] == "partial"
        ],
    }
    expected_counts = {
        "population_1_all_valid_accepted_pairs": 104,
        "population_2_assistance_type_preserved_yes": 78,
        "population_3_assistance_type_preserved_partial": 26,
    }

    # 4. CUE scorer (fit once; IMPLEMENTATION DECISION 1).
    cue_scorer, cue_provenance = build_cue_scorer()

    # 5. Feature registry.
    feature_specs = _feature_specs(formatting_config)

    # 6. Per-population analysis.
    results = {}
    pair_integrity_populations = {}
    for pop_name, pop_rows in populations.items():
        expected = expected_counts[pop_name]
        actual = len(pop_rows)
        pair_integrity_populations[pop_name] = {
            "expected_pairs": expected,
            "valid_pairs": actual,
            "excluded_pairs": expected - actual,
            "note": (
                "expected_pairs is the subset size C-A section 7.3 currently "
                "observed (104/78/26). If a future re-freeze changes "
                "assistance_type_preserved membership, this excluded_pairs "
                "value reflects the difference from that historical "
                "expectation, not a row-level exclusion reason."
            ),
        }
        results[pop_name] = analyze_population(
            pop_rows,
            feature_specs,
            cue_scorer,
            args.bootstrap_seed,
            args.n_bootstrap,
            args.permutation_seed,
            args.n_permutations,
        )

    analysis = {
        "task": "C-B paired delta analysis (R104 c_paired quadrant-C construction)",
        "spec_reference": "logs/c_existing_construction_audit_spec.md section 7",
        "code_version": get_code_version(),
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "config": {
            "review_csv": _display_path(review_csv_path),
            "benchmark_latest": _display_path(benchmark_latest_path),
            "gate_config": _display_path(gate_config_path),
            "formatting_config_source": _display_path(formatting_config_source_path),
            "bootstrap_seed": args.bootstrap_seed,
            "n_bootstrap": args.n_bootstrap,
            "permutation_seed": args.permutation_seed,
            "n_permutations": args.n_permutations,
            "gate_config_min_token_recognition_fraction": gate_config.get("min_token_recognition_fraction"),
        },
        "pinned_input_hashes_verified": verified_hashes,
        "additional_recorded_input_hashes_not_pinned_by_c_a": additional_input_hashes,
        "provenance_integrity": provenance_integrity,
        "pair_integrity": {
            "review_status_filter": {
                "expected_pairs": len(review_rows),
                "valid_pairs": len(valid_rows),
                "excluded_pairs": len(excluded_review_status),
                "exclusion_reasons": excluded_review_status,
            },
            "populations": pair_integrity_populations,
        },
        "cue_scorer_provenance": cue_provenance,
        "results": results,
        "implementation_decisions": [
            "IMPLEMENTATION DECISION 1: CUE score fit with StrongREJECT held "
            "out of the H reference (LOSO), since R104 text is 100% "
            "StrongREJECT-provenance and scoring it with a reference that "
            "includes StrongREJECT would be circular. See module docstring.",
            "IMPLEMENTATION DECISION 2: the section-7.4 low-coverage "
            "exclusion rule is applied to fightin_words, fw_z_score, "
            "lexical_diversity, lexical_risk_hit_count, and "
            "cue_tfidf_logreg_margin (all frequency/lexicon-based), and not "
            "to any structural count or formatting indicator. See module "
            "docstring.",
        ],
        "explicit_non_actions": [
            "did not create, rewrite, or modify any prompt",
            "did not modify any C-A section 7.1 pinned input",
            "did not run model inference or GPU code",
            "did not access the web",
            "did not begin B/D construction or common-CUE/contrastive construction",
            "did not assign a KEEP/KEEP-AS-SECONDARY/INCONCLUSIVE/DROP label "
            "(C-A section 7.7; out of scope for this implementation task)",
            "did not analyze R-AUTHORED (100% review_status=pending per C-A "
            "section 3; not part of the locked section 7 contract)",
        ],
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(build_markdown(analysis), encoding="utf-8")

    print(f"analysis_json={_display_path(out_json_path)}")
    print(f"analysis_md={_display_path(out_md_path)}")
    print(f"generation_commit={analysis['code_version'].get('generation_commit')}")
    for pop_name, pop_result in results.items():
        print(f"{pop_name}: n={pop_result['n_valid_pairs']}")

    return analysis


if __name__ == "__main__":
    main()
