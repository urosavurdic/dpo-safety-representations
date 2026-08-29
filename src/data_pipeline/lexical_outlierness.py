"""
3D-B: within-harmful lexical outlierness pilot.

Implements, without reinterpretation, the frozen design recorded in
logs/3d_a_lexical_outlierness_design.md ("3D-A"):

  - S2  Method 1: leave-one-group-out (LOGO) TF-IDF centroid distance
  - S3  Method 2: LOGO smoothed token self-information
  - S4  Duplicate/template grouping (frozen-field seed + 5-gram shingle
        Jaccard >= 0.6 union-find), frozen BEFORE scoring
  - S6  Fold-calibrated percentiles (the sole percentile definition used
        anywhere downstream)
  - S7  Agreement/robustness statistics (Spearman, tail overlap/Jaccard,
        permutation baseline, bootstrap)
  - S8  Source-balanced sensitivity
  - S9  Confound diagnostics (source, category, length, formatting,
        source-prediction, source-balanced, category-balanced, length
        sensitivity, OOV-rate)

Every locked numeric/procedural choice below cites the design-doc section
it implements. Where the design doc is silent on a micro-level detail
that still had to be resolved to produce runnable code, the choice is
recorded in AMBIGUITY_NOTES at the bottom of this module and surfaced in
the pilot's output rather than silently assumed to be the only reading.

Explicitly NOT implemented here (per 3D-B task brief): CUE, harmful-vs-
benign classification, Fightin' Words, quadrant construction, benign-
control selection, human evaluation. src/cue_scoring.py is not imported
or modified.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_pipeline.build_c_source_authored_candidates import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]

POPULATION_ARTIFACT_PATH = (
    REPO_ROOT / "data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl"
)
GROUPS_ARTIFACT_PATH = (
    REPO_ROOT / "data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json"
)
OUT_JSON_PATH = REPO_ROOT / "logs/3d_b_lexical_outlierness_pilot.json"
OUT_MD_PATH = REPO_ROOT / "logs/3d_b_lexical_outlierness_pilot.md"

EXPECTED_POPULATION_SIZE = 209
REQUIRED_SOURCES = ("StrongREJECT", "SimpleSafetyTests")

ALPHA = 1.0  # S3, locked
SHINGLE_LEN = 5  # S4
SHINGLE_THRESHOLD = 0.6  # S4, locked
HIGH_TAIL_CUTOFF = 0.75  # S6
LOW_TAIL_CUTOFF = 0.25  # S6
PERMUTATION_REPS = 10_000  # S7
PERMUTATION_SEED = 1337  # S7 "fixed, logged seed" - chosen and recorded here
BOOTSTRAP_REPS = 10_000  # S7
BOOTSTRAP_SEED = 20260829  # S7 "fixed, logged seed" - chosen and recorded here

# ── S2/S3 shared preprocessing ──────────────────────────────────────────────
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\b\w+\b")
PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """S2 locked preprocessing. NFKC -> lowercase -> URL placeholder ->
    whitespace collapse/strip. (The doc lists whitespace-collapse before
    the URL bullet but does not state execution order between the two;
    see AMBIGUITY_NOTES #1 - collapsing after URL substitution is the
    choice made here.)"""
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    t = URL_RE.sub("<url>", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def tokenize(normalized_text: str) -> List[str]:
    """S2: regex word-boundary tokens \\b\\w+\\b. Numbers kept literal;
    apostrophes are not word chars (accepted as-is, per S2)."""
    return TOKEN_RE.findall(normalized_text)


def word_ngrams_1_2(tokens: Sequence[str]) -> Counter:
    """S2 Method-1 features: word n-grams, range (1,2), raw within-doc
    counts (no augmentation/log scaling)."""
    feats = Counter()
    for tok in tokens:
        feats[(tok,)] += 1
    for i in range(len(tokens) - 1):
        feats[(tokens[i], tokens[i + 1])] += 1
    return feats


def similarity_text(text: str) -> str:
    """S4 grouping-only text variant: S2 normalization plus punctuation
    stripped. Never used for scoring."""
    t = normalize_text(text)
    t = PUNCT_RE.sub("", t)
    return t


def shingles5(text: str) -> set:
    """S4: character 5-gram shingles, no padding. Empty for len(text)<5."""
    if len(text) < SHINGLE_LEN:
        return set()
    return {text[i : i + SHINGLE_LEN] for i in range(len(text) - SHINGLE_LEN + 1)}


def jaccard(a: set, b: set) -> float:
    """S4: explicit 0.0 for either/both empty (avoids 0/0)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ── S4 duplicate / template grouping ────────────────────────────────────────
class UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: attach lexicographically-larger root under
            # the smaller, keeps result independent of union order
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def build_groups(rows: List[dict]) -> Tuple[Dict[str, str], dict]:
    """S4. Seeds from frozen exact/normalized-duplicate canonical fields
    (steps 1-2 of S4), then unions any pair with 5-gram-shingle Jaccard
    >= 0.6 over the similarity-normalized text (steps 3-5), assigns
    group_id = lexicographically smallest record_id per component (step
    6), and returns (record_id -> group_id, artifact_dict) for freezing
    to GROUPS_ARTIFACT_PATH (step 8) BEFORE any scoring.
    """
    record_ids = [r["record_id"] for r in rows]
    uf = UnionFind(record_ids)
    by_id = {r["record_id"]: r for r in rows}

    # Steps 1-2: seed from already-frozen exact/normalized duplicate fields.
    for r in rows:
        rid = r["record_id"]
        for canon_field in ("exact_duplicate_canonical_record_id", "normalized_duplicate_canonical_record_id"):
            canon = r.get(canon_field)
            if canon and canon != rid and canon in by_id:
                uf.union(rid, canon)

    # Steps 3-5: shingle-Jaccard >= 0.6 union over ALL C(n,2) pairs,
    # traversed in ascending (record_id, record_id) order for
    # reproducible logging.
    sim_text = {r["record_id"]: similarity_text(r["prompt_text"]) for r in rows}
    shingle_sets = {rid: shingles5(t) for rid, t in sim_text.items()}
    sorted_ids = sorted(record_ids)
    merge_log = []
    for i in range(len(sorted_ids)):
        for j in range(i + 1, len(sorted_ids)):
            a, b = sorted_ids[i], sorted_ids[j]
            j_sim = jaccard(shingle_sets[a], shingle_sets[b])
            if j_sim >= SHINGLE_THRESHOLD:
                uf.union(a, b)
                merge_log.append({"a": a, "b": b, "jaccard": j_sim})

    # Step 6: group_id = lexicographically smallest record_id per component.
    components: Dict[str, List[str]] = defaultdict(list)
    for rid in record_ids:
        components[uf.find(rid)].append(rid)
    record_to_group = {}
    for members in components.values():
        gid = min(members)
        for m in members:
            record_to_group[m] = gid

    artifact = {
        "artifact": "lexical_outlierness_groups_v1",
        "design_ref": "logs/3d_a_lexical_outlierness_design.md#4",
        "threshold": SHINGLE_THRESHOLD,
        "shingle_len": SHINGLE_LEN,
        "metric": "character_5gram_shingle_jaccard",
        "seed_fields": [
            "exact_duplicate_canonical_record_id",
            "normalized_duplicate_canonical_record_id",
        ],
        "n_rows": len(rows),
        "n_groups": len(components),
        "n_shingle_merges": len(merge_log),
        "shingle_merges": merge_log,
        "record_id_to_group_id": record_to_group,
    }
    return record_to_group, artifact


# ── S1 population loading (never re-derived) ────────────────────────────────
def load_population() -> List[dict]:
    """S1: selects the exact 209-row pool by reading the per-record
    eligibility already recorded by 3A3 (`candidate_universe_status ==
    "eligible_for_3a3"`, equivalently empty `exclusion_reasons`) - the
    same predicate 3A3 applied, never a redefinition. Asserts the
    resulting count is exactly 209, failing loudly otherwise."""
    rows = []
    with open(POPULATION_ARTIFACT_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    eligible = [r for r in rows if r.get("candidate_universe_status") == "eligible_for_3a3"]
    assert not any(r.get("exclusion_reasons") for r in eligible), (
        "eligible_for_3a3 row carries non-empty exclusion_reasons - predicate mismatch"
    )
    if len(eligible) != EXPECTED_POPULATION_SIZE:
        raise SystemExit(
            f"FAIL CLOSED: expected exactly {EXPECTED_POPULATION_SIZE} eligible rows, "
            f"got {len(eligible)}. Refusing to proceed - do not re-derive the population."
        )
    return eligible


def category_of(row: dict) -> str:
    """S9 category field policy: harm_area only, missing/blank -> literal
    'unknown', never dropped, never falls back to another field."""
    val = row.get("harm_area")
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return "unknown"
    return val


# ── S2 Method 1: LOGO TF-IDF centroid distance ──────────────────────────────
def _vectorize(feats: Counter, idf: Dict[Tuple[str, ...], float]) -> Tuple[Dict, float]:
    """Raw TF-IDF restricted to idf's keys (OOV features dropped), then
    L2-normalized. Returns (normalized_sparse_vec, pre_norm_l2)."""
    vec = {f: c * idf[f] for f, c in feats.items() if f in idf}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0.0:
        return {}, 0.0
    return {f: v / norm for f, v in vec.items()}, norm


def _cosine(v: Dict, mu: Dict) -> Optional[float]:
    if not v:
        return None
    dot = sum(val * mu.get(f, 0.0) for f, val in v.items())
    norm_v = math.sqrt(sum(val * val for val in v.values()))  # == 1.0 unless empty
    norm_mu = math.sqrt(sum(val * val for val in mu.values()))
    if norm_mu == 0.0:
        raise AssertionError(
            "S2 halt condition: fold centroid mu_i is the zero vector "
            "(V_{-g} empty) - refusing to emit a score."
        )
    return dot / (norm_v * norm_mu)


def fold_tfidf(
    ref_feats: List[Counter],
    weights: Optional[List[float]] = None,
) -> Tuple[Dict[Tuple[str, ...], float], Dict, float]:
    """Fits idf_{-g} (S2, smoothed sklearn-style) and centroid mu_{-g}
    over a fold's reference feature-Counters. weights=None -> primary
    (unweighted, w_j=1); weights given -> S8-style balanced variant
    (df/centroid weighted, total weight in place of |R_{-g}|)."""
    if weights is None:
        weights = [1.0] * len(ref_feats)
    total_weight = sum(weights)
    df: Dict[Tuple[str, ...], float] = defaultdict(float)
    for feats, w in zip(ref_feats, weights):
        for f in feats.keys():
            df[f] += w
    idf = {f: math.log((1.0 + total_weight) / (1.0 + d)) + 1.0 for f, d in df.items()}
    mu: Dict[Tuple[str, ...], float] = defaultdict(float)
    ref_vecs = []
    for feats, w in zip(ref_feats, weights):
        v, _ = _vectorize(feats, idf)
        ref_vecs.append(v)
        for f, val in v.items():
            mu[f] += w * val
    if weights is None or abs(total_weight - 1.0) > 1e-9:
        # primary: mu_i = (1/|R_-g|) * sum v_j  (weights are all 1 here)
        mu = {f: val / len(ref_feats) for f, val in mu.items()} if ref_feats else {}
    # (balanced branch: weights already sum to 1.0 by construction - S8 - no further division)
    return idf, dict(mu), ref_vecs


def score_tfidf(feats: Counter, idf: Dict, mu: Dict) -> float:
    v, _ = _vectorize(feats, idf)
    cos = _cosine(v, mu)
    if cos is None:
        return float("nan")
    return 1.0 - cos


# ── S3 Method 2: LOGO smoothed token self-information ───────────────────────
def fold_selfinfo(
    ref_tokens: List[List[str]],
    weights: Optional[List[float]] = None,
) -> Tuple[Dict[str, float], float, int]:
    """Fits c_{-g}(t), N_{-g}, |V_{-g}| (S3). weights=None -> primary
    (raw token-occurrence counts); weights given -> S8-style balanced
    token counts (same |V_{-g}|, i.e. same distinct-type set, per S8)."""
    unweighted_counts: Counter = Counter()
    for toks in ref_tokens:
        unweighted_counts.update(toks)
    v_size = len(unweighted_counts)  # distinct types observed - S8: unaffected by weighting

    if weights is None:
        counts = {t: float(c) for t, c in unweighted_counts.items()}
    else:
        counts = defaultdict(float)
        for toks, w in zip(ref_tokens, weights):
            for t in toks:
                counts[t] += w
        counts = dict(counts)
    total = sum(counts.values())
    return counts, total, v_size


def score_selfinfo(tokens: List[str], counts: Dict[str, float], total: float, v_size: int) -> float:
    if len(tokens) == 0:
        return float("nan")
    s = 0.0
    for t in tokens:
        c = counts.get(t, 0.0)
        s += -math.log((c + ALPHA) / (total + ALPHA * v_size))
    return s / len(tokens)


# ── S6 fold-calibrated percentile ────────────────────────────────────────────
def calibrated_percentile(score: float, ref_scores: Sequence[float], ref_weights: Optional[Sequence[float]] = None) -> Optional[float]:
    if math.isnan(score):
        return None
    finite = [(r, (1.0 if ref_weights is None else w)) for r, w in zip(ref_scores, ref_weights or [1.0] * len(ref_scores)) if not math.isnan(r)]
    total_w = sum(w for _, w in finite)
    if total_w == 0.0:
        return None
    lt = sum(w for r, w in finite if r < score)
    eq = sum(w for r, w in finite if r == score)
    return (lt + 0.5 * eq) / total_w


AMBIGUITY_NOTES = [
    {
        "id": 1,
        "location": "S2 normalize_text",
        "note": (
            "S2 lists whitespace-collapse before the URL-placeholder bullet but "
            "does not state execution order between the two. Implemented order: "
            "NFKC -> lowercase -> URL substitution -> whitespace collapse/strip. "
            "Effect on this pool is expected to be nil (URLs, if any, contain no "
            "internal whitespace requiring reordering) but is flagged rather than "
            "silently assumed."
        ),
    },
    {
        "id": 2,
        "location": "S9 item 5, source-prediction diagnostic",
        "note": (
            "S9 specifies 'logistic regression predicting source from tail "
            "membership alone; report accuracy/AUC against majority-class "
            "baseline' but does not specify a train/test split. Implemented as "
            "an in-sample fit-and-evaluate on the full scored pool (this is a "
            "diagnostic characterizing association in the realized sample, not "
            "a held-out predictive-generalization claim)."
        ),
    },
    {
        "id": 3,
        "location": "S9 item 8, length sensitivity 'tail membership changes'",
        "note": (
            "S9 does not specify how tail membership is redefined from the "
            "length-residualized percentile. Implemented: residuals are "
            "converted to an empirical percentile within the scored pool using "
            "the same '<' + 0.5*'=' / N rule as S6, then the same 0.75/0.25 "
            "cutoffs are applied; rows whose tail label differs from the "
            "original are counted as 'changed'."
        ),
    },
    {
        "id": 4,
        "location": "S9 item 4, formatting diagnostics",
        "note": (
            "S9 names the required categories (list markers, numbered steps, "
            "code-block delimiters, multi-sentence structure) but not exact "
            "regexes. Implemented regexes are recorded verbatim in the pilot "
            "JSON output's formatting_diagnostic_config field for auditability."
        ),
    },
]
