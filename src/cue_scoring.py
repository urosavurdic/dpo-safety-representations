"""
CUE (lexical harmful-association) scorers - two independent methods.

Implements the two-method reference-data design recorded in
logs/cue_reference_audit.md for the revised 2x2 (intent x CUE) benchmark:

    Method 1: TF-IDF (word + character n-grams) -> Logistic Regression.
              See TfidfLogRegCueScorer.
    Method 2: smoothed lexical log-odds / Fightin' Words - REUSED
              unchanged from src/corpus_discrimination.py.FightinWords,
              not reimplemented here.

CUE, per project definition, is an OPERATIONAL lexical harmful-association
score computed from two independent lexical methods; agreement between
them is used to identify a robust tail. This is NOT a claim that CUE is a
pure or independently identified latent construct - the same caveat
src/corpus_discrimination.py states for Fightin' Words applies equally to
everything in this module.

Both methods are prompt-text-only. Feature extraction never sees source
metadata, category metadata, intent labels, or model responses - only the
raw prompt string. Source/record metadata is used ONLY to build
leakage-safe folds (see below), never as a model feature.

── Leakage control (per logs/cue_reference_audit.md "Source-held-out
   scoring plan") ──────────────────────────────────────────────────────
- Harmful reference sources (HarmBench / StrongREJECT / SimpleSafetyTests)
  rotate under 3-fold leave-one-source-out (LOSO): scoring any item drawn
  from one of these sources fits both CUE methods on the OTHER two
  harmful sources (plus the fixed benign sides below), never on the
  source the item itself was drawn from. See leave_one_source_out_folds
  and score_harmful_reference_sources_loso.
- Benign, high-register reference (XSTest) has NO LOSO fold available -
  it is the only accessible source at that register. This is a disclosed
  limitation, not silently papered over: see XSTEST_LIMITATION_NOTE. This
  module does not produce an XSTest CUE score, because doing so would
  require either a circular reference (fit including XSTest itself) or
  an asymmetric one (drop the benign-high-register side); neither is
  performed silently.
- Benign, low-register reference (quadrant D pool) is unchanged and
  fixed, per the existing Fightin' Words design in
  src/corpus_discrimination.py.
- A generic dedup-group-aware fold helper (grouped_kfold_indices) is
  provided for any future pool that is NOT cleanly split into distinct
  named sources, so duplicate/template rows are never split across a
  train/held-out boundary. It reuses stripped_sha256/normalized_sha256
  verbatim from src/data_pipeline/build_c_source_authored_candidates.py -
  it does not recompute a new hashing/dedup convention.

All preprocessing/tokenizer/model parameters are declared in
FROZEN_CUE_CONFIG below and must not change after any real candidate
batch is scored - freeze a new config_version instead of editing this one
in place (mirrors the "predeclared, do not modify after creation"
convention already used by logs/benchmark_gate_config.json and by
FightinWords itself).
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion

from src.corpus_discrimination import (
    FightinWords,
    TOKENIZER_VERSION,
    assign_strata,
    empirical_rank,
    word_tokenize,
)
from src.data_pipeline.build_c_source_authored_candidates import (
    normalized_sha256,
    stripped_sha256,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_PATH = REPO_ROOT / "data/processed/controlled_eval.jsonl"
C_SOURCE_AUTHORED_VALIDATED_PATH = (
    REPO_ROOT / "data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl"
)
GATE_CONFIG_PATH = REPO_ROOT / "logs/benchmark_gate_config.json"

HARMFUL_LOSO_SOURCES = ("HarmBench", "StrongREJECT", "SimpleSafetyTests")
BENIGN_HIGH_REGISTER_SOURCE = "XSTest"  # no LOSO fold available - see below
BENIGN_LOW_REGISTER_QUADRANT = "D"

XSTEST_LIMITATION_NOTE = (
    "XSTest is the only accessible benign-high-register reference source "
    "(logs/cue_reference_audit.md); no leave-one-source-out fold exists "
    "for it. Any CUE score for XSTest-drawn material would necessarily be "
    "fit on a reference that either includes XSTest itself (circular) or "
    "drops the benign-high-register side entirely (asymmetric). This "
    "module does not silently produce an XSTest CUE score for that "
    "reason; XSTest is used only as a fixed component of the H reference "
    "when scoring the three harmful LOSO sources."
)

# ── Frozen configuration ────────────────────────────────────────────────
# Predeclared before any real candidate batch is scored with this module,
# per the task requirement to freeze preprocessing/tokenizer/model
# parameters before final candidate scoring. Treat as read-only once a
# real scoring run has happened - bump config_version instead of editing.
FROZEN_CUE_CONFIG = {
    "config_version": "cue_scorer_v1",
    "tfidf_logreg": {
        "word_analyzer": "word",
        "word_ngram_range": [1, 2],
        "word_min_df": 2,
        "char_analyzer": "char_wb",
        "char_ngram_range": [3, 5],
        "char_min_df": 2,
        "lowercase": True,
        "sublinear_tf": True,
        "max_features_per_view": 20000,
        "logreg_C": 1.0,
        "logreg_penalty": "l2",
        "logreg_class_weight": "balanced",
        "logreg_max_iter": 2000,
        "logreg_random_state": 20260829,
        "positive_class_role": "H (harmful reference and/or benign-high-register reference)",
        "negative_class_role": "D (benign low-register reference, quadrant D pool)",
        "score_field_semantics": (
            "tfidf_logreg_score_margin is the raw LogisticRegression "
            "decision_function value (signed distance to the separating "
            "hyperplane) - monotonic in H-association by construction. "
            "The sigmoid of this margin is reported separately and is NOT "
            "treated as a calibrated probability (no calibration step is "
            "performed)."
        ),
    },
    "fightin_words": {
        "prior_strength_per_token": 0.01,
        "min_count": 1,
        "min_token_recognition_fraction": 0.5,
        "tokenizer_version": TOKENIZER_VERSION,
        "note": "Unchanged from src/corpus_discrimination.py FightinWords defaults - not reimplemented here.",
    },
    "leakage_control": {
        "harmful_loso_sources": list(HARMFUL_LOSO_SOURCES),
        "benign_high_register_source": BENIGN_HIGH_REGISTER_SOURCE,
        "benign_high_register_held_out": False,
        "benign_low_register_quadrant": BENIGN_LOW_REGISTER_QUADRANT,
        "dedup_group_key": "prompt_normalized_sha256",
        "default_max_metric_rank_disagreement": 0.25,
    },
    "_note": (
        "Predeclared per logs/cue_reference_audit.md 'Source-held-out "
        "scoring plan' before any candidate batch is scored. Freeze a new "
        "config_version instead of editing this dict in place once a real "
        "scoring run has happened."
    ),
}


# ── Method 1: TF-IDF (word + char n-grams) -> Logistic Regression ──────────
class TfidfLogRegCueScorer:
    """
    TF-IDF word/character n-grams -> Logistic Regression.

    Fits on the same H (positive, label=1) / D (negative, label=0)
    reference roles FightinWords uses. Produces a MONOTONIC
    harmful-association score - the raw decision_function margin - rather
    than treating an uncalibrated sigmoid as a literal probability.

    Only prompt text is used as a feature. Parameters come from `config`
    (defaults to FROZEN_CUE_CONFIG["tfidf_logreg"]) and must stay fixed
    across every fold of one scoring run.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(config) if config is not None else dict(FROZEN_CUE_CONFIG["tfidf_logreg"])
        self.is_fitted = False
        self._vectorizer = None
        self._clf = None
        self.reference_h_sha256_: Optional[str] = None
        self.reference_d_sha256_: Optional[str] = None
        self.n_h_: int = 0
        self.n_d_: int = 0

    def _build_vectorizer(self) -> FeatureUnion:
        cfg = self.config
        word_vec = TfidfVectorizer(
            analyzer=cfg["word_analyzer"],
            ngram_range=tuple(cfg["word_ngram_range"]),
            min_df=cfg["word_min_df"],
            lowercase=cfg["lowercase"],
            sublinear_tf=cfg["sublinear_tf"],
            max_features=cfg["max_features_per_view"],
        )
        char_vec = TfidfVectorizer(
            analyzer=cfg["char_analyzer"],
            ngram_range=tuple(cfg["char_ngram_range"]),
            min_df=cfg["char_min_df"],
            lowercase=cfg["lowercase"],
            sublinear_tf=cfg["sublinear_tf"],
            max_features=cfg["max_features_per_view"],
        )
        return FeatureUnion([("word_tfidf", word_vec), ("char_tfidf", char_vec)])

    def fit(self, h_texts: Sequence[str], d_texts: Sequence[str]) -> "TfidfLogRegCueScorer":
        """Fit on H (label 1) and D (label 0) prompt text only. Records
        reference corpus hashes for provenance, mirroring FightinWords.fit."""
        h_texts = list(h_texts)
        d_texts = list(d_texts)
        if not h_texts or not d_texts:
            raise ValueError("TfidfLogRegCueScorer.fit requires non-empty H and D text lists.")

        texts = h_texts + d_texts
        labels = [1] * len(h_texts) + [0] * len(d_texts)

        vectorizer = self._build_vectorizer()
        X = vectorizer.fit_transform(texts)

        clf = LogisticRegression(
            C=self.config["logreg_C"],
            penalty=self.config["logreg_penalty"],
            class_weight=self.config["logreg_class_weight"],
            max_iter=self.config["logreg_max_iter"],
            random_state=self.config["logreg_random_state"],
        )
        clf.fit(X, labels)

        self._vectorizer = vectorizer
        self._clf = clf
        self.n_h_, self.n_d_ = len(h_texts), len(d_texts)
        h_joined = "\n".join(sorted(h_texts))
        d_joined = "\n".join(sorted(d_texts))
        self.reference_h_sha256_ = hashlib.sha256(h_joined.encode("utf-8")).hexdigest()
        self.reference_d_sha256_ = hashlib.sha256(d_joined.encode("utf-8")).hexdigest()
        self.is_fitted = True
        return self

    def score(self, text: str) -> Dict:
        """Deterministic given a fitted model: same text -> same output."""
        assert self.is_fitted, "Call fit() first."
        X = self._vectorizer.transform([text])
        margin = float(self._clf.decision_function(X)[0])
        # Monotonic in the margin, but NOT a calibrated probability - no
        # calibration step (e.g. Platt scaling on held-out data) is run.
        uncalibrated_sigmoid = 1.0 / (1.0 + math.exp(-margin))
        n_tokens = len(word_tokenize(text))
        return {
            "tfidf_logreg_score_margin": round(margin, 6),
            "tfidf_logreg_uncalibrated_sigmoid": round(uncalibrated_sigmoid, 6),
            "tfidf_logreg_score_is_calibrated_probability": False,
            "tokens_total_count": n_tokens,
            "empty_or_short_prompt_flag": n_tokens == 0,
            "config_version": FROZEN_CUE_CONFIG["config_version"],
            "reference_h_sha256": self.reference_h_sha256_,
            "reference_d_sha256": self.reference_d_sha256_,
            "reference_n_h": self.n_h_,
            "reference_n_d": self.n_d_,
        }

    def score_batch(self, texts: Sequence[str]) -> List[Dict]:
        return [self.score(t) for t in texts]


# ── Fold construction (leakage control) ─────────────────────────────────────
def leave_one_source_out_folds(sources: Dict[str, Sequence[str]]) -> List[Dict]:
    """
    Leave-one-source-out folds over named text pools.

    `sources` must map source name -> list of prompt texts (>= 2 distinct
    sources required). Returns one fold dict per source, in the same
    order `sources` was given (a regular dict preserves insertion order in
    Python 3.7+) - deterministic, not sorted/shuffled, so callers control
    reproducible ordering by how they build the input dict.

    Each fold dict has:
        held_out_source   - the source name excluded from training
        held_out_texts    - that source's own texts (to be scored out-of-fold)
        train_sources     - the other source names, in their given order
        train_texts       - concatenation of the other sources' texts
    """
    names = list(sources.keys())
    if len(names) < 2:
        raise ValueError(
            f"leave_one_source_out_folds requires >= 2 distinct sources, got {len(names)}."
        )
    folds = []
    for held_out in names:
        train_sources = [n for n in names if n != held_out]
        train_texts: List[str] = []
        for n in train_sources:
            train_texts.extend(sources[n])
        folds.append(
            {
                "held_out_source": held_out,
                "held_out_texts": list(sources[held_out]),
                "train_sources": train_sources,
                "train_texts": train_texts,
            }
        )
    return folds


def grouped_kfold_indices(
    items: Sequence[Dict], text_field: str, n_splits: int
) -> List[Tuple[List[int], List[int]]]:
    """
    Fallback leakage-safe fold builder for pools that do NOT come from
    cleanly distinct named sources (prefer leave_one_source_out_folds
    whenever a source label is available - see module docstring).

    Groups items by prompt_normalized_sha256 (reused verbatim from
    src/data_pipeline/build_c_source_authored_candidates.py - NOT a new
    dedup convention) so that duplicated/template-related rows always
    land in the same fold - never split across a train/held-out boundary.

    Deterministic: sklearn's GroupKFold assigns groups to folds by a
    fixed greedy balancing rule over the group order given (no shuffling,
    no random_state), so the same `items` order always yields the same
    split.

    Returns a list of (train_indices, test_indices) tuples, one per
    split, as positions into `items`.
    """
    if n_splits < 2:
        raise ValueError("grouped_kfold_indices requires n_splits >= 2.")
    groups = [normalized_sha256(item[text_field]) for item in items]
    n_unique_groups = len(set(groups))
    if n_unique_groups < n_splits:
        raise ValueError(
            f"Cannot build {n_splits} leakage-safe folds: only "
            f"{n_unique_groups} distinct dedup group(s) available across "
            f"{len(items)} item(s) (need >= n_splits so every fold can get "
            f"at least one whole group). Reduce n_splits or grow the pool."
        )
    gkf = GroupKFold(n_splits=n_splits)
    return [
        (list(train_idx), list(test_idx))
        for train_idx, test_idx in gkf.split(range(len(items)), groups=groups)
    ]


# ── Agreement (wires the predeclared, previously-unwired gate field) ───────
def compute_agreement(rank_a: float, rank_b: float, max_disagreement: float) -> Dict:
    """
    Compares two methods' empirical ranks for the same item and applies
    the predeclared logs/benchmark_gate_config.json
    `max_metric_rank_disagreement` threshold (read by the caller, never
    modified here - see load_gate_config).
    """
    disagreement = abs(rank_a - rank_b)
    return {
        "rank_disagreement": round(disagreement, 6),
        "max_metric_rank_disagreement": max_disagreement,
        "methods_agree_within_threshold": disagreement <= max_disagreement,
    }


def load_gate_config(path: Path = GATE_CONFIG_PATH) -> Dict:
    """Read-only load of the frozen gate config - never writes to it."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Orchestration: LOSO scoring of the three harmful reference sources ─────
def score_harmful_reference_sources_loso(
    harmful_source_texts: Dict[str, Sequence[str]],
    xstest_texts: Sequence[str],
    quadrant_d_texts: Sequence[str],
    gate_config: Optional[Dict] = None,
    tfidf_config: Optional[Dict] = None,
) -> List[Dict]:
    """
    Implements the LOSO "Source-held-out scoring plan" from
    logs/cue_reference_audit.md for the three harmful reference sources.

    For each of HARMFUL_LOSO_SOURCES, both CUE methods are fit on:
        H = (the OTHER two harmful sources) + xstest_texts
        D = quadrant_d_texts
    (excluding the held-out source's own text), then every item of the
    held-out source is scored out-of-fold by both methods. Per-item
    empirical rank (within that fold's held-out source) and agreement
    (using max_metric_rank_disagreement) are computed for each method.

    xstest_texts is a FIXED component of H in every fold (never held
    out) - see XSTEST_LIMITATION_NOTE. This function does not score
    xstest_texts or quadrant_d_texts themselves.
    """
    missing = set(HARMFUL_LOSO_SOURCES) - set(harmful_source_texts)
    if missing:
        raise ValueError(f"Missing required harmful LOSO source(s): {sorted(missing)}")

    gate_config = gate_config or {}
    max_disagreement = gate_config.get(
        "max_metric_rank_disagreement",
        FROZEN_CUE_CONFIG["leakage_control"]["default_max_metric_rank_disagreement"],
    )
    min_tok_frac = gate_config.get(
        "min_token_recognition_fraction",
        FROZEN_CUE_CONFIG["fightin_words"]["min_token_recognition_fraction"],
    )

    ordered_sources = {name: harmful_source_texts[name] for name in HARMFUL_LOSO_SOURCES}
    folds = leave_one_source_out_folds(ordered_sources)

    all_scored: List[Dict] = []
    for fold in folds:
        train_h = list(fold["train_texts"]) + list(xstest_texts)
        train_d = list(quadrant_d_texts)

        tfidf_scorer = TfidfLogRegCueScorer(config=tfidf_config).fit(train_h, train_d)
        fw = FightinWords(
            prior_strength=FROZEN_CUE_CONFIG["fightin_words"]["prior_strength_per_token"],
            min_count=FROZEN_CUE_CONFIG["fightin_words"]["min_count"],
        ).fit(train_h, train_d)

        held_out_texts = fold["held_out_texts"]
        tfidf_scores = [tfidf_scorer.score(t) for t in held_out_texts]
        fw_scores = [fw.score(t, min_tok_frac) for t in held_out_texts]

        tfidf_margins = [s["tfidf_logreg_score_margin"] for s in tfidf_scores]
        fw_unnorm = [s["fightin_words_score_unnormalized"] for s in fw_scores]

        for i, text in enumerate(held_out_texts):
            tfidf_rank = empirical_rank(tfidf_margins[i], tfidf_margins)
            fw_rank = empirical_rank(fw_unnorm[i], fw_unnorm)
            agreement = compute_agreement(tfidf_rank, fw_rank, max_disagreement)
            all_scored.append(
                {
                    "held_out_source": fold["held_out_source"],
                    "train_sources": fold["train_sources"],
                    "prompt_text": text,
                    "prompt_sha256": stripped_sha256(text),
                    "held_out_from_reference": True,
                    "tfidf_logreg": tfidf_scores[i],
                    "fightin_words": fw_scores[i],
                    "tfidf_logreg_source_rank": tfidf_rank,
                    "fightin_words_source_rank": fw_rank,
                    "tfidf_logreg_source_strata": assign_strata(tfidf_rank),
                    "fightin_words_source_strata": assign_strata(fw_rank),
                    **agreement,
                }
            )
    return all_scored


# ── Reference-data loading (reuses existing tracked artifacts only) ────────
def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_reference_texts_from_repo() -> Dict[str, List[str]]:
    """
    Loads the reference-data pools exactly as scoped in
    logs/cue_reference_audit.md "Selected sources for CUE reference
    fitting": HarmBench (quadrant A of the frozen eval set), StrongREJECT
    and SimpleSafetyTests (the eligible_for_3a3 rows of the existing,
    already-validated C-source-authored candidate universe - reused
    as-is, not recomputed), XSTest (quadrant B), and the quadrant D pool.
    Prompt text only - no other field is returned.

    Reads only already-tracked, already-committed repository artifacts
    (data/processed/controlled_eval.jsonl,
    data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl) -
    does not touch data/raw/ or re-run any acquisition/validation step.
    """
    eval_rows = _read_jsonl(EVAL_SET_PATH)
    harmbench_texts = [
        r["prompt"] for r in eval_rows if r.get("quadrant") == "A" and r.get("source") == "HarmBench"
    ]
    xstest_texts = [
        r["prompt"] for r in eval_rows if r.get("quadrant") == "B" and r.get("source") == "XSTest"
    ]
    quadrant_d_texts = [r["prompt"] for r in eval_rows if r.get("quadrant") == "D"]

    candidate_rows = _read_jsonl(C_SOURCE_AUTHORED_VALIDATED_PATH)
    strongreject_texts = [
        r["prompt_text"]
        for r in candidate_rows
        if r.get("source_dataset") == "StrongREJECT" and r.get("candidate_universe_status") == "eligible_for_3a3"
    ]
    simplesafetytests_texts = [
        r["prompt_text"]
        for r in candidate_rows
        if r.get("source_dataset") == "SimpleSafetyTests"
        and r.get("candidate_universe_status") == "eligible_for_3a3"
    ]

    return {
        "HarmBench": harmbench_texts,
        "StrongREJECT": strongreject_texts,
        "SimpleSafetyTests": simplesafetytests_texts,
        "XSTest": xstest_texts,
        "quadrant_D": quadrant_d_texts,
    }
