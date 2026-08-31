"""
Task C-F-B -- Joint-geometry analysis for A/B/C/D (+ R104-source,
R-AUTHORED).

This module implements EXACTLY the locked analysis contract in
`logs/cf_joint_geometry_spec.md` (C-F-A). It is an IMPLEMENTATION of that
contract, not a new design: every input path, hash, population
definition, feature definition, view-construction rule, seed, and
statistical procedure below is copied from that document, not invented
here. Where the spec leaves an implementation detail open, that decision
is stated explicitly (see the IMPLEMENTATION DECISION notes below) rather
than silently resolved, mirroring the convention already used in
`src/analysis/c_b_paired_delta_analysis.py`.

Population roles (C-F-A section 2; onboarding table):
  A, B, C, D    -- the four benchmark quadrants (`prompt` field), used for
                   every centroid/contrast/PCA/distribution-distance
                   computation. C is EXACTLY the R104 candidate side --
                   not a separate fit or a separate text source.
  R104-source   -- auxiliary, `source_prompt` column of
                   data/review/c_review_queue.csv. Projection-only, never
                   included in any A/B/C/D contrast or in any fit.
  R-AUTHORED    -- auxiliary, `candidate_prompt` column of
                   data/review/c_source_authored_review_queue.csv.
                   Projection-only, never included in any A/B/C/D
                   contrast or in any fit. Q25-selected, review_status
                   pending (C-D/C-E); this module repeats both caveats
                   alongside any R-AUTHORED-derived number (C-F-A
                   section 6).

Fit population: the structural standardizer, both TF-IDF vectorizers, the
TruncatedSVD, the common FightinWords instance, the n-gram vocabularies,
and the combined-view PCA are ALL fit on the pooled A/B/C/D rows only
(n=654 at the pinned commit). R104-source and R-AUTHORED are transformed
(never fit) into that space (C-F-A section 2 -- "the single most
consequential design choice in this document").

What this module explicitly does NOT do (C-F-A section 9 / task brief):
  - does not modify any benchmark input, candidate wording, or frozen
    file;
  - does not redefine any feature or add a metric beyond section 5's
    five required analysis blocks;
  - does not add alternative preprocessing after inspecting results, or
    make any procedure below adaptive to observed data;
  - does not train a supervised classifier to define an axis, and does
    not run contrastive/representation-learning training of any kind;
  - does not run GPU inference; the optional section-7 embeddings check
    is CPU-eligible in principle but is not run in this sandbox (no
    network access to the Hugging Face Hub) and is reported as
    "not_run", per the same fail-closed disclosure convention already
    used for R104's untested near-duplicate check (C-A section 2);
  - does not decide whether C is valid, whether the intended 2x2
    construct holds, or whether R-AUTHORED should be promoted -- every
    PCA angle, distance, and divergence value below is reported as a
    geometric/descriptive fact about this specific fitted
    representation, never as evidence for or against latent
    psychological independence of intent and surface cue (C-F-A
    section 5.2);
  - does not print raw prompt text, matched lexicon terms, or
    per-token/per-bigram JSD contributions to any committed output --
    only aggregate statistics and, where a row is excluded, its
    `record_id` (C-F-A section 5.5's explicit prohibition).

IMPLEMENTATION DECISION 1 (empty/OOV fallback within the retained
structural-only row, C-F-A section 4.5): the spec states that a
zero-token row's `word_count`/`character_count`/`sentence_count` "remain
well-defined at 0" in the structural-only view, but does not state a
value for `mean_word_length`/`lexical_diversity` (undefined as a ratio
of two zeros) for that same retained row. This module extends the same
"well-defined at 0" convention to those two features (0.0) so the row can
still be standardized and centroided in the structural-only view, exactly
as the spec's own retention rule requires -- rather than leaving a NaN
that would silently propagate into every downstream centroid. Verified
against the actual populations at run time: `n_empty_token_rows` is
reported per population, and this branch is exercised only if that count
is nonzero (expected 0, per section 4.5).

IMPLEMENTATION DECISION 2 (residualization target, C-F-A section 5.6
item 1): the spec says to "residualize every section 4.1/4.2-derived
feature on word_count ... recompute section 5.1 centroids and section 5.4
statistics on the residualized structural-only and combined views" but
does not state whether residualization happens before or after the
section-4.4 standardization/SVD/L2-normalization steps that build those
views. This module residualizes the ALREADY-BUILT view matrices'
columns directly (post-standardization for structural-only; post-L2-norm-
and-concatenation for combined) against each population's own raw
`word_count`, rather than reconstructing the views from residualized raw
features. This is the simpler of the two readings, is fully determined by
the already-locked view definitions, and does not introduce a second,
separately-fit standardization step.

IMPLEMENTATION DECISION 3 (FightinWords None-score rows, not addressed by
the spec): `FightinWords.score()` returns
`fightin_words_score_normalized = None` when a row has zero *recognized*
vocabulary tokens (`tokens_recognized_count == 0`), which is a narrower
condition than section 4.5's "zero tokens from `word_tokenize`" OOV rule.
This module treats a None `fw_score_common_v1` the same way as a
zero-token row: retained in the structural-only view, excluded from the
lexical-only and combined views (its z-scored value would otherwise be
undefined). Expected to affect 0 rows for real English-language prompts;
implemented defensively so a future re-freeze cannot silently divide by
an undefined value.

Exact reproduction command (C-F-A section 8; also every flag's default,
so running with no arguments from the repository root reproduces it):
    python -m src.analysis.cf_joint_geometry \\
        --benchmark-latest data/frozen_v2/LATEST_BENCHMARK.json \\
        --review-csv data/review/c_review_queue.csv \\
        --r-authored-csv data/review/c_source_authored_review_queue.csv \\
        --gate-config logs/benchmark_gate_config.json \\
        --formatting-config-source logs/3d_b_lexical_outlierness_pilot.json \\
        --svd-seed 20260905 \\
        --permutation-seed 20260903 --n-permutations 10000 \\
        --out-md logs/cf_joint_geometry_analysis.md \\
        --out-json logs/cf_joint_geometry_analysis.json

Fails closed (raises `AuditFailClosed`, a `SystemExit`) if any C-F-A
section 1 pinned hash does not match at load time, or if any population's
observed row count does not match the section 2 pinned count -- no output
is written in that case.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas
import scipy
import sklearn
from scipy import sparse
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.corpus_discrimination import FightinWords, word_tokenize
from src.diagnostics.score_lexical_risk_cues import score_prompt
from src.v2_io import load_json, load_jsonl, resolve_benchmark

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── C-F-A section 1: pinned input hashes (abort if any differs) ───────────
PINNED_INPUT_HASHES = {
    "data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl":
        "e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b",
    "data/frozen_v2/LATEST_BENCHMARK.json":
        "817885c1c50dcbb5babddaec05b938f0f47067151ababa3c669e893f38ea937a",
    "data/processed/controlled_eval.jsonl":
        "e640c2fba47afe2853c8717ae8492c62bf26cce21f6ec677f68ea88b117c05af",
    "data/review/c_review_queue.csv":
        "8f6dfba182e5d3595d9ac6292d13956dd1a027b18770da01f4ef510f236787bb",
    "data/review/c_source_authored_review_queue.csv":
        "c62725ec37b3d950d7fda164c89d6e71a315cc5dad1a701147100c9f8e8e485a",
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
}

# ── C-F-A section 2: pinned population sizes ───────────────────────────────
EXPECTED_POPULATION_COUNTS = {
    "A": 150,
    "B": 250,
    "C": 104,
    "D": 150,
    "R104_source": 104,
    "R_AUTHORED": 52,
}
QUADRANT_ORDER = ["A", "B", "C", "D"]
AUXILIARY_ORDER = ["R104_source", "R_AUTHORED"]
ALL_POPULATIONS_ORDER = QUADRANT_ORDER + AUXILIARY_ORDER
PAIR_ORDER = ["AB", "AC", "AD", "BC", "BD", "CD"]

# ── C-F-A section 8: seeds (new, verified against every seed already in
# use elsewhere in this repository -- 42/43/45/1337/271828/20260829-902) ──
SVD_SEED = 20260905
SVD_N_COMPONENTS = 50
PERMUTATION_SEED = 20260903
N_PERMUTATIONS_DEFAULT = 10_000
HOLM_ALPHA = 0.05

# ── C-F-A section 4.3 ───────────────────────────────────────────────────
FW_PRIOR_STRENGTH = 0.01
FW_MIN_COUNT = 1

# ── C-F-A section 5.5 ───────────────────────────────────────────────────
JSD_MIN_DF = 2
JSD_ALPHA = 0.01
RARE_TOKEN = "<RARE>"

# ── C-F-A section 4.1 ───────────────────────────────────────────────────
STRUCTURAL_FEATURE_ORDER = [
    "word_count",
    "character_count",
    "sentence_count",
    "mean_word_length",
    "lexical_diversity",
    "has_bullet_marker",
    "has_numbered_step",
    "has_code_block",
    "multi_sentence_flag",
    "lexical_risk_hit_count",
]

DEFAULT_BENCHMARK_LATEST = "data/frozen_v2/LATEST_BENCHMARK.json"
DEFAULT_REVIEW_CSV = "data/review/c_review_queue.csv"
DEFAULT_R_AUTHORED_CSV = "data/review/c_source_authored_review_queue.csv"
DEFAULT_GATE_CONFIG = "logs/benchmark_gate_config.json"
DEFAULT_FORMATTING_CONFIG_SOURCE = "logs/3d_b_lexical_outlierness_pilot.json"
DEFAULT_OUT_MD = "logs/cf_joint_geometry_analysis.md"
DEFAULT_OUT_JSON = "logs/cf_joint_geometry_analysis.json"


class AuditFailClosed(SystemExit):
    """Raised (as SystemExit) when a fail-closed check fails. No output
    is written in this case."""


# ── hashing / provenance ────────────────────────────────────────────────
def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pinned_hashes(repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    """C-F-A section 1: fail closed if any pinned input hash differs.
    Returns the actually-observed hashes (never assumed to still match
    the table above without direct re-verification)."""
    verified = {}
    mismatches = []
    for rel_path, expected in PINNED_INPUT_HASHES.items():
        full_path = repo_root / rel_path
        if not full_path.exists():
            mismatches.append(f"{rel_path}: MISSING")
            continue
        actual = file_sha256(full_path)
        verified[rel_path] = actual
        if actual != expected:
            mismatches.append(f"{rel_path}: expected {expected}, got {actual}")
    if mismatches:
        raise AuditFailClosed(
            "FAIL CLOSED: C-F-A section 1 pinned-hash re-verification "
            "failed:\n  " + "\n  ".join(mismatches)
        )
    return verified


def get_code_version() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        )
    except Exception as exc:  # pragma: no cover - environment without git
        return {"generation_commit": None, "working_tree_dirty": None, "error": str(exc)}
    return {"generation_commit": commit, "working_tree_dirty": dirty}


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ── text-derived feature primitives (byte-identical rules to C-A/C-B) ───
def sentence_count(text: str) -> int:
    """Count of [.!?]+ matches on raw text -- identical rule to 3D-B's
    multi_sentence_rule, applied as a count (C-F-A section 4.1)."""
    return len(re.findall(r"[.!?]+", text))


def mean_word_length(text: str) -> Optional[float]:
    tokens = word_tokenize(text)
    if not tokens:
        return None
    return sum(len(t) for t in tokens) / len(tokens)


def lexical_diversity(text: str) -> Optional[float]:
    tokens = word_tokenize(text)
    if not tokens:
        return None
    return len(set(tokens)) / len(tokens)


def regex_hit(text: str, pattern: str) -> int:
    return 1 if re.search(pattern, text) else 0


def simple_word_count(text: str) -> int:
    return len(text.split())


def simple_char_count(text: str) -> int:
    return len(text)


def zero_token_mask(texts: Sequence[str]) -> np.ndarray:
    """C-F-A section 4.5: rows whose word_tokenize(prompt) is empty."""
    return np.array([len(word_tokenize(t)) == 0 for t in texts], dtype=bool)


# ── population loading (C-F-A section 2) ────────────────────────────────
def load_frozen_quadrants(benchmark_rows: List[dict]) -> Dict[str, List[dict]]:
    pops: Dict[str, List[dict]] = {q: [] for q in QUADRANT_ORDER}
    for row in benchmark_rows:
        q = row.get("quadrant")
        if q not in pops:
            continue
        pops[q].append({
            "record_id": row.get("record_id"),
            "text": row.get("prompt"),
            "word_count": float(row.get("word_count")),
            "character_count": float(row.get("character_count")),
            "source_dataset": row.get("source_dataset"),
            "project_category": row.get("project_category"),
        })
    return pops


def load_review_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_r104_source_population(review_csv_path: Path) -> List[dict]:
    """R104-source: `source_prompt` column of c_review_queue.csv, using
    that CSV's own `word_count_source`/`character_count_source` columns
    (already computed on source_prompt by construction -- unlike
    R-AUTHORED, this pairing is unambiguous, C-F-A section 2)."""
    rows = load_review_csv(review_csv_path)
    return [
        {
            "record_id": r["record_id"],
            "text": r["source_prompt"],
            "word_count": float(r["word_count_source"]),
            "character_count": float(r["character_count_source"]),
        }
        for r in rows
    ]


def check_r_authored_word_char_basis(raw_rows: List[dict]) -> dict:
    """C-F-A section 2's required pre-implementation check: confirm
    directly whether R-AUTHORED's existing `word_count`/`character_count`
    columns were computed on `candidate_prompt` (as needed here) or
    `source_prompt` -- the column names alone do not settle this."""
    n = len(raw_rows)
    wc_matches_candidate = sum(
        1 for r in raw_rows
        if int(float(r["word_count"])) == simple_word_count(r["candidate_prompt"])
    )
    wc_matches_source = sum(
        1 for r in raw_rows
        if int(float(r["word_count"])) == simple_word_count(r["source_prompt"])
    )
    cc_matches_candidate = sum(
        1 for r in raw_rows
        if int(float(r["character_count"])) == simple_char_count(r["candidate_prompt"])
    )
    cc_matches_source = sum(
        1 for r in raw_rows
        if int(float(r["character_count"])) == simple_char_count(r["source_prompt"])
    )
    basis_confirmed_candidate = (
        n > 0 and wc_matches_candidate == n and cc_matches_candidate == n
    )
    return {
        "n_rows": n,
        "word_count_matches_candidate_prompt": wc_matches_candidate,
        "word_count_matches_source_prompt": wc_matches_source,
        "character_count_matches_candidate_prompt": cc_matches_candidate,
        "character_count_matches_source_prompt": cc_matches_source,
        "basis_confirmed_candidate_prompt": basis_confirmed_candidate,
        "branch_taken": (
            "reuse_existing_columns" if basis_confirmed_candidate
            else "recompute_fresh_from_candidate_prompt"
        ),
    }


def load_r_authored_population(r_authored_csv_path: Path) -> Tuple[List[dict], dict]:
    """R-AUTHORED: `candidate_prompt` column of
    c_source_authored_review_queue.csv. Per C-F-A section 2, recomputes
    word_count/character_count fresh from candidate_prompt if the
    pre-implementation check does not confirm the existing columns'
    basis; records which branch was taken either way."""
    raw_rows = load_review_csv(r_authored_csv_path)
    precheck = check_r_authored_word_char_basis(raw_rows)
    reuse = precheck["basis_confirmed_candidate_prompt"]
    out = []
    for r in raw_rows:
        text = r["candidate_prompt"]
        if reuse:
            word_count = float(r["word_count"])
            character_count = float(r["character_count"])
        else:
            word_count = float(simple_word_count(text))
            character_count = float(simple_char_count(text))
        out.append({
            "record_id": r["record_id"],
            "text": text,
            "word_count": word_count,
            "character_count": character_count,
        })
    return out, precheck


def verify_population_counts(populations: Dict[str, List[dict]]) -> None:
    mismatches = []
    for name, expected in EXPECTED_POPULATION_COUNTS.items():
        actual = len(populations.get(name, []))
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, got {actual}")
    if mismatches:
        raise AuditFailClosed(
            "FAIL CLOSED: C-F-A section 2 population-size mismatch (a "
            "re-freeze may have occurred):\n  " + "\n  ".join(mismatches)
        )


# ── section 4.1: structural feature block ───────────────────────────────
def compute_structural_raw(
    rows: List[dict], formatting_config: dict
) -> Dict[str, List[float]]:
    bullet_re = formatting_config["bullet_marker_regex"]
    numstep_re = formatting_config["numbered_step_regex"]
    codeblock_re = formatting_config["code_block_regex"]

    feats: Dict[str, List[float]] = {name: [] for name in STRUCTURAL_FEATURE_ORDER}
    n_empty_token_rows = 0
    for r in rows:
        text = r["text"]
        feats["word_count"].append(r["word_count"])
        feats["character_count"].append(r["character_count"])
        feats["sentence_count"].append(float(sentence_count(text)))
        mwl = mean_word_length(text)
        ld = lexical_diversity(text)
        if mwl is None or ld is None:
            n_empty_token_rows += 1
        # IMPLEMENTATION DECISION 1 (module docstring): 0.0 fallback for
        # a zero-token row, so the row stays well-defined in this view.
        feats["mean_word_length"].append(mwl if mwl is not None else 0.0)
        feats["lexical_diversity"].append(ld if ld is not None else 0.0)
        feats["has_bullet_marker"].append(float(regex_hit(text, bullet_re)))
        feats["has_numbered_step"].append(float(regex_hit(text, numstep_re)))
        feats["has_code_block"].append(float(regex_hit(text, codeblock_re)))
        feats["multi_sentence_flag"].append(1.0 if sentence_count(text) >= 2 else 0.0)
        feats["lexical_risk_hit_count"].append(float(score_prompt(text)[0]))
    feats["_n_empty_token_rows"] = n_empty_token_rows  # type: ignore[assignment]
    return feats


def zero_variance_features(pooled_fit_raw: Dict[str, List[float]]) -> List[str]:
    """C-F-A section 4.1's predeclared zero-variance drop rule, re-run on
    the full A/B/C/D pool (not assumed to transfer from R104-scale)."""
    dropped = []
    for name in STRUCTURAL_FEATURE_ORDER:
        vals = np.asarray(pooled_fit_raw[name], dtype=float)
        if vals.size < 2 or np.var(vals) == 0.0:
            dropped.append(name)
    return dropped


def fit_standardizer(
    pooled_fit_raw: Dict[str, List[float]], surviving: List[str]
) -> Dict[str, Tuple[float, float]]:
    stats_: Dict[str, Tuple[float, float]] = {}
    for name in surviving:
        vals = np.asarray(pooled_fit_raw[name], dtype=float)
        stats_[name] = (float(vals.mean()), float(vals.std(ddof=1)))
    return stats_


def apply_standardizer(
    raw: Dict[str, List[float]],
    surviving: List[str],
    stats_: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    if not surviving:
        n = len(next(iter(raw.values()))) if raw else 0
        return np.zeros((n, 0))
    cols = []
    for name in surviving:
        mean, sd = stats_[name]
        vals = np.asarray(raw[name], dtype=float)
        z = (vals - mean) / sd if sd > 0 else np.zeros_like(vals)
        cols.append(z)
    return np.column_stack(cols)


# ── section 4.2: lexical (TF-IDF + TruncatedSVD) block ──────────────────
def build_tfidf_vectorizers() -> Tuple[TfidfVectorizer, TfidfVectorizer]:
    """Parameter VALUES reused from src/cue_scoring.py::FROZEN_CUE_CONFIG
    ["tfidf_logreg"] -- the fitted LogisticRegression model itself is NOT
    reused (C-F-A section 1 / section 4.2)."""
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2,
        lowercase=True, sublinear_tf=True, max_features=20000,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        lowercase=True, sublinear_tf=True, max_features=20000,
    )
    return word_vec, char_vec


def fit_lexical_block(
    abcd_texts: List[str], svd_seed: int, n_components: int
) -> Tuple[TfidfVectorizer, TfidfVectorizer, TruncatedSVD, np.ndarray]:
    word_vec, char_vec = build_tfidf_vectorizers()
    word_mat = word_vec.fit_transform(abcd_texts)
    char_mat = char_vec.fit_transform(abcd_texts)
    raw_block = sparse.hstack([word_mat, char_mat]).tocsr()
    # C-F-A section 4.2: n_components = min(50, n_fit_rows - 1).
    n_comp = max(1, min(n_components, raw_block.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, algorithm="arpack", random_state=svd_seed)
    abcd_reduced = svd.fit_transform(raw_block)
    return word_vec, char_vec, svd, abcd_reduced


def transform_lexical_block(
    texts: List[str],
    word_vec: TfidfVectorizer,
    char_vec: TfidfVectorizer,
    svd: TruncatedSVD,
) -> np.ndarray:
    """Never fit_transform() -- transform() only (R104-source,
    R-AUTHORED)."""
    word_mat = word_vec.transform(texts)
    char_mat = char_vec.transform(texts)
    raw_block = sparse.hstack([word_mat, char_mat]).tocsr()
    return svd.transform(raw_block)


# ── section 4.3: a fresh, single common FightinWords fit ────────────────
def build_common_fightin_words(
    a_texts: List[str], b_texts: List[str], d_texts: List[str],
    prior_strength: float = FW_PRIOR_STRENGTH, min_count: int = FW_MIN_COUNT,
) -> FightinWords:
    """H = A u B, D = quadrant D -- reuses build_fw_from_eval's existing
    H/D convention, unmodified FightinWords class, but a genuinely new
    fit (C-F-A section 4.3: neither existing fightin_words_score_
    normalized column is comparable across R104/R-AUTHORED)."""
    fw = FightinWords(prior_strength=prior_strength, min_count=min_count)
    fw.fit(list(a_texts) + list(b_texts), list(d_texts))
    return fw


def score_fw_common(fw: FightinWords, texts: List[str]) -> List[Optional[float]]:
    """Returns fw_score_common_v1 per text (None iff the row has zero
    recognized vocabulary tokens -- see IMPLEMENTATION DECISION 3)."""
    return [fw.score(t)["fightin_words_score_normalized"] for t in texts]


# ── section 4.4: three representation views ─────────────────────────────
def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    if X.shape[1] == 0:
        return X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return X / safe


def build_all_views(
    populations: Dict[str, List[dict]],
    formatting_config: dict,
    svd_seed: int = SVD_SEED,
    svd_n_components: int = SVD_N_COMPONENTS,
    fw_prior_strength: float = FW_PRIOR_STRENGTH,
    fw_min_count: float = FW_MIN_COUNT,
) -> dict:
    """Fits every transformation ONCE on A u B u C u D; transforms
    R104-source and R-AUTHORED. Returns structural/lexical/combined view
    matrices per population plus full fit provenance (C-F-A sections
    4.1-4.5)."""
    abcd_rows = [row for name in QUADRANT_ORDER for row in populations[name]]
    abcd_texts = [r["text"] for r in abcd_rows]

    # -- structural block (4.1) --
    structural_raw = {
        name: compute_structural_raw(populations[name], formatting_config)
        for name in populations
    }
    n_empty_token_rows = {
        name: structural_raw[name].pop("_n_empty_token_rows")
        for name in populations
    }
    pooled_fit_raw = {
        feat: [v for name in QUADRANT_ORDER for v in structural_raw[name][feat]]
        for feat in STRUCTURAL_FEATURE_ORDER
    }
    dropped_structural = zero_variance_features(pooled_fit_raw)
    surviving_structural = [
        f for f in STRUCTURAL_FEATURE_ORDER if f not in dropped_structural
    ]
    standardizer = fit_standardizer(pooled_fit_raw, surviving_structural)
    structural_view = {
        name: apply_standardizer(structural_raw[name], surviving_structural, standardizer)
        for name in populations
    }

    # -- lexical block (4.2) --
    word_vec, char_vec, svd, abcd_lex_reduced = fit_lexical_block(
        abcd_texts, svd_seed, svd_n_components
    )
    lexical_reduced: Dict[str, np.ndarray] = {}
    idx = 0
    for name in QUADRANT_ORDER:
        n = len(populations[name])
        lexical_reduced[name] = abcd_lex_reduced[idx: idx + n]
        idx += n
    for name in AUXILIARY_ORDER:
        lexical_reduced[name] = transform_lexical_block(
            [r["text"] for r in populations[name]], word_vec, char_vec, svd
        )

    # -- FightinWords common fit (4.3) --
    fw = build_common_fightin_words(
        [r["text"] for r in populations["A"]],
        [r["text"] for r in populations["B"]],
        [r["text"] for r in populations["D"]],
        fw_prior_strength, fw_min_count,
    )
    fw_scores = {
        name: score_fw_common(fw, [r["text"] for r in populations[name]])
        for name in populations
    }
    abcd_fw_vals = [
        v for name in QUADRANT_ORDER for v in fw_scores[name] if v is not None
    ]
    fw_mean = float(np.mean(abcd_fw_vals)) if abcd_fw_vals else 0.0
    fw_sd = float(np.std(abcd_fw_vals, ddof=1)) if len(abcd_fw_vals) > 1 else 0.0

    # -- OOV / undefined-score exclusion (4.5, IMPLEMENTATION DECISION 3) --
    oov_mask: Dict[str, np.ndarray] = {}
    for name in populations:
        texts = [r["text"] for r in populations[name]]
        tok_mask = zero_token_mask(texts)
        fw_none_mask = np.array([v is None for v in fw_scores[name]], dtype=bool)
        oov_mask[name] = tok_mask | fw_none_mask

    lri_idx = (
        surviving_structural.index("lexical_risk_hit_count")
        if "lexical_risk_hit_count" in surviving_structural else None
    )
    lexical_view: Dict[str, np.ndarray] = {}
    for name in populations:
        n = len(populations[name])
        fw_z = np.array([
            (v - fw_mean) / fw_sd if (v is not None and fw_sd > 0) else 0.0
            for v in fw_scores[name]
        ], dtype=float).reshape(-1, 1)
        if lri_idx is not None:
            lri_z = structural_view[name][:, lri_idx].reshape(-1, 1)
        else:
            lri_z = np.zeros((n, 1))
        lexical_view[name] = np.hstack([lexical_reduced[name], fw_z, lri_z])

    combined_view: Dict[str, np.ndarray] = {}
    for name in populations:
        s_norm = l2_normalize_rows(structural_view[name])
        l_norm = l2_normalize_rows(lexical_view[name])
        combined_view[name] = np.hstack([np.sqrt(0.5) * s_norm, np.sqrt(0.5) * l_norm])

    lexical_view_retained = {
        name: lexical_view[name][~oov_mask[name]] for name in populations
    }
    combined_view_retained = {
        name: combined_view[name][~oov_mask[name]] for name in populations
    }

    return {
        "structural_raw": structural_raw,
        "n_empty_token_rows": n_empty_token_rows,
        "dropped_structural_features": dropped_structural,
        "surviving_structural_features": surviving_structural,
        "structural_standardizer": standardizer,
        "structural_view": structural_view,
        "lexical_view": lexical_view_retained,
        "combined_view": combined_view_retained,
        "oov_excluded_row_counts": {
            name: int(oov_mask[name].sum()) for name in populations
        },
        "oov_mask": oov_mask,
        "word_vec": word_vec,
        "char_vec": char_vec,
        "svd": svd,
        "fw": fw,
        "fw_common_provenance": {
            "h_populations": ["A", "B"],
            "d_population": "D",
            "prior_strength": fw_prior_strength,
            "min_count": fw_min_count,
            "n_h": len(populations["A"]) + len(populations["B"]),
            "n_d": len(populations["D"]),
            "corpus_h_sha256": fw.corpus_h_sha256_,
            "corpus_d_sha256": fw.corpus_d_sha256_,
            "output_field_name": "fw_score_common_v1",
        },
        "tfidf_svd_provenance": {
            "word_ngram_range": [1, 2], "word_min_df": 2,
            "char_ngram_range": [3, 5], "char_min_df": 2,
            "max_features_per_view": 20000, "sublinear_tf": True, "lowercase": True,
            "svd_n_components": int(svd.n_components),
            "svd_seed": svd_seed, "svd_algorithm": "arpack",
        },
    }


# ── section 5.1: centroid geometry ──────────────────────────────────────
def centroid(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def all_centroids(view: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Equal quadrant weighting: one centroid per population regardless
    of n (C-F-A section 5.1) -- never row-pool before centroiding."""
    return {name: centroid(X) for name, X in view.items()}


def pairwise_centroid_distances(centroids: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        pair: float(np.linalg.norm(centroids[pair[0]] - centroids[pair[1]]))
        for pair in PAIR_ORDER
    }


# ── section 5.2: factorial contrasts ────────────────────────────────────
def factorial_contrasts(centroids: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mu_a, mu_b, mu_c, mu_d = (centroids[q] for q in QUADRANT_ORDER)
    intent_contrast = (mu_a + mu_c) / 2 - (mu_b + mu_d) / 2
    surface_contrast = (mu_a + mu_b) / 2 - (mu_c + mu_d) / 2
    return intent_contrast, surface_contrast


def cosine_angle_degrees(u: np.ndarray, v: np.ndarray) -> Optional[float]:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return None
    cos = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


# ── section 5.3: PCA (visualization only) ───────────────────────────────
def fit_pca_combined(abcd_combined_rows: np.ndarray) -> PCA:
    n_components = int(min(abcd_combined_rows.shape[0], abcd_combined_rows.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(abcd_combined_rows)
    return pca


def cumulative_explained_variance(pca: PCA, k: int) -> float:
    k = min(k, len(pca.explained_variance_ratio_))
    return float(np.sum(pca.explained_variance_ratio_[:k]))


# ── section 5.4: energy distance + label-permutation test ──────────────
def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Szekely & Rizzo energy distance (V-statistic form):
    E(X,Y) = (2/nm) sum||xi-yj|| - (1/n^2) sum||xi-xj|| - (1/m^2) sum||yi-yj||
    """
    n, m = X.shape[0], Y.shape[0]
    dxy = cdist(X, Y, metric="euclidean")
    dxx = cdist(X, X, metric="euclidean")
    dyy = cdist(Y, Y, metric="euclidean")
    return float(2.0 * dxy.mean() - dxx.sum() / (n * n) - dyy.sum() / (m * m))


def _energy_stat_from_full_matrix(full_D: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray) -> float:
    n, m = len(idx_x), len(idx_y)
    dxy = full_D[np.ix_(idx_x, idx_y)]
    dxx = full_D[np.ix_(idx_x, idx_x)]
    dyy = full_D[np.ix_(idx_y, idx_y)]
    return float(2.0 * dxy.mean() - dxx.sum() / (n * n) - dyy.sum() / (m * m))


def rng_for_pair(base_seed: int, pair_name: str) -> np.random.Generator:
    """C-F-A section 5.4: one independent permutation stream per pair,
    not one shared stream reused across all six pairs -- deterministic
    from (base_seed, pair_index) via SeedSequence's multi-entropy input."""
    pair_index = PAIR_ORDER.index(pair_name)
    return np.random.default_rng([int(base_seed), int(pair_index)])


def permutation_test_energy_distance(
    X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, n_permutations: int,
) -> Tuple[float, float]:
    """Unpaired, row-level, two-sided label-permutation test (C-F-A
    section 5.4): pool rows, repeatedly reassign to the two original
    group sizes, recompute the statistic, two-sided empirical p-value =
    fraction of permuted statistics >= observed."""
    n, m = X.shape[0], Y.shape[0]
    pooled = np.vstack([X, Y])
    full_D = cdist(pooled, pooled, metric="euclidean")
    idx_all = np.arange(n + m)
    observed = _energy_stat_from_full_matrix(full_D, idx_all[:n], idx_all[n:])
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n + m)
        stat = _energy_stat_from_full_matrix(full_D, perm[:n], perm[n:])
        if stat >= observed:
            count_ge += 1
    p_value = count_ge / n_permutations if n_permutations > 0 else None
    return observed, p_value


def holm_bonferroni(
    p_values_named: Dict[str, Optional[float]], alpha: float = HOLM_ALPHA
) -> Dict[str, dict]:
    """Standard Holm step-down procedure, applied SEPARATELY within each
    view's family of 6 pairwise tests (C-F-A section 5.4 -- three
    independent families, not one pooled family of 18)."""
    items = sorted(
        ((k, v) for k, v in p_values_named.items() if v is not None),
        key=lambda kv: kv[1],
    )
    m = len(items)
    results: Dict[str, dict] = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(max((m - i) * p, running_max), 1.0)
        running_max = adj
        results[name] = {
            "raw_p": p, "rank": i + 1, "adjusted_p_holm": adj,
            "reject_at_alpha": adj <= alpha,
        }
    for name, p in p_values_named.items():
        if p is None:
            results[name] = {
                "raw_p": None, "rank": None, "adjusted_p_holm": None,
                "reject_at_alpha": None,
            }
    return results


# ── section 5.5: token / n-gram JSD ─────────────────────────────────────
def get_bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def build_ngram_vocab(pooled_texts: Sequence[str], n: int, min_df: int = JSD_MIN_DF) -> set:
    """Common vocabulary from the pooled A/B/C/D corpus only, retaining a
    unigram/bigram only if it appears in >= min_df documents (C-F-A
    section 5.5 -- reuses FROZEN_CUE_CONFIG's min_df=2 convention)."""
    doc_freq: Counter = Counter()
    for t in pooled_texts:
        tokens = word_tokenize(t)
        grams = tokens if n == 1 else get_bigrams(tokens)
        for g in set(grams):
            doc_freq[g] += 1
    return {g for g, c in doc_freq.items() if c >= min_df}


def ngram_distribution(
    texts: Sequence[str], vocab: set, n: int, alpha: float = JSD_ALPHA
) -> Tuple[np.ndarray, List[str]]:
    """Fixed support = sorted(vocab) + <RARE>. Out-of-vocabulary
    tokens/bigrams for a given population map to the single shared
    <RARE> bucket (never dropped), then additive smoothing is applied to
    every population's count distribution -- including the <RARE>
    bucket -- before normalizing to sum to 1 (C-F-A section 5.5)."""
    support = sorted(vocab) + [RARE_TOKEN]
    index = {g: i for i, g in enumerate(support)}
    counts = np.zeros(len(support), dtype=float)
    for t in texts:
        tokens = word_tokenize(t)
        grams = tokens if n == 1 else get_bigrams(tokens)
        for g in grams:
            counts[index.get(g, index[RARE_TOKEN])] += 1
    counts += alpha
    counts /= counts.sum()
    return counts, support


def jsd_base2(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base-2, bounded in [0, 1].
    scipy.spatial.distance.jensenshannon returns the SQUARE ROOT of the
    divergence (a metric); squaring it recovers the divergence itself."""
    return float(jensenshannon(p, q, base=2) ** 2)


# ── section 5.6: confound sensitivity ───────────────────────────────────
def residualize_on_covariate(X: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Per-column OLS residuals of X on `covariate` (plus intercept),
    fit and applied together (no separate fit/apply split is specified
    for this diagnostic -- see IMPLEMENTATION DECISION 2)."""
    if X.shape[1] == 0:
        return X
    c = np.asarray(covariate, dtype=float).reshape(-1, 1)
    design = np.hstack([np.ones_like(c), c])
    coeffs, *_ = np.linalg.lstsq(design, X, rcond=None)
    fitted = design @ coeffs
    return X - fitted


def median_split_mask(pooled_word_counts: np.ndarray) -> Tuple[float, np.ndarray]:
    median = float(np.median(pooled_word_counts))
    return median, pooled_word_counts <= median


# ── section 7: optional embeddings check ────────────────────────────────
def optional_embeddings_check(run_embeddings: bool) -> dict:
    """Secondary, non-blocking robustness check (C-F-A section 7). Not
    run in this sandbox (no network access to the Hugging Face Hub, and
    not requested by default) -- reported as "not_run" rather than
    silently omitted, per the same fail-closed disclosure convention
    already used for R104's untested near-duplicate check (C-A section
    2)."""
    if not run_embeddings:
        return {
            "status": "not_run",
            "reason": "optional and non-blocking (section 7); not requested "
            "(--run-embeddings not set)",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        }
    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        return {
            "status": "not_run",
            "reason": f"sentence-transformers unavailable in this environment: {exc}",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        }
    return {
        "status": "not_run",
        "reason": "network access to the Hugging Face Hub is unavailable in "
        "this sandbox; the model cannot be downloaded (same constraint "
        "already logged in C-A section 2 / logs/agent_state.json)",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    }


# ── output assembly ──────────────────────────────────────────────────────
def build_markdown(analysis: dict) -> str:
    lines = [
        "# C-F -- Joint-Geometry Analysis (A/B/C/D + R104-source, R-AUTHORED)",
        "",
        "Status: implementation of the locked contract in "
        "`logs/cf_joint_geometry_spec.md` (C-F-A). Descriptive/geometric "
        "reporting only -- no construct-validity decision is made here "
        "(C-F-A section 0 / section 5.2's explicit non-inference rule).",
        "",
        f"Generation commit: `{analysis['code_version'].get('generation_commit')}`",
        "",
        "## Population counts",
        "",
        "| Population | n |",
        "|---|---|",
    ]
    for name in ALL_POPULATIONS_ORDER:
        lines.append(f"| {name} | {analysis['population_counts'].get(name)} |")
    lines += [
        "",
        "## Dropped / surviving structural features",
        "",
        f"Dropped (zero-variance on A u B u C u D): "
        f"{analysis['views']['dropped_structural_features']}",
        f"Surviving: {analysis['views']['surviving_structural_features']}",
        "",
        "## Centroid pairwise distances by view",
        "",
    ]
    for view_name, view_result in analysis["results_by_view"].items():
        lines.append(f"### {view_name}")
        lines.append("")
        for pair, dist in view_result["pairwise_centroid_distances"].items():
            lines.append(f"- {pair}: {dist:.4f}")
        angle = view_result["factorial_contrast_cosine_angle_degrees"]
        lines.append(
            f"- intent/surface contrast angle: "
            f"{angle:.2f} deg (geometric fact about this fit only; "
            "not evidence for/against latent independence)"
            if angle is not None else
            "- intent/surface contrast angle: undefined (a zero-norm contrast vector)"
        )
        lines.append("")
    lines += [
        "## R-AUTHORED caveats (repeated per C-F-A section 6)",
        "",
        "- review_status: pending (C-E)",
        "- Q25 rank-selection bias (C-D Gate 4)",
        "",
        "## Embeddings (section 7)",
        "",
        f"Status: {analysis['embeddings']['status']} -- "
        f"{analysis['embeddings']['reason']}",
        "",
        "## Explicit non-actions",
        "",
    ]
    for item in analysis["explicit_non_actions"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-latest", default=DEFAULT_BENCHMARK_LATEST)
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--r-authored-csv", default=DEFAULT_R_AUTHORED_CSV)
    parser.add_argument("--gate-config", default=DEFAULT_GATE_CONFIG)
    parser.add_argument("--formatting-config-source", default=DEFAULT_FORMATTING_CONFIG_SOURCE)
    parser.add_argument("--svd-seed", type=int, default=SVD_SEED)
    parser.add_argument("--permutation-seed", type=int, default=PERMUTATION_SEED)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS_DEFAULT)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument(
        "--run-embeddings", action="store_true", default=False,
        help="Attempt the optional section-7 embeddings robustness check "
        "(non-blocking; not part of the exact reproduction command).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    benchmark_latest_path = REPO_ROOT / args.benchmark_latest
    review_csv_path = REPO_ROOT / args.review_csv
    r_authored_csv_path = REPO_ROOT / args.r_authored_csv
    gate_config_path = REPO_ROOT / args.gate_config
    formatting_config_source_path = REPO_ROOT / args.formatting_config_source
    out_md_path = REPO_ROOT / args.out_md
    out_json_path = REPO_ROOT / args.out_json

    # 1. Fail-closed hash re-verification (C-F-A section 1).
    verified_hashes = verify_pinned_hashes()

    # 2. Load benchmark ONLY via v2_io.resolve_benchmark (never by
    #    filename), per section 1's explicit instruction.
    benchmark_path, benchmark_sha = resolve_benchmark(latest_path=benchmark_latest_path)
    pinned_benchmark_sha = PINNED_INPUT_HASHES[
        "data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl"
    ]
    if benchmark_sha != pinned_benchmark_sha:
        raise AuditFailClosed(
            "FAIL CLOSED: frozen benchmark hash does not match the C-F-A "
            f"section 1 pinned value (expected {pinned_benchmark_sha}, "
            f"got {benchmark_sha})."
        )
    benchmark_rows = load_jsonl(benchmark_path)

    # 3. Populations (section 2), fail closed on size mismatch.
    quadrants = load_frozen_quadrants(benchmark_rows)
    r104_source = load_r104_source_population(review_csv_path)
    r_authored, r_authored_precheck = load_r_authored_population(r_authored_csv_path)
    populations = {**quadrants, "R104_source": r104_source, "R_AUTHORED": r_authored}
    verify_population_counts(populations)

    gate_config = load_json(gate_config_path)
    formatting_source = load_json(formatting_config_source_path)
    formatting_config = formatting_source["confound_diagnostics"]["formatting_diagnostic_config"]

    # 4. Build all three views (sections 4.1-4.5).
    views = build_all_views(
        populations, formatting_config,
        svd_seed=args.svd_seed, svd_n_components=SVD_N_COMPONENTS,
    )

    results_by_view = {}
    for view_name in ("structural_only", "lexical_only", "combined"):
        view_matrices = views[
            "structural_view" if view_name == "structural_only"
            else "lexical_view" if view_name == "lexical_only"
            else "combined_view"
        ]
        cents = all_centroids({q: view_matrices[q] for q in QUADRANT_ORDER})
        aux_cents = {name: centroid(view_matrices[name]) for name in AUXILIARY_ORDER}
        pairwise = pairwise_centroid_distances(cents)
        intent_c, surface_c = factorial_contrasts(cents)
        angle = cosine_angle_degrees(intent_c, surface_c)

        p_values = {}
        observed_stats = {}
        for pair in PAIR_ORDER:
            X, Y = view_matrices[pair[0]], view_matrices[pair[1]]
            rng = rng_for_pair(args.permutation_seed, pair)
            obs, p = permutation_test_energy_distance(X, Y, rng, args.n_permutations)
            observed_stats[pair] = obs
            p_values[pair] = p
        corrected = holm_bonferroni(p_values)

        results_by_view[view_name] = {
            "centroids_abcd": {q: cents[q].tolist() for q in QUADRANT_ORDER},
            "centroids_auxiliary": {n: aux_cents[n].tolist() for n in AUXILIARY_ORDER},
            "pairwise_centroid_distances": pairwise,
            "intent_contrast": intent_c.tolist(),
            "surface_contrast": surface_c.tolist(),
            "factorial_contrast_cosine_angle_degrees": angle,
            "energy_distance_observed": observed_stats,
            "energy_distance_permutation_p_values": p_values,
            "energy_distance_holm_bonferroni": corrected,
        }

    # 5. PCA on the combined view (section 5.3).
    abcd_combined = np.vstack([views["combined_view"][q] for q in QUADRANT_ORDER])
    pca = fit_pca_combined(abcd_combined)
    pca_result = {
        "cumulative_explained_variance_pc1_pc2": cumulative_explained_variance(pca, 2),
        "cumulative_explained_variance_pc1_pc3": cumulative_explained_variance(pca, 3),
        "note": "PCA is reporting/visualization only, fit on A u B u C u D "
        "rows; R104-source/R-AUTHORED are projected, never fit "
        "(section 5.3).",
    }

    # 6. Token/n-gram JSD (section 5.5).
    abcd_texts_pooled = [r["text"] for q in QUADRANT_ORDER for r in populations[q]]
    jsd_result = {}
    for n, scope in ((1, "unigram"), (2, "bigram")):
        vocab = build_ngram_vocab(abcd_texts_pooled, n)
        dists = {
            q: ngram_distribution([r["text"] for r in populations[q]], vocab, n)[0]
            for q in QUADRANT_ORDER
        }
        primary = {
            pair: jsd_base2(dists[pair[0]], dists[pair[1]]) for pair in PAIR_ORDER
        }
        r104_dist = ngram_distribution([r["text"] for r in populations["R104_source"]], vocab, n)[0]
        r_auth_dist = ngram_distribution([r["text"] for r in populations["R_AUTHORED"]], vocab, n)[0]
        auxiliary = {
            "R104_source_vs_C": jsd_base2(r104_dist, dists["C"]),
            **{f"R_AUTHORED_vs_{q}": jsd_base2(r_auth_dist, dists[q]) for q in QUADRANT_ORDER},
        }
        jsd_result[scope] = {"primary_abcd_pairs": primary, "auxiliary_non_blocking": auxiliary}

    # 7. Confound sensitivity (section 5.6).
    confound_result = _confound_sensitivity(populations, views)

    # 8. Optional embeddings check (section 7).
    embeddings_result = optional_embeddings_check(args.run_embeddings)

    analysis = {
        "task": "C-F -- Joint-geometry analysis for A/B/C/D (+ R104-source, R-AUTHORED)",
        "spec_reference": "logs/cf_joint_geometry_spec.md",
        "code_version": get_code_version(),
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "config": {
            "benchmark_latest": _display_path(benchmark_latest_path),
            "review_csv": _display_path(review_csv_path),
            "r_authored_csv": _display_path(r_authored_csv_path),
            "gate_config": _display_path(gate_config_path),
            "formatting_config_source": _display_path(formatting_config_source_path),
            "svd_seed": args.svd_seed,
            "svd_n_components": SVD_N_COMPONENTS,
            "permutation_seed": args.permutation_seed,
            "n_permutations": args.n_permutations,
        },
        "pinned_input_hashes_verified": verified_hashes,
        "population_counts": {name: len(rows) for name, rows in populations.items()},
        "r_authored_precheck": r_authored_precheck,
        "views": {
            "dropped_structural_features": views["dropped_structural_features"],
            "surviving_structural_features": views["surviving_structural_features"],
            "structural_standardizer": views["structural_standardizer"],
            "n_empty_token_rows": views["n_empty_token_rows"],
            "oov_excluded_row_counts": views["oov_excluded_row_counts"],
            "fw_common_provenance": views["fw_common_provenance"],
            "tfidf_svd_provenance": views["tfidf_svd_provenance"],
        },
        "results_by_view": results_by_view,
        "pca": pca_result,
        "token_ngram_jsd": jsd_result,
        "confound_sensitivity": confound_result,
        "embeddings": embeddings_result,
        "r_authored_caveats": [
            "review_status: pending (C-E)",
            "Q25 rank-selection bias, not a representative sample of all "
            "source-authored candidates (C-D Gate 4)",
        ],
        "explicit_non_actions": [
            "did not modify any C-F-A section 1 pinned input, candidate "
            "wording, or frozen benchmark file",
            "did not redefine any feature or add a metric beyond section "
            "5's five required analysis blocks",
            "did not add alternative preprocessing after inspecting results",
            "did not train a supervised classifier to define an axis, and "
            "did not run contrastive/representation-learning training",
            "did not run GPU inference; section-7 embeddings reported as "
            "not_run unless --run-embeddings is set, and even then no "
            "network access is available in this sandbox",
            "did not characterize any PCA axis, distance, or divergence as "
            "evidence for or against latent psychological independence of "
            "intent and surface cue (section 5.2)",
            "did not decide whether C is valid, whether the intended 2x2 "
            "construct holds, or whether R-AUTHORED should be promoted",
            "did not print raw prompt text, matched lexicon terms, or "
            "per-token/per-bigram JSD contributions",
        ],
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.write_text(build_markdown(analysis), encoding="utf-8")

    print(f"analysis_json={_display_path(out_json_path)}")
    print(f"analysis_md={_display_path(out_md_path)}")
    print(f"generation_commit={analysis['code_version'].get('generation_commit')}")
    for name in ALL_POPULATIONS_ORDER:
        print(f"{name}: n={analysis['population_counts'][name]}")

    return analysis


def _confound_sensitivity(populations: Dict[str, List[dict]], views: dict) -> dict:
    """C-F-A section 5.6: length, source, category, formatting."""
    result: dict = {}

    # -- 1. length: residualize on word_count, recompute centroids/
    #    distances on the residualized structural-only and combined
    #    views (IMPLEMENTATION DECISION 2); plus a median-split check. --
    word_counts_by_pop = {
        q: np.array([r["word_count"] for r in populations[q]], dtype=float)
        for q in QUADRANT_ORDER
    }
    length_result = {}
    for view_name in ("structural_only", "combined"):
        view_matrices = views["structural_view"] if view_name == "structural_only" else views["combined_view"]
        residualized = {
            q: residualize_on_covariate(view_matrices[q], word_counts_by_pop[q])
            for q in QUADRANT_ORDER
        }
        cents = all_centroids(residualized)
        pairwise_resid = pairwise_centroid_distances(cents)
        pairwise_orig = pairwise_centroid_distances(all_centroids({q: view_matrices[q] for q in QUADRANT_ORDER}))
        length_result[view_name] = {
            "pairwise_centroid_distances_residualized": pairwise_resid,
            "pairwise_centroid_distances_original": pairwise_orig,
            "attenuation": {
                pair: pairwise_orig[pair] - pairwise_resid[pair] for pair in PAIR_ORDER
            },
        }
    pooled_word_counts = np.concatenate([word_counts_by_pop[q] for q in QUADRANT_ORDER])
    median, _ = median_split_mask(pooled_word_counts)
    median_split_result = {"pooled_word_count_median": median, "by_view": {}}
    for view_name in ("structural_only", "combined"):
        view_matrices = views["structural_view"] if view_name == "structural_only" else views["combined_view"]
        below, above = {}, {}
        for q in QUADRANT_ORDER:
            mask = word_counts_by_pop[q] <= median
            below[q] = view_matrices[q][mask] if mask.any() else None
            above[q] = view_matrices[q][~mask] if (~mask).any() else None
        below_pairwise = {
            pair: (
                float(np.linalg.norm(centroid(below[pair[0]]) - centroid(below[pair[1]])))
                if below[pair[0]] is not None and below[pair[1]] is not None else None
            )
            for pair in PAIR_ORDER
        }
        above_pairwise = {
            pair: (
                float(np.linalg.norm(centroid(above[pair[0]]) - centroid(above[pair[1]])))
                if above[pair[0]] is not None and above[pair[1]] is not None else None
            )
            for pair in PAIR_ORDER
        }
        median_split_result["by_view"][view_name] = {
            "below_or_at_median": below_pairwise, "above_median": above_pairwise,
        }
    result["length"] = {"residualized": length_result, "median_split": median_split_result}

    # -- 2. source: only D supports a within-quadrant source contrast. --
    d_rows = populations["D"]
    source_groups: Dict[str, List[int]] = {}
    for i, r in enumerate(d_rows):
        source_groups.setdefault(r.get("source_dataset"), []).append(i)
    source_result = {"applicable_quadrants": ["D"], "not_applicable": ["A", "B", "C"], "D": {}}
    for view_name, view_matrices in (
        ("structural_only", views["structural_view"]),
        ("lexical_only", views["lexical_view"]),
        ("combined", views["combined_view"]),
    ):
        d_matrix = view_matrices["D"]
        group_names = sorted(source_groups.keys())
        group_cents = {
            g: centroid(d_matrix[source_groups[g]]) for g in group_names
            if len(source_groups[g]) > 0
        }
        pairs = {}
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                if g1 in group_cents and g2 in group_cents:
                    pairs[f"{g1}_vs_{g2}"] = float(np.linalg.norm(group_cents[g1] - group_cents[g2]))
        source_result["D"][view_name] = pairs
    source_result["note"] = (
        "source_dataset is perfectly confounded with quadrant membership "
        "for A, B, and C (each 100% one upstream dataset) -- this cannot "
        "be checked away and is a standing limitation on any A/B/C/D "
        "geometry finding, not resolved here (section 3 / section 5.6 item 2)."
    )
    result["source"] = source_result

    # -- 3. category: only A-vs-C within-category comparison computable. --
    a_rows, c_rows = populations["A"], populations["C"]
    a_cats = {r.get("project_category") for r in a_rows}
    c_cats = {r.get("project_category") for r in c_rows}
    shared_cats = sorted(a_cats & c_cats)
    category_result = {
        "applicable_pair": "A_vs_C", "not_applicable": ["B", "D"],
        "shared_categories": shared_cats, "by_view": {},
    }
    for view_name, view_matrices in (
        ("structural_only", views["structural_view"]),
        ("lexical_only", views["lexical_view"]),
        ("combined", views["combined_view"]),
    ):
        per_cat = {}
        for cat in shared_cats:
            a_idx = [i for i, r in enumerate(a_rows) if r.get("project_category") == cat]
            c_idx = [i for i, r in enumerate(c_rows) if r.get("project_category") == cat]
            if a_idx and c_idx:
                dist = float(np.linalg.norm(
                    centroid(view_matrices["A"][a_idx]) - centroid(view_matrices["C"][c_idx])
                ))
                per_cat[cat] = {"n_A": len(a_idx), "n_C": len(c_idx), "distance": dist}
        category_result["by_view"][view_name] = per_cat
    category_result["note"] = (
        "B's project_category taxonomy is disjoint from A/C's (10 levels "
        "vs. 4) and D has no category values at all -- reported as not "
        "applicable per section 3, never silently omitted."
    )
    result["category"] = category_result

    # -- 4. formatting: re-check variance on the full A u B u C u D pool. --
    formatting_features = ["has_bullet_marker", "has_numbered_step", "has_code_block"]
    surviving = set(views["surviving_structural_features"])
    dropped = set(views["dropped_structural_features"])
    formatting_result = {}
    for feat in formatting_features:
        if feat in dropped:
            formatting_result[feat] = {"status": "dropped", "reason": "zero-variance on full A u B u C u D pool"}
        else:
            idx = views["surviving_structural_features"].index(feat)
            contributions = {}
            for pair in PAIR_ORDER:
                mu_i = views["structural_view"][pair[0]][:, idx].mean()
                mu_j = views["structural_view"][pair[1]][:, idx].mean()
                per_feature_sq = (mu_i - mu_j) ** 2
                total_sq = pairwise_centroid_distances(all_centroids(
                    {q: views["structural_view"][q] for q in QUADRANT_ORDER}
                ))[pair] ** 2
                contributions[pair] = (
                    per_feature_sq / total_sq if total_sq > 0 else None
                )
            formatting_result[feat] = {
                "status": "survived",
                "fractional_contribution_to_structural_only_squared_distance": contributions,
            }
    result["formatting"] = formatting_result

    return result


if __name__ == "__main__":
    main()
