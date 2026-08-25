"""
Corpus-discrimination diagnostics for the v2 benchmark pipeline.

Implements Fightin' Words (Monroe et al. 2008) as the primary lexical
diagnostic. H = A ∪ B (operational high-cue reference); D = benign corpus.

IMPORTANT: H and D differ in source, topic, domain, category, length,
register, and prompt function. These diagnostics do NOT identify a pure
latent "surface-risk" variable. If source and label are nearly perfectly
aligned, source_cue_effect_status = "not_identified".

All parameters are predeclared here before any candidate scores are
inspected.
"""

import json
import math
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Tokenizer ──────────────────────────────────────────────────────────────────
def word_tokenize(text: str) -> List[str]:
    """
    Consistent word tokenizer used for ALL lexical scores in this module.
    Lowercases, splits on whitespace and punctuation boundaries, drops
    purely-punctuation tokens. Version is recorded in every scored row.
    """
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    return tokens


TOKENIZER_VERSION = "word_tokenize_v1_lower_alphanum_apostrophe"


# ── Fightin' Words (Monroe et al. 2008) ────────────────────────────────────────
class FightinWords:
    """
    Implements Fightin' Words log-odds ratio with an informative Dirichlet prior.

    H = A ∪ B (high-cue corpus)
    D = benign reference corpus

    Positive δ_w → word favors H over D.
    Scores are operational screening tools, not semantic truth.

    Parameters are predeclared and must not be changed after scoring any
    real candidate batch.
    """

    def __init__(self, prior_strength: float = 0.01, min_count: int = 1):
        self.prior_strength = prior_strength   # α_0 / |V| per token (uniform)
        self.min_count = min_count
        self.is_fitted = False
        # Set after fit()
        self.vocab_: Optional[List[str]] = None
        self.n_H_: int = 0
        self.n_D_: int = 0
        self.counts_H_: Optional[Counter] = None
        self.counts_D_: Optional[Counter] = None
        self.corpus_h_sha256_: Optional[str] = None
        self.corpus_d_sha256_: Optional[str] = None

    # ── fit ────────────────────────────────────────────────────────────────────
    def fit(self, h_texts: List[str], d_texts: List[str]) -> "FightinWords":
        """Fit on H and D corpora. Records corpus hashes for provenance."""
        self.counts_H_ = Counter()
        self.counts_D_ = Counter()
        for t in h_texts:
            self.counts_H_.update(word_tokenize(t))
        for t in d_texts:
            self.counts_D_.update(word_tokenize(t))
        self.n_H_ = sum(self.counts_H_.values())
        self.n_D_ = sum(self.counts_D_.values())
        # Build vocab: tokens that appear ≥ min_count in H OR D
        all_tokens = set(self.counts_H_.keys()) | set(self.counts_D_.keys())
        self.vocab_ = sorted(
            w for w in all_tokens
            if self.counts_H_.get(w, 0) + self.counts_D_.get(w, 0) >= self.min_count
        )
        # Corpus hashes for provenance
        h_joined = "\n".join(sorted(h_texts))
        d_joined = "\n".join(sorted(d_texts))
        self.corpus_h_sha256_ = hashlib.sha256(h_joined.encode()).hexdigest()
        self.corpus_d_sha256_ = hashlib.sha256(d_joined.encode()).hexdigest()
        self.is_fitted = True
        return self

    # ── delta and z for one token ──────────────────────────────────────────────
    def _delta(self, w: str) -> Tuple[float, float]:
        """Returns (delta_w, z_w). z_w = 0 if counts are zero."""
        alpha_w = self.prior_strength
        alpha_0 = self.prior_strength * len(self.vocab_)
        n_wH = self.counts_H_.get(w, 0)
        n_wD = self.counts_D_.get(w, 0)
        N_H, N_D = self.n_H_, self.n_D_
        delta = (
            math.log((n_wH + alpha_w) / (N_H + alpha_0 - n_wH - alpha_w + 2 * alpha_w))
            - math.log((n_wD + alpha_w) / (N_D + alpha_0 - n_wD - alpha_w + 2 * alpha_w))
        )
        denom = math.sqrt(
            1.0 / (n_wH + alpha_w) + 1.0 / (n_wD + alpha_w)
        )
        z = delta / denom if denom > 0 else 0.0
        return delta, z

    # ── score one prompt ────────────────────────────────────────────────────────
    def score(self, text: str, min_token_recognition_fraction: float = 0.50) -> Dict:
        """
        Returns a dict of all required scoring fields for one prompt.
        Records low_coverage_flag if token_recognition_fraction < threshold.
        """
        assert self.is_fitted, "Call fit() first."
        vocab_set = set(self.vocab_)
        tokens = word_tokenize(text)
        n_total = len(tokens)
        recognized = [t for t in tokens if t in vocab_set]
        n_rec = len(recognized)
        rec_frac = n_rec / n_total if n_total > 0 else 0.0
        low_cov = rec_frac < min_token_recognition_fraction or n_rec == 0

        unnorm = sum(self._delta(w)[0] for w in recognized)
        z_sum = sum(self._delta(w)[1] for w in recognized)
        norm = (unnorm / n_rec) if n_rec > 0 else None

        return {
            "fightin_words_score_unnormalized": round(unnorm, 6),
            "fightin_words_score_normalized": round(norm, 6) if norm is not None else None,
            "fightin_words_z_score_aggregate": round(z_sum, 6),
            "tokens_recognized_count": n_rec,
            "tokens_total_count": n_total,
            "token_recognition_fraction": round(rec_frac, 4),
            "low_coverage_flag": low_cov,
            "tokenizer_version": TOKENIZER_VERSION,
            "prior_parameters": {
                "prior_strength_per_token": self.prior_strength,
                "min_count": self.min_count,
                "vocab_size": len(self.vocab_),
            },
        }

    # ── score a paired (source, candidate) ────────────────────────────────────
    def score_pair(
        self, source_text: str, candidate_text: str,
        min_token_recognition_fraction: float = 0.50
    ) -> Dict:
        """
        Scores both source and candidate; adds paired_score_difference.
        Positive paired_score_difference means candidate has lower H-similarity
        than source (desired for C-paired: candidate is more D-like).
        """
        src = self.score(source_text, min_token_recognition_fraction)
        cand = self.score(candidate_text, min_token_recognition_fraction)
        src_u = src["fightin_words_score_unnormalized"]
        cand_u = cand["fightin_words_score_unnormalized"]
        diff = round(src_u - cand_u, 6)   # positive ↔ candidate is lower
        return {
            "source": src,
            "candidate": cand,
            "paired_score_difference": diff,
        }

    def prior_config(self) -> Dict:
        return {
            "prior_strength_per_token": self.prior_strength,
            "min_count": self.min_count,
            "vocab_size": len(self.vocab_) if self.is_fitted else None,
        }


# ── Helper: load eval-set texts by quadrant ───────────────────────────────────
def load_quadrant_texts(eval_path: str, quadrant: str) -> List[str]:
    texts = []
    with open(eval_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("quadrant") == quadrant:
                texts.append(row["prompt"])
    return texts


# ── Build and fit a FightinWords from the live eval set ───────────────────────
def build_fw_from_eval(eval_path: str, prior_strength: float = 0.01) -> FightinWords:
    """
    H = A ∪ B from the eval set.
    D = quadrant D from the eval set (benign reference corpus).
    """
    a_texts = load_quadrant_texts(eval_path, "A")
    b_texts = load_quadrant_texts(eval_path, "B")
    d_texts = load_quadrant_texts(eval_path, "D")
    h_texts = a_texts + b_texts
    fw = FightinWords(prior_strength=prior_strength)
    fw.fit(h_texts, d_texts)
    return fw


# ── Empirical rank within a corpus ────────────────────────────────────────────
def empirical_rank(score: float, reference_scores: List[float]) -> float:
    """
    Fraction of reference scores that are <= this score.
    Lower rank = more D-like (lower H-similarity) = more desirable for C-source-authored.
    """
    if not reference_scores:
        return None
    return sum(1 for s in reference_scores if s <= score) / len(reference_scores)


def assign_strata(rank: Optional[float]) -> Dict:
    """Map global_rank to Q10/Q25/Q40 membership flags."""
    if rank is None:
        return {"in_Q10": None, "in_Q25": None, "in_Q40": None}
    return {
        "in_Q10": rank <= 0.10,
        "in_Q25": rank <= 0.25,
        "in_Q40": rank <= 0.40,
    }
