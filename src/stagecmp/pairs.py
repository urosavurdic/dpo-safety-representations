"""The frozen pair and contrast vocabulary for stage-vs-stage comparison.

Naming follows the same discipline as ``src/crossbranch/branches.py``: a
transition is named by what varies, never by what we hope it shows.

The 2x2 at the centre of this study crosses TRAINING OBJECTIVE against
INSTRUCTION CORPUS, holding initialisation and source data fixed:

                   Alpaca                  Dolly-15k
    safety SFT     M1     -> M2            M1_alt -> M2_alt
    DPO            M1     -> M3_direct     M1_alt -> M3_direct_alt

Both arms of each column start from the SAME checkpoint and consume the SAME
PKU-SafeRLHF material. They are NOT matched on training signal: DPO sees chosen
and rejected responses while SFT sees only the chosen one, and the recipes
differ in epochs and learning rate. The honest name for the contrast is
therefore an objective-and-recipe contrast at matched initialisation and matched
source corpus -- never "the DPO objective in isolation". ``CONTRAST_CAVEAT``
below is written into every artifact that reports it.
"""
from __future__ import annotations

from itertools import combinations, permutations

from src.common.stages import ALL_STAGES

# --- the 2x2 -------------------------------------------------------------- #

OBJECTIVE_SFT = "safety_sft"
OBJECTIVE_DPO = "dpo"
CORPUS_ALPACA = "alpaca"
CORPUS_DOLLY = "dolly"

#: transition key -> (pre, post, objective, corpus)
TRANSITIONS_2X2 = {
    "sft_alpaca": ("M1", "M2", OBJECTIVE_SFT, CORPUS_ALPACA),
    "dpo_alpaca": ("M1", "M3_direct", OBJECTIVE_DPO, CORPUS_ALPACA),
    "sft_dolly": ("M1_alt", "M2_alt", OBJECTIVE_SFT, CORPUS_DOLLY),
    "dpo_dolly": ("M1_alt", "M3_direct_alt", OBJECTIVE_DPO, CORPUS_DOLLY),
}

#: the fixed axis each arm is measured against, so both post-checkpoints in a
#: column are read with ONE ruler. A stage-specific axis makes the contrast
#: meaningless, because each post-stage would be measured with its own.
FIXED_REFERENCE = {CORPUS_ALPACA: "M1", CORPUS_DOLLY: "M1_alt"}

CONTRAST_CAVEAT = (
    "Objective-and-recipe contrast at matched initialisation and matched source "
    "corpus. M1->M2 and M1->M3_direct share the start checkpoint and the "
    "PKU-SafeRLHF material but differ in training signal (DPO consumes chosen "
    "AND rejected responses, SFT only the chosen one), in epochs (2 vs 1) and "
    "in learning rate (2e-4 vs 5e-5). Not an isolation of the DPO objective. "
    "n=2 corpora, one instantiation each: intervals on the corpus and "
    "interaction terms are over PROMPTS, not over corpora."
)

# --- other named transitions we care about -------------------------------- #

#: the mediated path, for completeness beside the direct one
TRANSITIONS_MEDIATED = {
    "mediated_alpaca": ("M2", "M3", OBJECTIVE_DPO, CORPUS_ALPACA),
    "mediated_dolly": ("M2_alt", "M3_alt", OBJECTIVE_DPO, CORPUS_DOLLY),
}

#: instruction tuning itself
TRANSITIONS_INSTRUCT = {
    "instruct_alpaca": ("M0", "M1", "instruction_sft", CORPUS_ALPACA),
    "instruct_dolly": ("M0", "M1_alt", "instruction_sft", CORPUS_DOLLY),
}

#: pairs that get bootstrap confidence intervals rather than point estimates
#: only. Eight, named in advance, so the full matrix does not turn into 36
#: uncorrected intervals per metric.
HEADLINE_PAIRS = (
    ("M1", "M2"),
    ("M1", "M3_direct"),
    ("M1_alt", "M2_alt"),
    ("M1_alt", "M3_direct_alt"),
    ("M2", "M3"),
    ("M2_alt", "M3_alt"),
    ("M3", "M3_direct"),
    ("M3_alt", "M3_direct_alt"),
)

#: layers reported in the headline tables. The full sweep covers every layer;
#: these two are where the existing causal and probe work already lives.
HEADLINE_LAYERS = (24, 28)

#: Stages where a causal ablation would be interpretable. M1/M1_alt withhold on
#: ~1.3% of quadrant A, so a null there means "nothing to remove", not "not
#: load-bearing". Deliberately NOT the preregistered INTERVENTION_STAGES, which
#: is frozen by tests/test_v2_io_binding_contracts.py and must not be edited.
SFT_INTERVENTION_STAGES = ("M2", "M2_alt")
FLOOR_EFFECT_STAGES = ("M0", "M1", "M1_alt")


def unordered_pairs(stages=None):
    """The 36 unordered stage pairs, for symmetric metrics."""
    return list(combinations(stages or ALL_STAGES, 2))


def ordered_pairs(stages=None):
    """The 72 ordered stage pairs, for asymmetric metrics such as rho_AD_perp
    (which depends on the PRE stage's own subspace) and probe transfer."""
    return list(permutations(stages or ALL_STAGES, 2))


def pair_key(pre: str, post: str, sep: str = "__to__") -> str:
    return f"{pre}{sep}{post}"


def transition(key: str) -> tuple[str, str, str, str]:
    """Look a named transition up in any of the tables above."""
    for table in (TRANSITIONS_2X2, TRANSITIONS_MEDIATED, TRANSITIONS_INSTRUCT):
        if key in table:
            return table[key]
    raise KeyError(
        f"unknown transition {key!r}; known: "
        f"{sorted({**TRANSITIONS_2X2, **TRANSITIONS_MEDIATED, **TRANSITIONS_INSTRUCT})}"
    )


def arms_by_factor(objective: str = None, corpus: str = None) -> dict:
    """Subset of the 2x2 matching the given factor levels."""
    out = {}
    for key, (pre, post, obj, corp) in TRANSITIONS_2X2.items():
        if objective is not None and obj != objective:
            continue
        if corpus is not None and corp != corpus:
            continue
        out[key] = (pre, post, obj, corp)
    return out
