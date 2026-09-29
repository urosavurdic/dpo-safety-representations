"""The pair vocabulary, and the frozen sets it must not collide with."""
from __future__ import annotations

import pytest

from src.common.stages import ALL_STAGES, INTERVENTION_STAGES
from src.stagecmp.pairs import (
    CONTRAST_CAVEAT,
    CORPUS_ALPACA,
    CORPUS_DOLLY,
    FIXED_REFERENCE,
    FLOOR_EFFECT_STAGES,
    HEADLINE_PAIRS,
    OBJECTIVE_DPO,
    OBJECTIVE_SFT,
    SFT_INTERVENTION_STAGES,
    TRANSITIONS_2X2,
    arms_by_factor,
    ordered_pairs,
    pair_key,
    transition,
    unordered_pairs,
)


def test_nine_stages_give_36_unordered_and_72_ordered_pairs():
    assert len(ALL_STAGES) == 9
    assert len(unordered_pairs()) == 36
    assert len(ordered_pairs()) == 72


def test_the_2x2_holds_initialisation_fixed_within_each_corpus():
    """Both arms of a column must start from the SAME checkpoint, or the
    contrast is not at matched initialisation."""
    for corpus in (CORPUS_ALPACA, CORPUS_DOLLY):
        pres = {v[0] for v in arms_by_factor(corpus=corpus).values()}
        assert len(pres) == 1, f"{corpus} arms start from {pres}"


def test_the_2x2_crosses_both_factors_exactly_once():
    cells = {(v[2], v[3]) for v in TRANSITIONS_2X2.values()}
    assert cells == {
        (OBJECTIVE_SFT, CORPUS_ALPACA),
        (OBJECTIVE_DPO, CORPUS_ALPACA),
        (OBJECTIVE_SFT, CORPUS_DOLLY),
        (OBJECTIVE_DPO, CORPUS_DOLLY),
    }


def test_direct_arms_skip_the_safety_sft_stage():
    assert TRANSITIONS_2X2["dpo_alpaca"] == ("M1", "M3_direct", OBJECTIVE_DPO, CORPUS_ALPACA)
    assert TRANSITIONS_2X2["dpo_dolly"][:2] == ("M1_alt", "M3_direct_alt")


def test_fixed_reference_is_each_corpus_start_checkpoint():
    for corpus, ref in FIXED_REFERENCE.items():
        pres = {v[0] for v in arms_by_factor(corpus=corpus).values()}
        assert pres == {ref}


def test_sft_intervention_stages_do_not_alias_the_frozen_set():
    """INTERVENTION_STAGES is frozen by test_v2_io_binding_contracts and named
    in the preregistration. This package must register its own list rather than
    edit that one."""
    assert set(SFT_INTERVENTION_STAGES).isdisjoint(INTERVENTION_STAGES)
    assert set(SFT_INTERVENTION_STAGES) <= set(ALL_STAGES)


def test_floor_effect_stages_are_excluded_from_the_intervention_list():
    """M1 withholds on ~1.3% of quadrant A, so a null there would mean
    'nothing to remove', not 'not load-bearing'."""
    assert set(FLOOR_EFFECT_STAGES).isdisjoint(SFT_INTERVENTION_STAGES)


def test_headline_pairs_are_real_stages_and_distinct():
    assert len(set(HEADLINE_PAIRS)) == len(HEADLINE_PAIRS) == 8
    for pre, post in HEADLINE_PAIRS:
        assert pre in ALL_STAGES and post in ALL_STAGES and pre != post


def test_headline_pairs_include_both_arms_of_every_2x2_column():
    for pre, post, _o, _c in TRANSITIONS_2X2.values():
        assert (pre, post) in HEADLINE_PAIRS


def test_transition_lookup_and_key_format():
    assert transition("sft_alpaca")[:2] == ("M1", "M2")
    assert pair_key("M1", "M2") == "M1__to__M2"
    with pytest.raises(KeyError, match="unknown transition"):
        transition("no_such_arm")


def test_the_caveat_states_what_is_not_matched():
    for phrase in ("chosen AND rejected", "epochs", "learning rate", "n=2 corpora"):
        assert phrase in CONTRAST_CAVEAT
