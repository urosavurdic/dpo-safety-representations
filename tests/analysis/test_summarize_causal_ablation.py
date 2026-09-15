"""Precedence contract for the composed classifier.

classify_completion lives in src.common.refusal_classifier, beside the three
predicates it composes, so these patch that module -- patching a module that
merely re-exports the function would not affect the names it closes over.
"""

from src.common import refusal_classifier as rc


def test_precedence_degenerate_wins_over_everything(monkeypatch):
    monkeypatch.setattr(rc, "is_degenerate", lambda text: True)
    monkeypatch.setattr(rc, "classify_refusal", lambda text: True)
    monkeypatch.setattr(rc, "is_soft_deflection", lambda text: True)
    assert rc.classify_completion("anything") == "degenerate"


def test_precedence_refusal_wins_over_soft_deflection(monkeypatch):
    monkeypatch.setattr(rc, "is_degenerate", lambda text: False)
    monkeypatch.setattr(rc, "classify_refusal", lambda text: True)
    monkeypatch.setattr(rc, "is_soft_deflection", lambda text: True)
    assert rc.classify_completion("anything") == "refusal"


def test_precedence_soft_deflection_when_not_refusal(monkeypatch):
    monkeypatch.setattr(rc, "is_degenerate", lambda text: False)
    monkeypatch.setattr(rc, "classify_refusal", lambda text: False)
    monkeypatch.setattr(rc, "is_soft_deflection", lambda text: True)
    assert rc.classify_completion("anything") == "soft_deflection"


def test_default_comply_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(rc, "is_degenerate", lambda text: False)
    monkeypatch.setattr(rc, "classify_refusal", lambda text: False)
    monkeypatch.setattr(rc, "is_soft_deflection", lambda text: False)
    assert rc.classify_completion("anything") == "comply"


def test_summarizer_exposes_the_same_function_object():
    """The summarizer must not shadow the frozen classifier with its own copy."""
    from src.analysis import summarize_causal_ablation as sca

    assert sca.classify_completion is rc.classify_completion
