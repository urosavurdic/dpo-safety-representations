from src.analysis.summarize_causal_ablation import classify_completion


def test_precedence_degenerate_wins_over_everything(monkeypatch):
    monkeypatch.setattr(classify_completion, "is_degenerate", lambda text: True)
    monkeypatch.setattr(classify_completion, "classify_refusal", lambda text: True)
    monkeypatch.setattr(classify_completion, "is_soft_deflection", lambda text: True)
    assert classify_completion("anything") == "degenerate"


def test_precedence_refusal_wins_over_soft_deflection(monkeypatch):
    monkeypatch.setattr(classify_completion, "is_degenerate", lambda text: False)
    monkeypatch.setattr(classify_completion, "classify_refusal", lambda text: True)
    monkeypatch.setattr(classify_completion, "is_soft_deflection", lambda text: True)
    assert classify_completion("anything") == "refusal"


def test_precedence_soft_deflection_when_not_refusal(monkeypatch):
    monkeypatch.setattr(classify_completion, "is_degenerate", lambda text: False)
    monkeypatch.setattr(classify_completion, "classify_refusal", lambda text: False)
    monkeypatch.setattr(classify_completion, "is_soft_deflection", lambda text: True)
    assert classify_completion("anything") == "soft_deflection"


def test_default_comply_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(classify_completion, "is_degenerate", lambda text: False)
    monkeypatch.setattr(classify_completion, "classify_refusal", lambda text: False)
    monkeypatch.setattr(classify_completion, "is_soft_deflection", lambda text: False)
    assert classify_completion("anything") == "comply"