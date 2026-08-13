from src.analysis import summarize_causal_ablation as sca

def test_precedence_degenerate_wins_over_everything(monkeypatch):
    monkeypatch.setattr(sca, "is_degenerate", lambda text: True)
    monkeypatch.setattr(sca, "classify_refusal", lambda text: True)
    monkeypatch.setattr(sca, "is_soft_deflection", lambda text: True)
    assert sca.classify_completion("anything") == "degenerate"


def test_precedence_refusal_wins_over_soft_deflection(monkeypatch):
    monkeypatch.setattr(sca, "is_degenerate", lambda text: False)
    monkeypatch.setattr(sca, "classify_refusal", lambda text: True)
    monkeypatch.setattr(sca, "is_soft_deflection", lambda text: True)
    assert sca.classify_completion("anything") == "refusal"


def test_precedence_soft_deflection_when_not_refusal(monkeypatch):
    monkeypatch.setattr(sca, "is_degenerate", lambda text: False)
    monkeypatch.setattr(sca, "classify_refusal", lambda text: False)
    monkeypatch.setattr(sca, "is_soft_deflection", lambda text: True)
    assert sca.classify_completion("anything") == "soft_deflection"


def test_default_comply_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(sca, "is_degenerate", lambda text: False)
    monkeypatch.setattr(sca, "classify_refusal", lambda text: False)
    monkeypatch.setattr(sca, "is_soft_deflection", lambda text: False)
    assert sca.classify_completion("anything") == "comply"