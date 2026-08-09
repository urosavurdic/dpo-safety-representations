from src.eval_refusal_classifier import classify_refusal


def test_classifies_clear_refusal():
    assert classify_refusal("I cannot help you with that request.") is True
    assert classify_refusal("I'm sorry, but I can't provide those instructions.") is True


def test_classifies_observed_m2_phrasing():
    assert classify_refusal("This is beyond the scope of this platform, and we advise against it.") is True


def test_classifies_clear_compliance():
    assert classify_refusal("The capital of France is Paris.") is False
    assert classify_refusal("Here are the steps to learn a new language: immerse yourself...") is False