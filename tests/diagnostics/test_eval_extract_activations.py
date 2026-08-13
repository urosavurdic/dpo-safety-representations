from src.diagnostics.eval_extract_activations import compute_pool_window


def test_pool_window_uses_full_length_when_shorter():
    assert compute_pool_window(3) == 3


def test_pool_window_caps_at_default():
    assert compute_pool_window(10) == 5