from src.utils.eval_stats import rate_with_ci


def test_rate_with_ci_basic():
    result = rate_with_ci(5, 10)
    assert result["rate"] == 0.5
    assert result["n"] == 10
    assert 0 <= result["ci_low"] < 0.5 < result["ci_high"] <= 1


def test_rate_with_ci_zero_total():
    result = rate_with_ci(0, 0)
    assert result["rate"] is None
    assert result["n"] == 0


def test_rate_with_ci_all_success():
    result = rate_with_ci(10, 10)
    assert result["rate"] == 1.0
    assert result["ci_high"] <= 1.0