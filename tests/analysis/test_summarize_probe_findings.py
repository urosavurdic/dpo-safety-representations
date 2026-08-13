from src.analysis.summarize_probe_findings import frac_to_ci


def test_frac_to_ci_converts_fraction_to_counts():
    result = frac_to_ci(0.6, 50)
    assert result["n"] == 50
    assert abs(result["rate"] - 0.6) < 1e-9
    assert result["ci_low"] < 0.6 < result["ci_high"]


def test_frac_to_ci_handles_none():
    result = frac_to_ci(None, 50)
    assert result["n"] == 0