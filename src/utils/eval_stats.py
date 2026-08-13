from statsmodels.stats.proportion import proportion_confint


def rate_with_ci(successes: int, total: int, confidence: float = 0.95):
    if total == 0:
        return {"rate": None, "ci_low": None, "ci_high": None, "n": 0}
    rate = successes / total
    ci_low, ci_high = proportion_confint(successes, total, alpha=1 - confidence, method="wilson")
    return {"rate": rate, "ci_low": ci_low, "ci_high": ci_high, "n": total}