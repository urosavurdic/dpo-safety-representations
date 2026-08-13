"""
Utilities for interpretability analysis.

Shared functions for loading, transforming, and formatting mechanistic analysis results.
"""

import numpy as np
from typing import Dict, List, Any


def compute_effect_size(baseline_rate: float, ablated_rate: float) -> Dict[str, float]:
    """
    Compute standard effect size measures.
    
    Args:
        baseline_rate: Baseline behavior rate (e.g., 0.80 for 80% soft-deflection)
        ablated_rate: Ablated behavior rate (e.g., 0.25 after intervention)
    
    Returns:
        Dictionary with effect sizes
    """
    # Absolute change
    abs_change = baseline_rate - ablated_rate
    
    # Relative change (percent reduction)
    rel_change = abs_change / baseline_rate if baseline_rate > 0 else 0
    
    # Cohen's h (for proportions)
    phi_0 = 2 * np.arcsin(np.sqrt(baseline_rate))
    phi_1 = 2 * np.arcsin(np.sqrt(ablated_rate))
    cohens_h = phi_0 - phi_1
    
    return {
        "absolute_change": float(abs_change),
        "relative_change_pct": float(rel_change * 100),
        "cohens_h": float(cohens_h),
        "cohens_h_interpretation": interpret_cohens_h(cohens_h)
    }


def interpret_cohens_h(cohens_h: float) -> str:
    """
    Interpret Cohen's h effect size for proportions.
    
    Standard thresholds:
    - 0.2: small
    - 0.5: medium
    - 0.8: large
    """
    abs_h = abs(cohens_h)
    if abs_h < 0.2:
        return "negligible"
    elif abs_h < 0.5:
        return "small"
    elif abs_h < 0.8:
        return "medium"
    else:
        return "large"


def format_ci_string(rate: float, ci_lower: float, ci_upper: float, count: int = None) -> str:
    """Format rate with confidence interval for reporting."""
    ci_str = f"{rate:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
    if count is not None:
        ci_str += f" ({count})"
    return ci_str


def summarize_behavioral_change(
    target_name: str,
    baseline_rate: float,
    ablated_rate: float,
    n_samples: int,
    discordant_pairs: Dict[str, int] = None,
    p_value: float = None
) -> Dict[str, Any]:
    """
    Generate a structured summary of behavioral change under intervention.
    
    Args:
        target_name: Name of target (e.g., "quadrant C soft-deflection")
        baseline_rate: Rate in baseline condition
        ablated_rate: Rate in ablated condition
        n_samples: Total sample size
        discordant_pairs: McNemar table (e.g., {"baseline_yes_ablated_no": 11})
        p_value: McNemar exact p-value, if available
    
    Returns:
        Structured summary dictionary
    """
    effect_size = compute_effect_size(baseline_rate, ablated_rate)
    
    summary = {
        "target": target_name,
        "baseline": {
            "rate": baseline_rate,
            "count": int(baseline_rate * n_samples)
        },
        "ablated": {
            "rate": ablated_rate,
            "count": int(ablated_rate * n_samples)
        },
        "effect": effect_size,
        "sample_size": n_samples,
        "significance": {
            "p_value": p_value,
            "is_significant_at_0_05": p_value < 0.05 if p_value else None
        }
    }
    
    if discordant_pairs:
        summary["discordant_pairs"] = discordant_pairs
    
    return summary
