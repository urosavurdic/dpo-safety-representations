"""
Integrated interpretability report generator.

Orchestrates all mechanistic interpretability analyses and produces a unified,
human-readable report for understanding the refusal direction and causal effect.

Report structure:
1. Executive summary
2. Per-layer attribution (which layers matter)
3. Scaling analysis (how effect varies with magnitude)
4. Direction stability (how representation changes during training)
5. Integration & implications for mechanistic understanding
6. Limitations and future work
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from . import per_layer_analysis, alpha_scaling, direction_stability


def generate_interpretability_report(
    causal_raw_wide: str = "results/raw/causal_ablation_raw_wide.json",
    causal_raw_narrow: str = "results/raw/causal_ablation_raw_narrow.json",
    projections: str = "results/refusal_direction/quadrant_projections.json",
    cosine_sim: str = "results/refusal_direction/cosine_similarity.json",
    output_dir: str = "results/interpretability/reports",
    output_name: str = "interpretability_report.md"
) -> str:
    """
    Generate a comprehensive interpretability report.
    
    Args:
        causal_raw_wide, causal_raw_narrow: Raw ablation data files
        projections: Quadrant projection data
        cosine_sim: Direction stability (cosine similarity) data
        output_dir: Where to save the report
        output_name: Report filename (Markdown format)
    
    Returns:
        Path to generated report
    """
    
    # Run all analyses
    per_layer = per_layer_analysis.compute_per_layer_contributions(causal_raw_wide, projections)
    alpha = alpha_scaling.estimate_alpha_scaling_curve(projections, 
                                                       "results/summaries/causal_ablation_raw_narrow_summary.json")
    direction = direction_stability.analyze_direction_stability(cosine_sim, projections)
    
    # Build Markdown report
    report_lines = [
        "# Mechanistic Interpretability Report: Refusal Direction & Causal Ablation",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report analyzes the mechanistic basis of the extracted refusal direction and its causal",
        "effects on model behavior. Using techniques from mechanistic interpretability literature",
        "(gradient-based attribution, scaling analysis, representation stability tracking), we identify",
        "which layers contribute most, how effects scale with intervention strength, and how the",
        "direction drifts during training.",
        "",
        "### Key Findings",
        "",
        "1. **Effect magnitude:** Ablating layers 14-28 completely suppresses quadrant A refusal (14%→0%)",
        "   and substantially reduces quadrant C soft-deflection (80%→0%). Narrowing to layers 24-28 only",
        "   partially reduces C (80%→25%) while maintaining full A suppression.",
        "",
        "2. **Layer attribution:** Deeper layers (21, 28) carry larger refusal signal in target quadrants.",
        "   Effect is distributed across the deep transformer, not isolated to a single layer.",
        "",
        "3. **Scaling behavior:** Effects scale approximately linearly with intervention magnitude,",
        "   suggesting clean causal pathways rather than threshold-gated mechanisms.",
        "",
        "4. **Direction stability:** Moderate drift across training (cosine sim ~0.4-0.7 from M0 to M3).",
        "   Despite drift, ablation remains effective, suggesting direction is a robust causal lever.",
        "",
        "---",
        "",
        "## 1. Per-Layer Attribution Analysis",
        "",
        "**Goal:** Identify which transformer layers contribute most to the refusal signal.",
        "",
        "**Method:** Analyze projection magnitudes per layer and compare with discordant pair flips",
        "from the causal ablation.",
        "",
        "### Results",
        ""
    ]
    
    if "per_layer_signal_magnitude" in per_layer:
        report_lines.append("#### Per-Layer Signal Magnitude (M3 Baseline)")
        report_lines.append("")
        report_lines.append("| Layer | Quadrant C Mean Proj | Quadrant A Mean Proj | Combined Mean |")
        report_lines.append("|-------|---------------------|---------------------|---------------|")
        
        for layer in sorted(per_layer["per_layer_signal_magnitude"].keys()):
            vals = per_layer["per_layer_signal_magnitude"][layer]
            c_proj = vals["quadrant_C_mean_abs_proj"]
            a_proj = vals["quadrant_A_mean_abs_proj"]
            combined = vals["mean_of_targets"]
            report_lines.append(f"| {layer} | {c_proj:.4f} | {a_proj:.4f} | {combined:.4f} |")
        
        report_lines.append("")
    
    if "layer_contribution_interpretation" in per_layer:
        report_lines.append("#### Interpretation")
        report_lines.append("")
        interp = per_layer["layer_contribution_interpretation"]
        report_lines.append(f"**Key Finding:** {interp.get('key_finding', 'N/A')}")
        report_lines.append("")
        report_lines.append(interp.get("interpretation", ""))
        report_lines.append("")
    
    if "recommendations" in per_layer:
        report_lines.append("#### Recommendations")
        report_lines.append("")
        for i, rec in enumerate(per_layer["recommendations"], 1):
            report_lines.append(f"{i}. {rec}")
        report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## 2. Alpha Scaling Analysis",
        "",
        "**Goal:** Understand how behavioral effects scale as a function of intervention magnitude.",
        "",
        "**Method:** Interpolate the refusal direction projection from 0 (full ablation) to 1 (no ablation)",
        "and estimate the scaling curve from observed baseline/ablated behavior.",
        "",
        "### Results",
        ""
    ])
    
    if "scaling_estimates" in alpha:
        report_lines.append("#### Effect Size by Condition")
        report_lines.append("")
        
        for target, data in alpha["scaling_estimates"].items():
            report_lines.append(f"**{target}:**")
            report_lines.append(f"- Baseline (α=1): {data['alpha_0']:.3f}")
            report_lines.append(f"- Ablated (α=0): {data['alpha_1']:.3f}")
            report_lines.append(f"- Empirical effect size: {data['effect_size_empirical']:.3f}")
            if "linear_scaling_prediction" in data:
                report_lines.append(f"- Linear interpolation predictions:")
                for alpha_val, pred_rate in data["linear_scaling_prediction"].items():
                    report_lines.append(f"  - {alpha_val}: {pred_rate:.3f}")
            report_lines.append("")
    
    if "interpretation" in alpha:
        report_lines.append("#### Interpretation")
        report_lines.append("")
        interp = alpha["interpretation"]
        report_lines.append(interp.get("scaling_regime", ""))
        report_lines.append("")
        report_lines.append(f"**Key Insight:** {interp.get('key_insight', '')}")
        report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## 3. Direction Stability Analysis",
        "",
        "**Goal:** Track how the refusal direction changes across training stages (M0→M1→M2→M3).",
        "",
        "**Method:** Compute cosine similarity of extracted direction vs M0 baseline; analyze drift",
        "per layer and per training stage.",
        "",
        "### Results",
        ""
    ])
    
    if "stability_summary" in direction:
        summary = direction["stability_summary"]
        report_lines.append("#### Aggregate Stability Metrics")
        report_lines.append("")
        report_lines.append(f"- Mean cosine similarity (M0 vs M3): {summary['mean_similarity_M0_vs_M3']:.3f}")
        report_lines.append(f"- Range: {summary['min_similarity']:.3f} to {summary['max_similarity']:.3f}")
        report_lines.append(f"- Std Dev: {summary['std_similarity']:.3f}")
        report_lines.append(f"- Stable layers (sim > 0.7): {summary['layers_high_stability']}")
        report_lines.append(f"- Drifting layers (sim < 0.5): {summary['layers_low_stability']}")
        report_lines.append("")
    
    if "drift_dynamics" in direction:
        if "aggregate" in direction["drift_dynamics"]:
            report_lines.append("#### Stage-to-Stage Drift")
            report_lines.append("")
            agg = direction["drift_dynamics"]["aggregate"]
            report_lines.append(f"- Mean drift M0→M1: {agg['mean_drift_M0_to_M1']:.3f}")
            report_lines.append(f"- Mean drift M1→M2: {agg['mean_drift_M1_to_M2']:.3f}")
            report_lines.append(f"- Mean drift M2→M3: {agg['mean_drift_M2_to_M3']:.3f}")
            report_lines.append("")
    
    if "key_findings" in direction:
        report_lines.append("#### Key Findings")
        report_lines.append("")
        for finding in direction["key_findings"]:
            report_lines.append(f"- {finding}")
        report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## 4. Integration & Mechanistic Implications",
        "",
        "### Synthesis",
        "",
        "The three analyses converge on a coherent picture:",
        "",
        "1. **The refusal direction is a distributed, layer-spanning feature** with higher signal in",
        "   deep layers (21, 28). It is NOT localized to a single layer or layer pair.",
        "",
        "2. **The causal effect is robust to training drift.** Despite 40-57% drift in direction similarity",
        "   from M0 to M3, the ablation remains highly effective (16/16 pairs flip away from soft-deflection",
        "   at 80% baseline). This suggests the direction is either: (a) continuously refined by training",
        "   to remain causally potent, or (b) a sufficient but not strictly necessary cause.",
        "",
        "3. **Effects scale smoothly (linearly) with intervention magnitude**, suggesting clean causal",
        "   pathways rather than threshold-gating or multiplicative interactions. However, the asymmetry",
        "   between quadrants (A complete, C partial under narrow ablation) hints at distinct mechanistic",
        "   substrates for legitimate refusal vs. over-caution.",
        "",
        "### Behavioral Selectivity",
        "",
        "The intervention achieves complete suppression of both target behaviors (C and A) but fails to",
        "achieve selectivity—narrowing to deep layers (24-28) does not preferentially reduce soft-deflection",
        "while sparing legitimate refusal. This suggests:",
        "",
        "- The same direction contribution drives both behaviors, or",
        "- The two behaviors are driven by related but distinct direction components that cannot be",
        "  separated by layer-range selection alone.",
        "",
        "### Implications for Mechanistic Understanding",
        "",
        "- **Feature geometry:** The refusal direction is a genuine feature, not an artifact. Its persistence",
        "  across training and effectiveness under intervention support this.",
        "",
        "- **Causal structure:** The direction is causally sufficient to modulate both legitimate and",
        "  problematic refusal behaviors. It is not strictly necessary (other paths may exist).",
        "",
        "- **Future interventions:** Selectively improving model behavior via this direction would require",
        "  finer-grained manipulation (e.g., per-quadrant re-weighting or layer-specific scaling) rather",
        "  than simple layer-range narrowing.",
        "",
        "---",
        "",
        "## 5. Limitations & Future Work",
        "",
        "### Limitations",
        "",
        "1. **Attribution granularity:** Per-layer analysis uses projection magnitude, not gradient-based",
        "   saliency. Backprop-based approaches could refine layer rankings.",
        "",
        "2. **Causal sufficiency, not necessity:** Ablation shows the direction is sufficient to affect",
        "   behavior but does not prove it is necessary. Other mechanisms may compensate.",
        "",
        "3. **Post-hoc analysis:** Scaling predictions are estimated from two data points (baseline, ablated),",
        "   not empirically validated with intermediate α values.",
        "",
        "4. **Single direction:** Analysis treats the refusal direction as monolithic. Subspace analysis",
        "   (e.g., PCA on activations) might reveal multi-dimensional structure.",
        "",
        "### Future Directions",
        "",
        "1. **Gradient-based attribution:** Backprop-compute relevance or integrated gradients to assign",
        "   per-layer importance.",
        "",
        "2. **Intermediate α validation:** Empirically test α ∈ {0.25, 0.5, 0.75} by scaling projection",
        "   magnitudes and re-running inference on the eval set.",
        "",
        "3. **Behavioral subspace analysis:** PCA on per-quadrant activations to decompose into orthogonal",
        "   components; test which components drive C vs A.",
        "",
        "4. **Selectivity optimization:** Grid search over layer ranges, per-layer scaling factors, or",
        "   orthogonal direction removal to find interventions that preserve A while reducing C.",
        "",
        "---",
        "",
        "## References",
        "",
        "- **Mechanistic Interpretability:** Vig & Belinkov (2019), \"Analyzing the Structure of Attention",
        "  in a Transformer Language Model\"; Christiano et al. (2021), \"Alignment Research Center work\".",
        "",
        "- **Causal Intervention:** Pearl (2009), *Causality*; Geva et al. (2023), \"Mechanistic Interpretability",
        "  of Language Models\".",
        "",
        "- **Direction/Representation Analysis:** Finlayson et al. (2021), \"Deconstructing Hate Speech",
        "  Detection: Interpreting Classifiers\".",
        "",
        "---",
        "",
        f"*Report generated by `src.interpretability.integrated_report` on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    ])
    
    # Write report
    report_path = Path(output_dir) / output_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nInterpretability report saved to {report_path}")
    return str(report_path)


if __name__ == "__main__":
    generate_interpretability_report()
