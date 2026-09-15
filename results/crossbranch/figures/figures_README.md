# Cross-branch figures -- provenance

All values are read from committed analysis JSON. No figure value is computed
here beyond arithmetic on those fields (point, ci_low, ci_high).

| file | source JSON | fields |
|---|---|---|
| `fig1_stage1_dose_response.png` | `crossbranch_AtoB_analysis.json` | `by_coefficient[*].dtv.own_delta_target`, `.own_normmatched_random` (point + percentile CI); `gate.gate_quadrant` |
| `fig2_stage2_gate_quadrant.png` | `crossbranch_AtoB_stage2_analysis.json` | `per_quadrant[q*].arms[*].dtv` (point + paired-bootstrap CI); `per_quadrant[q*].tv_baseline_to_reference` |
| `fig3_stage2_contrasts.png` | `crossbranch_AtoB_stage2_analysis.json` | `per_quadrant[q*].contrasts[*].dtv_diff` (point + paired-bootstrap CI) |
| `fig4_quadrant_summary.png` | `crossbranch_AtoB_stage2_analysis.json` | `per_quadrant[A..D].arms[*].dtv`; per-panel `n` |

`q*` is the Stage-1 gate quadrant (`stage1_gate_quadrant`), = **C**.

Negative $\Delta$TV = the arm's four-way label distribution moved toward
$B3$'s. Filled markers = 95% CI excludes 0. Bootstrap: B=10,000, seed
20260904, percentile, paired over identical `record_id`s within the quadrant.

`plotted_values.json` in this directory is the exact set of numbers drawn.
`fig*` also has a sensitivity twin if `--sensitivity` was passed
(`crossbranch_AtoB_stage2_analysis_sensitivity_extended_regex.json`).
