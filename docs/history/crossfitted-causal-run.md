# The cross-fitted causal run

The big overnight GPU run (`notebooks/08_final_gpu_run.ipynb`) executed for
real. What changed and what to know:

**New code (committed, 977 tests green):**
- `v2_pipeline causal --cross-fit K` — K-fold cross-fitted causal ablation.
  Each quadrant-A `direction_estimation` prompt generated under a direction
  (and matched-random-control magnitude) estimated from the other K−1
  folds. Only A folds (D centroid keeps all 120). Writes
  `causal_ablation_v2_{stage}_L24-28_xfit{K}.json`, conditions
  `{stage}_xfit_*`. Report "out-of-fold n=120", never "independent".
- `src/analysis/relabel_causal_conditions.py` — repairs a real bug: the
  `--all-ad-sensitivity` path leaked the shard-unit name into the row
  `condition`, so `_fullAD` files were written with `M3_baseline_fullAD` /
  `M3_ablated_random_fullAD` (only `ablated_AD` was relabelled). Effect:
  baseline arm fell out of judge scope → never scored; CF2 triples
  incomplete → `estimation_split_only`/`full_A` silently n=0. Fixed at
  source in `stage_causal` (relabel every condition) AND with this
  metadata-only repair script for files already generated.
- `behavioral_judges --resume-from <judged.json>` — carry good scores
  forward per judge; key includes response text so a regenerated response
  is re-scored.
- `behavioral_judges.RESPONSE_GLOBS` widened to `raw/causal_ablation_v2_*.json`
  (was anchored to `_L24-28.json`, silently excluded every `_fullAD`/`_xfit`
  file).
- `confirmatory_behavioral_endpoints`: `CF2_by_stage[*].cross_fitted` +
  `.estimation_split_only`, `CF2_crossfit_branch_contrasts` (paired
  bootstrap on cross-fitted per-prompt effects — M3-vs-each, 2×2
  corpus×history), `CF2_circularity_bias` (paired est−xf on the SAME 120
  rows). Keep-FIRST merge on duplicate `(record_id,stage,condition)` keys
  so the fullAD regeneration of the 30 held-out prompts can't move the
  preregistered CF2 number (verified: 90 dup rows skipped, CF2 primary
  byte-identical at +0.113623).

**Results (2026-09-07 judge file `behavioral_judges_v2_20260907T043919Z.json`,
gitignored; everything else committed under `results/`):**
- CF1 (−0.4008) and CF2 held-out M3 (+0.1136) **unchanged** — anchors intact.
- Cross-fitted out-of-fold n=120, ALL FOUR branches CI-excludes-0:
  M3 +0.154 [+0.105,+0.203], M3_alt +0.044 [+0.005,+0.085],
  M3_direct +0.025 [+0.013,+0.039], M3_direct_alt +0.012 [+0.003,+0.023].
- **Story shifted**: not "effect present/absent by branch" (the old n=30
  framing, a significance-pattern fallacy) → "effect present in all four,
  MAGNITUDE is path-dependent". 2×2 interaction +0.097 [+0.040,+0.150]
  excludes 0; corpus moves the effect +0.013 within direct-DPO vs +0.110
  within mediated. M3_alt-vs-M3_direct_alt spans 0 → mediation effect lives
  in the Alpaca branch.
- Circularity bias (est−xf, same rows): undetectable 3/4 branches, +0.005
  for M3_direct_alt → `full_A` reads at face value.
- McNemar re-run (authoritative, `results/summaries/mcnemar_direction_specificity.json`):
  quad C soft_deflection n=104 — M3 0/29 p<1e-6, M3_alt 3/14 p=0.013,
  M3_direct 3/8 p=0.227, M3_direct_alt 7/7 p=1.00. n=150 quad A/D
  (`refusal`, sensitivity) is noisy — don't lean on it.

**Paper** (`private/paper/main.tex`, gitignored): title **"Preserved but
Path-Dependent"**. Abstract, §6, §7, limitations, deviations table
rewritten for the above. 4 figures regenerated (`private/paper/figures/`,
fig3 redesigned held-out-vs-cross-fitted + branch contrasts; fig2 still
4/9 stages — needs `activations_bundle` from Drive or a 1-min re-run of
`factorial_direction_audit.py` on all 9). Ledger `docs/findings_654_synthesis.md`
fully updated. **Remaining `\pending` (author tasks only):** Arditi et al.
citation check; §10 instrument validation (human annotation).

**Local activation state:** M2/M3/M2_alt/M3_alt are fresh 654-row; the
other 5 stages are stale 370-row locally (the `activations_bundle` on
Drive has all 9 fresh).
