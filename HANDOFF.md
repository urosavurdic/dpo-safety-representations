# HANDOFF — DPO Safety Representations Project

Self-contained current-state snapshot. Full historical decision log in
PROJECT_CONTEXT.md if deeper context is needed.

## Research question
Does DPO give a language model genuinely richer internal safety
representations, or does it mainly amplify/reshape an existing refusal
direction? Chain: M0 (Qwen2.5-1.5B-Base) → M1 (SFT-Helpful) → M2 (SFT-Safety,
matched data) → M3 (DPO).

## Status: Phase 4 complete and validated. Repo reorganized. Ready for Phase 5 (write-up).
Not "needs retrying" — the components below are done, tested, and their
numbers are internally consistent between this file, the README, and the
committed results/ JSON. Two small items remain open (marked TODO below),
neither requires new experiments.

## What each component found

**C1 — Behavioral eval.** Quadrant C soft-deflection: M0 0% → M1 0% → M2 10%
→ M3 70% (non-overlapping CIs). Quadrant A hard-refusal: 6%→12%. DPO's effect
concentrates on disguised (neutrally-worded) harm specifically.

**C2 — Activation extraction.** Verified clean: 370×29×1536, no NaNs, all
stages. Merge-cascade correctness independently confirmed (layer-0 cosine
similarity = 1.0000 across all stage pairs, as expected since LoRA never
touches embeddings).

**C3 — Linear probes.** Naive CV accuracy retired (saturates near 1.0 even
for untrained M0 — dataset-fingerprint confound). Real signal: held-out
flagging rate at the final layer. Quadrant D: M0 0.600 → M3 0.180
(non-overlapping). Quadrant C: M0 0.000 → M1/M2/M3 0.85/0.80/0.75. **M1
already flags 85% of quadrant C despite 0% behavioral change — representation
precedes behavior.**

**C4 — Refusal direction (diff-in-means, A vs D — not PCA).** Adjacent-stage
cosine similarity, deep layers: M0→M1 ≈0.50–0.56 (largest rotation), M1→M2
≈0.94–0.95 (near-stationary), M2→M3 ≈0.84–0.86 (real but smaller rotation than
M0→M1). Quadrant C's normalized position on the D→A axis: 0.39 (M0) → 0.67
(M1) → 0.56 (M2) → 0.68 (M3) — big jump at M1, roughly flat after.

**C5 — Causal ablation, two layer ranges tested, both McNemar-confirmed.**
- Wide (14–28): C 80%→0% (16/16 flip, p=0.000031); A 14%→0% (7/7 flip, p=0.015625).
- Narrow (24–28): C 80%→25% (11/16 flip, p=0.000977); A 14%→0% (7/7 flip, p=0.015625).
- Reading: A's suppression is fully explained by layers 24–28 alone. C's is
  not — layers 14–23 carry real additional signal. Neither range achieves
  selectivity (preserving A while reducing only C); the intervention is
  causally load-bearing for both, just not equally deep-concentrated.
- **TODO:** quadrant B's soft-deflection rate under the *narrow* ablation
  specifically (known under wide: 4.8%→0%). No new experiment needed — already
  in `results/raw/causal_ablation_raw_narrow.json`, just needs
  `summarize_causal_ablation.py` run against it and the number pulled.

**Interpretability module.** `alpha_scaling.py`, `per_layer_analysis.py`,
`integrated_report.py` removed — their "findings" didn't reproduce against
real data (JSON key mismatches meant they silently computed on empty/error
data) and the headline "linear scaling" claim was a tautology of interpolating
between two points, not an empirical result. `direction_stability.py` kept and
rewritten to match the real `cosine_similarity.json` schema, now tested.

## Current overall verdict
Post-training doesn't create a new, DPO-specific safety representation from
scratch — sensitivity to disguised harm is already present after generic
instruction-tuning (C3). DPO measurably reshapes the refusal-associated
direction more than safety-SFT does and changes how strongly it converts into
behavior (C4), but the causal ablation (C5) shows this mechanism isn't
separable from legitimate refusal at the layer-range resolution tested. Closer
to "coupling/amplification" (Hypothesis B) than "genuinely new representation"
(Hypothesis A) — a real, nuanced result, not a clean binary one.

## Known limitations, stated not hidden
LoRA-rank confound (not fully resolved); single direction only; ablation shows
sufficiency not necessity; n=20 for quadrant C; 1.5B scale; M1's Alpaca data
may itself skew safe, confounding Finding 3's "instruction-tuning, not safety
training" reading. Full list in README.

## Repo state
Reorganized (see README "Project Structure"). All moves + import fixes need
verification: `pytest tests/ -v` should pass with zero failures after the
reorg commands are applied and paths updated.

## Next steps, in order
1. Apply the reorg (commands given separately) and fix the resulting import/path breaks.
2. `pytest tests/ -v` — confirm clean.
3. Fill in the two TODOs above (quadrant B narrow rate; direction_stability's fresh 29-layer aggregate).
4. Full clean-environment reproduction pass (validates the reorg didn't silently break anything) — see README Quick Start.
5. Write-up: expand "Current overall verdict" into the actual report, using README + this file as the source of truth.

## Working conventions (still apply)
One component at a time, tests before moving on. Wilson CIs via
`src/eval_stats.py`'s `rate_with_ci`. Don't relitigate locked decisions
without a specific new reason.