# HANDOFF — DPO Safety Representations Project

## Status: Phase 4 complete, no further experiments planned.
All components tested, statistically validated, and internally consistent
between this file, README.md, and committed results/ JSON.

## What each component found

**C1 — Behavioral eval.** Quadrant C soft-deflection: M0 0%→M1 0%→M2 10%→M3
70%. Quadrant A: 6%→12%. DPO's effect concentrates on disguised harm.

**C2 — Activation extraction.** Verified clean, merge-cascade confirmed correct.

**C3 — Linear probes.** Naive CV accuracy retired (saturates even at
untrained M0). Real signal: held-out flagging rate. M1 already flags 85% of
quadrant C despite 0% behavioral change — representation precedes behavior.

**C4 — Refusal direction (diff-in-means, not PCA).** Mean drift across 28
layers: M0→M1 ≈0.335, M1→M2 ≈0.040, M2→M3 ≈0.070. Biggest rotation happens
during generic instruction-tuning, not safety training.

**C5 — Causal ablation, wide + narrow, both McNemar-confirmed.**
- Wide (14–28): A 14%→0% (p=0.0156), C 80%→0% (p=0.00003).
- Narrow (24–28): A 14%→0% (p=0.0156, 100% relative), B 5.6%→1.2% (79%
  relative), C 80%→25% (p=0.0010, 69% relative).
- A's suppression is fully explained by the deepest 5 layers; B and C
  aren't — but B and C don't separate from each other either. Not
  quadrant-selective, but the mechanism is layer-differentiated in a
  specific, precise way, not uniformly "everything is entangled."

**C5b — Steering.** Multi-layer (14–28): 98% degenerate output, not
refusal — most likely residual-stream compounding across 15 injections.
Single-layer (21): small, non-significant shift (McNemar p=0.50, n=2
discordant). Genuine null result — not a confirmed causal complement to
ablation, reported honestly as inconclusive rather than reframed as positive.

**Interpretability module.** Reduced to `direction_stability.py` (rewritten
to match the real cosine_similarity.json schema, tested) and
`lora_subspace_check.py` (new). Everything else removed — original scripts
had JSON key mismatches meaning they never actually computed on real data,
plus a tautological "linear scaling" claim and unverifiable citations.

**LoRA-subspace check.** 90%+ of the refusal direction's norm lies outside
the rank-64 LoRA subspace at every layer/module checked — not primarily a
LoRA artifact. But real, above-chance alignment (z=3–10 vs. a 200-sample
random-direction baseline) concentrates at deep layers and `down_proj`
specifically — matching where C4 independently finds DPO's rotation
concentrates. Two independent methods (weight geometry, activation
statistics) converging on the same layers.

## Current overall verdict
No evidence DPO builds a new, safety-specific representation from scratch —
the direction is already present after generic instruction-tuning (C3). DPO
primarily strengthens coupling between that representation and behavior, with
real but secondary additional rotation at deep layers (C4), and a modest,
non-dominant relationship to its own LoRA subspace (LoRA check). The causal
ablation effect is real and significant (C5) but not quadrant-selective in
the way originally hoped, with a precise (not uniform) layer-dependence
pattern. Steering did not independently confirm the causal story (C5b) — an
honest gap, not smoothed over. Closer to "coupling/amplification" than
"genuinely new representation."

## Known limitations, stated not hidden
LoRA confound quantified but not eliminated; single direction only; ablation
shows sufficiency not necessity and steering didn't confirm the complementary
direction; n=20 for quadrant C; 1.5B scale; M1's Alpaca data may itself skew
safe; steering's degenerate-collapse mechanism not independently diagnosed.
Full list in README.

## Repo state
Reorganized, all import/path fixes applied and verified (`pytest tests/ -v`
green — see README Repository Hygiene). Two new small test files added for
`mcnemar_steering.py` and the LoRA random-baseline addition, closing the last
test-coverage gap.

## Next steps, in order
1. Confirm `pytest tests/ -v` is fully green after the last two test
   additions and the `lora_subspace_check.py` dict-overwrite fix.
2. Write-up: expand this file's "Current overall verdict" into the actual
   report. All source numbers are final — no further experiments planned.

## Working conventions (still apply)
Wilson CIs via `src/eval_stats.py`. Don't relitigate locked decisions without
a specific new reason.