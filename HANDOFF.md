# HANDOFF — DPO Safety Representations Project

Single-file current-state snapshot. Paste this at the start of any new
session — it's self-contained. Full historical decision log lives in
PROJECT_CONTEXT.md in the repo (github.com/urosavurdic/dpo-safety-representations)
if deeper context is ever needed, but this file should be enough on its own.

## Research question
Does DPO give a language model genuinely richer internal safety
representations, or does it mainly amplify/reshape an existing refusal
direction? Four-model chain: M0 (Qwen2.5-1.5B-Base) → M1 (SFT-Helpful) →
M2 (SFT-Safety, matched data) → M3 (DPO).

## Status: Phase 4 essentially complete, one open experimental question
Everything is done. Needs validation. 

## What each component found (the numbers that matter)

**C1 — Behavioral eval.** Soft-deflection on quadrant C (harmful intent,
neutral wording): M0 0% → M1 0% → M2 10% → M3 70% (non-overlapping CIs).
Quadrant A (obviously harmful) hard-refusal only moves 6%→12%. DPO's
effect is concentrated on neutrally-worded harm, not harm in general.

**C2 — Activation extraction.** Verified clean (370×29×1536, no NaNs, all
stages). Model-delta/merge handling independently confirmed correct via
cross-stage cosine check (layer 0 = 1.0000 always, expected since LoRA
doesn't touch embeddings; deep layers diverge as expected with
training-chain distance).

**C3 — Linear probes.** Naive CV accuracy saturates near 1.0 at nearly
every layer for every stage including untrained M0 (dataset/style
fingerprint confound, not fixable with the available quadrant data) —
retired as a metric. Real signal: fraction of held-out quadrants flagged
"unsafe" at the final layer. Quadrant D (benign): M0 0.600 [.46,.72] vs M3
0.180 [.10,.31]. Quadrant C (harmful, neutral wording): M0 0.000 [0,.16]
vs M1/M2/M3 0.850/0.800/0.750 (all well above M0). **Key finding: M1
already representationally flags 85% of quadrant C despite 0% behavioral
soft-deflection** — representation appears before behavior changes.

**C4 — Refusal direction (diff-in-means, A vs D).** Adjacent-stage cosine
similarity at deep layers: M0→M1 ≈0.50-0.56 (biggest rotation, generic
instruction-tuning), M1→M2 ≈0.94-0.95 (barely rotates, looks like
amplification), M2→M3 ≈0.84-0.86 (moderate additional rotation, more than
SFT-safety but far less than instruction-tuning). Quadrant C's
scale-normalized position on the D→A axis: M0 0.39 → M1 0.67 → M2 0.56 →
M3 0.68 at the final layer — big jump at M1, roughly flat M1→M2→M3. The
big M2→M3 *behavioral* jump isn't matched by an equally big M2→M3
*representational* jump on this metric — suggests DPO changes how the
representation gets read out into behavior more than it repositions the
representation itself.

**C5 — Causal ablation (first attempt, layers 14-28; and narrow 24-28 refinement).**

 - This one is a bit sus not bad to check it

Wide ablation (layers 14–28): quadrant C soft-deflection 80% (16/20) → 0% (0/20);
quadrant A refusal 14% (7/50) → 0% (0/50). Paired McNemar exact tests:
C (wide) 16→0 discordant pairs, p = 0.000031; A 7→0 discordant pairs, p = 0.015625.

Narrow ablation (layers 24–28): quadrant C soft-deflection 80% (16/20) → 25% (5/20);
paired McNemar exact p = 0.000977 (11 switched away, 5 stayed). Quadrant A again
collapses 7/50 → 0/50 (p = 0.015625).

Interpretation: the causal intervention is clearly load-bearing for the target
behavior (quadrant C soft-deflection) and for legitimate refusal (quadrant A).
Narrowing the ablation to the deepest 5 layers reduces but does not eliminate
the effect on quadrant C while leaving quadrant A suppression unchanged. In other
words, the intervention is effective but not behaviorally selective: the narrow
ablation produces a partial rescue for over-caution but still suppresses
legitimate refusal. This timeboxed refinement provides a clear, reproducible
answer and closes the last planned experimental variation for Phase 4.

Quadrant C soft-deflection 80%→0%; quadrant A refusal also
14%→0%. Not selective. Confirmed with a paired McNemar's exact test (not
just Wilson CIs): C, 16/16 discordant pairs switched away from
soft-deflection under ablation, p=0.000031. A, 7/7 discordant pairs
switched away from refusal, p=0.015625. Both effects are individually
significant and complete (100% flip, 0% reverse-flip) — the ablation's
lack of selectivity is now statistically solid, not just visually
apparent. One narrower-layer attempt (24-28 instead of 14-28) is the last
open experimental question before Phase 5.



## Current overall verdict (concise, cautious)

 - All experiments need to be retried this is preliminary too.

The results suggest that post-training alters how a pre-existing refusal-
related direction is read out, rather than creating an entirely new,
isolated safety module. Instruction-tuning already produces a measurable
refusal-like direction, and later training reshapes how that signal is
used to generate behavior. DPO changes both representations and their
readout in behavior, but the causal ablation evidence indicates the effect
is not cleanly separable from legitimate refusal: the intervention is
causally important, yet the tested ablations produce side effects that
reduce selectivity. This is a nuanced outcome that argues against a simple
"new module" interpretation and motivates careful follow-ups.

Post-training doesn't appear to create a new, DPO-specific safety
representation from scratch — sensitivity to neutrally-worded harm is
already present after generic instruction-tuning (M1). DPO does
measurably reshape (not just amplify) the refusal-associated direction
more than SFT-safety does, and changes how strongly that representation
converts into refusal behavior. Closer to "coupling/amplification"
(Hypothesis B) than "genuinely new representation" (Hypothesis A), with
real nuance, not a clean binary result.


## Repo state / sync warnings (keep in mind)
Git synchronization between local development and remote/Colab runs remains an operational hazard. Local edits and testing have been performed; confirm the intended commits are pushed to `main` before rerunning experiments in a fresh Colab runtime. The current local edits include the narrow ablation analysis and small script fixes — double-check remote state before assuming equivalence.

Known hygiene debt: a few smoke-test artifacts are tracked in history (~tens of MB). These should be cleaned or archived before a formal release; see the todo list for concrete steps.

## Next steps, in order
1. Check if the whole repo is fine and if anything has to be changed
2. Redo experiments and confirm findings and reports
3. Decide on the next steps and possibly more question.


## Working conventions (still apply)
One component at a time, tests before moving on. PowerShell/VS Code local
+ Colab T4 for GPU. Wilson CIs via `src/eval_stats.py`'s `rate_with_ci` —
reuse it, don't reimplement. Don't relitigate locked design decisions
without a specific new reason.