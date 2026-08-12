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
Components 1-4 done and closed out. Component 5 (causal ablation) has one
result in hand; one refinement attempt is in flight (narrower-layer
ablation + paired McNemar's test). After that: Phase 5 (write-up + repo
cleanup), no more new experiments planned.

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

**C5 — Causal ablation (first attempt, layers 14-28, full projection
removal).** Quadrant C soft-deflection 80%→0% under ablation — but
quadrant A's legitimate refusal also collapses 14%→0%, and B's small
signal disappears too. Real causal effect on refusal-related behavior,
but NOT selective — can't yet claim this specific direction is uniquely
responsible for the neutral-wording effect. Verified consistent between
the printed transcript output and the committed
`causal_ablation_summary.json`.

## Current overall verdict (honest, not a clean binary)
Post-training doesn't appear to create a new, DPO-specific safety
representation from scratch — sensitivity to neutrally-worded harm is
already present after generic instruction-tuning (M1). DPO does
measurably reshape (not just amplify) the refusal-associated direction
more than SFT-safety does, and changes how strongly that representation
converts into refusal behavior. Closer to "coupling/amplification"
(Hypothesis B) than "genuinely new representation" (Hypothesis A), with
real nuance, not a clean binary result.

## Repo state / sync warnings (real, recurring issue — check every session)
Three independent sessions have each found GitHub behind local/Colab
state at different points. As of this writing: Components 1-3 should be
on `main`; Components 4-5 (refusal direction, causal ablation) are on
`phase4-wip`, not yet merged — merging is the #1 next action. Always
verify actual repo contents before trusting any log, including this one.
Known hygiene debt, not urgent but real: ~56MB of committed smoke-test
binaries under `outputs/smoke_test_m1/`.

## Next steps, in order
1. `git checkout main && git merge phase4-wip && git push` — do this first, unblocks everything else.
2. Run `src/mcnemar_causal_ablation.py` (local, no Colab, uses data you already have) — paired significance test on the existing ablation result.
3. One narrower-layer ablation attempt (layers 24-28 instead of 14-28, per Component 4's finding that DPO's effect concentrates deep) — needs one more Colab GPU run. Time-boxed: if it's not selective either, stop and write up the honest negative result — don't keep tuning.
4. Phase 5: update PROJECT_CONTEXT.md's decision log with C4/C5 findings, reorganize results/ (behavioral/, probes/, refusal_direction/, causal_ablation/, figures/), remove the smoke-test binaries from git, rewrite README as the scientific story, then the actual write-up.

## Working conventions (still apply)
One component at a time, tests before moving on. PowerShell/VS Code local
+ Colab T4 for GPU. Wilson CIs via `src/eval_stats.py`'s `rate_with_ci` —
reuse it, don't reimplement. Don't relitigate locked design decisions
without a specific new reason.