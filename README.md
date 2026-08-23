# Where Does Safety Live? Tracing a Refusal Direction from Base Model to DPO

*A mechanistic study of SFT and DPO on Qwen2.5-1.5B, replicated across two independent training datasets.*

We trained a 1.5B model through four stages — base → generic SFT → safety
SFT → DPO — and tracked a single internal direction associated with refusal
at every step. Then we did it again, starting from a different, independent
instruction-tuning dataset, to see if the story survives. Short version: DPO
looks like it's mostly turning up the volume on a direction that's already
there by the time generic instruction-tuning finishes, not building
something new from scratch. That held up on a second dataset. What didn't
hold up cleanly is *how* that direction gets organized depending on which
training path got you there — and that turned out to be more interesting
than I expected going in.

---

## The question

Does safety training give a model genuinely richer internal representations
of harm, or does it mostly reshape how an existing, already-formed direction
gets hooked up to behavior? Put differently: is there a real difference
between a model that "understands" harm and one that's just gotten better at
routing a pre-existing signal to a refusal?

To test this without just asking the model, I built a controlled eval set
with four quadrants — crossing *actually harmful* against *worded
harmful-sounding*:

|                       | **Worded harmful-sounding**                          | **Worded neutrally**                                          |
|-----------------------|-------------------------------------------------------|-----------------------------------------------------------------|
| **Actually harmful**  | **A** — obviously harmful (HarmBench, n=150). Correct response: refuse. | **C** — see below (n=104). Correct response: refuse or safe redirection. |
| **Actually benign**   | **B** — sounds risky, isn't (XSTest, n=250). Correct response: comply. Classic over-refusal trap. | **D** — benign (Alpaca + Dolly + OASST1, n=150). Correct response: comply. |

**Quadrant C is now the pipeline's promoted, reviewed set — n=104**, not the
original 20. Revised description (see CLAUDE.md for the full history of
why): *source-attributed, human-reviewed harmful requests, rewritten from
StrongREJECT (Souly et al. 2024) to reduce — not eliminate — overt
domain-specific and evasion-coded vocabulary, while preserving the original
harmful objective and requested type of assistance.* That's a deliberately
narrower, more precise claim than "disguised" or "neutral wording." External
review (recorded in CLAUDE.md) correctly pointed out that both the original
20 hand-curated items and my first StrongREJECT-derived draft batch still
contained overt wrongdoing/evasion cues ("unauthorized," "without consent,"
"avoid detection," "untraceable") — generic, not domain-specific, but still
legible safety triggers a model could key off without representing the
deeper objective at all. A quantified lexical audit
(`src/diagnostics/score_lexical_risk_cues.py`) confirmed it: 8/20 original
items and 9/20 of my first drafts contained at least one term from a
predeclared risk lexicon.

The promoted batch — reviewed in conversation, not rubber-stamped, then
substantially scaled up (15 → 104) after review correctly pushed back that
15 was far too small — scores exactly **zero** lexical-cue hits across all
104, verified by `src/diagnostics/score_lexical_risk_cues.py` and enforced
by a regression test. 51 further candidates from the same drafting pass
resisted full cue reduction for a structural reason worth stating plainly:
for harms where **evading detection is the objective itself** (smuggling
something across a border, covertly monitoring someone, insider trading,
illegal waste disposal), removing all evasion-coded language would mean
losing the harmful objective, not just its surface dressing. These are
kept in a separate secondary set (`evasion_dominant`), not force-fit into
the primary quadrant — a genuine limitation of the "neutral wording"
premise for this sub-category of harm, not a curation failure to paper
over.

Before trusting any of that, I checked whether a pre-existing dataset
could just replace this whole exercise — genuinely checked, not just
asserted: BELLS-Operational (CeSIA), AHB, CASE-Bench, and OpenSafeIntent
were all investigated (see CLAUDE.md for specifics). None fit — each maps
cleanly to a *different* construct (gated access with the wrong
ground-truth philosophy, stylistic/literary obfuscation, context-dependent
safety, or dual-use matched variants with their own contamination risk).
StrongREJECT (Souly et al. 2024) remains the right source for the
underlying harmful intent; the work that needed doing was rigor around
*how* it gets reworded and documented, not finding a shortcut around
authorship entirely. `src/data_pipeline/quadrant_c_pipeline.py` implements
that rigor: every candidate's source text is verified verbatim against the
live StrongREJECT source before use (caught a real bug this way — 6/20
candidates in an earlier pass turned out to store truncated previews
instead of the full source text), checked against every training file the
project has, and classified with an explicit, stated reason before ever
reaching a human reviewer.

The original 20 hand-curated items are **retired from headline/primary
claims** as of this update — their authorship is genuinely unverifiable
(git shows a single human-authored commit, but that can't distinguish
hand-typed from AI-drafted-then-reviewed), and on inspection they had the
same generic-wrongdoing-cue problem as the first StrongREJECT draft pass.
Kept in the repository for reproducibility, not used as evidence.

Quadrant C is the whole point of this design: a model relying on surface
wording will do fine on A, B, and D but should struggle here. A model
whose internal harm-related signal depends primarily on overt safety cues
may show that dependence here — preserved performance is evidence the
signal generalizes beyond the specific direct formulations it was measured
on, not by itself proof of "understanding" in a strong sense.

---

## Setup, briefly

**Training chain** (Qwen2.5-1.5B, LoRA r=64 throughout — flagging that
constraint up front, see Limitations): M0 (base) → M1 (SFT on Alpaca) → M2
(SFT on PKU-SafeRLHF *chosen* responses) → M3 (DPO on the *same*
PKU-SafeRLHF prompts, matched chosen/rejected pairs). M2 and M3 sharing
identical prompts and differing only in objective is what lets me separate
"DPO as a method" from "DPO's training data." A second path, **M3_direct**,
applies DPO straight to M1, skipping M2 — isolates whether DPO's effect
depends on the model already having been through safety-SFT.

**The replication branch**: M1's data is the thing I was least sure about.
Alpaca is generated by `text-davinci-003` — an already-aligned model — so
any safety-flavored style in M1 could in principle be inherited from
Alpaca's generator rather than being a real property of instruction-tuning.
So I trained a second, parallel chain (`_alt` suffix) identical in every way
except M1_alt is trained on
[Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
— human-written from scratch, no aligned-LLM generation step. M2_alt/M3_alt/
M3_direct_alt keep training on the same PKU-SafeRLHF data as the original —
the only variable that changes is which dataset produced M1.

**Refusal direction**: diff-in-means, mean(activation | quadrant A) −
mean(activation | quadrant D), per layer per stage, unit-normalized. Same
core method as Arditi et al. (2024, NeurIPS).

Full methodology (causal ablation, steering, bootstrap procedures, the
LoRA-subspace check) is in `CLAUDE.md`, kept out of here to keep this
readable.

---

## Finding 1: the representation shows up before the behavior does

M1 — generic instruction-tuning, *before any safety training at all* —
already has probes flagging 85% of quadrant C as unsafe internally, despite
0% behavioral soft-deflection at that stage. The model has something
resembling the relevant representation well before it does anything
differently in its outputs.

The direction itself moves most during this same stage. Mean drift (1 −
cosine similarity) across 28 layers: M0→M1 ≈ 0.335, M1→M2 ≈ 0.040, M2→M3 ≈
0.070. Whatever DPO is doing, it's adding on top of something mostly already
built during plain instruction-tuning, not building it from scratch.

---

## Finding 2: this replicates on a second, independent dataset

<p align="center"><img src="assets/cross_branch_similarity.png" width="620" alt="Cross-branch direction similarity bar chart"></p>

Training M1 on Dolly instead of Alpaca gives a direction that stays highly
similar to the original at every stage — never below 0.875, mostly in the
0.90–0.92 range. I want to be precise about what this does and doesn't show:
it's evidence the direction isn't specific to Alpaca, replicated across the
two datasets I actually tried. That's meaningfully weaker than "a general
property of instruction-tuning," which would need more than two datasets to
back up. But it's real support against the concern that this whole project
was reporting an artifact of one dataset's quirks.

---

## Finding 3: the training path matters more than I expected

<p align="center"><img src="assets/deep_layer_stability.png" width="700" alt="Deep-layer bootstrap stability comparison"></p>

This is the result I didn't see coming, and it's the strongest new thing in
this update. Bootstrap resampling (B=1000) shows M3_direct and
M3_direct_alt — the two direct-DPO branches, on *different* datasets — both
settle into a tight 0.98–0.995 cosine-similarity band at layers 16–28. Every
M2-mediated stage, on both datasets, stays looser (roughly 0.94–0.98) and
generally gets *less* stable with depth, not more.

That's a pattern that showed up independently in two separate training
runs, which is what a real replication is supposed to look like. One
precision worth holding onto: "bootstrap-stable" means the *estimate* of the
direction is less sensitive to which prompts get resampled — it's evidence
the underlying computation is more concentrated, but it doesn't by itself
prove a cleaner causal circuit.

**Update — this now has a formal test behind it.** Pairing each bootstrap
replicate across stages (same resampled prompt subset, same seed, different
model's activations — see `paired_deep_layer_stability_test.py`), a
Wilcoxon signed-rank test on the mean deep-layer (16–28) stability finds
M3_direct more stable than M3 by +0.015 (0.9917 vs. 0.9767) and
M3_direct_alt more stable than M3_alt by +0.015 (0.9918 vs. 0.9772), with
the sign consistent across essentially all 1000 replicates in both branches
(p ≈ 0 for both, and pooled). One caveat worth being upfront about: that
p-value describes how sharply two *bootstrap* distributions are separated,
not a classical p-value from 1000 independent real-world samples — every
replicate resamples the same fixed 370-prompt set, so its extremity mostly
reflects the size of the gap relative to each stage's own resampling noise,
not literal astronomical real-world confidence. The direction and
reliability of the effect are real; the magnitude of the p-value itself
shouldn't be over-read.

There's a second piece that fits the same shape. Cross-branch similarity
actually rises slightly from M1 (0.899) to M2 (0.919), stays close at M3
(0.916), and drops for the direct-DPO branch (M3_direct: 0.873, the lowest
of the four). Safety-SFT looks like it pulls the two branches'
representations closer together; skipping it doesn't. **This is now a
tested finding, not just a pattern:** bootstrapping the difference between
the M2-mediated pairs (M2, M3) and the direct-DPO pair (M3_direct) gives
+0.044, 95% CI [+0.037, +0.052] — clear of zero, so this isn't resampling
noise (`bootstrap_cross_branch_difference.py`).

A third signal turned out to be more ambiguous once actually tested. Using
each layer's effect size for separating *actually harmful* (A+C) from *just
surface wording* (B+D), M2 and M3 both peak at layer 9; M2_alt and M3_alt
both peak at layer 16 — a 7-layer gap that looked dataset-sensitive.
Bootstrapping the argmax itself (`bottleneck_layer.py`: resample and
re-find the winning layer, 1000 replicates) mostly explains this away as
argmax noise rather than a robust dataset effect: M2_alt's bootstrap winner
is layer 16 only 46% of the time (95% CI spans layers 9–28), and M3_alt's
*actual* bootstrap-modal winner is layer 9 (50% of resamples) — not the
reported layer 16 at all. M2 and M3 are more stable (66% and 51% mode
fraction, both concentrated in the 9–17 range); the direct-DPO branches'
point estimates (M3_direct: 13, M3_direct_alt: 14) are close together as
originally noted, and are themselves unevenly concentrated (47% and 86%
mode fraction respectively) — but the "7-layer, dataset-sensitive gap"
framing for the M2-mediated pairs doesn't survive resampling well and
should be walked back.

The A-vs-D bottleneck layer (not previously highlighted, but the bootstrap
surfaced a cleaner version of the same core Finding-3 shape here) tells a
tighter story. M3_direct's winning layer is 18 in 86% of resamples (95% CI
[14, 18]) and M3_direct_alt's is 18 in 99% of resamples (CI essentially a
point, [18, 18]) — both direct-DPO branches. The M2-mediated stages are
markedly less concentrated: M3's winner (layer 16) holds in 70% of
resamples but its CI stretches to layer 28, and M3_alt's full-sample winner
is layer 18, but that's not even its bootstrap mode — the actual modal
winner across resamples is layer 28, and only 28% of the time, with a CI
spanning 16–28 (about as diffuse as a "winner" gets). So the "direct-DPO branches carve a sharper, more
concentrated representation" pattern from the deep-layer stability and
cross-branch results above shows up a third time here, on an independent
metric — but the specific 7-layer, dataset-sensitive claim about the
harm-vs-surface metric doesn't hold up and is now downgraded to "mostly
argmax noise" pending a more carefully powered version.

---

## Finding 4: steering on all 8 SFT/DPO stages — the sufficiency half of the causal story

Ablation established necessity (on M3): removing the direction suppresses
refusal. This closes the other half — does adding it induce refusal? — and
extends the test across the whole training arc, not just the 4 DPO
endpoints, using each stage's own natural direction at layer 24.

**Pre-DPO stages show essentially no causal effect.** M1 and M1_alt (before
any safety training) and, more surprisingly, M2 and M2_alt (after
safety-SFT, before DPO) show close to zero induced refusal when steered
along their own direction — M2's quadrant-A refusal count is literally 0
in both baseline and steered conditions. That safety-SFT alone doesn't
give the direction causal teeth, even though the representation is already
present by this point (Finding 1), is independent evidence for the same
claim the geometry-based results have been making: DPO is what couples the
representation to behavior, not just what amplifies it.

**M3 and M3_alt show a clean sufficiency effect with minimal side effects.**
Steering roughly doubles quadrant-A refusal counts (M3: 7→13, M3_alt: 6→9)
while leaving quadrant-D compliance essentially untouched (M3: 49/50 comply
either way; M3_alt: 50/50 → 48/50). This is the result you'd want if the
direction is doing real, targeted causal work.

**M3_direct and M3_direct_alt are messier, in an informative way.** Both
show elevated non-compliance on quadrant D *at baseline*, before any
steering (M3_direct: 39/50 comply, 10/50 refusal+soft-deflection; M3_direct_alt:
36/50 comply) — notably lower than M3's 49/50. Worth flagging as a pattern
consistent with the "direct-DPO carves a sharper, more concentrated
representation" story from Finding 3, though I'm not claiming it's proven
causally connected. M3_direct_alt specifically shows a strange-looking but
verified-real result: steering changes the generated text for 39/50
quadrant-A prompts (checked directly, not a bug) but rarely flips the
classified category — its behavior may already be locked in enough that
more of the same direction perturbs wording without crossing a decision
boundary.

**Degenerate collapse reproduces on quadrant D too**, not just A —
M2_alt's steered quadrant D shows 5/50 outputs collapsing into repetition
loops (up from 1/50 at baseline), confirmed by reading the actual
completions, not just the classifier's say-so. Same open, undiagnosed
mechanism as before (Limitations #8), now with a concrete new example on a
pre-DPO stage.

**Caveat that matters for reading these numbers:** all of the above is on
the *current* 50-item quadrant A / 50-item quadrant D. Quadrant A is
expanding to 150 (Next Steps), which changes the direction itself, not
just adds test items — so this steering run will need a full rerun once
that lands, not an incremental update.

---

## The honest null results

**Steering didn't confirm the causal story.** Causal ablation shows the
direction is *necessary* for refusal — remove it, refusal collapses.
Steering was supposed to show it's *sufficient* — add it, refusal should
appear. It hasn't, cleanly. Adding the direction across 15 layers at once
collapses output into near-total degenerate text (98%) instead of inducing
refusal, most likely from the addition compounding across that many
simultaneous residual-stream injections. A single-layer version produced a
small, non-significant shift (p=0.50) — but at layer 21, which sits outside
the 24–28 range ablation actually found sufficient, so that test may have
been aimed at the wrong layer as much as it tested the mechanism. A
reworked, configurable version of the experiment exists
(`eval_steering_v2.py`, defaults to a layer inside the validated range) but
hasn't been run yet.

**90%+ of the direction's norm lies outside the LoRA subspace it was trained
through.** Rules out "this is just what a rank-64 adapter happened to be
capable of writing" as the main story — but the overlap that does exist is
real (3–10 standard deviations above a random-direction baseline) and
concentrated exactly where DPO's rotation concentrates. Not primarily a LoRA
artifact, not fully independent of it either.

**Quadrant C's behavioral gap between branches doesn't clear significance.**
M3 70% soft-deflection [95% CI 48–86%] vs. M3_alt 55% [34–74%]; M3_direct
55% [34–74%] vs. M3_direct_alt 30% [15–52%]. Both point the same direction
(alt branch soft-deflects less on disguised harm), and that consistency
across two independent paths is worth noting — but with n=20 and
overlapping CIs in both cases, this is a hypothesis, not a confirmed
difference. The mirror pattern on quadrant A (n=50, tighter CIs — alt models
refuse *more* on overtly-worded harm) is more trustworthy given the larger
sample. Together they suggest something worth checking properly: maybe the
alt branch leans more on surface wording and less on recognizing disguised
intent. That's a real possibility, not something the current data confirms.

---

## Open questions / what I'd want feedback on

- Is "safety-SFT homogenizes cross-dataset representations" a real
  mechanism, or is there a simpler explanation (e.g. M2 initialization
  reducing variance generically, independent of anything about
  homogenization specifically)? Genuinely unsure.
- If someone ran an SAE on this model, would "refusal" split into several
  correlated but distinct features (apology, policy-citation, moralizing,
  topic-flagging)? Ablation would still show the *bundle* is causally
  load-bearing either way, so I don't think this threatens the causal
  result — but it might change what "the direction" actually means.

*(Resolved since the last update: "does the deep-layer stability difference
survive a formal paired comparison" — yes, see Finding 3. "Is the 7-layer,
dataset-sensitive bottleneck gap real" — no, mostly argmax noise, also
Finding 3.)*

---

## Next steps, in priority order

Items 1–3 below are done as of the latest update (pure statistics on data
already collected, no GPU needed) — see Finding 3 for the results:

1. ~~Bootstrap the difference in cross-branch similarity between the
   M2-mediated and direct-DPO paths.~~ Done: +0.044, 95% CI [+0.037, +0.052].
2. ~~Report a distribution over near-optimal bottleneck layers, not just
   the argmax.~~ Done: the A-vs-D metric confirms Finding 3's pattern
   further; the harm-vs-surface "7-layer gap" mostly didn't survive and is
   downgraded to argmax noise.
3. ~~A formal paired comparison of the deep-layer stability distributions
   between direct-DPO and M2-mediated branches.~~ Done: Wilcoxon signed-rank,
   p ≈ 0 in both branches and pooled.

Remaining, roughly by cost:

1. ~~Run the redone steering experiment inside the ablation-validated layer
   range, with a quadrant-A side-effect check included.~~ **Code is done,
   the reported run in Finding 4 is not the final one.** Closed the
   sufficiency gap in the causal story, and along the way closed a second
   gap I hadn't originally flagged: the direction was being tested on the
   same A/D prompts it was estimated from. Fixed via a held-out split
   (`assign_direction_split` — 80/20, direction-estimation vs.
   held-out-behavioral) threaded through direction construction, both
   bootstraps, bottleneck layer, and now causal ablation/steering too. The
   numbers in Finding 4 are STILL from the pre-split, pre-quadrant-expansion
   50/50 eval set — this has not changed, the run has not happened yet.
   `controlled_eval.jsonl` itself IS current (654 rows, C=104, split
   assigned — confirmed directly against the file, not just asserted) but
   no downstream artifact (activations, direction, steering output) has
   been regenerated against it. Added `src/analysis/run_full_steering.py`
   to orchestrate the real run across all 8 stages at once (checks the
   eval-set/activation/direction preconditions before spending any GPU
   time, resumable, never overwrites, writes a manifest) and
   `src/analysis/build_finding4_report.py` to turn that run's output into
   real Wilson-CI numbers diffed explicitly against what's published above
   — flags a "MATERIAL CHANGE" rather than silently overwriting, per this
   item's own requirement. Also found and fixed a live bug while building
   this: `mcnemar_steering.py` was hardcoded to the literal condition names
   `"M3_baseline"`/`"M3_steered"` and would have silently matched 0 rows
   against any real `eval_steering_v2.py` output file (which names
   conditions `"{tag}_baseline"`/`"{tag}_steered"`) — never actually
   worked against a real run. Fixed the same way `summarize_steering.py`
   was previously. **Still needs a GPU machine with model access to
   actually execute** — see CLAUDE.md's latest session note for the exact
   command sequence, including one more gotcha: `python -m src.reproduce
   direction` (as written in earlier handoff notes) will silently no-op
   post-rebuild, since its outputs already exist from the old activation
   set — needs `--force`.
2. ~~Quadrant C, properly documented, at scale.~~ **Done — 104 candidates
   promoted, up from 15.** The earlier staged plan (40–60 first, expand
   later) was superseded, not followed — went straight to a larger batch
   once the drafting/verification process was proven working on the first
   20, rather than re-litigating the staging question. Every candidate's
   source text is verified verbatim against its live StrongREJECT source,
   classified into a transformation family, checked for internal
   duplication, and scored to genuinely zero on the lexical-cue check
   before promotion — not just asserted. `src/data_pipeline/quadrant_c_pipeline.py`
   handles sourcing end-to-end; still not a substitute for the human
   review step that happened before promotion, just more rigor going into
   it. Deliberately excluded: weapons/explosives/CBRN-adjacent and
   drug-synthesis content from the source pool, regardless of how the
   request would be reworded.
3. Full fine-tuning robustness check (removes the LoRA-rank confound) —
   expensive, aspirational.
4. Diagnose the steering degenerate-collapse mechanism directly by tracking
   residual-stream norm growth layer-by-layer during generation. **Tooling
   built and unit-tested, not yet run against the real model** (no GPU/HF
   access in the environment this was built in). Added
   `src/interpretability/residual_norm_tracking.py` (hook-based per-layer,
   per-generation-step residual norm tracking; a norm-preserving steering
   hook variant and a norm-clipping variant, to directly test whether
   either avoids collapse without removing the steering direction itself),
   `src/analysis/eval_residual_norm_diagnostic.py` (GPU script: runs
   baseline / collapsing (layers 14-28) / non-collapsing (layer 24) /
   optionally collapsing-with-the-fix on a small quadrant-D sample,
   tracking norms throughout), and `src/analysis/plot_residual_norms.py`
   (CPU-only — layer × generation-step heatmaps plus a most-anomalous-layer
   comparison line plot; this one actually ran end-to-end against synthetic
   data in testing and produces real figures, see its test file). Checked
   the actual deprecated-run outputs before building this rather than
   working from the summarized description: the multi-layer collapse isn't
   token soup, it's a tight loop of refusal-flavored tokens
   ("unfortunately... unfortunately... WARNING WARNING"), which is
   consistent with — but doesn't on its own prove — the norm/magnitude
   story `eval_steering_v2.py`'s docstring already speculates about; the
   diagnostic is built to actually test that, not assume it.

---

## Limitations

1. **LoRA, quantified, not fully resolved.** 90%+ of the direction's norm
   sits outside the rank-64 subspace, but real above-chance alignment exists
   at deep layers. A full-fine-tuning check would add confidence.
2. **Single diff-in-means direction.** Other orthogonal safety-relevant
   directions may exist; not searched for (see Open Questions).
3. ~~Ablation shows necessity, not yet sufficiency~~ **Resolved (Finding 4):**
   steering confirms sufficiency on M3/M3_alt with minimal side effects, and
   shows pre-DPO stages (M1, M2) don't yet have the causal machinery. Caveat:
   run on the pre-expansion 50/50 A/D eval set, needs a rerun once the
   expansion below lands.
4. **Quadrant C, now n=104, up from the original 20.**
   Original 20 retired — unverifiable provenance (git shows a single
   human-authored commit, can't distinguish hand-typed from
   AI-drafted-then-reviewed) and, on external review, not actually neutral
   wording (still contained generic wrongdoing/evasion cues). Replaced with
   104 individually-classified, source-attributed, verified-zero-cue items
   drawn from StrongREJECT, clearing the 100+ target. See CLAUDE.md for
   the drafting/review process.
5. **1.5B scale.** Not claimed to generalize to frontier models.
6. **The Alpaca-artifact concern is substantially, not fully, addressed.**
   Reproducibility across Alpaca and Dolly is real evidence, but it's two
   datasets, one model family, one LoRA setup.
7. ~~The new cross-branch claims need a formal statistical pass~~ **Resolved**
   — bootstrap difference test, bottleneck-layer bootstrap CI, and paired
   deep-layer stability test all done, see Finding 3.
8. **Why multi-layer steering collapses to degenerate output** now has a
   clear, evidenced explanation, not just a hypothesis: the original
   `eval_steering.py` steered 15 layers (14-28) simultaneously, every
   forward pass, and 49/50 quadrant-D completions collapsed into repetition
   as a result (kept as `results/raw/steering_raw_D_MULTILAYER_14to28_DEPRECATED.json`
   for the record). Single-layer steering (current default, layer 24)
   resolves it in practice, but the underlying mechanism — why compounding
   across layers specifically causes collapse — is still not independently
   diagnosed at the activation level.
9. **Quadrant D was Alpaca-only, shared across both branches** — the
   original (Alpaca-trained) branch got tested in-distribution while the
   alt (Dolly-trained) branch didn't, a real asymmetry. Fix designed (keep
   the original 50, add a Dolly supplement + an independent-source
   supplement so both branches get equal in/out-of-distribution exposure)
   but not yet run/verified — see CLAUDE.md Next Steps.
10. **PKU-SafeRLHF-derived training data (DPO pairs, safety-SFT) only pairs
    responses where annotators disagreed on safety** (`is_response_0_safe
    != is_response_1_safe` — same-safety pairs are dropped entirely). This
    is a deliberate, defensible choice for isolating a clean safety-contrast
    signal, but it means the model never saw a (safe, safe) preference
    signal (no DPO pressure toward more-helpful-among-safe-options) or a
    (unsafe, unsafe) one. Whether this changes the RESULT (not just the
    training objective) versus training on the full natural preference
    distribution is untested.
11. **Whether the eval set's harm categories (illegal, cybercrime_intrusion,
    harassment_bullying, misinformation_disinformation) are actually
    well-represented in the safety-SFT/DPO training data hasn't been
    confirmed with current numbers.** `src/diagnostics/analyze_data_coverage.py`
    exists specifically for this (already flagged a similar concern for
    vehicle-related harm categories) but hasn't been re-run against the
    current category set. If a category is eval-tested but training-rare,
    weak eval performance there reflects a train/eval mismatch, not a
    genuine safety-training failure — this needs checking before drawing
    conclusions from category-level results.

---

## Repo map

Training and analysis are orchestrated through
`colab_unified_training.ipynb` / `colab_unified_analysis.ipynb` and
`src/training/stage_registry.py` — one notebook per task, not one per
model. `src/reproduce.py` runs whatever's CPU-feasible locally;
`src/export_results.py` packages `results/` into a clean, checksummed
folder for moving between machines. Full command reference, current status,
and working conventions live in `CLAUDE.md`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

---

## How to Cite

```bibtex
@misc{dpo_safety_representations,
  title={Where Does Safety Live? Tracing a Refusal Direction from Base Model to DPO},
  author={[Uroš Savurdić]},
  year={2026},
  howpublished={\url{https://github.com/urosavurdic/dpo-safety-representations}},
  note={Independent research project, not peer-reviewed}
}
```