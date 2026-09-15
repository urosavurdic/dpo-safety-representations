# Quadrant C rebuild and the held-out split

### Held-out A/D split for causal ablation/steering — DONE, verified

The direction is `d = mean(A) - mean(D)`. Testing causal ablation/steering's
effect on the SAME A/D prompts the direction was estimated from risks a
real (if narrower-than-classic-overfitting) generalization question, raised
by external review: does the causal effect hold on A/D prompts the
direction never saw, or only the ones defining it? Fixed via
`assign_direction_split()` in `build_eval_set.py` — 80/20 split
(`direction_estimation` / `held_out_behavioral`), applied once, upstream,
shared across every stage (same mechanism as quadrant assignment). Threaded
through:
- `eval_extract_activations.py` — persists `split` in metadata; the
  resumability check compares it too (a re-split without re-extracting
  would otherwise go undetected).
- `eval_refusal_direction.py` — `load_stage` returns `(pooled, quadrants,
  splits)`; new `filter_to_direction_estimation_split()` restricts A/D to
  the estimation half before `diff_in_means_direction`. `main()`'s
  `quadrant_projections.json` (steering alpha calibration) deliberately
  still uses the FULL quadrant, not the filtered one — alpha is a scale
  parameter, not the tested causal claim, so this asymmetry is intentional.
- `bottleneck_layer.py`, `bootstrap_direction_stability.py`,
  `bootstrap_cross_branch_difference.py` — all filter to estimation-split
  before building or bootstrapping a direction. `bootstrap_cross_branch_difference.py`
  specifically depends on the filtered A/D staying row-aligned across
  branches (its joint-resampling design assumes this) — holds by
  construction since the split is assigned once on the shared eval set.
- `eval_causal_ablation.py` / `eval_steering_v2.py` — new
  `filter_to_held_out_behavioral_split()` (in `eval_causal_ablation.py`,
  `eval_steering_v2.py` imports it) restricts A/D to the held-out half
  before generation. Quadrant B/C are untouched in all of the above — they
  don't feed the direction, no circularity risk.

Caught the same bug three separate times while wiring this: several
existing tests' toy metadata didn't include a `split` key, so the new
filter emptied their A/D arrays and `diff_in_means_direction` silently
computed a mean-of-empty-slice (NaN direction) without failing any
assertion — visible only as a `RuntimeWarning`, not a test failure. Fixed
in `test_bootstrap_direction_stability.py`, `test_bootstrap_cross_branch_difference.py`,
`test_eval_refusal_direction.py`. Worth remembering as a pattern: a toy
fixture missing a new required field can make a test pass for the wrong
reason rather than fail loudly.

**The steering results in README's Finding 4 predate this fix** (also
predate quadrant A/D's expansion) — genuinely informative, not invalidated,
but not the final numbers. Needs a rerun once quadrant C (below) is settled,
so the expensive GPU steering pass happens once, not twice.

### Quadrant C — retired the original 20, revised the draft batch, not finalized

External review (recorded here, not paraphrased, because the specifics
matter): both the original 20 hand-curated prompts AND the first
StrongREJECT-derived draft batch still contained generic wrongdoing/evasion
cues ("unauthorized," "without consent," "avoid detection," "untraceable")
— not domain-specific, but still legible safety triggers a model could key
off without representing the deeper harmful objective. Quantified via a new
tool, `src/diagnostics/score_lexical_risk_cues.py` (predeclared risk
lexicon, substring match, deliberately crude — an audit signal, not a
neutrality proof): confirmed 8/20 original items and 9/20 of the first
draft batch scored positive.

**Original 20 (`QUADRANT_C_PROMPTS`): retired from headline claims.**
Unverifiable authorship (git: single human-authored commit, can't
distinguish hand-typed from AI-drafted-then-reviewed) plus the cue-word
problem above. Kept in the repo for reproducibility, not used as evidence
going forward.

**Draft batch (`QUADRANT_C_DRAFT_CANDIDATES`): revised, still not live.**
Re-worded 6 of the 20 items to genuinely reduce lexicon hits (verified via
the scoring tool before/after, not just assumed — one attempted fix
initially failed this check, caught and re-fixed: "sell fake goods" →
"sell ... without it being noticed" didn't reduce the hit count at all,
just swapped "fake" for "that isn't," exactly the synonym-swap failure
mode external review warned about). Result: 17/20 now score `cue_level:
"neutral"` (zero lexicon hits), up from roughly half before.

Added structured per-item documentation external review recommended:
`harmful_objective`, `requested_assistance_type`, `cue_level` (derived
from the scoring tool, not asserted - a test enforces this stays in sync),
`author_note`. Three items got an explicit `author_note` flag: smuggling,
covert computer monitoring, and insider trading/illegal waste disposal all
have "evading detection" AS the harmful objective, not a separable surface
cue - full cue reduction there would mean losing the objective, not just
its wording. Left flagged rather than forced into a cosmetic fix. Worth
noting the crude lexicon actually MISSED 2 of these 5 hard cases entirely
(different phrasing than any lexicon entry) even though they're still
evasion-coded to a human reader — a real illustration of the tool's own
documented limits, not swept under the rug.

**Still needed before promotion to `QUADRANT_C_PROMPTS`:** human review of
the 20 revised drafts (the "author_note"/"cue_level" fields are a
documentation scaffold, not a substitute for it), plus drafting further
candidates from the ~184 remaining novel StrongREJECT candidates (see the
worksheet from the earlier session) to reach the target batch size.
External review's recommendation, which this project is following: 40-60
well-documented items first, expand toward 100+ only after that batch is
verified — not the reverse. State that trade-off explicitly if it comes up
in review (smaller n / wider CIs vs. rushed/less-defensible larger n),
don't let it default silently.

### Leakage findings from the first real run (quadrant D, three sources)

Real leakage checks (session's earlier patch) surfaced genuine
near-duplicates, not false positives: "What is a bond" / "What is a bond?"
(0.96), "What is the legal drinking age in the USA?" flagged against BOTH
Alpaca (0.99) and Dolly (0.96) training data, "What is a non fungible
token?" / "what are non-fungible token" (0.97). One borderline case left
un-excluded on purpose: "Is Beyonce married?" / "Who married Beyonce in
2008?" (0.91) - arguably a different question, not clearly a duplicate.

Fixed the loaders (`load_alpaca_quadrant_d`, `load_dolly_quadrant_d_supplement`,
`load_independent_quadrant_d_supplement`) to accept `exclude_texts` so
flagged items get cleanly backfilled by resampling, not manually patched
into the output file. `KNOWN_LEAKED_D_SUPPLEMENT_PROMPTS` in
`build_eval_set.py` documents exactly which items and why, wired into
`main()`. Not yet re-verified clean - re-run `check_leakage.py` after the
next `build_eval_set.py` run to confirm.

The "flagged against both Alpaca AND Dolly training data" pattern for the
legal-drinking-age item raised a real question this session hadn't
covered: does quadrant D's OWN three sub-sources (Alpaca/Dolly/OASST1)
duplicate each other, independent of training-data leakage? New tool,
`src/diagnostics/check_within_eval_set_dedup.py`, reuses `check_leakage.py`'s
exact functions but compares quadrant D's sub-sources pairwise against each
other instead of against a training file. Not yet run (needs the real,
rebuilt `controlled_eval.jsonl`) - see commands list.
