# The 8-stage steering run and the collapse diagnostic

Worked from a handoff describing two Next Steps items. No GPU and no
HuggingFace Hub network access in this environment (confirmed directly -
`AutoTokenizer.from_pretrained` 403s against api.anthropic.com's egress
proxy, matching every prior session's documented sandbox limitation), so
nothing below includes a real experimental number - this is tooling,
tested against toy/synthetic/CPU-only data, same as this project's
established pattern for agent sessions without GPU access. torch/
transformers/peft/etc. WERE installed here (pip has no such restriction,
only egress to huggingface.co does) to get real import-level test
coverage instead of guessing - baseline before any of this session's
changes: 254 passed, 6 failed (1 already-documented pre-existing
test-isolation bug in `test_summarize_cross_branch.py`, 2 HF-network
failures, 3 `trl`-version-mismatch failures from installing latest-not-
pinned packages rather than the exact pinned versions - none of these are
this session's concern or doing).

**Verified the handoff's claims against the actual repo before building
anything** (see the note appended above this section) - all checked out:
654-row eval set, split assigned, leakage resolved to 0 exact/1
intentional near-dup. What did NOT check out: CLAUDE.md's own prose
narrating the quadrant-C session said the eval-set rebuild was still
pending, when it had actually already landed in the same commit. Not a
data problem, just a documentation-lag problem - noted above rather than
rewritten, to avoid touching the quadrant-C narrative directly per this
handoff's own constraint about not touching that work.

**Task 1 - orchestrating the real run:**
- `src/analysis/run_full_steering.py`: loops `eval_steering_v2.py` across
  all 8 non-M0 stages, quadrants A+D. Checks three preconditions before
  spending any GPU time on a stage: the live eval set's A/D rows all have
  a `split` key, that stage's `results/activations/{stage}_metadata.json`
  matches the live eval set exactly (byte-for-byte, same check
  `eval_extract_activations.py` uses internally), and that stage's
  direction `.npy` exists. Resumable (skips a stage whose output file
  already exists, `--force` to rerun), writes a manifest to
  `results/manifests/` in the same style `src/reproduce.py` already uses.
  Deliberately has NO torch/transformers import at module level - it only
  shells out to `eval_steering_v2.py` as a subprocess per stage - so it
  stays collectible/testable in a torch-less environment; the eval-set-
  loading logic is duplicated rather than imported from
  `eval_extract_activations.py`, for the same reason that file and
  `eval_causal_ablation.py` already duplicate `load_controlled_eval()`
  from each other instead of sharing it across a torch-importing module
  boundary. `--dry-run` prints the full plan (what would run/skip/block
  and why) without touching a subprocess - this is what got tested here,
  against the REAL current repo state, not synthetic data (see below).
- Found and fixed a real, live, previously-undiscovered bug while building
  the stats step this needs: `mcnemar_steering.py` was hardcoded to the
  literal condition names `"M3_baseline"`/`"M3_steered"` and defaulted
  `--file` to the old exploratory `steering_raw_D_L21.json`. Every real
  `eval_steering_v2.py` output names its conditions
  `"{tag}_baseline"`/`"{tag}_steered"` (e.g.
  `"M3_L24_quadrant_a_projection_coef1_QAD_baseline"`) - the literal-string
  version would have silently matched 0 paired prompts against ANY real
  run, for any stage, ever, no error, just a p-value computed from an
  empty contingency table without complaint. This is the exact same bug
  class `summarize_steering.py` was already fixed for (see that section
  above) - that fix didn't get applied here too at the time. Fixed the
  same way: condition names are now parameters (derived via
  `summarize_steering.find_condition_pairs`, not reimplemented), `--file`
  and `--quadrant` are both required with no default (mirrors
  `bootstrap_causal_effect.py`'s existing convention - pooling quadrants
  A and D under one refusal-rate test would conflate two prompt sets with
  different baseline rates and different intended questions, so there's
  no sane default to silently pick).
- `src/analysis/build_finding4_report.py`: takes a completed run (via
  `run_full_steering.py`'s manifest, or an explicit file list) and
  computes real Wilson-CI stats per stage/quadrant/category (reuses
  `classify_completion`/`rate_with_ci` exactly as
  `summarize_causal_ablation.py` does, doesn't reimplement
  classification), then diffs each number against a hand-transcribed
  snapshot of README's CURRENTLY-PUBLISHED Finding 4 figures
  (`OLD_FINDING4` constant, sourced once from the README text itself -
  several stages only have a qualitative claim in the old text, e.g. "M1:
  close to zero induced refusal" with no exact count, and those are
  reported as "no precise old figure to compare against" rather than
  inventing one to diff). Flags >10-point rate swings as `material_change`
  explicitly, per this Next Steps item's own requirement not to silently
  overwrite. This script computes nothing from imagination - every number
  in its output either comes from a real input file or is the explicitly-
  labeled old snapshot.
- **A real gotcha in the handoff's own literal command sequence**: `python
  -m src.reproduce direction` (no `--force`) will silently do nothing
  useful once activations are re-extracted, because
  `results/refusal_direction/cosine_similarity.json` and the other five
  files that component produces already exist (built from the OLD
  370-prompt activation set, currently committed) - `reproduce.py`'s
  resumability check sees them as "already produced" and skips, no error,
  no warning beyond a one-line "skipping (--force to rerun)" that's easy
  to miss in a longer command sequence. Confirmed directly:
  `already_produced("direction")` returns `True` against this checkout
  right now. **The corrected Task 1 command sequence, GPU machine, in
  order:**
  ```
  python -m src.data_pipeline.build_m1_data --dataset alpaca
  python -m src.data_pipeline.build_m1_data --dataset dolly
  python -m src.analysis.eval_extract_activations
  python -m src.analysis.eval_behavioral
  python -m src.reproduce --components direction --force
  python -m src.analysis.run_full_steering --dry-run          # confirm the plan first
  python -m src.analysis.run_full_steering                    # the real 8-stage run
  python -m src.analysis.build_finding4_report --manifest results/manifests/full_steering_<timestamp>.json
  ```
  Then update README's Finding 4 by hand from that report's output,
  quoting any `MATERIAL CHANGE` lines explicitly rather than silently
  replacing the currently-published numbers - this was an explicit
  requirement of the handoff, and `build_finding4_report.py` computes the
  diff but deliberately does not write to README.md itself (a human should
  read the material-change lines before they go in, not have them silently
  auto-applied).

**Task 2 - collapse diagnostic:**
- Pulled the actual deprecated-run outputs before building anything, not
  just the summarized description: the multi-layer collapse
  (`steering_raw_D_MULTILAYER_14to28_DEPRECATED.json`) isn't token soup,
  it's the model stuck in a tight loop of refusal-flavored tokens
  ("unfortunately... unfortunately... WARNING WARNING"). Consistent with
  (but doesn't by itself prove) the magnitude/compounding story
  `eval_steering_v2.py`'s own docstring already speculates about - a
  distribution-collapse-under-greedy-decoding explanation is also
  consistent with the same observation and isn't mutually exclusive. The
  diagnostic built here is meant to actually test the norm-specific part
  of the hypothesis, not assume it's confirmed just because it's
  plausible.
- `src/interpretability/residual_norm_tracking.py`: `ResidualNormTracker`
  registers a forward hook per decoder layer recording the last token
  position's L2 norm at every forward call (= every generation step,
  under `model.generate()`'s KV-cache path, with "step 0" being the
  post-prefill/end-of-prompt call). `compute_baseline_range` /
  `compare_to_baseline` / `first_step_exceeding_p99` turn a pooled
  unsteered baseline into a per-layer "trained-typical" range (mean/std/
  p50/p95/p99) and then quantify how far a steered run's norms deviate
  from it, and at which generation step the deviation first crosses p99 -
  directly answers "is this already out of range at token 1, or does it
  build up over generation" (different mechanisms, different fixes).
  Also added two alternative steering hooks to actually test a fix rather
  than just describe one: `make_norm_preserving_steering_hook` (injects
  the direction, then rescales the result back to the EXACT pre-steering
  norm - isolates whether magnitude growth specifically, as opposed to
  the direction's mere presence, drives collapse) and
  `make_norm_clipped_steering_hook` (gentler - only rescales vectors that
  exceed a given ceiling, e.g. the baseline's own p99 at that layer,
  leaves normal-magnitude tokens untouched). 17 tests, all pure
  torch-CPU + fake decoder-layer `nn.Module`s (same pattern
  `tests/analysis/test_eval_causal_ablation.py` already uses for
  `get_decoder_layers`/`register_ablation_hooks`) - no real model needed
  to verify the hook math and the tracker's bookkeeping are correct.
- `src/analysis/eval_residual_norm_diagnostic.py`: GPU script, runs
  baseline / collapsing (layers 14-28, uncorrected alpha - replicates the
  historical deprecated config as closely as this script's defaults
  allow) / non-collapsing (layer 24, this repo's current default) /
  optionally `--also-test-fix` (collapsing layer set, but with
  `make_norm_preserving_steering_hook` substituted for the normal additive
  hook - if THIS condition's degenerate rate drops back toward the
  non-collapsing baseline while the uncorrected collapsing condition
  stays high, that's real evidence for the magnitude hypothesis, not just
  a plausible story) on a small (`--n-prompts`, default 8) quadrant-D
  sample, tracking norms throughout via the tracker above. NOT executed
  against the real model here (no GPU/HF access) - import-checked and its
  pure-logic helpers (`build_steering_hooks`, `summarize_config`,
  `build_norm_summary`) unit-tested (7 tests), same split between
  "testable pure logic" and "GPU-only orchestration in main()" that
  `eval_steering_v2.py`/`eval_causal_ablation.py`'s own test files already
  use (they never test `main()` either, for the same reason).
- `src/analysis/plot_residual_norms.py`: CPU-only, no torch. Reads the
  diagnostic script's output JSON, produces a layer × generation-step
  heatmap per condition for one representative prompt, plus a line plot
  at whichever layer showed the single largest z-score against baseline
  across all conditions, with a horizontal reference line at that layer's
  baseline p99. Unlike everything else in this session, this one actually
  RAN end-to-end here, against synthetic data built to match the real
  diagnostic script's exact output schema (7 tests, `tests/analysis/
  test_plot_residual_norms.py`) - produces real, non-empty PNGs (checked
  file size, not just "didn't crash"), and one was visually inspected
  during this session: a baseline line flat near the reference p99, a
  "noncollapsing" line drifting slightly, and a "collapsing" line growing
  roughly linearly and crossing well above the p99 line within a few
  steps - exactly the qualitative shape the real diagnosis should produce
  if the magnitude hypothesis holds, built into the test fixture as the
  scenario to check the plotting code renders correctly, NOT as a claim
  about what the real model will do.
  ```
  python -m src.analysis.eval_residual_norm_diagnostic --stage M3 --also-test-fix
  python -m src.analysis.plot_residual_norms --file results/raw/residual_norm_diagnostic_M3.json
  ```

**Test status after this session**: 317 passed, 3 failed (same 3
pre-existing environment-gap failures as the documented baseline above,
none touched or caused by this session's changes) when running the full
suite minus the 3 `trl`-version-mismatch tests in `test_train_dpo.py`
(latest-not-pinned `trl`, unrelated to this session, not investigated
further since out of scope). 89 new tests added across 6 new/changed test
files, all passing, verified against a genuinely fresh `git clone` of this
repo (not just the working copy changes were made in).

**What this session did NOT do, explicitly**: run any GPU code for real,
produce a single real steering number, or touch quadrant C/the data
pipeline. Both Task 1 and Task 2's actual experimental results still need
a human with a GPU machine and HF Hub access to run the commands above and
look at what comes out.
