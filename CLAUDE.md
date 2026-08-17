# CLAUDE.md — working notes for whoever (human or AI) picks this project up next

This file is the internal working document — design decisions, conventions,
current status, exact commands, and known gotchas. `README.md` is the
polished, portfolio-facing science writeup; if you're about to *do* work on
this repo rather than *read about* it, this is the file to actually follow.

---

## Research question

**Short:** Does DPO give a language model genuinely richer internal
representations of safety, or does it mainly amplify an existing
low-dimensional refusal mechanism, causing over-refusal on ambiguous
prompts?

**Long:** Across a Base → SFT-Helpful → SFT-Safety → DPO training chain, do
internal representations of safe/unsafe/ambiguous prompts become more
linearly separable and semantically richer (Hypothesis A), or does the
model mainly strengthen sensitivity along a pre-existing refusal direction,
pulling ambiguous prompts toward the unsafe cluster (Hypothesis B)? Current
evidence favors B, with real nuance added by the data-dependence extension
(see README).

---

## Core design decisions (locked in — don't relitigate without a real reason)

1. **Training chain:** M0 (Qwen2.5-1.5B-Base) → M1 (SFT-Helpful, Alpaca) →
   M2 (SFT-Safety) → M3 (DPO). M3_direct = DPO applied straight to M1,
   skipping M2. Alt branch (`_alt` suffix) mirrors the whole chain, M1_alt
   trained on Dolly-15k instead of Alpaca, everything downstream keeps
   training on the *same* PKU-SafeRLHF data as the original branch —
   single-variable design, only M1's source dataset changes.
2. **Matched-data design between M2 and M3:** DPO chosen/rejected pairs and
   M2's SFT-safety data both come from PKU-SafeRLHF, same prompts, matched.
   Isolates "DPO the method" from "DPO the data."
3. **LoRA rank confound, quantified not eliminated:** r=64 throughout. The
   LoRA-subspace check (90%+ of the direction's norm is outside the rank-64
   subspace) bounds but doesn't remove this as a limitation.
4. **Template-matching protocol:** M0 has no chat template; M1+ do. M0's
   prompts get wrapped in the same literal template tokens before
   activation extraction, so template surface form isn't a confound.
5. **Controlled 4-quadrant eval set** (`data/processed/controlled_eval.jsonl`,
   370 prompts, fixed — do not regenerate, do not add/remove examples
   without a strong reason and updating every downstream result):
   A=50 (HarmBench), B=250 (XSTest), C=20 (hand-curated, **the power
   bottleneck — see Next Steps**), D=50 (Alpaca reserved).
6. **Stack:** Transformers + PEFT + TRL (`ref_model=None` reference-free
   DPO) + Accelerate. Raw forward hooks for activations (not
   TransformerLens). scikit-learn for probes, scipy/statsmodels for CIs.
7. **Checkpointing:** Git for code, Drive for training checkpoints
   (disposable once pushed to HF — see below), HF Hub for versioned
   adapters, W&B for training metrics.
8. **Smoke-test convention:** fast, CPU-only, tiny-data tests that check
   plumbing, not scientific correctness. Run `pytest tests/ -v` before
   every push and after any `src/` change.
9. **Probe metric:** use `quadrant_c_flagged_unsafe_frac` (held-out flagging
   rate), never `cv_accuracy_mean` to pick a "best layer" — CV accuracy
   saturates near 1.0 everywhere including untrained M0, and picking by it
   silently returns whichever layer ties first (was actually a live bug,
   see "Bugs already found and fixed" below).

---

## Repo structure that matters

```
configs/                  one YAML per training stage + its GPU dry-run sibling
  m{1,2,3}_*.yaml          original (Alpaca) branch
  m3_direct_*.yaml         direct-DPO control
  m{1,2,3}_alt_*.yaml      alt (Dolly) branch
  m3_direct_alt_*.yaml     direct-DPO on the alt branch

src/training/
  train_sft.py, train_dpo.py    config-driven, --config <path>, identical CLI
  model.py                      STAGE_ADAPTER_CHAINS (runtime adapter-merge
                                 order) + load_stage_model / try_load_stage_model
  stage_registry.py             TRAINING_STAGES (training orchestration:
                                 dependency order, config paths, data-prep
                                 requirements) - single source of truth for
                                 the unified training notebook AND src/reproduce.py

src/data_pipeline/
  build_eval_set.py             builds controlled_eval.jsonl (run once, don't rerun)
  build_m1_data.py              --dataset {alpaca,dolly}, builds M1's SFT data
  data_prep.py                  builds PKU-SafeRLHF matched pairs (dpo_pairs.jsonl,
                                 sft_safety.jsonl) - shared by M2/M3/M3_direct
                                 and their alt counterparts

src/analysis/
  eval_behavioral.py            Component 1, all 9 stages, resumable
  eval_extract_activations.py   Component 2, all 9 stages, resumable
  eval_probes.py                Component 3, all 9 stages, resumable
  eval_refusal_direction.py     Component 4 - STAGES (extraction) vs
                                 SEQUENTIAL_STAGES (M0-M3 only, the TRUE
                                 chain) vs ALT_SEQUENTIAL_STAGES vs
                                 CROSS_BRANCH_PAIRS - see docstring, this
                                 file computes cosine_similarity.json's
                                 vs_M0/vs_M3/adjacent/adjacent_alt/
                                 direct_branch/cross_branch sections
  eval_causal_ablation.py       Component 5, --stage (any STAGE_ADAPTER_CHAINS
                                 key), GPU
  eval_steering.py              Component 5b v1, M3/L21 or 14-28 only, the
                                 one with the documented null result
  eval_steering_v2.py           Component 5b v2, --stage/--layers/--alpha-*/
                                 --quadrants, never overwrites, built but
                                 NOT YET RUN
  summarize_causal_ablation.py  --file --stage, Wilson CIs, persists them now
  mcnemar_causal_ablation.py    --file --conditions --quadrant --category
  bootstrap_causal_effect.py    --file --quadrant --category, absolute +
                                 relative effect, 95% CI
  summarize_cross_branch.py     pulls behavioral + probe + direction data
                                 together into one original-vs-alt report

src/interpretability/
  direction_stability.py            reads cosine_similarity.json
  bootstrap_direction_stability.py  B=1000, all layers, all 9 stages
  bottleneck_layer.py               Cohen's d per layer, A-vs-D and
                                     (A+C)-vs-(B+D), argmax per stage

src/reproduce.py            local CPU-only orchestrator (--list / --components)
src/export_results.py       packages results/ into a clean, checksummed export

colab_unified_training.ipynb   config-driven, all 8 stages, one notebook
colab_unified_analysis.ipynb   Components 1-5b, one notebook
```

Note: `colab_*.ipynb` files are NOT git-tracked (matches the project's
existing convention — they live wherever you keep Colab notebooks, e.g.
Drive).

---

## Exact commands

**Training** (Colab, GPU): use `colab_unified_training.ipynb`. Set
`STAGES_TO_RUN` (any subset of `TRAINING_STAGES` keys) and `DRY_RUN`.
Prerequisites are resolved and ordered automatically.

**Full analysis pass** (Colab, GPU + CPU components mixed): use
`colab_unified_analysis.ipynb`. Toggle `COMPONENTS_TO_RUN`; causal
ablation/steering loop over `STAGES_FOR_CAUSAL` automatically.

**CPU-only, local, once activations exist:**
```bash
python -m src.reproduce --list                         # status, runs nothing
python -m src.reproduce --components all                # everything CPU-feasible
python -m src.analysis.summarize_cross_branch            # original-vs-alt report
python -m src.export_results                             # clean checksummed export
```

**Manual component-by-component** (what the notebooks actually call):
```bash
python -m src.analysis.eval_behavioral
python -m src.analysis.reclassify_behavioral
python -m src.analysis.eval_extract_activations
python -m src.analysis.eval_probes
python -m src.analysis.summarize_probe_findings
python -m src.analysis.eval_refusal_direction
python -m src.interpretability.direction_stability
python -m src.interpretability.bootstrap_direction_stability
python -m src.interpretability.bottleneck_layer
python -m src.analysis.summarize_cross_branch
python -m src.analysis.eval_causal_ablation --stage M3
python -m src.analysis.summarize_causal_ablation --file results/raw/causal_ablation_raw_narrow.json --stage M3
python -m src.analysis.mcnemar_causal_ablation --file results/raw/causal_ablation_raw_narrow.json --conditions M3_baseline M3_ablated
python -m src.analysis.bootstrap_causal_effect --file results/raw/causal_ablation_raw_narrow.json --quadrant C --category soft_deflection
```

---

## Bugs already found and fixed (don't rediscover these)

- `write_json()`'s real signature is `(data, path)` — a call site had it
  backwards, would crash on first real invocation.
- Several `STAGES` lists used `"M3-direct"` (hyphen) while
  `STAGE_ADAPTER_CHAINS` uses `"M3_direct"` (underscore) — crashes on load.
- `eval_behavioral.py` wrote to `results/behavioral_eval_raw.json` (flat)
  while the real data lives at `results/behavioral_eval/raw.json` (nested,
  matching `reclassify_behavioral.py`) — meant it could never find/resume
  existing results.
- `eval_probes.py`/`summarize_cross_branch.py` picked "best layer" by
  `cv_accuracy_mean`, which saturates at 1.0 everywhere — ties broke to the
  first (shallowest, least informative) layer, silently zeroing out every
  reported quadrant-C/D flagging rate. Fixed to pick by
  `quadrant_c_flagged_unsafe_frac`.
- Cross-branch cosine-similarity means were diluted by layer 0 (always
  exactly 0.0, a known template-token artifact) — fixed to exclude it.
- Dry-run configs for a brand-new branch (alt) pointed `init_from_adapter`
  at the real HF repo, which doesn't exist until the real (non-dry-run)
  training actually pushes — dry-run chains need to point at the
  prerequisite stage's own local dry-run output instead. Only fixed for the
  alt branch; the original branch's dry-run configs correctly point at real
  HF repos since those already exist for real — don't "fix" those unless
  they're actually failing for you.
- `mcnemar_causal_ablation.py`'s `main()` called `build_paired_outcomes()`
  missing the `target_category` argument — silently shifted every later
  argument. Nothing had ever tested `main()` end-to-end, which is how it
  slipped through. Now has a real regression test.
- `pathlib.Path` renders with the HOST OS's separator on `str()` — a test
  comparing a `Path`-built string against a Linux/Colab config path value
  failed on Windows. Use `posixpath` for anything that's data (a remote
  path string), not an actual local filesystem path.
- `eval_refusal_direction.py`'s adjacent-chain computation looped over the
  full `STAGES` list including `M3_direct`/alt stages, producing a
  `"M3_vs_M3_direct"` entry labeled as if it were a real sequential training
  step. Fixed with `SEQUENTIAL_STAGES`/`ALT_SEQUENTIAL_STAGES` kept separate
  from `STAGES`.

## Testing status

`pytest tests/ -v` should be fully green. If you're in a lightweight sandbox
without `torch`/`transformers`/`trl`/`peft`/`datasets`/`sentence-transformers`
installed, everything CPU-pure (stats, interpretability, data-pipeline logic,
config consistency) runs fine; anything importing `torch` at module level
will error on collection — that's an environment gap, not a real failure.
Run the full suite in the actual project `.venv` to be sure.

---

## Next steps, in priority order (mirrors README, more implementation detail here)

1. **Bootstrap the difference in cross-branch similarity** between
   M2-mediated and direct-DPO paths. Implementation sketch: resample prompt
   pairs the same way `bootstrap_causal_effect.py` already does, but on the
   per-layer cosine similarity arrays in `cosine_similarity.json`'s
   `cross_branch` section — needs the underlying per-prompt activations,
   not just the saved direction vectors, so this reuses
   `eval_refusal_direction.load_stage` + `diff_in_means_direction`, resampled.
2. **Bootstrap CI over near-optimal bottleneck layers**, not just the
   argmax — extend `bottleneck_layer.py`'s `find_bottleneck_layer` to also
   report a distribution over which layer wins across bootstrap resamples
   of the same Cohen's d computation.
3. **Formal paired comparison of deep-layer stability distributions**
   between direct-DPO and M2-mediated branches — the raw per-replicate
   bootstrap cosine similarities already exist in memory during
   `bootstrap_direction_stability.py`'s run; persist them (not just
   mean/median/std) for at least the deep layers so a direct paired test is
   possible without rerunning the bootstrap.
4. **Run `eval_steering_v2.py`** — flip `COMPONENTS_TO_RUN["steering"]` to
   `True` in the analysis notebook, or run directly:
   `python -m src.analysis.eval_steering_v2 --stage M3 --layers 24 --quadrants A D`
5. **Quadrant C, n=20 → 100+.** Seed corpus:
   [StrongREJECT](https://github.com/alexandrasouly/strongreject) (313
   human-curated, category-labeled harmful prompts). Workflow: take an
   already-published harmful request, reword its surface phrasing to sound
   mundane/legitimate while preserving the underlying ask, keep the harm
   category label. Do NOT bulk-generate new disguised-harm prompts from
   scratch with an LLM — curate by hand or with careful human review of any
   AI-assisted drafts. Once curated, integration is cheap: extend
   `build_eval_set.py`'s quadrant C section, rerun the eval pipeline (no
   retraining needed, existing models are fine).
6. Full fine-tuning robustness check (expensive, aspirational).
7. Diagnose the steering degenerate-collapse mechanism (track residual-
   stream norm growth layer-by-layer during multi-layer generation).

---

## Onboarding a new agent

See `ONBOARDING_PROMPT.md` for a ready-to-paste prompt that points a fresh
AI agent session at this file, README.md, and the current git history, and
tells it what to do first.