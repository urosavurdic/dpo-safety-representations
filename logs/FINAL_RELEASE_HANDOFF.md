# FINAL RELEASE HANDOFF — v2 C-quadrant benchmark, pre-Colab

Generated: 2026-09-01T20:40:06Z, after Task 4 (final CPU verification) passed
with no blockers.

## Verdict

**COLAB READY**

## Identity

- Repo: https://github.com/urosavurdic/dpo-safety-representations
- Branch: `agent/c-quadrant-end-to-end-e0e2317a`
- Final HEAD: `3e668107fd966f1af0f0eea13c95e2626d2aedf9`
  ("Retire Arm-2 (c_source_authored) from active release path; archive to
  legacy/quadrant_c_arm2")
- Working tree: clean at HEAD (verified in-session and from an independent
  fresh clone).

## Benchmark / split

- Benchmark pointer: `data/frozen_v2/LATEST_BENCHMARK.json` →
  `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl`
- Benchmark SHA-256: `e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b`
  (recomputed directly from the file; matches pointer and manifest)
- Row counts: A=150, B=250, C=104, D=150 (654 total) — identical to the
  pre-Task-3 (20260825) frozen manifest, so A/B/D are confirmed preserved.
- Split manifest: `logs/direction_split_manifest.json`
- Split SHA-256 (self-hash of canonical JSON): `880381606de7aa2ffbdb8f7c75303cf4937167ed1a2e1b417afeb33761fcf8f1`
  — verified by regenerating the manifest from the frozen benchmark
  (`python -m src.create_direction_split_manifest`) and diffing: byte-identical.
  Seed 45, 80/20 direction-estimation/held-out split on A+D (240/60).
- R104 / C-paired: sole active C construction, 104/104 rows reviewed and
  accepted, 0 rejected, 0 pending. Provenance:
  `data/quadrant_c_pipeline/candidate_records_v2.jsonl`.
- Zero Arm-2 rows in the active benchmark: confirmed by direct grep for
  `c_source_authored` in `benchmark_v2_20260826T212909Z.jsonl` (0 matches)
  and by the manifest's `c_counts_by_construction: {"c_paired": 104}`.

## Companion mechanism

`src/analysis/v2_pipeline.py`'s `--namespace` flag routes a companion
eval set's artifacts to `results/companions/<namespace>/` (paired with a
non-default `--latest-pointer`) so a companion run's activations can never
collide with the main run's `results/`. Untouched by Task 3; its tests
(`test_v2_pipeline.py`, `test_v2_compat.py`) pass. Not used by the default
run — no companion pointer is set for this release.

## Training chain / analysis scope (unchanged)

- Confirmed present in the `run` dry-run stage plan: `M0 → M1 → M2 → M3`,
  plus `M3_direct`, and the full `_alt` (Dolly) mirror
  (`M1_alt → M2_alt → M3_alt → M3_direct_alt`).
- Causal-ablation scope: `INTERVENTION_STAGES` (the four DPO endpoints:
  M3, M3_direct, M3_alt, M3_direct_alt) — untouched by Task 3
  (`git show --stat` on the Task-3 commit shows only
  `legacy/quadrant_c_arm2/*` and `logs/release_gap_audit.md` changed).
- Steering scope: `STEERING_STAGES` (every trained stage), layer 24 by
  default, `alpha_source=direction_estimation_only` fixed for `run` by
  design — untouched by Task 3.
- Calibration split / statistical definitions (`src/eval_stats.py`,
  Wilson CIs, McNemar): untouched by Task 3.
- Full dry-run graph (`--dry-run --regenerate --with-probes --with-norm-diag`,
  matching the live-run cell's exact flags) confirmed to contain all nine
  expected elements per stage: extraction, direction, probes, behavioral,
  causal ablation, steering, norm diagnostic, checkpoint/resume, deadline.
  (Note: plain `--with-probes` without `--with-norm-diag` omits norm_diag
  from the printed plan — that's the flag gating it, not a wiring gap.)

## Arm-2 status

Deferred, not deleted. Task 3 (commit `3e66810`) moved the three
Arm-2-only implementation scripts and their tests into
`legacy/quadrant_c_arm2/`, fixed the cross-imports/`REPO_ROOT` depth that
relocation broke, and added `legacy/quadrant_c_arm2/README.md` documenting
provenance and deferral. Left in place (per the task's dependency
decision, unchanged by this handoff):
`src/data_pipeline/build_c_source_authored_candidates.py`, the raw/validated
candidate JSONLs, `data/review/c_source_authored_review_queue.csv`, the
shared CUE/lexical-outlierness modules, and historical comparative audit
reports — these are shared with active R104 tooling or are provenance
records, not Arm-2-exclusive.

## Colab operational details

- Notebook: `notebooks/colab_unified_analysis.ipynb` (git-tracked in this
  repo, despite `CLAUDE.md`/`PROJECT_CONTEXT.md` describing untracked
  Colab notebooks elsewhere as convention — this one is the exception,
  verify `git ls-files notebooks/` if that matters to you). Runs
  `src/analysis/v2_pipeline.py`, `v2_shards.py`, `v2_compat.py`,
  `validate_benchmark_v2.py` only — no legacy mutable-eval-set pipeline.
- Pinned commit: notebook cell 4 defines `PINNED_COMMIT =
  'REPLACE_AFTER_PUSH_WITH_COMMIT_SHA'` as an intentional placeholder.
  **Before the live-run cell (cell 22) will execute, set
  `PINNED_COMMIT = '3e668107fd966f1af0f0eea13c95e2626d2aedf9'`** — the
  cell asserts the placeholder was replaced and that the checked-out
  commit matches exactly.
- Persistent storage: Google Drive at
  `/content/drive/MyDrive/dpo_safety_v2/` — `results/` is symlinked there
  (survives disconnects/fresh runtimes) and `HF_HOME` points at a Drive-backed
  HF cache (`hf_cache/`) so the base model + LoRA adapters download once
  total, not once per session.
- Calibration command: `python -m src.analysis.v2_pipeline calibrate
  --stage M3 --probe-capacity` (default stage M3, default 32 prompts;
  `--probe-capacity` additionally measures the largest OOM-safe
  forward/generation batch size and records it for `--act-batch auto` /
  `--gen-batch auto`).
- Resumable run command (what the live-run cell executes when `RUN_GPU =
  True`): `bash rerun_mechanistic_v2.sh --regenerate --with-probes
  --with-norm-diag --act-batch auto --gen-batch auto --deadline-minutes
  <SESSION_DEADLINE_MINUTES>`.
- Deadline/session behavior: default session budget 300 minutes
  (`DEFAULT_DEADLINE_MINUTES`); the runner stops cleanly at a shard
  boundary once spent. Progress is shard-checkpointed under
  `results/behavioral_eval/v2_shards/`, `results/raw/v2_causal_shards/`,
  `results/raw/v2_steering_shards/`, each shard bound to the
  benchmark+split SHA above — a killed/disconnected session resumes from
  the next unfinished shard rather than restarting the stage; reopening
  the notebook and running top-to-bottom again is the intended recovery
  path.

## R104 interpretation

R104 = `c_paired`, the 104-row, individually-classified, source-attributed,
verified-zero-cue quadrant-C construction drawn from StrongREJECT,
replacing the original 20-row hand-curated set (unverifiable provenance,
and on external review not actually neutral wording). It is the sole
active C construction for this release; Arm-2 (`c_source_authored`) is
deferred, not part of R104, and contributes zero rows to the active
benchmark.

## Known limitations (carried from README, not re-derived here)

LoRA confound quantified but not eliminated (90%+ of the direction's norm
sits outside the rank-64 subspace, but real above-chance deep-layer
alignment exists); single diff-in-means direction only, other orthogonal
directions not searched for; ablation-sufficiency claim (Finding 4) was
run on the pre-expansion 50/50 A/D eval set and needs a rerun against the
current split; 1.5B scale, not claimed to generalize to frontier models;
Alpaca-vs-Dolly reproducibility is two datasets/one model family/one LoRA
setup, not fully general; quadrant D was Alpaca-only pre-expansion (fix
designed, not yet run); PKU-SafeRLHF training data only pairs
safety-disagreement responses, so no same-safety preference signal was
trained on; eval harm-category coverage in training data not yet
re-confirmed against the current category set. Full detail in README.md
§Limitations.

## GPU results status

**No GPU results exist yet for this release.** Every activation,
behavioral, causal-ablation, steering, and norm-diagnostic artifact in
`results/` is either absent or stale (pre-dates the 654-row frozen
benchmark). `artifact_freshness_pass` in the benchmark gate is expected to
read `False` until the Colab live run produces fresh activations against
`benchmark_v2_20260826T212909Z.jsonl` — that is what the live run is for,
not a defect in the current state. `HANDOFF.md`'s "Phase 4 complete"
verdict describes analysis of the prior (pre-v2-expansion) results and
has not yet been re-run against the current R104/654-row benchmark.
