# Release Gap Audit

Audit date (UTC): 2026-09-01
Branch: `agent/c-quadrant-end-to-end-e0e2317a`
HEAD at audit time: `b679f998fc57b86feab0bcd547d9b7c7b30fd6e6`
("R104 human review: blind packet generation")

Method: repository state was inspected directly (fresh clone, `git status`,
`git log`, file/hash verification, focused CPU-only test execution,
`python -m compileall src`, and a real `--dry-run` invocation of
`rerun_mechanistic_v2.sh`). No GPU work was run. `pytest tests/ -q` (full
suite) was **not** run, per task instructions; only files relevant to this
audit's claims were exercised. Old prompts/plans (the onboarding pack and
prompt pack) were used only as a checklist, not as a source of truth — every
claim below was independently re-verified against current repository bytes.

---

## 1. CURRENT STATE

- Working tree is clean (`git status --short` empty except standard
  `__pycache__`/`.pytest_cache` ignores).
- 105 commits on the branch. Most recent work is the **R104 blind
  human-review packet** (`data/review/r104_human_review_blind.csv`,
  104 rows, `decision` column **empty for all rows** — this review has not
  been done yet).
- Frozen benchmark: `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl`,
  SHA-256 `e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b`.
  Verified by direct `sha256sum`: matches `LATEST_BENCHMARK.json`, the
  manifest, and `benchmark_validation_status.json`. Counts: A=150, B=250,
  D=150, C=104 (`c_paired` only — the source-authored arm is **not** in the
  frozen benchmark yet).
- `logs/benchmark_validation_status.json`: `technical_benchmark_status =
  FAIL`, driven entirely by `artifact_freshness_pass = false` (all M0–M3
  and the five alt/direct-DPO stages are missing `*_final.npy`/
  `*_pooled.npy`/`*_metadata_binding.json` — verified: `results/activations/`
  contains only 9 `*_metadata.json` files, zero `.npy` arrays).
- All other static/structural gates currently pass:
  `schema_integrity_pass`, `prompt_integrity_pass`, `c_review_pass`,
  `c_review_mapping_pass`, `benchmark_hash_pass`, `split_benchmark_hash_pass`,
  `split_hash_pass` are all `true`.
- `logs/agent_state.json` and `logs/RESUME_PROMPT.md` are **stale**: both
  date to commit `59f533e` (2026-08-26), ~60 commits and 6 days behind
  HEAD. They still list "C-source-authored arm: not built," which is false
  as of commit `cb255f0` (Milestone-3A4-equivalent work). Do not treat
  either file as current status going forward.

## 2. R104 HUMAN REVIEW — EXPLICIT STATUS (do not assume)

Two distinct review artifacts exist and must not be conflated:

- **`data/review/c_review_queue.csv`** (104 rows, `review_status=accept`
  for all 104): a researcher-authored commit (`59f533e`/`c40620b`,
  2026-08-26, author `Uros Savurdic`, real git identity — not an agent).
  This is what the *currently frozen benchmark* was built from
  (`review_summary.accepted = 104` in the manifest), and what
  `c_review_pass` currently checks against.
- **`data/review/r104_human_review_blind.csv`** (104 rows, blind,
  neutral `R001`–`R104` IDs, shuffled order, `decision` column **all
  empty**): generated 2026-08-31 (`b679f998`, HEAD) as the review called
  for by `logs/c_existing_construction_decision.md`'s Gate-3 verdict
  (**"KEEP FOR HUMAN REVIEW"** — "read a sample of the 104 pairs... to
  judge whether the rewrite plausibly reduces surface harm-signaling").
  That verdict postdates the accept-all review by five audit stages
  (C-D → C-E → C-F). The accept-all pass was a coarse pre-audit gate, not
  the calibrated review the later audits determined was actually needed.

**Conclusion: the R104 human review the current methodology calls for is
NOT complete.** The frozen benchmark's `c_review_pass=true` reflects the
earlier, now-superseded accept-all pass, not the blind review. Treat
`c_review_pass=true` as "structurally valid input," not as "the
scientifically-motivated human review is done." This is `HUMAN ONLY`.

## 3. COMPLETED REQUIREMENTS

| Item | Status |
|---|---|
| `.gitattributes` LF normalization (`* text=auto eol=lf` + per-extension rules) | `DONE` |
| POSIX-path handling in `src/v2_io.py` (`normalize_json_path`, `.as_posix()` in `binding()`) | `DONE` |
| `LATEST_BENCHMARK.json` / manifest / `benchmark_validation_status.json` use POSIX paths | `DONE` (verified: no backslashes present) |
| Benchmark hash / split-manifest hash / review-CSV hash internal consistency | `DONE` (re-verified via `sha256sum`, not assumed) |
| Fresh-clone benchmark resolution (this audit *is* a fresh clone) | `DONE` |
| Missing-activation-metadata → stale freshness logic | `DONE` (`stale_activation_files` correctly lists all 27 missing artifacts; `artifact_freshness_pass=false`) |
| Shard/checkpoint infrastructure (`src/analysis/v2_shards.py`: `Deadline`, `ShardStore`, `plan_shards`, `run_sharded`, `run_with_oom_backoff`, `probe_batch_capacity`) | `DONE` |
| Deadline enforcement + shard-boundary stop + resume-with-zero-recompute | `DONE` — proven by `tests/analysis/test_v2_resumability.py`, an explicit CPU-only, fake-clock, byte-identical-merge acceptance test (Milestone 2D). Ran it directly: passes. |
| Length-sorted batching + `record_id`-based order restoration | `DONE` (`tests/analysis/test_v2_pipeline.py` / `plan_shards` sorting; covered by passing tests) |
| v2 compatibility bridge to legacy CPU statistics filenames (`src/analysis/v2_compat.py`) | `DONE` — binding-checked, non-destructive, fails closed against un-bridged legacy files; `tests/analysis/test_v2_compat.py` passes |
| Direction-family outputs (`adjacent`, `adjacent_alt`, `direct_branch`, `cross_branch`) feeding `src/interpretability/*` | `DONE` — all five named scripts (`direction_stability.py`, `bootstrap_direction_stability.py`, `bottleneck_layer.py`, `bootstrap_cross_branch_difference.py`, `paired_deep_layer_stability_test.py`) and `summarize_cross_branch.py` exist with matching tests |
| Causal-ablation scope = exactly the 4 DPO endpoints (M3, M3_direct, M3_alt, M3_direct_alt) | `DONE` — confirmed directly in the dry-run stage table |
| Steering scope = all 8 non-M0 stages | `DONE` — confirmed directly in the dry-run stage table |
| `--steering-stage` independent of causal-ablation stage selection | `DONE` (separate `item["steering"]`/`item["causal"]` flags in `stage_plan`) |
| Norm-collapse diagnostic gated behind `--with-norm-diag`, benchmark-bound, uses held-out Quadrant-D | `DONE` (`stage_norm_diag`, gated at `args.with_norm_diag and stage in steering`) |
| Colab notebook: Drive mount, persistent `HF_HOME`/results, resolved-path printing, status/calibration/resume cells, gate verification cell, v2-runner-only invocation | `DONE` — all present as distinct, correctly-ordered cells |
| Arm 2 (source-authored) candidate construction: fetch, hash, category-match, dedup vs Quadrant A, Fightin' Words scoring, Q25 stratum, 150-cap, contamination check vs the 4 named SFT/DPO files | `DONE` — `logs/3a4_scoring.md` shows a legitimate, documented reason the queue is 52 rows, not 150 (Q25 stratum yielded only 52 candidates; `Capped by limit: False`, not an error) |
| `src/reproduce.py` `probes`/`direction`/`behavioral_stats` components | `DONE` (paths match what `v2_compat.py` bridges into) |
| `python -m compileall src` | `DONE` — clean, ran directly |

## 4. ACTUAL REMAINING REQUIREMENTS

In priority order (see §9 for sequencing):

1. **Complete the R104 blind human review** (`data/review/r104_human_review_blind.csv`) — `HUMAN ONLY`. Nothing downstream of R104 should be treated as final until this is done, per §2.
2. **Milestone 3B (Arm-2 review-queue validation) has no corresponding artifact.** `c_source_authored_review_queue.csv` has not changed since the 3A4 scoring commit (`cb255f0`); there is no dedicated 3B validation log/commit. The queue's contents look individually correct on inspection (correct schema, `review_status=pending` throughout, contamination/overlap columns populated, no wording changes possible to detect without re-deriving from source), but the explicit audit step itself was never run/recorded. `NEEDS VERIFICATION`.
3. **Milestone 3C: human review of the 52-row Arm-2 queue** — `HUMAN ONLY`, not started (`review_status=pending` for all 52 rows, confirmed directly).
4. **Milestone 4A (final C-arm integration) cannot currently run as documented.** `src/finalize_benchmark.py` takes a single `--review-csv` and a single `--provenance` file (default `data/quadrant_c_pipeline/candidate_records_v2.jsonl`). That default provenance file contains **only** `c_paired` records (verified: 155/155 rows are `c_construction=c_paired`, zero `c_source_authored`); Arm-2 provenance actually lives in a separate file (`data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl`) that `finalize_benchmark.py` never reads. The script *does* already know how to place a `c_source_authored` row in the benchmark once it sees one (lines handling `construction == "c_source_authored"` exist), but there is currently no way to hand it Arm-2 rows with resolvable provenance in a single invocation. This needs either (a) a merged/reconciled review+provenance input, or (b) a second `--provenance`/`--review-csv` pair, before Milestone 4A can run. `NEEDS FIX` (blocked behind items 1 and 3 regardless).
5. **Milestone 4B (paired companion set) not started.** No companion JSONL/manifest/`LATEST_COMPANION.json` exists anywhere in the repo (`find` confirms zero matches). The supporting infrastructure it should reuse (`ArtifactPaths.for_namespace`, `--namespace`, `--latest-pointer` in `v2_pipeline.py`) is already built and unused, exactly as intended by the "reuse existing pointer architecture" requirement. Correctly blocked behind 3C/4A, not itself broken. `NOT APPLICABLE` yet (blocked, not failing).
6. **`--with-probes` has no effect on `--dry-run` output.** Verified directly: `diff` between `--dry-run --regenerate --with-probes` and the same command without `--with-probes` is empty. `main_run()` returns immediately after `describe_plan()` when `args.dry_run` is set, before reaching the `if args.with_probes: compute_probes(...)` branch — so the dry run never reflects whether probes were requested, contradicting the "dry-run must show... probes where requested" acceptance requirement. Low severity (doesn't affect the live run, only the dry-run's informational completeness). `NEEDS FIX`.
7. **`PINNED_COMMIT` placeholder in `notebooks/colab_unified_analysis.ipynb` was never set.** Still reads `'REPLACE_AFTER_PUSH_WITH_COMMIT_SHA'`. This is fail-closed by design (the live-run cell asserts `commit == PINNED_COMMIT` and refuses to proceed while the placeholder is present), so it is not unsafe, but it is a genuine outstanding action — its own markdown says it should be set "after Milestones 2C/2D, 5A, and 6A land," and those have now landed. `NEEDS FIX` (mechanical — set to the real final integration commit at handoff, Milestone 8C).
8. **`artifacts` is a stray committed empty regular file, not a directory**, and this breaks the exact patch-output path `logs/RESUME_PROMPT.md` instructs the researcher to use (`artifacts/patches/...`). Verified directly: `mkdir -p artifacts/patches` fails with `mkdir: cannot create directory 'artifacts': Not a directory`. Introduced in commit `b94d9b7` ("Lst chngs"). `NEEDS FIX`.
9. **`src/reproduce.py`'s `causal_stats` component is broken end-to-end**, in two independent, confirmed ways (see §4 "BROKEN / FAILING CHECKS" for the exact repro). `NEEDS FIX`.
10. **Semantic near-duplicate check (embedding-based) has never been run**, for either R104-vs-Quadrant-A or Arm-2-vs-Quadrant-A. Every relevant log (`c_existing_construction_decision.md`, the Arm-2 `contamination_status="checked_exact_zero_near_unknown"` column) states this consistently — exact-string dedup is done, semantic near-dup is not. This is a real, currently-open integrity gap, not a documentation lag. `NEEDS VERIFICATION`/`HUMAN ONLY` scientific call on whether it's required before the GPU run, or can ship as a documented limitation.
11. **GPU rerun itself** — extraction, behavioral eval, causal ablation, steering, probes, norm-diag across all 9 stages against the *current* frozen benchmark. This is the actual T4 work everything above exists to make safe. `NOT APPLICABLE` to this audit (explicitly out of scope) but is the terminal remaining requirement once 1–10 are resolved.
12. **Milestone 8A/8B/8C (final verification, readiness audit, handoff) have not formally run** as their own discrete step — this audit is effectively a partial 8A/8B pass, but no `COLAB READY`/`NOT COLAB READY` declaration or `logs/agent_state.json`/`logs/RESUME_PROMPT.md` refresh has happened since `59f533e`. `NEEDS VERIFICATION` (this audit's own bottom line, see §9).

## 5. BROKEN / FAILING CHECKS

Two confirmed, reproduced breakages (both re-run just now, not inferred):

**A. `src/reproduce.py`'s `causal_stats` component is broken.**
```
$ python -m src.analysis.summarize_causal_ablation \
    --file results/raw/causal_ablation_raw_narrow.json --stage M3
usage: summarize_causal_ablation.py [-h] [--file FILE]
summarize_causal_ablation.py: error: unrecognized arguments: --stage M3
```
Even with `--stage` removed, it fails differently:
```
$ python -m src.analysis.summarize_causal_ablation \
    --file results/raw/causal_ablation_raw_narrow.json
...
KeyError: 'stage'
```
Root cause: `results/raw/causal_ablation_raw_narrow.json` uses the field
name `model_stage` (confirmed: `dict_keys(['prompt', 'quadrant', 'source',
'model_stage', 'response'])`), while `summarize_causal_ablation.py`,
`mcnemar_causal_ablation.py`, and `bootstrap_causal_effect.py` all read
`row["stage"]`. The sibling file `results/raw/causal_ablation_raw_wide.json`
does use `stage` (confirmed) — so the three summarization scripts are
compatible with the *wide* file's schema but not the *narrow* file's, and
`src/reproduce.py`'s `causal_stats` component specifically points at the
*narrow* file with an argument (`--stage M3`) none of the three scripts
even accept. `tests/analysis/test_summarize_causal_ablation.py` only
exercises `classify_completion()`, so this was never caught. This is the
same defect `logs/RESUME_PROMPT.md` flagged on 2026-08-26 ("`model_stage`
vs `stage` mismatch... needs fix in summarization code if that path is
used") — it is now confirmed **still present and actively wired into the
one-command reproduction entry point**, not merely a stale note.
`NEEDS FIX`.

**B. `artifacts/patches/` (the documented patch-output directory) cannot be created.**
```
$ mkdir -p artifacts/patches
mkdir: cannot create directory 'artifacts': Not a directory
```
`artifacts` is tracked as a 0-byte regular file (introduced in `b94d9b7`,
"Lst chngs"), not a directory, blocking the exact path
`logs/RESUME_PROMPT.md` instructs future agents/the researcher to write
patches to. `NEEDS FIX`.

Everything else exercised in this audit — 103 focused CPU-only tests
(`test_v2_io`, `test_validate_benchmark_v2`, `test_v2_shards`,
`test_v2_resumability`, `test_v2_pipeline_deadline`, `test_v2_compat`,
`test_v2_direction_family`, `test_build_r104_human_review_packet`,
`test_config_consistency`), `python -m compileall src`, and the
`--dry-run --regenerate --with-probes` invocation — **passed / succeeded**
on direct execution.

## 6. STALE OR REDUNDANT ARTIFACTS

| Artifact | Staleness | Classification |
|---|---|---|
| `logs/agent_state.json` | Dated `59f533e` (2026-08-26); ~60 commits behind HEAD; incorrectly claims Arm 2 "not built" | `NEEDS FIX` (regenerate at next handoff milestone) |
| `logs/RESUME_PROMPT.md` | Same commit; its exact patch path (`artifacts/patches/...`) is currently broken (§5B) | `NEEDS FIX` |
| `logs/researcher_runbook_colab.md`, `logs/researcher_runbook_local.md` | Same commit (`59f533e`, 2026-08-26); predate all of Milestones 3A4 onward | `NEEDS VERIFICATION` |
| `CLAUDE.md` | Last touched `313942d`, 2026-08-25 | `NEEDS VERIFICATION` |
| `HANDOFF.md` | Last touched `23518bf`, 2026-08-14 | `NEEDS VERIFICATION` |
| `PROJECT_CONTEXT.md` | Last touched `c5ac374`, 2026-08-13 | `NEEDS VERIFICATION` |
| `results/refusal_direction/*_direction.npy` (un-suffixed), `results/probes/`, `results/behavioral_eval/raw.json`, `results/raw/*.json`, `results/interpretability/*` | Pre-v2 ("370-row era" per `logs/agent_state.json`) legacy results, not bound to the current frozen benchmark; will be superseded by the v2 GPU rerun + `v2_compat.py` bridge | `NOT APPLICABLE` to touch now — correctly left in place per onboarding rules ("mark stale, do not delete"); the `_v2_compat_binding.json` safety mechanism already prevents silent overwrite |
| `results/raw/steering_raw_D_L21_exploratory_DEPRECATED.json`, `results/raw/steering_raw_D_MULTILAYER_14to28_DEPRECATED.json` | Explicitly named `DEPRECATED`, clearly superseded by the current L14–28/L24 spec | `NOT APPLICABLE` — correctly labeled, no action needed |
| `assets/c_quadrant_colab_fixes.patch` + `apply_colab_fixes.py` | One-shot historical batch-fix script pinned to `BASE_COMMIT = faee3317...`; `assert_clean_enough()` fail-closes if HEAD ≠ that commit, so it cannot run against current HEAD | `NOT APPLICABLE` to fix now; candidate for archival, see §7 |

## 7. LEGACY CODE CANDIDATES

- `apply_colab_fixes.py` (96K, top-level) + `assets/c_quadrant_colab_fixes.patch` (88K): a one-time historical patch-generation script hard-pinned to an old base commit. Nothing in the current runner, notebook, or tests references it. Not currently harmful (fails closed rather than silently misapplying), but it is dead weight in the repo root. Recommend archival in a future cleanup-scoped task — **not** in scope for this audit or for scientific/engineering milestones.
- `notebooks/colab_unified_training.ipynb`: not inspected in depth for this audit (training for M0–M3/alt/direct is already complete and committed); flagging only that it exists alongside the analysis notebook and was out of this audit's assigned scope (`notebooks/colab_unified_analysis.ipynb` only, per the original Milestone 7 series). `NOT APPLICABLE` to this audit.

## 8. REPRODUCIBILITY RISKS

- **Semantic near-duplicate checking is absent** for both C arms (§4.10). This is the single largest open *scientific* reproducibility/validity risk currently carried forward without resolution — every audit stage that touched it (C-A through C-F, and the Arm-2 3A4 log) explicitly flags it as unresolved rather than passing or failing it.
- **`artifacts/patches/` cannot be created** (§5B) — blocks the documented reproducible-patch workflow until fixed. Any agent following `logs/RESUME_PROMPT.md` literally today would fail at the final step.
- **`src/reproduce.py causal_stats`** is a documented, single-command reproduction path that currently cannot execute (§5A) — a researcher following the script's own `--list`/usage text would hit a stack trace, not a result.
- The accept-all R104 review (§2) sitting inside the frozen benchmark's provenance chain, while a stricter review is pending, is not itself a reproducibility bug (the benchmark is internally hash-consistent and was validly built from what existed at the time) but is a **scientific-validity risk** if the frozen benchmark is used for the GPU rerun before the blind review concludes and is reconciled.

## 9. COLAB/T4 READINESS RISKS

- `artifact_freshness_pass=false` is the sole current blocker on `technical_benchmark_status`, and it is *expected* to be false pre-GPU-rerun — not a defect, per the notebook's own inline comment ("expected False before this session's GPU pass produces fresh activations").
- `PINNED_COMMIT` is unset (§4.7) — the notebook will correctly refuse a live run until this is set to a real commit SHA at final handoff.
- The dry-run itself (`--dry-run --regenerate --with-probes`) runs cleanly and shows the correct 9-stage, stage-major plan with the correct causal/steering scope, correct shard/resume paths, and a live 300-minute session budget — the core execution graph is sound.
- `--with-probes` not being reflected in dry-run output (§4.6) is cosmetic, not a blocker to the live run itself.

## 10. EXACT ORDER OF NEXT TASKS

*Note: items 2, 6, 7, and 8 below involve Arm-2 / `c_source_authored`,
which is deferred from the active release (R104 / `c_paired` is the sole
active C construction). They are marked `DEFERRED` and are not required
to reach this release's Colab run. See `legacy/quadrant_c_arm2/README.md`.*

1. **`HUMAN ONLY`** — Complete the R104 blind review (`data/review/r104_human_review_blind.csv`, 104 empty `decision` cells) and reconcile its outcome against the existing accept-all `c_review_queue.csv` before the frozen benchmark is treated as final for the GPU rerun.
2. **`DEFERRED`** — Milestone 3C: review the 52-row `data/review/c_source_authored_review_queue.csv`. Not required for the active release; Arm-2 is deferred, not in the active benchmark.
3. **Engineering, small, independent of 1–2** — Fix the `artifacts` stray-file / `artifacts/patches/` breakage (§5B). Trivial, no scientific risk, should happen ASAP since every subsequent milestone's patch step depends on it.
4. **Engineering, small, independent of 1–2** — Fix the `causal_stats` `stage`/`model_stage` mismatch in `src/reproduce.py`'s wiring (§5A): either point it at `causal_ablation_raw_wide.json`, or normalize the narrow file's field name, and drop the unsupported `--stage` flag from the invocation.
5. **Engineering, small, independent of 1–2** — Make `--with-probes` visible in `--dry-run` output (§4.6).
6. **`DEFERRED`** — Milestone 3B: produce the missing explicit Arm-2 review-queue validation artifact (§4.2). Not required for the active release.
7. **`DEFERRED`** — Milestone 4A: extend `src/finalize_benchmark.py` (or its inputs) to actually integrate Arm 2 (§4.4). Not required for the active release; the active benchmark uses R104 / `c_paired` only.
8. **`DEFERRED`** — Milestone 4B: build the paired companion set (infrastructure is ready and unused). Not required for the active release.
9. **Blocked on 1–8, or an explicit scientific decision to proceed without it** — resolve or explicitly accept-as-limitation the semantic near-duplicate gap (§4.10, §8) — `HUMAN ONLY` scientific call.
10. **After 1–9 (or an explicit, documented decision to defer some of them)** — refresh `logs/agent_state.json` and `logs/RESUME_PROMPT.md` (currently ~6 days stale), set `PINNED_COMMIT` in the notebook, and run the actual Milestone 8A/8B/8C final-verification / readiness-audit / handoff sequence.
11. **Only after 1–10** — the real T4 GPU rerun.

**This audit's own bottom line:** `NOT COLAB READY`. The engineering
machinery (shards, deadline, resume, batching, compat layer, dry-run,
notebook persistence/status/calibration cells) is in solid, tested shape.
The blockers are (a) two small, independent, already-diagnosed code fixes
(§5), and (b) benchmark finalization that is genuinely gated on pending
human review of both C arms, not on missing engineering.
