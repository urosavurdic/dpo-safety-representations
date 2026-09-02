# EMERGENCY INTEGRITY AUDIT — dpo-safety-representations

Audit-only session. No code changed, no GPU run, no artifacts regenerated.
All findings below are grounded in: file contents in the working tree,
`git log`/`git show`, and one CPU-only, no-GPU execution of
`src/analysis/summarize_probe_findings.py` and the (already-fixed)
layer-selection function against the real committed `results/probes/*.json`
files, to confirm the reported symptom rather than guess at it.

---

## 1. REPOSITORY STATE

- HEAD: `e689092ef5e63c91a9dffdc76e356a76628bcb1f`
  ("Final release handoff: Colab-ready verdict, benchmark/split/Arm-2
  status, resume prompt")
- Branch: `agent/c-quadrant-end-to-end-e0e2317a` (tracks
  `origin/agent/c-quadrant-end-to-end-e0e2317a`, up to date)
- Working tree: clean, no uncommitted changes.
- Note: `logs/RESUME_PROMPT.md` and `logs/agent_state.json` both claim the
  "final" HEAD is `3e668107f...` — HEAD is actually one commit *past* that
  (`e689092`, which is itself the commit that added those two files). Not a
  problem, just means those two files describe the state as of their own
  authoring commit, not literally "current HEAD" — expected for a handoff
  doc, not a defect.

## 2. BENCHMARK STATE

- Pointer: `data/frozen_v2/LATEST_BENCHMARK.json` →
  `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl`
- SHA-256 (recomputed directly, matches pointer):
  `e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b`
- Row counts (verified by direct count): **A=150, B=250, C=104, D=150,
  total=654.** Matches the frozen target exactly.
- Older artifact: `data/frozen_v2/benchmark_v2_20260825T223843Z.jsonl` also
  exists, also 654/A150/B250/C104/D150 — same composition, just an earlier
  freeze. `LATEST_BENCHMARK.json` correctly points at the later
  (`20260826`) file. Not ambiguous, just two snapshots on disk.
- `data/processed/controlled_eval.jsonl` (the file the *legacy* analysis
  scripts actually read) currently has the **same 654 rows / same prompt
  set** as the frozen benchmark (verified by diffing sorted prompt lists —
  identical). This is coincidence-by-discipline, not an enforced
  invariant: `src/data_pipeline/build_eval_set.py` builds
  `controlled_eval.jsonl` independently (its own quadrant-C constant,
  its own A/B/D loaders) and `src/finalize_benchmark.py` freezes
  `controlled_eval.jsonl` → `frozen_v2` as a **one-way, one-time
  snapshot**. Nothing re-checks that `controlled_eval.jsonl` still
  matches `frozen_v2` after the freeze. If either file is edited alone in
  the future, they silently diverge again with no error from either
  pipeline.
- Ambiguity/mismatch: none in the frozen artifacts themselves. The
  mismatch is structural (two independently-maintained files that happen
  to currently agree), not a row-count defect today.

## 3. PROBE STATE

- **Two parallel, non-interoperating Component-2/3 implementations exist
  in this repo right now:**

  | | Legacy (pre-v2) | v2 (current) |
  |---|---|---|
  | Entry point | `src/analysis/eval_extract_activations.py`, `eval_probes.py`, `summarize_probe_findings.py` | `src/analysis/v2_pipeline.py` (via `rerun_mechanistic_v2.sh` / the Colab notebook) |
  | Eval-set source | `data/processed/controlled_eval.jsonl`, read directly, no hash check | `data/frozen_v2/LATEST_BENCHMARK.json`, SHA-256-verified via `src/v2_io.py` |
  | Activation output | `results/activations/{stage}_final.npy` / `_pooled.npy` / `_metadata.json` | **Same directory and same three filenames** for a default (non-`--namespace`) run, **plus** `{stage}_metadata_binding.json` |
  | Freshness check | Self-referential only (compares to its own last snapshot) | Triple-checked: binding-hash + metadata-snapshot equality + array row-count |
  | Probe output | `results/probes/{stage}_probe_results.json` | `results/probes_v2/{stage}_probe_results.json` (different dir — no collision here) |
  | Still documented as "the commands" in | `CLAUDE.md` §"Manual component-by-component" (lines 153–169) | `RESUME_PROMPT.md`, `FINAL_RELEASE_HANDOFF.md`, the notebook itself |

  `logs/FINAL_RELEASE_HANDOFF.md` states outright: *"Runs
  `src/analysis/v2_pipeline.py`, `v2_shards.py`, `v2_compat.py`,
  `validate_benchmark_v2.py` only — no legacy mutable-eval-set pipeline."*
  i.e. the actual Colab notebook never touches the legacy scripts. But
  `CLAUDE.md` — the file an agent or user is most likely to read for "how
  do I run this" — still lists the legacy scripts as the "manual"
  commands, with no deprecation note next to them. This divergence is the
  most direct repository-integrity problem found in this audit.

- **C. Why does the final-layer Component 3 report still show `Quadrant C
  (n=20)`?** — Reproduced directly, no GPU needed.
  `src/analysis/summarize_probe_findings.py:26-30` hardcodes:
  ```python
  QUADRANTS = {
      "holdout_b_flagged_unsafe_frac": ("B (held-out)", 200),
      "quadrant_c_flagged_unsafe_frac": ("C", 20),
      "quadrant_d_flagged_unsafe_frac": ("D", 50),
  }
  ```
  These `n` values are Wilson-CI denominators used **only for display** —
  never derived from the benchmark file, the split manifest, or even the
  probe result file's own row count. I ran the script against this repo's
  actual committed `results/probes/*.json` and it printed, verbatim:
  `--- Quadrant C (n=20) ---` and `--- Quadrant D (n=50) ---`. D's
  hardcode is also wrong under the frozen benchmark (should be 150, not
  50). Only B's (200 = 250 − 50 held out) happens to still be correct.
  This script has **no freshness check of any kind** — unlike
  `eval_probes.py`, it doesn't compare metadata snapshots, doesn't check
  SHAs, doesn't check anything. It will print this exact wrong header
  forever, regardless of what real data underlies the numbers, until the
  dict is hand-edited or derived from `data/frozen_v2/LATEST_BENCHMARK.json`.

- **D. Exactly which file determines the probe dataset and C
  membership?** — Two different answers depending on which pipeline runs
  (see table above): `data/processed/controlled_eval.jsonl` for the
  legacy scripts (hardcoded path, `eval_extract_activations.py:32`), or
  `data/frozen_v2/LATEST_BENCHMARK.json` → the SHA-pinned benchmark file
  for `v2_pipeline.py` (`src/v2_io.py::resolve_benchmark`).

- **A. Why did the probe rerun report `Best layer 0` for all stages?**
  Partially reproduced, partially UNVERIFIABLE. The historical bug is
  real and documented in-repo (`CLAUDE.md` "Bugs already found and
  fixed"): `pick_most_informative_layer()` used to pick by
  `cv_accuracy_mean`, which saturates near 1.0 at almost every layer
  including untrained M0 — ties resolve to Python `max()`'s first match,
  i.e. layer 0, for every stage. **This is already fixed in the code at
  current HEAD** — `eval_probes.py:85-97` now picks by
  `quadrant_c_flagged_unsafe_frac`, and `v2_pipeline.py`'s own
  `compute_probes` doesn't pick a "best layer" at all (keeps every
  layer, explicitly notes `"layer_selection": "none; ... C not used for
  selection"`).
  I ran the current (fixed) selection function directly against the
  actual committed `results/probes/*_probe_results.json` files: **8 of 9
  stages correctly resolve to layer 24–28** (M1: 28, M2: 28, M3: 28,
  M3_direct: 27, M1_alt: 28, M2_alt: 28, M3_alt: 28, M3_direct_alt: 24).
  Only **M0** resolves to layer 0 — because M0 (the untrained baseline)
  has `quadrant_c_flagged_unsafe_frac == 0.0` at *every* layer (an
  untrained model flags nothing as unsafe anywhere), so every layer ties
  and `max()` still defaults to the shallowest. That's a genuine, minor,
  currently-unfixed tie-break edge case for stages with literally zero
  signal — but it is **not** "layer 0 for all stages." I could not
  reproduce the literal "all stages" symptom from current code + current
  committed data. Two possible explanations I cannot distinguish without
  either the original transcript or an actual GPU run (out of scope this
  session): (1) the observed report predates the documented fix landing
  on this branch, or (2) a genuine fresh GPU run produced near-zero
  quadrant-C flagging at every layer for every trained stage too, which
  would be a real finding, not a pipeline defect. **Marked
  UNVERIFIABLE** per the audit's evidence rule.

- **B. Is layer 0 a valid scientific layer here, or a known artifact?**
  Known artifact. Layer index 0 = raw embedding output (index 0 of
  `output_hidden_states`, before any transformer block runs). The repo's
  own code/docs treat it that way everywhere it appears: `eval_probes.py`
  calls it "the shallowest, LEAST informative layer"; `CLAUDE.md`
  separately notes cross-branch cosine similarities were "diluted by
  layer 0 (always exactly 0.0, a known template-token artifact) — fixed
  to exclude it." No part of this codebase treats layer 0 as a
  legitimate "best" layer; every appearance of it winning a selection is
  flagged as degenerate tie-breaking, not signal.

- **E. Which artifact binds activations to the frozen v2 benchmark?**
  Only within the v2 pipeline: `results/activations/{stage}_metadata_binding.json`
  (default run) or `results/companions/<namespace>/activations/{stage}_metadata_binding.json`
  (companion run), written by `v2_pipeline.py::stage_extract`, checked by
  `activations_bound()` / `load_bound_activation()` via
  `src/v2_io.py::assert_binding()` against the benchmark SHA (from
  `LATEST_BENCHMARK.json`) and the split-manifest SHA — plus independent
  metadata-snapshot equality and array-row-count checks. Confirmed none
  of these `_metadata_binding.json` files exist yet anywhere in the repo
  (also confirmed independently by `logs/benchmark_validation_status.json`'s
  own `stale_activation_files` list, which names all 9 stages' binding
  files as missing). **The legacy pipeline has no such artifact at all.**

- **F. Are current activations actually based on the 654-row
  benchmark?** No activations (`.npy` arrays) exist anywhere in this
  checkout — both `results/activations/*.npy` and
  `results/companions/*/activations/*.npy` are gitignored, and no GPU run
  has produced them yet. The only committed artifacts are orphaned
  `results/activations/{stage}_metadata.json` files, which describe **370
  rows (A=50, B=250, C=**20**, D=50)** — a pre-v2 eval set, not the current
  654-row/C=104 benchmark — and `results/probes/{stage}_probe_results.json`,
  computed from that same 370-row set. This matches
  `logs/FINAL_RELEASE_HANDOFF.md`'s own statement verbatim: "Every
  activation, behavioral, causal-ablation, steering, and norm-diagnostic
  artifact in `results/` is either absent or stale (pre-dates the
  654-row frozen benchmark)."

- **G. Can a stale activation/probe result silently pass as current?**
  In the v2 pipeline: no — triple-checked, and the top-level gate
  (`logs/benchmark_validation_status.json`, produced by
  `validate_benchmark_v2.py`) independently and correctly currently
  reports `"artifact_freshness_pass": false`. In the legacy pipeline:
  **yes, demonstrated directly this session** — `summarize_probe_findings.py`
  has zero freshness check and printed a report with a hardcoded, wrong
  quadrant-size denominator against real repo data; `eval_probes.py`'s
  freshness check is self-referential (compares only to its own prior
  run) and never checks the frozen benchmark's SHA at all, so a stale
  `controlled_eval.jsonl` could pass its freshness check indefinitely if
  nothing else ever changes it.

## 4. PERSISTENCE STATE

- Model checkpoints: orchestrated by `src/training/stage_registry.py` /
  `src/training/model.py` (`STAGE_ADAPTER_CHAINS`, `try_load_stage_model`)
  — **not re-audited this session** (training/model persistence is
  outside "audit only, no model regeneration" scope; flagging only what's
  already on record, see §5/§6).
- Activation save paths: `results/activations/` (legacy AND default v2
  run — same directory, see §3 table) or
  `results/companions/<namespace>/activations/` (v2 companion run).
- Behavioral save paths: legacy `eval_behavioral.py` →
  `results/behavioral_eval/raw.json`; v2 →
  `results/behavioral_eval/v2_shards/`. Same parent directory, different
  filenames — lower collision risk than activations, but still shared
  namespace.
- Drive vs ephemeral (from `notebooks/colab_unified_analysis.ipynb` +
  `logs/FINAL_RELEASE_HANDOFF.md`): `DRIVE_ROOT =
  /content/drive/MyDrive/dpo_safety_v2/`. The entire top-level `results/`
  directory is symlinked into this Drive root, and `HF_HOME` points at a
  Drive-backed `hf_cache/`. The git checkout itself
  (`/content/dpo-safety-representations`) is **not** Drive-backed — it's
  re-cloned and re-pinned to `PINNED_COMMIT` each session.
  **Consequence worth flagging:** because `results/` is symlinked as a
  whole, Drive persistence does not distinguish hash-bound v2 artifacts
  from legacy orphaned ones — both would persist and both would keep
  accumulating in the same Drive folder across sessions if the legacy
  scripts are ever run there.

## 5. RESUME STATE

- Training resume: not re-audited this session (out of scope; see §4).
- Activation/shard resume (v2 pipeline): hardened in commit `c615204`
  ("Task 2A: harden shard-plan identity and stale-artifact freshness in
  v2 runner") — fixed a real defect where `ShardStore.declare_unit` only
  rejected a changed shard *count*, not a changed row *count*, between
  sessions (row-count changes with the same shard count were silently
  invisible before the fix). Per that commit and
  `logs/FINAL_RELEASE_HANDOFF.md`: shard writes are atomic, completed-shard
  state is filesystem-cross-checked, and each shard carries the
  benchmark+split SHA binding. Not independently re-executed this session
  (would require GPU/real shard data, out of scope) — this assessment
  relies on the code + the specific hardening commit + the handoff doc's
  explicit claim, not a fresh reproduction.
- Behavioral/causal/steering shard resume: same mechanism
  (`src/analysis/v2_shards.py`), same commit.
- **Legacy activation resume** (`eval_extract_activations.py`):
  self-referential only — correctly detects when `controlled_eval.jsonl`
  has grown/changed relative to its own last snapshot
  (`eval_set_matches_saved_metadata`, itself a documented past fix, see
  `CLAUDE.md`), but has **no awareness of `frozen_v2` at all**, so
  "resumed correctly" and "resumed against the right benchmark" are two
  different claims here — only the first is checked.

## 6. STALE-ARTIFACT RISKS

1. **Confirmed, demonstrated:** `results/probes/*.json` (9 files) and
   `results/activations/*_metadata.json` (9 files) are stale, pre-v2
   artifacts (370 rows, C=20) still committed to the branch. They don't
   corrupt the v2 pipeline (different binding-file requirement), but
   they will be read as current by `summarize_probe_findings.py` forever
   with no warning, and by `eval_probes.py`'s cache path if a
   `results/probes/{stage}_metadata.json` snapshot were ever added
   without also fixing the underlying eval set.
2. **Confirmed by direct file-path comparison:** the legacy
   `eval_extract_activations.py` and the *default* (non-`--namespace`)
   `v2_pipeline.py` run write to the **exact same directory and exact
   same filenames** for `{stage}_final.npy` / `{stage}_pooled.npy` /
   `{stage}_metadata.json` (`results/activations/`). Running both against
   the same checkout — plausible, since `CLAUDE.md` still documents the
   legacy commands as "what the notebooks actually call" (they don't,
   see §3) — risks one overwriting the other's `.npy` arrays. Today this
   would likely just waste GPU time rather than corrupt anything
   permanently (v2's own `_metadata_binding.json` requirement means it
   would re-extract regardless), but it is a real footgun, and it is
   exactly the kind of Colab/T4-session-wasting failure mode flagged as a
   hard constraint for this project.
3. **Documentation drift:** `CLAUDE.md` §"Manual component-by-component"
   (lines 153–169) is stale relative to the actual release. It documents
   the legacy Components 1–3 scripts with no note that the supported,
   Colab-ready path is `v2_pipeline.py` / `rerun_mechanistic_v2.sh`
   instead. This is the most plausible root cause of the original
   "Best layer 0" / "Quadrant C n=20" reports if they came from someone
   following `CLAUDE.md` literally rather than `RESUME_PROMPT.md` or the
   notebook.
4. **Pre-existing, already flagged by a prior agent, not re-investigated
   this session (in `logs/agent_state.json`'s `unresolved_issues`):**
   "model_stage vs stage mismatch in causal ablation raw files."
   UNVERIFIABLE further within this session's scope.
5. `data/processed/controlled_eval.jsonl` and `data/frozen_v2/...jsonl`
   currently agree by construction discipline, not by an enforced check
   (§2) — a future edit to either file alone would silently desynchronize
   them with no error from any script that currently exists.

## 7. EXACT MINIMUM FIXES REQUIRED

*(Not performed this session — audit only. Listed for the next session.)*

1. `src/analysis/summarize_probe_findings.py`: derive `QUADRANTS`' `n`
   values from the actual data (e.g. from the loaded probe-result file's
   row provenance, or from `data/frozen_v2/LATEST_BENCHMARK.json`'s
   counts) instead of the hardcoded `20`/`50`/`200`.
2. Either delete/relocate the stale `results/probes/*.json` and
   `results/activations/*_metadata.json` (9+9 files, 370-row/C=20
   provenance) out of the active `results/` tree, or add a loud
   freshness banner so a stale read is visually unmistakable.
3. `CLAUDE.md` §"Manual component-by-component": add a deprecation note
   pointing at `v2_pipeline.py` / `rerun_mechanistic_v2.sh`, or remove
   the legacy commands from that list entirely, so no one — human or
   agent — runs them expecting current results again.
4. Either make the legacy `eval_extract_activations.py` refuse to run
   (or write to a clearly-legacy path) now that `v2_pipeline.py` owns
   `results/activations/`, to remove the same-path collision risk in
   §6.2.
5. (Minor, low priority) `pick_most_informative_layer()`'s all-zero tie
   case (M0-only, §3.A) could explicitly report "no signal at any layer"
   rather than silently returning layer 0 — cosmetic, not a correctness
   bug, since M0 is expected to show no safety signal.

## 8. WHAT DOES NOT NEED TO BE FIXED

- `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl` and
  `LATEST_BENCHMARK.json`: correct, SHA-verified, 654/A150/B250/C104/D150.
  **Do not touch** (per `RESUME_PROMPT.md`'s own explicit instruction).
- `src/v2_io.py`, `v2_pipeline.py`'s activation-binding logic
  (`activations_bound`/`load_bound_activation`): correctly designed and
  currently correctly reporting "not yet bound" (nothing to fix; it's
  doing its job).
- `logs/direction_split_manifest.json`, `logs/benchmark_gate_config.json`:
  frozen, verified, out of scope.
- `eval_probes.py`'s layer-selection logic itself: already fixed
  (picks by quadrant-C flagging rate, not accuracy) — the remaining M0
  tie-break is cosmetic (§7.5), not the original bug.
- `legacy/quadrant_c_arm2/`: intentionally archived, not part of the
  active benchmark (0 rows contributed) — leave as-is per
  `RESUME_PROMPT.md`.
- The v2 shard-resume/atomic-write machinery: already hardened in commit
  `c615204`; no further action identified this session.

## 9. RECOMMENDED NEXT SESSION

Given the constraints (limited agent sessions, limited paid compute, no
wasted Colab/T4 time): do **not** open Colab yet. First, in a cheap
CPU-only session, apply fixes §7.1–§7.4 (all pure text/file edits, no
GPU, no model or activation regeneration) and re-run only
`python -m src.analysis.summarize_probe_findings` locally (no GPU needed)
plus the two targeted test files already covering this machinery
(`tests/test_v2_io.py`, `tests/test_validate_benchmark_v2.py`) to confirm
nothing regressed. Only after that should the real T4 run in
`notebooks/colab_unified_analysis.ipynb` be started, exactly as
`RESUME_PROMPT.md` describes — that part of the pipeline (v2_pipeline.py
+ v2_io.py + v2_shards.py) audited clean and is genuinely the part
"colab_ready" refers to; the problem is entirely in the parallel legacy
scripts and the documentation still pointing at them.

---

AUDIT COMPLETE — SAFE TO PATCH
