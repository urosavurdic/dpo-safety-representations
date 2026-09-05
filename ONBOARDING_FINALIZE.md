# Onboarding: finish the DPO-safety-representations project (finalization phase)

Paste this whole file as your first message to a fresh Claude Code session in
this repo. It replaces re-deriving context from a long, expensive prior
session. Read `CLAUDE.md` and `docs/audit/analysis_plan.md` next, in that
order, before doing anything else.

## Who you're talking to — read this before writing a single message

The user is **not** an ML/ops person and gets genuinely, legitimately angry
at ambiguity, partial solutions, or being asked to manually fill in a
placeholder. Concrete rules, learned the hard way this session — do not
relearn them:

- **Never hand over a notebook/script with a "put your setup cell here"
  placeholder or any manual fill-in-the-blank.** If you give a notebook, it
  must be 100% runnable via Runtime → Run all, with real values baked in
  (actual repo URL, actual branch name, actual pinned commit — check
  `git rev-parse HEAD` yourself, don't guess or reuse an old one).
- **Deliver files via `git add`/`commit`/`push` to the current branch, not
  via a file-attachment tool.** A file-card delivery silently failed to
  reach the user earlier this session; git push → their `git pull` is the
  channel proven to work, every time, for this user.
- Give **numbered, single-path, copy-paste-ready** instructions. No "if
  X then do A, else do B" branching dumped on the user at once — pick the
  most likely path, state it plainly, and only branch if you must.
- **Verify claims against the actual live state (Drive contents, file
  timestamps, `v2_pipeline status` output) before telling the user
  something "is already done."** This session burned enormous time and
  trust on wrong assumptions (stale local-PC file timestamps mistaken for
  Drive state; a Colab Drive mount silently showing a stale/incomplete
  view while claiming "absent" when data existed elsewhere). When the user
  pastes a status/log output, read it literally — don't pattern-match to
  what you expect it to say.
- Money/tokens are a real, explicit concern to the user. Don't ask for
  another back-and-forth round if you can resolve something yourself
  (e.g. by reading a file, checking git, or running a local CPU test).

## Current state as of this handoff (verify before trusting — see rule above)

Branch: `agent/c-quadrant-end-to-end-e0e2317a`. Latest commits include
`confirmatory_behavioral_endpoints` per-branch CF2 support and
`notebooks/06_option2_finish_run.ipynb` (self-contained Colab notebook, real
setup cells baked in, pinned to a specific commit — check its current pin
against `git log` since more commits may have landed after this handoff).

**The confirmatory science, established and solid (do not re-derive, do not
distrust without new evidence):**
- **CF1** (C, M2→M3 behavioral transition): `Δ_C = -0.4008`, 95% CI
  `[-0.4603, -0.3396]`, n=104. Real, computed once already from a genuine
  judge run. (The raw data behind this specific number may currently be
  gone from the `dpo_v2` Drive folder — see below — but the number itself,
  and the fact it was computed correctly, is not in question.)
- **CF2** (M3, direction-specific causal effect): `+0.1136`, 95% CI
  `[+0.0279, +0.2064]`, n=30 held-out-A triples. WildGuard secondary agrees
  in direction (`+0.2`, CI excludes 0).
- Interpretation both endpoints support: DPO amplified an existing
  low-dimensional refusal direction (Hypothesis B/H2) rather than building
  a rich, general safety representation. See `docs/audit/analysis_plan.md`
  §3 for the exact claim language this project is bound to — do not use
  stronger language than it allows (e.g. never "DPO created a new safety
  representation").

**What "Option 2" (this finalization push) is extending:** CF2-style causal
ablation + steering to the other 3 DPO-endpoint branches (`M3_direct`,
`M3_alt`, `M3_direct_alt`), so the direct-DPO-vs-M2-mediated and
alt-vs-original-branch comparisons have real causal evidence, not just
descriptive/geometric evidence. `confirmatory_behavioral_endpoints.py`
already supports this (`CF2_by_stage`, tested, committed) — no more code
work needed there.

**GPU/Drive saga (read once, don't repeat the investigation):** the user's
Google Drive (`dpo_v2` folder, free tier, 15GB) hit its quota mid-session,
which caused real data loss / inconsistent visibility across the session —
not a mount bug that "just needed a refresh," actual missing files
(metadata written, `.npy` payloads not). Everything got resolved by
manually re-running extraction for the affected stages. **Before assuming
any file exists on Drive, have the user run and read the real output of:**
```
python -m src.analysis.v2_pipeline status
```
Do not trust local-PC file timestamps as a proxy for Drive state — they are
NOT the same filesystem and were wrongly conflated once already this
session, at real cost.

**Storage is still tight (was at 13.23/15 GB, 88%).** If more quota errors
happen, the fix that already worked: delete `dpo_v2/hf_cache` (fully
disposable — models just re-download, costing minutes not data) via the
Drive web UI, not by guessing at what else to delete.

## Exact remaining steps, in order

1. **Let the currently-running `notebooks/06_option2_finish_run.ipynb`
   finish completely**, if it hasn't already by the time you read this.
   Ask the user for its final output (especially cell 8's
   `confirmatory_endpoints.json` printout and cell 9/10's output) rather
   than assuming.
2. **CF1 follow-up** (behavioral responses for M2/M2_alt/M3/M3_alt were
   found missing from this Drive folder despite CF1 having been computed
   once already elsewhere/earlier — regenerate them here so this folder is
   self-consistent):
   ```
   python -m src.analysis.v2_pipeline behavior --stages M2 M2_alt M3 M3_alt
   python -m src.analysis.behavioral_judges \
     --response-manifest results/manifests/consolidated_judge.json \
     --from-results-dir results \
     --out-dir results/behavioral_judges_v2 \
     --run-live --scope confirmatory
   python -m src.analysis.confirmatory_behavioral_endpoints \
     --judged <the new behavioral_judges_v2_<ts>.json — use the newest one> \
     --benchmark data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl \
     --out results/summaries/confirmatory_endpoints.json
   ```
   After this, `CF2_by_stage` should be populated for M3 (confirmatory) and
   ideally M3_direct/M3_alt/M3_direct_alt (secondary) too, depending on
   whether cells 5-6 of the notebook completed for those branches — check,
   don't assume.
3. **CPU-only battery** (cell 9/10 of the notebook, or run standalone once
   `results/` is synced anywhere with all 9 stages' fresh activations):
   `subspace_geometry`, `projection_trajectory`, `direction_decodability`
   (**CF3** — see below), `representation_robustness`,
   `bottleneck_layer`, `bootstrap_direction_stability`,
   `bootstrap_cross_branch_difference`,
   `paired_deep_layer_stability_test --seed 20260904`,
   `summarize_probe_findings`, `summarize_cross_branch`, plus per-branch
   `summarize_causal_ablation`/`mcnemar_causal_ablation`/
   `bootstrap_causal_effect`/`summarize_steering` for the 3 new causal
   files and 2 new steering files. Exact commands are in the notebook and
   in `docs/REPRODUCE.md`'s POST-T4 section.
4. **Human annotation** (parallel/independent of the above — the user's own
   time, not GPU time): once `results/behavioral_judges_v2/<latest>.json`
   exists and is final,
   ```
   python -m src.analysis.build_human_review_packet \
     --responses results/behavioral_judges_v2/<latest>.json \
     --judged results/behavioral_judges_v2/<latest>.json \
     --packet-out results/human_review/packet.json \
     --key-out <somewhere OUTSIDE the repo>/SEALED_KEY.json
   ```
   The rubric + bilingual (English/Serbian) instructions are already
   written: `docs/human_review_instructions.md`. Point the user at it — do
   not re-explain the rubric from scratch, and do not change the rubric
   (it's frozen, `analysis_plan.md` §5.3). After annotation:
   ```
   python -m src.analysis.check_behavioral_agreement \
     --sealed-key <path>/SEALED_KEY.json \
     --annotations <path>/annotations.json \
     --judged results/behavioral_judges_v2/<latest>.json \
     --out results/behavioral_judges_v2/agreement_report.json
   ```
5. **README.md Findings rewrite**, sourced only from real numbers in
   `results/summaries/confirmatory_endpoints.json` and the CPU-battery
   outputs above, using the frozen claim language in
   `docs/audit/analysis_plan.md` §3. Do not carry forward old 370-era or
   pre-this-session numbers without flagging the change explicitly (this
   project's own convention — see CLAUDE.md's "bottleneck-layer walk-back"
   precedent for how to report a number changing between sessions).

## On a NeurIPS workshop abstract

The user asked whether drafting one is realistic in under a day. Honest
answer to give them, don't oversell: a short **abstract/extended-abstract**
(a few hundred words to ~2 pages, the typical workshop-track length) is
realistic within a day **once the numbers above are final** — the
motivation, method, and framing already exist in polished form in
`README.md` and `docs/audit/analysis_plan.md`; writing the abstract is
mostly compression, not new work. A full paper (related work, multiple
formal ablation write-ups, camera-ready formatting) is **not** realistic in
a day. Ask which specific workshop/deadline/template before committing to a
page count or claiming a hard deadline is hittable — that detail was never
established and changes what's actually achievable.

## Do not touch

`src/analysis/crossbranch/`, `tests/analysis/crossbranch/`,
`notebooks/crossbranch_stage0_stage1.ipynb`, `results/crossbranch/` — a
separate, parallel agent's uncommitted work. Leave it alone regardless of
its state when you look.
