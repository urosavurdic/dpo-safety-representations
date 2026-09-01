# RESUME PROMPT — v2 Benchmark Pipeline

Use this prompt to start a new Claude session that picks up exactly where
the previous agent left off. The agent must NOT rely on any prior
conversation memory.

---

## Context

Repository: https://github.com/urosavurdic/dpo-safety-representations
Branch: `agent/c-quadrant-end-to-end-e0e2317a`
Final HEAD as of this handoff: `3e668107fd966f1af0f0eea13c95e2626d2aedf9`

All CPU-side release engineering is done: the R104/c_paired benchmark is
frozen and validated, Arm-2 is deferred and archived under
`legacy/quadrant_c_arm2/`, `pytest`/`compileall` are clean (modulo missing
heavy ML deps in CPU-only sandboxes — not a code issue), and a fresh clone
resolves everything without manual intervention. Full detail:
`logs/FINAL_RELEASE_HANDOFF.md`.

**Nothing here needs further CPU/release-engineering work.** The only
remaining step is the GPU run in Colab.

---

## Your first actions (mandatory, in order)

```bash
# 1. Orient
git fetch origin
git checkout agent/c-quadrant-end-to-end-e0e2317a
cat logs/agent_state.json
cat logs/FINAL_RELEASE_HANDOFF.md
git status --short   # expect clean

# 2. Confirm you're at (or past) the handoff commit
git log -1 --format='%H'   # expect 3e668107fd966f1af0f0eea13c95e2626d2aedf9 or a descendant
```

If `git status --short` is not empty, or HEAD is not
`3e668107fd966f1af0f0eea13c95e2626d2aedf9` (or a descendant of it), stop
and reconcile before doing anything else — do not assume this prompt's
claims still hold.

## What's actually left: the Colab GPU run

1. Open `notebooks/colab_unified_analysis.ipynb` in Colab.
2. Runtime > Change runtime type > T4 GPU.
3. In the "Clone and pin the exact commit" cell, set:
   `PINNED_COMMIT = '3e668107fd966f1af0f0eea13c95e2626d2aedf9'`
   (or the current HEAD if this branch has moved since).
4. Run top-to-bottom through the dry-run/calibration cells — safe to
   repeat, costs only a small calibration pass, not the full GPU budget.
5. In the live-run cell, set `RUN_GPU = True` and re-run it. This invokes
   `bash rerun_mechanistic_v2.sh --regenerate --with-probes
   --with-norm-diag --act-batch auto --gen-batch auto --deadline-minutes
   <SESSION_DEADLINE_MINUTES>` under the hood.
6. If the session disconnects or the (default 300-minute) deadline is
   hit: reopen the notebook, run top-to-bottom again. Completed
   shards/stages under `results/behavioral_eval/v2_shards/`,
   `results/raw/v2_causal_shards/`, `results/raw/v2_steering_shards/` are
   skipped automatically; the run resumes from the first unfinished
   shard. `results/` and the HF cache are bound to a persistent Drive
   folder (`/content/drive/MyDrive/dpo_safety_v2/`), so this survives
   disconnects and fresh runtimes.

## Current blockers

None on the CPU/release-engineering side. The only reason no GPU results
exist yet is that the live run hasn't been started — see above.

## Files the next agent MUST NOT overwrite

- `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl` and
  `data/frozen_v2/LATEST_BENCHMARK.json` (frozen; changing either
  invalidates every downstream hash-bound artifact)
- `logs/direction_split_manifest.json` (regenerate only via
  `python -m src.create_direction_split_manifest`, never hand-edit)
- `logs/benchmark_gate_config.json` (frozen before validation)
- Any existing `results/` file (mark stale, do not delete — the shard
  resume logic depends on what's already there)
- `legacy/quadrant_c_arm2/**` (archived Arm-2 code — move back into the
  active path only on an explicit decision to un-defer Arm-2, not as
  routine cleanup)

## Known limitations carried forward

See `README.md` §Limitations and `logs/FINAL_RELEASE_HANDOFF.md` for the
current list (LoRA confound, single direction, 1.5B scale, etc.) — not
repeated here to avoid drift between copies.
