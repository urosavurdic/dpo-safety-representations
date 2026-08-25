# RESUME PROMPT — v2 Benchmark Pipeline

Use this prompt to start a new Claude session that picks up exactly where the
previous agent left off. The agent must NOT rely on any prior conversation memory.

---

## Context

Repository: https://github.com/urosavurdic/dpo-safety-representations
Branch: `agent/c-quadrant-end-to-end-e0e2317a`
Base commit: `e0e2317a52e89b0d614b99152a9ca71758baf489`

The researcher has just reviewed and committed `data/review/c_review_queue.csv`
with `review_status` set to `accept` or `reject` for all 104 rows.

---

## What was done before this prompt

Phase 0–7 are complete:
- Full reconciliation audit (see `logs/reconciliation_report.{json,md}`)
- C-paired provenance reconstructed in `data/quadrant_c_pipeline/candidate_records_v2.jsonl`
- Gate config frozen at `logs/benchmark_gate_config.json`
- Direction split manifest at `logs/direction_split_manifest.json`
- Review queue generated at `data/review/c_review_queue.csv` (104 rows, all c_paired)
- Scripts: `src/finalize_benchmark.py`, `src/validate_benchmark_v2.py`, `rerun_mechanistic_v2.sh`

---

## Your first actions (mandatory, in order)

```bash
# 1. Orient
git fetch origin
git checkout agent/c-quadrant-end-to-end-e0e2317a
cat logs/agent_state.json
git status --short

# 2. Verify the reviewed CSV was committed and has no pending rows
sha256sum data/review/c_review_queue.csv
python3 -c "
import csv
rows = list(csv.DictReader(open('data/review/c_review_queue.csv')))
pending = [r for r in rows if r['review_status'].strip() == 'pending']
print(f'Total: {len(rows)}, pending: {len(pending)}')
accepted = [r for r in rows if r['review_status'].strip() == 'accept']
print(f'Accepted: {len(accepted)}')
"

# 3. Finalize the frozen benchmark
python -m src.finalize_benchmark \
    --review-csv data/review/c_review_queue.csv \
    --gate-config logs/benchmark_gate_config.json

# 4. Run validation
BENCH=$(python3 -c "import json; print(json.load(open('data/frozen_v2/LATEST_BENCHMARK.json'))['benchmark_path'])")
python -m src.validate_benchmark_v2 \
    --benchmark "$BENCH" \
    --review-csv data/review/c_review_queue.csv \
    --gate-config logs/benchmark_gate_config.json \
    --split-manifest logs/direction_split_manifest.json

# 5. Check status
cat logs/benchmark_validation_status.json | python3 -m json.tool | grep -E "technical_benchmark|reduced_cue|wording_only"

# 6. Generate patch
BASE=e0e2317a
CODE=$(git rev-parse HEAD | cut -c1-8)
git diff --binary $BASE HEAD \
    -- . ':(exclude)artifacts/patches/*.patch' \
    > artifacts/patches/c_quadrant_${BASE}_${CODE}.patch
sha256sum artifacts/patches/c_quadrant_${BASE}_${CODE}.patch
```

---

## Current blockers (must report, not ignore)

1. `c_review_pass` will FAIL until all rows are accepted or rejected.
2. `artifact_freshness_pass` will FAIL until GPU rerun regenerates activations against the frozen benchmark.
3. C-source-authored arm is NOT built — declare it unavailable unless you acquire HarmBench CSV locally.
4. `model_stage` vs `stage` mismatch in `results/raw/causal_ablation_raw_narrow.json` — needs fix in summarization code if that path is used.

---

## Files the next agent MUST NOT overwrite

- `data/processed/controlled_eval.jsonl` (any old result artifacts)
- `logs/benchmark_gate_config.json` (frozen before validation)
- Any existing `results/` file (mark stale, do not delete)

---

## Unresolved issues (from logs/agent_state.json)

- C-source-authored arm not built
- Activation artifacts stale (370-row era); GPU rerun required
- StrongREJECT source CSV SHA-256 not recorded
- Near-duplicate check not run (embedding model unavailable)

---

## Queue SHA-256 (verify before using)

`data/review/c_review_queue.csv` SHA-256 when generated:
`9e0f82592bb6e87953b05bcb65177a6a7651152e087edef43c2429b5cbe1ef01`

If the SHA after researcher edit differs, that's expected (review fields changed).
Verify that ONLY the review fields changed — no prompt text was edited.
