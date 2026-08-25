#!/usr/bin/env bash
# =============================================================================
# rerun_mechanistic_v2.sh — GPU-gate and orchestrator for the v2 mechanistic
# rerun. Reads ONLY JSON files. Aborts if any required gate field is not true.
#
# Usage:
#   bash rerun_mechanistic_v2.sh --dry-run
#   bash rerun_mechanistic_v2.sh --stage M3 --stage M3_alt
#
# Required JSON files:
#   logs/benchmark_validation_status.json
#   logs/benchmark_gate_config.json
#   data/frozen_v2/LATEST_BENCHMARK.json
#   logs/direction_split_manifest.json
# =============================================================================
set -euo pipefail

DRY_RUN=0
STAGES=()

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stage)   STAGES+=("$2"); shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Default stages if none specified
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=("M3" "M3_alt")
fi

# ── Helper: read JSON field ─────────────────────────────────────────────────
json_get() { python3 -c "import json,sys; d=json.load(open('$1')); print(d.get('$2','MISSING'))" ; }
json_get_nested() { python3 -c "import json,sys; d=json.load(open('$1')); keys='$2'.split('.'); v=d; [v.__setitem__(k,v.get(k,{})) or v.update({k:v.pop(k)}) for k in keys[:-1]]; print(v.get(keys[-1],'MISSING'))" 2>/dev/null || echo "MISSING" ; }

# ── Required JSON files ──────────────────────────────────────────────────────
for f in \
  "logs/benchmark_validation_status.json" \
  "logs/benchmark_gate_config.json" \
  "data/frozen_v2/LATEST_BENCHMARK.json" \
  "logs/direction_split_manifest.json"
do
  if [[ ! -f "$f" ]]; then
    echo "ABORT: required file missing: $f" >&2
    exit 1
  fi
done

# ── Load key values ──────────────────────────────────────────────────────────
TBS=$(json_get logs/benchmark_validation_status.json technical_benchmark_status)
BENCH_PATH=$(json_get data/frozen_v2/LATEST_BENCHMARK.json benchmark_path)
BENCH_SHA=$(json_get data/frozen_v2/LATEST_BENCHMARK.json benchmark_sha256)
SPLIT_SHA=$(json_get logs/direction_split_manifest.json split_manifest_sha256)

echo ""
echo "========================================"
echo " rerun_mechanistic_v2.sh"
echo "========================================"
echo "  Benchmark:    $BENCH_PATH"
echo "  Bench SHA:    $BENCH_SHA"
echo "  Split SHA:    $SPLIT_SHA"
echo "  Stages:       ${STAGES[*]}"
echo ""

# ── Print warnings (warning-only gate fields) ────────────────────────────────
python3 - << 'PYEOF'
import json, sys
status = json.load(open("logs/benchmark_validation_status.json"))
gate   = json.load(open("logs/benchmark_gate_config.json"))
warn_fields = gate.get("warning_only_gate_fields", [])
print("WARNING-ONLY gate fields:")
for f in warn_fields:
    v = status.get(f, "null")
    marker = "⚠ " if v in (None, False, "null", "MISSING", "INCONCLUSIVE") else "✓ "
    print(f"  {marker}{f}: {v}")
PYEOF

# ── Required gate fields — abort if not exactly True ─────────────────────────
python3 - << 'PYEOF'
import json, sys
status = json.load(open("logs/benchmark_validation_status.json"))
gate   = json.load(open("logs/benchmark_gate_config.json"))
required = gate.get("required_gate_fields", [])
failed = []
for f in required:
    v = status.get(f)
    if v is not True:
        failed.append(f"{f}={v}")
if failed:
    print(f"ABORT: required gate fields not True: {', '.join(failed)}", file=sys.stderr)
    sys.exit(1)
print("All required gate fields: PASS")
PYEOF

# ── Technical benchmark status ────────────────────────────────────────────────
if [[ "$TBS" != "PASS" ]]; then
  echo "ABORT: technical_benchmark_status=$TBS (must be PASS)" >&2
  exit 1
fi

# ── Benchmark file exists and hash matches ────────────────────────────────────
if [[ ! -f "$BENCH_PATH" ]]; then
  echo "ABORT: benchmark file not found: $BENCH_PATH" >&2
  exit 1
fi
ACTUAL_SHA=$(sha256sum "$BENCH_PATH" | awk '{print $1}')
if [[ "$ACTUAL_SHA" != "$BENCH_SHA" ]]; then
  echo "ABORT: benchmark SHA mismatch!" >&2
  echo "  Expected: $BENCH_SHA" >&2
  echo "  Actual:   $ACTUAL_SHA" >&2
  exit 1
fi
echo "  Benchmark hash: ✓"

# ── Dry-run mode ──────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "=== DRY RUN — commands that WOULD execute ==="
  for STAGE in "${STAGES[@]}"; do
    echo ""
    echo "── Stage: $STAGE ──"
    echo "  python -m src.experiments.extract_activations \\"
    echo "      --eval-set $BENCH_PATH \\"
    echo "      --benchmark-sha256 $BENCH_SHA \\"
    echo "      --stage $STAGE \\"
    echo "      --force-regen"
    echo ""
    echo "  python -m src.experiments.compute_refusal_direction \\"
    echo "      --eval-set $BENCH_PATH \\"
    echo "      --split-manifest logs/direction_split_manifest.json \\"
    echo "      --stage $STAGE \\"
    echo "      --force-regen"
    echo ""
    echo "  python -m src.experiments.run_behavioral_eval \\"
    echo "      --eval-set $BENCH_PATH \\"
    echo "      --benchmark-sha256 $BENCH_SHA \\"
    echo "      --stage $STAGE"
    echo ""
    echo "  python -m src.experiments.run_causal_ablation \\"
    echo "      --eval-set $BENCH_PATH \\"
    echo "      --benchmark-sha256 $BENCH_SHA \\"
    echo "      --split-manifest logs/direction_split_manifest.json \\"
    echo "      --stage $STAGE \\"
    echo "      --conditions baseline ablated norm_matched_control"
  done
  echo ""
  echo "=== END DRY RUN ==="
  exit 0
fi

# ── Live run (GPU required) ───────────────────────────────────────────────────
echo ""
echo "Starting live GPU run for stages: ${STAGES[*]}"

for STAGE in "${STAGES[@]}"; do
  echo ""
  echo "═══ Stage: $STAGE ═══"

  # Activation extraction
  python -m src.experiments.extract_activations \
      --eval-set "$BENCH_PATH" \
      --benchmark-sha256 "$BENCH_SHA" \
      --stage "$STAGE" \
      --force-regen

  # Refusal direction
  python -m src.experiments.compute_refusal_direction \
      --eval-set "$BENCH_PATH" \
      --split-manifest logs/direction_split_manifest.json \
      --stage "$STAGE" \
      --force-regen

  # Behavioral eval
  python -m src.experiments.run_behavioral_eval \
      --eval-set "$BENCH_PATH" \
      --benchmark-sha256 "$BENCH_SHA" \
      --stage "$STAGE"

  # Causal ablation
  python -m src.experiments.run_causal_ablation \
      --eval-set "$BENCH_PATH" \
      --benchmark-sha256 "$BENCH_SHA" \
      --split-manifest logs/direction_split_manifest.json \
      --stage "$STAGE" \
      --conditions baseline ablated norm_matched_control
done

echo ""
echo "GPU run complete. Commit ONLY small artifacts (manifests, hashes, summaries)."
echo "Store large .npy arrays in Drive and record their SHA-256 and benchmark hash."
