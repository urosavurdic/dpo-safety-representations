#!/usr/bin/env bash
set -euo pipefail
exec python -m src.analysis.v2_pipeline run "$@"
