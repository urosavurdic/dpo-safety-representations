#!/usr/bin/env bash
set -euo pipefail
exec python -m src.pipeline.frozen_run_pipeline run "$@"
