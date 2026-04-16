#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

bash "$PROJECT_ROOT/scripts/run.sh" \
  --stage all \
  --encoder conv \
  --min-content-ratio 0.001 \
  --checkpoint-dir "$PROJECT_ROOT/checkpoints_conv" \
  --output-dir "$PROJECT_ROOT/outputs_conv"
