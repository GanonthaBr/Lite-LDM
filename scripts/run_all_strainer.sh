#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Optional STRAINER encoder weights path:
# export STRAINER_WEIGHTS=/path/to/strainer_encoder_weights.pth

if [[ -n "${STRAINER_WEIGHTS:-}" ]]; then
  bash "$PROJECT_ROOT/scripts/run.sh" \
    --stage all \
    --encoder strainer \
    --min-content-ratio 0.001 \
    --checkpoint-dir "$PROJECT_ROOT/checkpoints_strainer" \
    --output-dir "$PROJECT_ROOT/outputs_strainer" \
    --strainer-weights "$STRAINER_WEIGHTS"
else
  bash "$PROJECT_ROOT/scripts/run.sh" \
    --stage all \
    --encoder strainer \
    --min-content-ratio 0.001 \
    --checkpoint-dir "$PROJECT_ROOT/checkpoints_strainer" \
    --output-dir "$PROJECT_ROOT/outputs_strainer"
fi
