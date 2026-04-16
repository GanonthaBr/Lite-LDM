#!/usr/bin/env bash
set -euo pipefail

# Optional STRAINER encoder weights path:
# export STRAINER_WEIGHTS=/path/to/strainer_encoder_weights.pth

if [[ -n "${STRAINER_WEIGHTS:-}" ]]; then
  bash scripts/run.sh \
    --stage all \
    --encoder strainer \
    --min-content-ratio 0.001 \
    --checkpoint-dir ./checkpoints_strainer \
    --output-dir ./outputs_strainer \
    --strainer-weights "$STRAINER_WEIGHTS"
else
  bash scripts/run.sh \
    --stage all \
    --encoder strainer \
    --min-content-ratio 0.001 \
    --checkpoint-dir ./checkpoints_strainer \
    --output-dir ./outputs_strainer
fi
