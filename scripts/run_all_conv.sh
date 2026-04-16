#!/usr/bin/env bash
set -euo pipefail

bash scripts/run.sh \
  --stage all \
  --encoder conv \
  --min-content-ratio 0.001 \
  --checkpoint-dir ./checkpoints_conv \
  --output-dir ./outputs_conv
