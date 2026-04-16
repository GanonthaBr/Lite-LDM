#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# LiteLDM Bash runner for remote VMs.
# Usage examples:
#   bash scripts/run.sh --stage all
#   bash scripts/run.sh --stage train-vae --encoder strainer --strainer-weights /path/to/strainer_encoder_weights.pth
#   bash scripts/run.sh --stage generate --encoder strainer

STAGE="all"
ENCODER="conv"
STRAINER_WEIGHTS=""
LOCAL_PATH=""
TOKEN_ENV="huggingface_token"
MIN_CONTENT_RATIO="0.001"
VAE_EPOCHS="50"
DIFF_EPOCHS="100"
NUM_GENERATE="4"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
OUTPUT_DIR="$PROJECT_ROOT/outputs"
SAVE_TENSORS="1"
SHOW_PLOTS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --encoder)
      ENCODER="$2"
      shift 2
      ;;
    --strainer-weights)
      STRAINER_WEIGHTS="$2"
      shift 2
      ;;
    --local)
      LOCAL_PATH="$2"
      shift 2
      ;;
    --token-env)
      TOKEN_ENV="$2"
      shift 2
      ;;
    --min-content-ratio)
      MIN_CONTENT_RATIO="$2"
      shift 2
      ;;
    --vae-epochs)
      VAE_EPOCHS="$2"
      shift 2
      ;;
    --diff-epochs)
      DIFF_EPOCHS="$2"
      shift 2
      ;;
    --num-generate)
      NUM_GENERATE="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --no-save-tensors)
      SAVE_TENSORS="0"
      shift
      ;;
    --show-plots)
      SHOW_PLOTS="1"
      shift
      ;;
    -h|--help)
      echo "LiteLDM Bash Runner"
      echo "  --stage <preflight|data|train-vae|train-diffusion|generate|recon-check|all>"
      echo "  --encoder <conv|strainer>"
      echo "  --strainer-weights <path>"
      echo "  --local <path>"
      echo "  --token-env <env_var_name>"
      echo "  --min-content-ratio <float>"
      echo "  --vae-epochs <int>"
      echo "  --diff-epochs <int>"
      echo "  --num-generate <int>"
      echo "  --checkpoint-dir <path>"
      echo "  --output-dir <path>"
      echo "  --no-save-tensors"
      echo "  --show-plots"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Run with --help for usage."
      exit 1
      ;;
  esac
done

CMD=(python "$PROJECT_ROOT/scripts/run_pipeline.py"
  --stage "$STAGE"
  --encoder-backbone "$ENCODER"
  --token-env "$TOKEN_ENV"
  --min-content-ratio "$MIN_CONTENT_RATIO"
  --vae-epochs "$VAE_EPOCHS"
  --diff-epochs "$DIFF_EPOCHS"
  --num-generate "$NUM_GENERATE"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --output-dir "$OUTPUT_DIR")

if [[ -n "$LOCAL_PATH" ]]; then
  CMD+=(--local "$LOCAL_PATH")
fi

if [[ -n "$STRAINER_WEIGHTS" ]]; then
  CMD+=(--strainer-encoder-weights "$STRAINER_WEIGHTS")
fi

if [[ "$SAVE_TENSORS" == "1" ]]; then
  CMD+=(--save-tensors)
fi

if [[ "$SHOW_PLOTS" == "1" ]]; then
  CMD+=(--show-plots)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
