#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:1
#SBATCH -J liteldm-strainer
#SBATCH -o /ocean/projects/cis260093p/gpayang/Lite-LDM/slurm-%j.out
#SBATCH -e /ocean/projects/cis260093p/gpayang/Lite-LDM/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260093p/gpayang/Lite-LDM"
cd "$PROJECT_ROOT"

mkdir -p "$PROJECT_ROOT/logs"

# Keep data and artifacts on the project storage path.
export LOCAL="${LOCAL:-$PROJECT_ROOT/local_storage}"
mkdir -p "$LOCAL"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  echo "Missing Python environment at $PROJECT_ROOT/.venv"
  echo "Run setup first from login node:"
  echo "  PYTHON_BIN=\$(which python) VENV_DIR=$PROJECT_ROOT/.venv bash scripts/setup_vm.sh"
  exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

python - <<'PY'
import sys
import torch
print(f"Python: {sys.version.split()[0]}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible on this node. Exiting.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

bash "$PROJECT_ROOT/scripts/run_all_strainer.sh"
