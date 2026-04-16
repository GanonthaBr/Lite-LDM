#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after upload to the VM.
USER_PYTHON_BIN="${PYTHON_BIN:-}"
PYTHON_BIN=""

# Priority:
# 1) Explicit override via PYTHON_BIN env var
# 2) Active conda env python (if any)
# 3) python3.11, python3.10, python3
if [[ -n "$USER_PYTHON_BIN" ]]; then
  PYTHON_BIN="$USER_PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" ]] && command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No suitable Python interpreter found."
  echo "Install Python 3.10+ or activate a conda env with Python 3.10+."
  exit 1
fi

echo "Using interpreter: $PYTHON_BIN"

PY_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION##*.}"

if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 10) )); then
  echo "Python $PY_VERSION detected, but this project requires Python 3.10+ for pinned dependencies."
  echo "Activate a newer conda env (recommended) or set PYTHON_BIN to python3.10/python3.11."
  exit 1
fi

$PYTHON_BIN -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Install project dependencies from both requirement files.
for req_file in requirements.txt requirementss.txt; do
  if [[ -f "$req_file" ]]; then
    echo "Installing dependencies from $req_file"
    python -m pip install -r "$req_file"
  else
    echo "Missing required dependency file: $req_file"
    exit 1
  fi
done

# Optional: if your VM image does not already include a CUDA-enabled torch build,
# install a matching wheel from https://pytorch.org/get-started/locally/
# Example (CUDA 12.1):
# python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo "VM setup complete. Activate env with: source .venv/bin/activate"
