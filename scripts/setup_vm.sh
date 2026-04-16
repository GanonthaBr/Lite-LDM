#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after upload to the VM.
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel

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
