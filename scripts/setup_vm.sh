#!/usr/bin/env bash
set -euo pipefail

# Run from repository root after upload to the VM.
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel

# Install project dependencies.
python -m pip install -r requirements.txt

# Optional: if your VM image does not already include a CUDA-enabled torch build,
# install a matching wheel from https://pytorch.org/get-started/locally/
# Example (CUDA 12.1):
# python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo "VM setup complete. Activate env with: source .venv/bin/activate"
