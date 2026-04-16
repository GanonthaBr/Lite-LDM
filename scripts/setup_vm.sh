#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Keep environment inside project root by default unless VENV_DIR is explicitly provided.
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
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

if [[ -x "$VENV_DIR/bin/python" ]]; then
  VENV_VERSION="$($VENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$VENV_VERSION" != "$PY_VERSION" ]]; then
    echo "Existing $VENV_DIR uses Python $VENV_VERSION; recreating with Python $PY_VERSION"
    rm -rf "$VENV_DIR"
  fi
fi

mkdir -p "$(dirname "$VENV_DIR")"
$PYTHON_BIN -m venv --clear "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

install_requirements_linewise() {
  local req_file="$1"
  local strict_mode="$2"
  local tmp_file
  local failed_file
  local had_failures=0

  tmp_file="$(mktemp)"
  failed_file="$(mktemp)"

  # Some exported requirement files are UTF-16. Convert when needed.
  if iconv -f utf-16 -t utf-8 "$req_file" > "$tmp_file" 2>/dev/null; then
    :
  else
    cp "$req_file" "$tmp_file"
  fi

  # Normalize CRLF line endings.
  sed -i 's/\r$//' "$tmp_file"

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="$(echo "$raw_line" | sed 's/^\s*//;s/\s*$//')"

    if [[ -z "$line" ]] || [[ "$line" == \#* ]]; then
      continue
    fi

    # Skip pip options/references in linewise mode.
    if [[ "$line" == -* ]]; then
      continue
    fi

    echo "  -> Installing: $line"
    if ! python -m pip install "$line"; then
      echo "$line" >> "$failed_file"
      had_failures=1
    fi
  done < "$tmp_file"

  if [[ "$had_failures" -eq 1 ]]; then
    if [[ "$strict_mode" == "strict" ]]; then
      echo "Failed to install one or more dependencies from $req_file:"
      cat "$failed_file"
      rm -f "$tmp_file" "$failed_file"
      exit 1
    else
      echo "Warning: some entries in $req_file are not pip-installable in this environment (often conda-only packages)."
      echo "Skipped entries:"
      cat "$failed_file"
    fi
  fi

  rm -f "$tmp_file" "$failed_file"
}

# Install project dependencies from both requirement files.
for req_file in requirements.txt requirementss.txt; do
  if [[ ! -f "$req_file" ]]; then
    echo "Missing required dependency file: $req_file"
    exit 1
  fi

  echo "Installing dependencies from $req_file"
  if [[ "$req_file" == "requirements.txt" ]]; then
    # This file may include conda-exported entries unavailable to pip.
    install_requirements_linewise "$req_file" "warn"
  else
    # This file is expected to be fully pip-installable.
    install_requirements_linewise "$req_file" "strict"
  fi
done

# Optional: if your VM image does not already include a CUDA-enabled torch build,
# install a matching wheel from https://pytorch.org/get-started/locally/
# Example (CUDA 12.1):
# python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo "VM setup complete. Activate env with: source $VENV_DIR/bin/activate"
