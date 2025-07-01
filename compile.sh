#!/usr/bin/env bash
set -euo pipefail

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
  echo "Error: uv is not installed. Run 'pip install uv' or check https://github.com/astral-sh/uv"
  exit 1
fi

echo "[+] Compiling pyproject.toml into requirements.txt..."
uv pip compile pyproject.toml > requirements.txt
echo "[✓] requirements.txt generated successfully"
