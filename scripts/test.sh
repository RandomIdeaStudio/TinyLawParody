#!/usr/bin/env bash
# Run the whole test suite (Python + JavaScript parity). Linux, cloud and CI.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${VIRTUAL_ENV:-}" ] && [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi
command -v node >/dev/null || { echo "Node.js is required (JavaScript parity tests)" >&2; exit 1; }
for f in web/*.js tests/js/*.js; do node --check "$f"; done
"$PY" -m pytest -q "$@"
