#!/bin/bash
# Post-hoc failure analysis of eval results JSONL files.
#
# Usage:
#   ./scripts/analyze-failures.sh -i <results.jsonl> [options]
#   ./scripts/analyze-failures.sh -i 'outputs/**/runs/*.jsonl'
#   ./scripts/analyze-failures.sh -i outputs/data/zapier/20260411_zapier_clean/runs/
#
# All arguments pass through to src.data.analyze_failures.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python -m src.data.analyze_failures "$@"
