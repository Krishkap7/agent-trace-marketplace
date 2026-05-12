#!/usr/bin/env bash
# Launch the Streamlit trace viewer against the local SQLite DB.
#
# Usage: bash scripts/run_ui.sh [extra streamlit args]
set -euo pipefail

cd "$(dirname "$0")/.."
exec uv run streamlit run trace_marketplace/ui/app.py "$@"
