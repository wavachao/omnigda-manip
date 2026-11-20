#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8080}"
BACKEND="${2:-tensorrt}"
exec "$(dirname "$0")/drun.sh" python -m omnigda.web.server --host 0.0.0.0 --port "${PORT}" --backend "${BACKEND}" --include-novel
