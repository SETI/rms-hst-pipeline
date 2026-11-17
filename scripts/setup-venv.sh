#!/usr/bin/env bash
#
# rms-hst-pipeline - Virtual Environment Bootstrap
#
# Creates the virtualenv that scripts/run-all-checks.sh expects and installs the project
# in editable mode with its development and documentation extras. Safe to re-run: an
# existing environment is reused and its packages upgraded.
#
# Usage:
#   ./scripts/setup-venv.sh [options]
#
# Options:
#   -r, --recreate         Delete an existing virtualenv and build a fresh one
#   -p, --python CMD       Interpreter used to create the venv (default: python3)
#   -h, --help             Show this help message
#
# Environment:
#   VENV or VENV_PATH      Path to virtualenv (default: $PROJECT_ROOT/venv)
#
# Exit codes:
#   0 - Environment ready
#   1 - Bootstrap failed
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV:-${VENV_PATH:-$PROJECT_ROOT/venv}}"
PYTHON_CMD="python3"
RECREATE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--recreate) RECREATE=1; shift ;;
        -p|--python)   PYTHON_CMD="$2"; shift 2 ;;
        -h|--help)     sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}" >&2; exit 1 ;;
    esac
done

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo -e "${RED}Interpreter not found: $PYTHON_CMD${NC}" >&2
    exit 1
fi

if [[ $RECREATE -eq 1 && -d "$VENV_PATH" ]]; then
    echo -e "${YELLOW}Removing existing environment at $VENV_PATH${NC}"
    rm -rf "$VENV_PATH"
fi

if [[ ! -d "$VENV_PATH" ]]; then
    echo -e "${BLUE}Creating virtualenv at $VENV_PATH${NC}"
    "$PYTHON_CMD" -m venv "$VENV_PATH"
else
    echo -e "${BLUE}Reusing existing virtualenv at $VENV_PATH${NC}"
fi

echo -e "${BLUE}Upgrading pip and build tooling${NC}"
"$VENV_PATH/bin/python" -m pip install -q --upgrade pip setuptools wheel

echo -e "${BLUE}Installing rms-hst-pipeline with dev and docs extras${NC}"
"$VENV_PATH/bin/python" -m pip install -e "$PROJECT_ROOT[dev]"

cat <<MSG

$(echo -e "${GREEN}Environment ready.${NC}")

Activate it with:

    source $VENV_PATH/bin/activate

The pipeline reads three environment variables at import time, with no defaults. Set them
before running the tests or any task:

    export HST_STAGING=/path/to/staging
    export HST_PIPELINE=/path/to/pipeline
    export HST_BUNDLES=/path/to/bundles

MSG
