#!/usr/bin/env bash
#
# rms-hst-pipeline - Run All Checks
#
# Runs the checks that gate a merge. With no arguments it runs everything that is
# currently enforced; pass individual flags to run a subset.
#
# Usage:
#   ./scripts/run-all-checks.sh [options]
#
# Options:
#   --tests            Run the test suite, excluding the `network` tests
#   --network          Run only the `network` tests (queries MAST; slow)
#   --coverage         Run the test suite with a coverage report
#   --pyroma           Check the packaging metadata
#   --markdown         Lint the Markdown files
#   --sphinx           Build the documentation with warnings as errors
#   --ruff             Run ruff (not yet enforced in CI; expect findings)
#   --flake8-cont      Run the continuation-line indent check (not yet enforced)
#   -h, --help         Show this help message
#
# Environment:
#   VENV or VENV_PATH  Path to virtualenv (default: $PROJECT_ROOT/venv)
#
# Exit codes:
#   0 - All selected checks passed
#   1 - At least one check failed
#

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV:-${VENV_PATH:-$PROJECT_ROOT/venv}}"

if [[ -x "$VENV_PATH/bin/python" ]]; then
    PY="$VENV_PATH/bin/python"
else
    echo -e "${YELLOW}No virtualenv at $VENV_PATH; using the python3 on PATH.${NC}"
    echo -e "${YELLOW}Run ./scripts/setup-venv.sh to create one.${NC}"
    PY="python3"
fi

RUN_TESTS=0 RUN_NETWORK=0 RUN_COVERAGE=0 RUN_PYROMA=0 RUN_MARKDOWN=0 RUN_RUFF=0 RUN_FLAKE8=0
RUN_SPHINX=0
EXPLICIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tests)       RUN_TESTS=1;    EXPLICIT=1; shift ;;
        --network)     RUN_NETWORK=1;  EXPLICIT=1; shift ;;
        --coverage)    RUN_COVERAGE=1; EXPLICIT=1; shift ;;
        --pyroma)      RUN_PYROMA=1;   EXPLICIT=1; shift ;;
        --markdown)    RUN_MARKDOWN=1; EXPLICIT=1; shift ;;
        --sphinx)      RUN_SPHINX=1;   EXPLICIT=1; shift ;;
        --ruff)        RUN_RUFF=1;     EXPLICIT=1; shift ;;
        --flake8-cont) RUN_FLAKE8=1;   EXPLICIT=1; shift ;;
        -h|--help)     sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}" >&2; exit 1 ;;
    esac
done

# Default: everything CI currently enforces. Ruff, the continuation-line check and the
# network tests stay opt-in, matching the commented-out steps in run-tests.yml.
if [[ $EXPLICIT -eq 0 ]]; then
    RUN_TESTS=1 RUN_PYROMA=1 RUN_MARKDOWN=1 RUN_SPHINX=1
fi

for var in HST_STAGING HST_PIPELINE HST_BUNDLES; do
    if [[ -z "${!var:-}" ]]; then
        echo -e "${RED}$var is not set.${NC} The package reads it at import time, so"
        echo -e "${RED}test collection fails without it. See scripts/setup-venv.sh.${NC}"
        exit 1
    fi
done

FAILED=()

run_check() {
    local name="$1"; shift
    echo -e "\n${BLUE}=== $name ===${NC}"
    if "$@"; then
        echo -e "${GREEN}PASS: $name${NC}"
    else
        echo -e "${RED}FAIL: $name${NC}"
        FAILED+=("$name")
    fi
}

[[ $RUN_TESTS    -eq 1 ]] && run_check "Tests (non-network)" "$PY" -m pytest -m "not network and not targets_data" "$PROJECT_ROOT/tests"
[[ $RUN_NETWORK  -eq 1 ]] && run_check "Tests (network)"     "$PY" -m pytest -m network -v -n 0 "$PROJECT_ROOT/tests"
[[ $RUN_COVERAGE -eq 1 ]] && run_check "Coverage"            "$PY" -m pytest -m "not network and not targets_data" --cov=src --cov-report=term-missing --cov-report=html "$PROJECT_ROOT/tests"
[[ $RUN_PYROMA   -eq 1 ]] && run_check "Pyroma"              "$PY" -m pyroma "$PROJECT_ROOT"
[[ $RUN_MARKDOWN -eq 1 ]] && run_check "PyMarkdown"          "$PY" -m pymarkdown scan "$PROJECT_ROOT/README.md" "$PROJECT_ROOT/CONTRIBUTING.md" "$PROJECT_ROOT/CLAUDE.md"
[[ $RUN_SPHINX   -eq 1 ]] && run_check "Sphinx"               "$PY" -m sphinx -W -b html "$PROJECT_ROOT/docs" "$PROJECT_ROOT/docs/_build"
[[ $RUN_RUFF     -eq 1 ]] && run_check "Ruff"                "$PY" -m ruff check "$PROJECT_ROOT/src" "$PROJECT_ROOT/tests"
[[ $RUN_FLAKE8   -eq 1 ]] && run_check "Flake8 continuation" "$PY" -m flake8 --select=E12,E13 "$PROJECT_ROOT/src" "$PROJECT_ROOT/tests"

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo -e "${GREEN}All checks passed.${NC}"
    exit 0
fi
echo -e "${RED}Failed: ${FAILED[*]}${NC}"
exit 1
