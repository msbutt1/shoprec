#!/bin/bash
# Run all linting, formatting, and testing checks
#
# Usage:
#   ./scripts/lint_all.sh [--check] [--skip-tests] [--fix]
#
# Options:
#   --check: Only check formatting (don't modify files)
#   --skip-tests: Skip running pytest
#   --fix: Auto-fix formatting issues (default behavior)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
CHECK_MODE=false
SKIP_TESTS=false
FIX_MODE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_MODE=true
            FIX_MODE=false
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --fix)
            FIX_MODE=true
            CHECK_MODE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "ShopRec Code Quality Checks"
echo "============================================================"
echo ""

ALL_PASSED=true

# Function to run a command and check result
run_check() {
    local description=$1
    shift
    local cmd=("$@")
    
    echo "============================================================"
    echo "Running: $description"
    echo "Command: ${cmd[*]}"
    echo "============================================================"
    echo ""
    
    if "${cmd[@]}"; then
        echo ""
        echo -e "${GREEN}✓${NC} $description passed"
        echo ""
        return 0
    else
        echo ""
        echo -e "${RED}✗${NC} $description failed"
        echo ""
        return 1
    fi
}

# 1. Run isort (import sorting)
if [ "$CHECK_MODE" = true ]; then
    if ! run_check "isort (import sorting)" isort src tests scripts --check-only --diff; then
        ALL_PASSED=false
    fi
else
    if ! run_check "isort (import sorting)" isort src tests scripts --check; then
        echo "Attempting to auto-fix import sorting..."
        if run_check "isort (auto-fix)" isort src tests scripts; then
            echo -e "${GREEN}Import sorting fixed!${NC}"
        else
            ALL_PASSED=false
        fi
    fi
fi

# 2. Run black (code formatting)
if [ "$CHECK_MODE" = true ]; then
    if ! run_check "black (code formatting)" black src tests scripts --check; then
        ALL_PASSED=false
    fi
else
    if ! run_check "black (code formatting)" black src tests scripts --check; then
        echo "Attempting to auto-fix code formatting..."
        if run_check "black (auto-fix)" black src tests scripts; then
            echo -e "${GREEN}Code formatting fixed!${NC}"
        else
            ALL_PASSED=false
        fi
    fi
fi

# 3. Run pytest (if not skipped)
if [ "$SKIP_TESTS" = false ]; then
    if ! run_check "pytest (tests)" pytest tests/ -v; then
        ALL_PASSED=false
    fi
fi

# Summary
echo "============================================================"
if [ "$ALL_PASSED" = true ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "============================================================"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please fix the issues above.${NC}"
    echo "============================================================"
    echo ""
    exit 1
fi

