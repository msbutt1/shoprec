#!/usr/bin/env python3
"""Run all linting, formatting, and testing checks.

This script runs black, isort, and pytest in sequence to ensure code quality.
Can be run from the project root directory.

Usage:
    python scripts/lint_all.py [--check] [--skip-tests] [--fix]

Options:
    --check: Only check formatting (don't modify files)
    --skip-tests: Skip running pytest
    --fix: Auto-fix formatting issues (default behavior)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful.
    
    Args:
        cmd: Command to run as list of strings
        description: Human-readable description of what's being run
    
    Returns:
        True if command succeeded (exit code 0), False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=False,
        )
        
        if result.returncode == 0:
            print(f"\n✓ {description} passed\n")
            return True
        else:
            print(f"\n✗ {description} failed (exit code: {result.returncode})\n")
            return False
            
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"  Make sure the command is installed and in your PATH\n")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        return False


def main() -> int:
    """Main entry point for linting script.
    
    Returns:
        Exit code: 0 if all checks passed, 1 otherwise
    """
    parser = argparse.ArgumentParser(
        description="Run linting, formatting, and testing checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check formatting (don't modify files)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix formatting issues (default behavior)",
    )
    
    args = parser.parse_args()
    
    # Determine if we're checking or fixing
    check_mode = args.check and not args.fix
    
    print("\n" + "="*60)
    print("ShopRec Code Quality Checks")
    print("="*60)
    
    all_passed = True
    
    # 1. Run isort (import sorting)
    isort_cmd = ["isort", "src", "tests", "scripts"]
    if check_mode:
        isort_cmd.extend(["--check-only", "--diff"])
    else:
        isort_cmd.append("--check")  # Check first, then we'll fix if needed
    
    if not run_command(isort_cmd, "isort (import sorting)"):
        if not check_mode:
            # Try to fix it
            fix_cmd = ["isort", "src", "tests", "scripts"]
            print("Attempting to auto-fix import sorting...")
            if run_command(fix_cmd, "isort (auto-fix)"):
                print("Import sorting fixed!")
            else:
                all_passed = False
        else:
            all_passed = False
    
    # 2. Run black (code formatting)
    black_cmd = ["black", "src", "tests", "scripts"]
    if check_mode:
        black_cmd.append("--check")
    
    if not run_command(black_cmd, "black (code formatting)"):
        if not check_mode:
            # Try to fix it
            fix_cmd = ["black", "src", "tests", "scripts"]
            print("Attempting to auto-fix code formatting...")
            if run_command(fix_cmd, "black (auto-fix)"):
                print("Code formatting fixed!")
            else:
                all_passed = False
        else:
            all_passed = False
    
    # 3. Run pytest (if not skipped)
    if not args.skip_tests:
        pytest_cmd = ["pytest", "tests/", "-v"]
        if not run_command(pytest_cmd, "pytest (tests)"):
            all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed!")
        print("="*60 + "\n")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

