# Run all linting, formatting, and testing checks
#
# Usage:
#   .\scripts\lint_all.ps1 [-Check] [-SkipTests] [-Fix]
#
# Options:
#   -Check: Only check formatting (don't modify files)
#   -SkipTests: Skip running pytest
#   -Fix: Auto-fix formatting issues (default behavior)

param(
    [switch]$Check,
    [switch]$SkipTests,
    [switch]$Fix
)

$ErrorActionPreference = "Stop"

# Determine mode
$checkMode = $Check -and -not $Fix
$fixMode = $Fix -or -not $Check

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "ShopRec Code Quality Checks" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$allPassed = $true

# Function to run a command and check result
function Run-Check {
    param(
        [string]$Description,
        [string[]]$Command
    )
    
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "Running: $Description" -ForegroundColor Yellow
    Write-Host "Command: $($Command -join ' ')" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        & $Command[0] $Command[1..($Command.Length-1)]
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Host ""
            Write-Host "✓ $Description passed" -ForegroundColor Green
            Write-Host ""
            return $true
        } else {
            Write-Host ""
            Write-Host "✗ $Description failed (exit code: $exitCode)" -ForegroundColor Red
            Write-Host ""
            return $false
        }
    } catch {
        Write-Host ""
        Write-Host "✗ Error running $Description : $_" -ForegroundColor Red
        Write-Host ""
        return $false
    }
}

# 1. Run isort (import sorting)
if ($checkMode) {
    $result = Run-Check "isort (import sorting)" @("isort", "src", "tests", "scripts", "--check-only", "--diff")
    if (-not $result) {
        $allPassed = $false
    }
} else {
    $result = Run-Check "isort (import sorting)" @("isort", "src", "tests", "scripts", "--check")
    if (-not $result) {
        Write-Host "Attempting to auto-fix import sorting..." -ForegroundColor Yellow
        $fixResult = Run-Check "isort (auto-fix)" @("isort", "src", "tests", "scripts")
        if ($fixResult) {
            Write-Host "Import sorting fixed!" -ForegroundColor Green
        } else {
            $allPassed = $false
        }
    }
}

# 2. Run black (code formatting)
if ($checkMode) {
    $result = Run-Check "black (code formatting)" @("black", "src", "tests", "scripts", "--check")
    if (-not $result) {
        $allPassed = $false
    }
} else {
    $result = Run-Check "black (code formatting)" @("black", "src", "tests", "scripts", "--check")
    if (-not $result) {
        Write-Host "Attempting to auto-fix code formatting..." -ForegroundColor Yellow
        $fixResult = Run-Check "black (auto-fix)" @("black", "src", "tests", "scripts")
        if ($fixResult) {
            Write-Host "Code formatting fixed!" -ForegroundColor Green
        } else {
            $allPassed = $false
        }
    }
}

# 3. Run pytest (if not skipped)
if (-not $SkipTests) {
    $result = Run-Check "pytest (tests)" @("pytest", "tests/", "-v")
    if (-not $result) {
        $allPassed = $false
    }
}

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
if ($allPassed) {
    Write-Host "✓ All checks passed!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    exit 0
} else {
    Write-Host "✗ Some checks failed. Please fix the issues above." -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

