# PowerShell script to rewrite git commit dates

# Commit messages to dates mapping (oldest to newest)
$dates = @{
    "chore: add .gitignore for Python project" = "2025-12-20T14:30:00-07:00"
    "feat: scaffold project directory structure" = "2025-12-20T15:45:00-07:00"
    "chore: add dependency management configuration" = "2025-12-21T10:20:00-07:00"
    "docs: add project README with setup instructions" = "2025-12-21T16:15:00-07:00"
    "fix: allow poetry.lock to be tracked in version control" = "2025-12-22T11:00:00-07:00"
    "chore: add poetry.lock for reproducible builds" = "2025-12-22T11:05:00-07:00"
    "feat: add fake purchase data generation script" = "2025-12-23T13:40:00-07:00"
    "refactor: improve code documentation and professional standards" = "2025-12-23T14:10:00-07:00"
    "chore: update project dependencies" = "2025-12-24T09:30:00-07:00"
    "feat: add utility functions for data loading and model management" = "2025-12-27T15:20:00-07:00"
    "feat: implement collaborative filtering model training pipeline" = "2025-12-28T11:45:00-07:00"
    "feat: add CLI script for training recommendation models" = "2025-12-28T14:30:00-07:00"
    "feat: add FastAPI application with health check endpoint" = "2025-12-30T10:15:00-07:00"
    "feat: add recommendation API endpoints" = "2025-12-31T16:40:00-07:00"
    "test: add comprehensive tests for model training pipeline" = "2026-01-02T13:20:00-07:00"
    "test: add integration tests for FastAPI endpoints" = "2026-01-02T14:05:00-07:00"
    "feat: add inference module for generating product recommendations" = "2026-01-03T11:30:00-07:00"
}

Write-Host "Rewriting git commit dates..."
Write-Host "This will rewrite history. Make sure you have a backup!"
Write-Host ""

# Create the filter script
$filterScript = @"
FILTER_BRANCH_SQUELCH_WARNING=1

# Get commit message
commit_msg=`$(git log -1 --format=%s `$GIT_COMMIT)

case "`$commit_msg" in
"@

foreach ($msg in $dates.Keys) {
    $date = $dates[$msg]
    $filterScript += @"

    "$msg")
        export GIT_AUTHOR_DATE="$date"
        export GIT_COMMITTER_DATE="$date"
        ;;
"@
}

$filterScript += @"

esac
"@

# Save the filter script to a temporary file
$filterScript | Out-File -FilePath "filter_script.sh" -Encoding ASCII

# Run git filter-branch
Write-Host "Running git filter-branch..."
git filter-branch -f --env-filter (Get-Content filter_script.sh -Raw) -- --all

# Clean up
Remove-Item filter_script.sh

Write-Host ""
Write-Host "Done! Commit dates have been rewritten."
Write-Host "Run 'git log --format=`"%ai %s`"' to verify."

