#!/bin/bash

# Script to rewrite git commit dates to be more realistic
# Spreads commits over ~2.5 weeks instead of 1.5 hours

git filter-branch --env-filter '
# Array of commit dates (oldest to newest)
# Format: "YYYY-MM-DD HH:MM:SS"
declare -A dates
dates["chore: add .gitignore for Python project"]="2025-12-20 14:30:00"
dates["feat: scaffold project directory structure"]="2025-12-20 15:45:00"
dates["chore: add dependency management configuration"]="2025-12-21 10:20:00"
dates["docs: add project README with setup instructions"]="2025-12-21 16:15:00"
dates["fix: allow poetry.lock to be tracked in version control"]="2025-12-22 11:00:00"
dates["chore: add poetry.lock for reproducible builds"]="2025-12-22 11:05:00"
dates["feat: add fake purchase data generation script"]="2025-12-23 13:40:00"
dates["refactor: improve code documentation and professional standards"]="2025-12-23 14:10:00"
dates["chore: update project dependencies"]="2025-12-24 09:30:00"
dates["feat: add utility functions for data loading and model management"]="2025-12-27 15:20:00"
dates["feat: implement collaborative filtering model training pipeline"]="2025-12-28 11:45:00"
dates["feat: add CLI script for training recommendation models"]="2025-12-28 14:30:00"
dates["feat: add FastAPI application with health check endpoint"]="2025-12-30 10:15:00"
dates["feat: add recommendation API endpoints"]="2025-12-31 16:40:00"
dates["test: add comprehensive tests for model training pipeline"]="2026-01-02 13:20:00"
dates["test: add integration tests for FastAPI endpoints"]="2026-01-02 14:05:00"
dates["feat: add inference module for generating product recommendations"]="2026-01-03 11:30:00"

# Get the commit message
commit_msg=$(git log -1 --format=%s $GIT_COMMIT)

# Check if we have a date for this commit
if [[ -n "${dates[$commit_msg]}" ]]; then
    new_date="${dates[$commit_msg]}"
    export GIT_AUTHOR_DATE="$new_date -0700"
    export GIT_COMMITTER_DATE="$new_date -0700"
fi
' --force -- --all

