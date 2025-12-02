# cleanup_root.ps1 - Clean up GitHub root folder

Write-Host "Cleaning up repository root..." -ForegroundColor Cyan
Write-Host ""

# Step 1: List what will be deleted
Write-Host "[1/4] Items to delete:" -ForegroundColor Yellow
$toDelete = @("src", "tests", "data", ".coverage", "test_results.xml", "setup_github.ps1", "validate_phase1.ps1")
$toDelete | ForEach-Object { 
    if (Test-Path $_) {
        Write-Host "  ❌ $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "[2/4] Deleting duplicates..." -ForegroundColor Cyan

# Step 2: Delete items
$toDelete | ForEach-Object {
    if (Test-Path $_) {
        if ((Get-Item $_).PSIsContainer) {
            Remove-Item -Path $_ -Recurse -Force
            Write-Host "  ✓ Deleted folder: $_" -ForegroundColor Green
        } else {
            Remove-Item -Path $_ -Force
            Write-Host "  ✓ Deleted file: $_" -ForegroundColor Green
        }
    }
}

# Step 3: Verify clean structure
Write-Host ""
Write-Host "[3/4] Final structure:" -ForegroundColor Cyan
Get-ChildItem -Path . -Exclude .git,.mypy_cache,.pytest_cache,htmlcov | Select-Object Name | ForEach-Object {
    Write-Host "  ✓ $_" -ForegroundColor Green
}

# Step 4: Commit and push
Write-Host ""
Write-Host "[4/4] Committing changes..." -ForegroundColor Cyan
git add -A
git commit -m "cleanup: remove duplicate files at root level

- Remove duplicate src/, tests/, data/ folders (kept in archive)
- Remove .coverage, test_results.xml files (in archive)
- Remove empty setup_github.ps1, validate_phase1.ps1 scripts
- Root now contains only: Starting Module 1 archive, .gitignore, README.md"

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ CLEANUP COMPLETE!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Repository now clean and organized:" -ForegroundColor Cyan
    Write-Host "  ✓ https://github.com/nishadrahul12/Telecom-AI" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Latest commits:" -ForegroundColor Green
    git log --oneline -3
} else {
    Write-Host "❌ Push failed!" -ForegroundColor Red
}
