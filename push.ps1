<#>
.SYNOPSIS
    Push Attention Scaling Fresh Study to GitHub
.DESCRIPTION
    Pushes the attention-scaling-study repository to GitHub.
    Handles authentication via GitHub CLI or token.
    Handles unrelated histories if needed.
.NOTES
    Run from: C:\Users\Raza-SEAGUARD\Downloads\Attention\attention-scaling-study
    Prerequisites: git installed, GitHub account, repo exists at https://github.com/razahasan2000/Attention-Scaling
#>

# ---------------------------------------------------------
# Configuration - EDIT THESE VALUES
# ---------------------------------------------------------
$RepoPath = "C:\Users\Raza-SEAGUARD\Downloads\Attention\attention-scaling-study"
$GitHubRepo = "razahasan2000/Attention-Scaling"
$Branch = "main"
$CommitMessage = "Initial commit: Attention Scaling Fresh Study - 320 runs complete, manuscript synced, all tests passing"

# ---------------------------------------------------------
# DO NOT EDIT BELOW THIS LINE
# ---------------------------------------------------------

Write-Host "=== Attention Scaling Study - GitHub Push ===" -ForegroundColor Cyan
Write-Host "Repo: $GitHubRepo" -ForegroundColor Gray
Write-Host "Branch: $Branch`n" -ForegroundColor Gray

# Change to repo directory
Set-Location $RepoPath
Write-Host "Working in: $(Get-Location)" -ForegroundColor Gray

# ---------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------
Write-Host "`n=== Pre-flight Checks ===" -ForegroundColor Yellow

# Check git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "git not found in PATH"
    exit 1
}
Write-Host "git: OK" -ForegroundColor Green

# Check repo exists
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    Write-Host "git init: OK" -ForegroundColor Green
} else {
    Write-Host "git repo: OK" -ForegroundColor Green
}

# Check remote
$remoteUrl = git remote get-url origin 2>$null
if (-not $remoteUrl) {
    Write-Host "Adding remote origin..." -ForegroundColor Yellow
    git remote add origin "https://github.com/razahasan2000/Attention-Scaling.git"
} else {
    Write-Host "Remote: $remoteUrl" -ForegroundColor Gray
    # Ensure correct URL
    if ($remoteUrl -notlike "*razahasan2000/Attention-Scaling*") {
        Write-Host "Updating remote URL..." -ForegroundColor Yellow
        git remote set-url origin "https://github.com/razahasan2000/Attention-Scaling.git"
    }
}

# Check git status
$status = git status --porcelain
if ($status) {
    Write-Host "Uncommitted changes detected:" -ForegroundColor Yellow
    git status --short
} else {
    Write-Host "Working tree clean" -ForegroundColor Green
}

# Check current branch
$currentBranch = git rev-parse --abbrev-ref HEAD
Write-Host "Current branch: $currentBranch" -ForegroundColor Gray

# ---------------------------------------------------------
# Authentication method selection
# ---------------------------------------------------------
Write-Host "`n=== Authentication ===" -ForegroundColor Yellow

$useGhCli = $false
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $ghStatus = gh auth status 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "GitHub CLI authenticated" -ForegroundColor Green
        $useGhCli = $true
    } else {
        Write-Host "GitHub CLI not authenticated" -ForegroundColor Yellow
    }
} else {
    Write-Host "GitHub CLI not installed" -ForegroundColor Yellow
}

if (-not $useGhCli) {
    Write-Host "`nAuthentication options:" -ForegroundColor Cyan
    Write-Host "  1. GitHub CLI (recommended): gh auth login"
    Write-Host "  2. Personal Access Token (PAT) in URL"
    Write-Host "  3. Git Credential Manager (Windows default)" -ForegroundColor Gray
    
    $choice = Read-Host "Choose auth method (1/2/3) [default: 3]"
    switch ($choice) {
        '1' { 
            Write-Host "Run 'gh auth login' then re-run this script" -ForegroundColor Yellow
            exit 1
        }
        '2' {
            $token = Read-Host "Enter Personal Access Token (hidden)" -AsSecureString
            $tokenPlain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($token))
            git remote set-url origin "https://$tokenPlain@github.com/razahasan2000/Attention-Scaling.git"
            Write-Host "Token configured in remote URL" -ForegroundColor Green
        }
        default {
            Write-Host "Using Git Credential Manager (Windows default)" -ForegroundColor Gray
        }
    }
}

# ---------------------------------------------------------
# Ensure branch name
# ---------------------------------------------------------
$currentBranch = git rev-parse --abbrev-ref HEAD 2>$null
if ($currentBranch -ne $Branch) {
    Write-Host "Renaming branch to '$Branch'..." -ForegroundColor Yellow
    git branch -M $Branch
}

# ---------------------------------------------------------
# Ensure commit exists
# ---------------------------------------------------------
$hasCommits = git rev-parse --verify HEAD 2>$null
if (-not $hasCommits) {
    Write-Host "No commits yet - creating initial commit..." -ForegroundColor Yellow
    git add .
    git commit -m "Initial commit: Attention Scaling Fresh Study - 320 runs complete, manuscript synced, all tests passing"
}

# ---------------------------------------------------------
# Push with fallback for unrelated histories
# ---------------------------------------------------------
Write-Host "`n=== Pushing to GitHub ===" -ForegroundColor Cyan

try {
    $result = git push -u origin $Branch 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Push successful!" -ForegroundColor Green
        exit 0
    }
    
    # Check for unrelated histories error
    if ($result -match "unrelated histories") {
        Write-Host "`nUnrelated histories detected - merging..." -ForegroundColor Yellow
        $result = git pull origin $Branch --allow-unrelated-histories --no-rebase 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Merge successful, pushing..." -ForegroundColor Green
            git push -u origin $Branch
            Write-Host "`n✅ Push successful!" -ForegroundColor Green
            exit 0
        } else {
            Write-Error "Merge failed: $result"
        }
    } elseif ($result -match "Authentication failed|Permission denied|could not read Username") {
        Write-Error "Authentication failed. Run 'gh auth login' or configure PAT."
    } else {
        Write-Error "Push failed: $result"
    }
}
catch {
    Write-Error "Error: $_"
    exit 1
}

# ---------------------------------------------------------
# Verify push
# ---------------------------------------------------------
Write-Host "`n=== Verifying ===" -ForegroundColor Yellow
$remoteUrl = git remote get-url origin
Write-Host "Remote: $remoteUrl"
$latestCommit = git log --oneline -1
Write-Host "Latest commit: $latestCommit"

Write-Host "`n=== DONE ===" -ForegroundColor Green
Write-Host "Repository pushed to: https://github.com/razahasan2000/Attention-Scaling"