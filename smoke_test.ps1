$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$tasks = @("easy", "medium", "hard")
$hasError = $false

function Assert-Condition {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        Write-Host "[FAIL] $Message" -ForegroundColor Red
        $script:hasError = $true
    }
}

foreach ($task in $tasks) {
    Write-Host "--- Running task: $task ---" -ForegroundColor Cyan

    $rawOutput = & $pythonExe "inference.py" "--task" $task 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ($rawOutput -join "`n")
        Assert-Condition $false "Command failed for task '$task'"
        continue
    }

    $lines = @($rawOutput | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ -ne "" })
    $startLine = $lines | Where-Object { $_ -match '^\[START\]' } | Select-Object -First 1
    $stepLines = @($lines | Where-Object { $_ -match '^\[STEP\]' })
    $endLine = $lines | Where-Object { $_ -match '^\[END\]' } | Select-Object -Last 1

    Assert-Condition ($null -ne $startLine) "Missing [START] for task '$task'"
    Assert-Condition ($stepLines.Count -ge 1) "Missing [STEP] lines for task '$task'"
    Assert-Condition ($null -ne $endLine) "Missing [END] for task '$task'"

    $rewardFromSteps = @()
    foreach ($line in $stepLines) {
        Assert-Condition ($line -match '^\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d{2} done=(true|false) error=(null|.+)$') "Bad [STEP] format: $line"
        if ($line -match ' reward=(-?\d+\.\d{2}) ') {
            $rewardFromSteps += $Matches[1]
        }
    }

    if ($endLine -match '^\[END\] success=(true|false) steps=(\d+) score=(-?\d+\.\d{2}) rewards=(.*)$') {
        $stepsFromEnd = [int]$Matches[2]
        $scoreFromEnd = [double]$Matches[3]
        $rewardsFromEnd = $Matches[4]
        $expectedRewards = ($rewardFromSteps -join ',')

        Assert-Condition ($stepsFromEnd -eq $stepLines.Count) "steps mismatch for task '$task' (END=$stepsFromEnd, STEP lines=$($stepLines.Count))"
        Assert-Condition ($scoreFromEnd -gt 0.0 -and $scoreFromEnd -lt 1.0) "score out of range for task '$task' (score=$scoreFromEnd)"
        Assert-Condition ($rewardsFromEnd -eq $expectedRewards) "rewards mismatch for task '$task' (END='$rewardsFromEnd', STEP='$expectedRewards')"
    }
    else {
        Assert-Condition $false "Bad [END] format: $endLine"
    }

    Write-Host ($lines -join "`n")
    Write-Host ""
}

if ($hasError) {
    Write-Host "Smoke test failed." -ForegroundColor Red
    exit 1
}

Write-Host "Smoke test passed for easy, medium, hard." -ForegroundColor Green
exit 0
