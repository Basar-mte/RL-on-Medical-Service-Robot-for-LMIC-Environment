# Fleet completion for the v3 campaign: the two vanguards (cwn seed0,
# fixed_equal seed0) are already training in runs/. This launches the other
# 16 runs as two run_all groups, waits until ALL 18 run dirs contain
# final_model.zip, then runs the evaluation chain exactly once.

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"
$env:SDL_VIDEODRIVER = "dummy"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

$py = (Resolve-Path ".venv\Scripts\python.exe").Path
$LogFile = "runs\campaign_fleet_$(Get-Date -Format yyyyMMdd_HHmmss).log"
"[fleet] start: $(Get-Date -Format s)" | Out-File $LogFile -Append -Encoding utf8

# Group A: 12 baseline runs (wave scheduler caps concurrency at 6)
$argsA = "train\run_all.py --agents nav_heavy,vanilla,he2025,attention " +
         "--seeds 0,1,2 --parallel --max_concurrent 6 --n_envs 2 " +
         "--timesteps 1000000 --no_camera --no_check_env"
# Group B: remaining adaptive/fixed seeds (cwn+fixed_equal seeds 1,2)
$argsB = "train\run_all.py --agents cwn,fixed_equal --seeds 1,2 " +
         "--parallel --max_concurrent 2 --n_envs 2 " +
         "--timesteps 1000000 --no_camera --no_check_env"

$pA = Start-Process -FilePath $py -ArgumentList $argsA -NoNewWindow -PassThru `
        -RedirectStandardOutput "runs\fleet_groupA.log" -RedirectStandardError "runs\fleet_groupA.err"
$pB = Start-Process -FilePath $py -ArgumentList $argsB -NoNewWindow -PassThru `
        -RedirectStandardOutput "runs\fleet_groupB.log" -RedirectStandardError "runs\fleet_groupB.err"
"[fleet] group A pid $($pA.Id), group B pid $($pB.Id)" | Out-File $LogFile -Append -Encoding utf8

# Wait for ALL 18 completed runs (vanguards + fleet), then evaluate once.
while ($true) {
    $done = (Get-ChildItem runs -Directory -ErrorAction SilentlyContinue |
             Where-Object { Test-Path (Join-Path $_.FullName "final_model.zip") } |
             Measure-Object).Count
    "[fleet] $(Get-Date -Format HH:mm) completed runs: $done/18" | Out-File $LogFile -Append -Encoding utf8
    if ($done -ge 18) { break }
    Start-Sleep -Seconds 300
}

"[fleet] training done: $(Get-Date -Format s) - starting evaluation" | Out-File $LogFile -Append -Encoding utf8
& $py eval\post_training.py --runs_dir runs --n_episodes 100 2>&1 | Out-File $LogFile -Append -Encoding utf8
& $py eval\run_ablations.py --runs_dir runs --n_episodes 50 2>&1 | Out-File $LogFile -Append -Encoding utf8
& $py eval\log_trajectories.py --runs_dir runs 2>&1 | Out-File $LogFile -Append -Encoding utf8
"[fleet] ALL DONE: $(Get-Date -Format s)" | Out-File $LogFile -Append -Encoding utf8
