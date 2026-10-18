# DMCH-redesign campaign launcher (fully isolated).
# Trains all 6 agents (seed 0, 500K) on the redesign env (prayer_experiment/
# dmch_redesign.yaml) with the rich 20-d context, writing only to
# prayer_experiment/redesign_runs/. n_envs=1 so the in-process patches hold.
# When all runs finish, auto-runs eval_redesign.py.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File prayer_experiment\run_redesign.ps1
#
# Stop:  Get-Process python* | Where-Object { $_.CommandLine -like '*redesign_train*' } | Stop-Process

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"; $env:SDL_VIDEODRIVER = "dummy"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"

$py = (Resolve-Path ".venv\Scripts\python.exe").Path
$outroot = "prayer_experiment\redesign_runs"
New-Item -ItemType Directory -Force $outroot | Out-Null

$STEPS = 500000
$agents = @("cwn", "attention", "vanilla", "fixed_equal", "nav_heavy", "he2025")
$procs = @()
foreach ($a in $agents) {
  $log = "$outroot\${a}_seed0_train.log"
  $args = "prayer_experiment\redesign_train.py --agent $a --seed 0 " +
          "--timesteps $STEPS --n_envs 1 --no_camera --no_check_env " +
          "--checkpoint_freq 100000 --config prayer_experiment\dmch_redesign.yaml --log_dir $outroot"
  $procs += Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -PassThru `
             -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  Start-Sleep -Seconds 5
}
"[redesign] launched $($procs.Count) runs ($STEPS steps) at $(Get-Date -Format s)" |
  Out-File "$outroot\launch.log" -Append -Encoding utf8
$procs | Wait-Process
"[redesign] all training done: $(Get-Date -Format s) - starting eval" |
  Out-File "$outroot\launch.log" -Append -Encoding utf8

& $py "prayer_experiment\eval_redesign.py" *> "$outroot\eval.log"
"[redesign] eval done: $(Get-Date -Format s)" | Out-File "$outroot\launch.log" -Append -Encoding utf8
