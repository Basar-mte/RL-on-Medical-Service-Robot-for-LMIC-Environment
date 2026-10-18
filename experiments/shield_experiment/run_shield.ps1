# Shield-in-loop bonus experiment launcher (fully isolated).
# Trains attention/cwn/vanilla x 3 seeds WITH the reactive shield in the loop,
# writing only to shield_experiment/runs/. Does not touch the main pipeline.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File shield_experiment\run_shield.ps1
#
# Monitor:  powershell -File scripts\watch_progress.ps1   # (edit RUNS var to shield_experiment\runs) OR use the commands below
# Stop:     Get-Process python* | Where-Object { $_.CommandLine -like '*shield_train*' } | Stop-Process

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"; $env:SDL_VIDEODRIVER = "dummy"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"

$py = (Resolve-Path ".venv\Scripts\python.exe").Path
$outroot = "shield_experiment\runs"
New-Item -ItemType Directory -Force $outroot | Out-Null

# adaptive agents (cwn, attention) must use n_envs=1; vanilla uses 2.
$jobs = @(
  @{a="attention"; s=0; n=1}, @{a="attention"; s=1; n=1}, @{a="attention"; s=2; n=1},
  @{a="cwn"; s=0; n=1}, @{a="cwn"; s=1; n=1}, @{a="cwn"; s=2; n=1},
  @{a="vanilla"; s=0; n=2}, @{a="vanilla"; s=1; n=2}, @{a="vanilla"; s=2; n=2}
)
$procs = @()
foreach ($j in $jobs) {
  $log = "$outroot\$($j.a)_seed$($j.s)_train.log"
  $args = "shield_experiment\shield_train.py --agent $($j.a) --seed $($j.s) " +
          "--timesteps 1000000 --n_envs $($j.n) --no_camera --no_check_env " +
          "--checkpoint_freq 100000 --log_dir $outroot"
  $procs += Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -PassThru `
             -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  Start-Sleep -Seconds 4
}
"[shield-exp] launched $($procs.Count) runs into $outroot" | Out-File "$outroot\launch.log" -Append -Encoding utf8
$procs | Wait-Process
"[shield-exp] all done: $(Get-Date -Format s)" | Out-File "$outroot\launch.log" -Append -Encoding utf8
