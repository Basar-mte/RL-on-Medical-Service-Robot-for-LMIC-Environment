# 6-term weighter probe launcher (fully isolated; ASCII only).
# Trains CWN with a 6-way weighter (nav/safe/prayer/gender/attendant/task) on the
# redesign env, 3 seeds x 500K, n_envs=1, then auto-evaluates vs the 4-term CWN /
# fixed_equal. Writes only prayer_experiment/sixterm_runs/. No main-code edits.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File prayer_experiment\run_sixterm.ps1
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"; $env:SDL_VIDEODRIVER = "dummy"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"
$py = (Resolve-Path ".venv\Scripts\python.exe").Path
$outroot = "prayer_experiment\sixterm_runs"
New-Item -ItemType Directory -Force $outroot | Out-Null

$procs = @()
foreach ($s in 0, 1, 2) {
  $log = "$outroot\cwn_seed${s}_train.log"
  $args = "prayer_experiment\sixterm_train.py --agent cwn --seed $s " +
          "--timesteps 500000 --n_envs 1 --no_camera --no_check_env " +
          "--checkpoint_freq 100000 --config prayer_experiment\dmch_redesign.yaml --log_dir $outroot"
  $procs += Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -PassThru `
             -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  Start-Sleep -Seconds 5
}
"[sixterm] launched $($procs.Count) runs at $(Get-Date -Format s)" | Out-File "$outroot\launch.log" -Append -Encoding utf8
$procs | Wait-Process
"[sixterm] training done: $(Get-Date -Format s) - starting eval" | Out-File "$outroot\launch.log" -Append -Encoding utf8
& $py "prayer_experiment\eval_sixterm.py" *> "$outroot\eval.log"
"[sixterm] COMPLETE: $(Get-Date -Format s)" | Out-File "$outroot\launch.log" -Append -Encoding utf8
