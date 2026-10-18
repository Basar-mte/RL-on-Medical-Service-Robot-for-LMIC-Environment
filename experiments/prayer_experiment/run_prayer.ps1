# Prayer-ENGAGEMENT probe launcher (fully isolated).
# Trains cwn / attention / fixed_equal (seed 0) on the ENGAGED benchmark (prayer
# gate on the female-ward west door + richer 20-d context), writing only to
# prayer_experiment/runs/. Touches no main-pipeline file. n_envs=1 so the
# in-process monkeypatches hold (DummyVecEnv, no unpatched subprocess workers).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File prayer_experiment\run_prayer.ps1
#
# Stop:  Get-Process python* | Where-Object { $_.CommandLine -like '*prayer_train*' } | Stop-Process

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"; $env:SDL_VIDEODRIVER = "dummy"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"

$py = (Resolve-Path ".venv\Scripts\python.exe").Path
$outroot = "prayer_experiment\runs"
New-Item -ItemType Directory -Force $outroot | Out-Null

$STEPS = 500000
$jobs = @(
  @{a = "cwn";         s = 0 },
  @{a = "attention";   s = 0 },
  @{a = "fixed_equal"; s = 0 }
)
$procs = @()
foreach ($j in $jobs) {
  $log = "$outroot\$($j.a)_seed$($j.s)_train.log"
  $args = "prayer_experiment\prayer_train.py --agent $($j.a) --seed $($j.s) " +
          "--timesteps $STEPS --n_envs 1 --no_camera --no_check_env " +
          "--checkpoint_freq 100000 --log_dir $outroot"
  $procs += Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -PassThru `
             -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  Start-Sleep -Seconds 4
}
"[prayer-exp] launched $($procs.Count) runs ($STEPS steps) into $outroot at $(Get-Date -Format s)" |
  Out-File "$outroot\launch.log" -Append -Encoding utf8
$procs | Wait-Process
"[prayer-exp] all done: $(Get-Date -Format s)" | Out-File "$outroot\launch.log" -Append -Encoding utf8
