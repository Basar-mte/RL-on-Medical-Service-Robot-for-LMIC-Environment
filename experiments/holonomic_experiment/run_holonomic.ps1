# Holonomic-kinematics ablation launcher (fully isolated).
# Retrains cwn / attention / vanilla x 3 seeds under a HOLONOMIC (unicycle)
# motion model, writing only to holonomic_experiment/runs/. Does not touch the
# main pipeline. All runs use n_envs=1 so the in-process kinematics monkeypatch
# holds (DummyVecEnv, no unpatched subprocess workers).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File holonomic_experiment\run_holonomic.ps1
#
# Stop:  Get-Process python* | Where-Object { $_.CommandLine -like '*holonomic_train*' } | Stop-Process

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"; $env:SDL_VIDEODRIVER = "dummy"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"

$py = (Resolve-Path ".venv\Scripts\python.exe").Path
$outroot = "holonomic_experiment\runs"
New-Item -ItemType Directory -Force $outroot | Out-Null

$jobs = @(
  @{a="cwn"; s=0},       @{a="cwn"; s=1},       @{a="cwn"; s=2},
  @{a="attention"; s=0}, @{a="attention"; s=1}, @{a="attention"; s=2},
  @{a="vanilla"; s=0},   @{a="vanilla"; s=1},   @{a="vanilla"; s=2}
)
$procs = @()
foreach ($j in $jobs) {
  $log = "$outroot\$($j.a)_seed$($j.s)_train.log"
  $args = "holonomic_experiment\holonomic_train.py --agent $($j.a) --seed $($j.s) " +
          "--timesteps 1000000 --n_envs 1 --no_camera --no_check_env " +
          "--checkpoint_freq 100000 --log_dir $outroot"
  $procs += Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -PassThru `
             -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  Start-Sleep -Seconds 4
}
"[holo-exp] launched $($procs.Count) runs into $outroot" | Out-File "$outroot\launch.log" -Append -Encoding utf8
$procs | Wait-Process
"[holo-exp] all done: $(Get-Date -Format s)" | Out-File "$outroot\launch.log" -Append -Encoding utf8
