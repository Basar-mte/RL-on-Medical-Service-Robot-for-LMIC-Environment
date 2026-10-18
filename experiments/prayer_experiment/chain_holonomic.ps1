# Chain runner: wait for the DMCH-redesign campaign (training + eval) to finish
# and free the GPU, then clean the aborted holonomic partial runs and launch the
# holonomic (kinematics) ablation + its eval. Fully autonomous; ASCII only.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File prayer_experiment\chain_holonomic.ps1

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$py = (Resolve-Path ".venv\Scripts\python.exe").Path
New-Item -ItemType Directory -Force "holonomic_experiment\runs" | Out-Null
function Log($m) { "[chain] $m  $(Get-Date -Format s)" | Out-File "holonomic_experiment\runs\chain.log" -Append -Encoding utf8 }

Log "watcher started; waiting for redesign campaign to finish"

# 1. wait until redesign training processes are gone
while ($true) {
  $r = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue |
       Where-Object { $_.CommandLine -like '*redesign_train*' -and $_.CommandLine -match '--agent' }
  if (-not $r) { break }
  Start-Sleep -Seconds 60
}
Log "redesign training finished; waiting for its eval (redesign_eval.json), up to 30 min"

# 2. wait for redesign eval to complete (or time out) so the GPU is fully clear
$deadline = (Get-Date).AddMinutes(30)
while (-not (Test-Path "prayer_experiment\redesign_runs\redesign_eval.json") -and (Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 30
}
Log "proceeding to holonomic ablation"

# 3. remove the aborted ~7% partial runs so eval picks only fresh runs
Get-ChildItem "holonomic_experiment\runs" -Directory -EA SilentlyContinue | Remove-Item -Recurse -Force -EA SilentlyContinue
Get-ChildItem "holonomic_experiment\runs\*_train.log*" -EA SilentlyContinue | Remove-Item -Force -EA SilentlyContinue
Log "cleaned aborted partial runs; launching run_holonomic.ps1 (cwn/attention/vanilla x3 seeds x 1M)"

# 4. run the ablation (blocks until all 9 runs finish), then evaluate
powershell -NoProfile -ExecutionPolicy Bypass -File "holonomic_experiment\run_holonomic.ps1"
Log "holonomic training done; running eval_holonomic.py"
& $py "holonomic_experiment\eval_holonomic.py" *> "holonomic_experiment\runs\eval.log"
Log "HOLONOMIC ABLATION COMPLETE (see holonomic_experiment/runs/holonomic_eval.json)"
