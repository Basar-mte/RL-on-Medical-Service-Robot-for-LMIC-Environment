# (Re)launch the FINAL campaign: record provenance, register the 5-minute watchdog
# task, start the orchestrator (hidden window, logs in runs_v5/campaign_<ts>.log).
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_v5.ps1
# Idempotent: the orchestrator skips complete runs and resumes the rest from checkpoints.
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\python.exe"

& $py (Join-Path $root "scripts\provenance_v5.py")
if ($LASTEXITCODE -ne 0) { Write-Error "provenance failed"; exit 1 }

$wd = Join-Path $root "scripts\watchdog_v5.ps1"
schtasks /Create /SC MINUTE /MO 5 /TN HospitalNavV5Watchdog /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $wd" /F
if ($LASTEXITCODE -ne 0) { Write-Error "could not register the watchdog task"; exit 1 }

& powershell -NoProfile -ExecutionPolicy Bypass -File $wd
Write-Host "campaign launched. Dashboard:  .venv\Scripts\python.exe scripts\dashboard_v5.py --watch"
Write-Host "Stop:  schtasks /Delete /TN HospitalNavV5Watchdog /F ; then stop python processes running campaign_v5_stable.py / train_sac.py"
