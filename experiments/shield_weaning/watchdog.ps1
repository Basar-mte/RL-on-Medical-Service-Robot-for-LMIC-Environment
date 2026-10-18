# Auto-restart for the shield-weaning campaign (scheduled task
# "HospitalNavWeanWatchdog", every 5 minutes while logged in). Relaunches
# campaign.py if it is not running and runs/ALL_DONE does not exist; the
# campaign itself resumes every run from its newest checkpoint.
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runs = Join-Path $PSScriptRoot "runs"
New-Item -ItemType Directory -Force $runs | Out-Null

if (Test-Path (Join-Path $runs "ALL_DONE")) { exit 0 }

$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*shield_weaning*campaign.py*' }
if ($alive) { exit 0 }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Process -FilePath (Join-Path $root ".venv\Scripts\python.exe") `
    -ArgumentList "experiments\shield_weaning\campaign.py" `
    -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $runs "campaign_$ts.log") `
    -RedirectStandardError (Join-Path $runs "campaign_$ts.err")
Add-Content (Join-Path $runs "watchdog.log") "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') orchestrator (re)started -> campaign_$ts.log"
