# Auto-restart for the stable-recipe paper re-run (scheduled task
# "HospitalNavV5Watchdog", every 5 minutes while logged in). Relaunches
# scripts/campaign_v5_stable.py if it is not running and runs_v5/ALL_DONE does
# not exist; the campaign resumes every run from its newest checkpoint.
$root = Split-Path $PSScriptRoot -Parent
$runs = Join-Path $root "runs_v5"
New-Item -ItemType Directory -Force $runs | Out-Null

if (Test-Path (Join-Path $runs "ALL_DONE")) { exit 0 }

$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*campaign_v5_stable.py*' }
if ($alive) { exit 0 }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Process -FilePath (Join-Path $root ".venv\Scripts\python.exe") `
    -ArgumentList "scripts\campaign_v5_stable.py" `
    -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $runs "campaign_$ts.log") `
    -RedirectStandardError (Join-Path $runs "campaign_$ts.err")
Add-Content (Join-Path $runs "watchdog.log") "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') orchestrator (re)started -> campaign_$ts.log"
