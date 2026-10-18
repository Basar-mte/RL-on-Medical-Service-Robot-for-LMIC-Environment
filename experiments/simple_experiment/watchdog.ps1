# Auto-restart for the simplified-experiment campaign.
# Run by the scheduled task "HospitalNavSimpleWatchdog" every 5 minutes and at
# logon. If the orchestrator is not running and the campaign is not finished,
# it relaunches campaign.py, which resumes every run from its newest checkpoint.
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runs = Join-Path $PSScriptRoot "runs"
New-Item -ItemType Directory -Force $runs | Out-Null

if (Test-Path (Join-Path $runs "ALL_DONE")) { exit 0 }

$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*simple_experiment*campaign.py*' }
if ($alive) { exit 0 }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
Start-Process -FilePath (Join-Path $root ".venv\Scripts\python.exe") `
    -ArgumentList "experiments\simple_experiment\campaign.py" `
    -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $runs "campaign_$ts.log") `
    -RedirectStandardError (Join-Path $runs "campaign_$ts.err")
Add-Content (Join-Path $runs "watchdog.log") "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') orchestrator (re)started -> campaign_$ts.log"
