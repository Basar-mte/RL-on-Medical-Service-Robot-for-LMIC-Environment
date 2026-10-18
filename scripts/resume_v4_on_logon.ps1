# Auto-resume the V4 campaign after a power outage / reboot.
# Registered as a logon scheduled task; safe to run anytime: campaign_v4.py
# is idempotent (skips complete runs, resumes from checkpoints, never
# double-launches live runs), and this guard skips even starting it if an
# orchestrator is already running.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'campaign_v4' }
if (-not $running) {
    Start-Process -FilePath "d:\Roky\lmic-hospitalnav\lmic-hospitalnav\.venv\Scripts\python.exe" `
        -ArgumentList "scripts\campaign_v4_2m.py" `
        -WorkingDirectory "d:\Roky\lmic-hospitalnav\lmic-hospitalnav" `
        -WindowStyle Hidden `
        -RedirectStandardOutput "d:\Roky\lmic-hospitalnav\lmic-hospitalnav\runs_v4\campaign_resume.log" `
        -RedirectStandardError "d:\Roky\lmic-hospitalnav\lmic-hospitalnav\runs_v4\campaign_resume.err"
}
