# Live campaign dashboard: per-run progress, fps, ETA, GPU health.
# Refreshes every 30 s. Ctrl+C to exit (training is unaffected).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_progress.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\watch_progress.ps1 -Once   # single snapshot

param([switch]$Once)

Set-Location (Join-Path $PSScriptRoot "..")
$TOTAL = 1000000

while ($true) {
    if (-not $Once) { Clear-Host }
    "=== LMIC-HospitalNav campaign status  $(Get-Date -Format 'HH:mm:ss') ==="
    ""
    $sumFps = 0; $sumSteps = 0; $running = 0; $done = 0
    $queued = 0
    Get-ChildItem runs\*_train.log -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.Name -replace '_train.log',''
        $runDir = Get-ChildItem runs -Directory -Filter "${name}_*" -ErrorAction SilentlyContinue | Select-Object -First 1
        $fin = $runDir -and (Test-Path (Join-Path $runDir.FullName 'final_model.zip'))
        # A run is LIVE only if its log was written very recently; a queued
        # or dead run's log is frozen (its stale tail must NOT be read as
        # live progress — that made queued runs look like they were running
        # at their pre-outage percentage).
        $ageSec = ((Get-Date) - $_.LastWriteTime).TotalSeconds
        $t = Get-Content $_.FullName -Tail 60
        $s = ($t | Select-String 'total_timesteps\s+\|\s+(\d+)').Matches | Select-Object -Last 1
        $srm = ($t | Select-String 'success_rate\s+\|\s+([\d.]+)').Matches | Select-Object -Last 1
        $f = ($t | Select-String 'fps\s+\|\s+(\d+)').Matches | Select-Object -Last 1
        $steps = if ($s) { [int]$s.Groups[1].Value } else { 0 }
        $fps = if ($f) { [int]$f.Groups[1].Value } else { 0 }
        $sr = if ($srm) { [double]$srm.Groups[1].Value } else { -1 }
        $srTxt = if ($sr -ge 0) { "SR {0,5:P1}" -f $sr } else { "SR     ?" }
        if ($fin) {
            $done += 1
            "{0,-28} DONE" -f $name
        } elseif ($ageSec -gt 180) {
            $queued += 1
            "{0,-28} QUEUED / not training (log idle {1:N0}s; last ckpt ~{2:N0})" -f $name, $ageSec, $steps
        } else {
            $running += 1; $sumFps += $fps; $sumSteps += $steps
            $pct = [math]::Round(100.0 * $steps / $TOTAL, 1)
            $eta = if ($fps -gt 0) { [math]::Round(($TOTAL - $steps) / $fps / 3600.0, 1) } else { "?" }
            "{0,-28} {1,9:N0} /1M  {2,5}%  {3,3} fps  {4}  ~{5} h left" -f $name, $steps, $pct, $fps, $srTxt, $eta
        }
    }
    ""
    "runs: $running training, $queued queued, $done done | live fps: $sumFps"
    $gpu = (nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader) 2>$null
    if ($gpu) { "gpu : $gpu  (temp under ~85C and power under 600W = healthy)" }
    ""
    "latest campaign log lines:"
    # Most-recently-written orchestrator log across ALL naming schemes
    # (campaign_*, recover, fleet_*) — pinning to campaign_* alone showed a
    # frozen pre-outage log and looked like a stall.
    $clog = Get-ChildItem runs\recover.log, runs\campaign_*.log, runs\fleet_*.log -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($clog) { "  [$($clog.Name)]"; Get-Content $clog.FullName -Tail 3 | ForEach-Object { "  $_" } }
    if ($Once) { break }
    Start-Sleep -Seconds 30
}
