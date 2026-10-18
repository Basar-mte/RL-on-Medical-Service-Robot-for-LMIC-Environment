# Live dashboard for the shield-in-loop bonus experiment.
# Same look as scripts\watch_progress.ps1 but pointed at shield_experiment\runs\.
# Refreshes every 30 s. Ctrl+C to exit (training is unaffected).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File shield_experiment\dashboard.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File shield_experiment\dashboard.ps1 -Once

param([switch]$Once)

Set-Location (Join-Path $PSScriptRoot "..")
$RUNS = "shield_experiment\runs"
$TOTAL = 1000000

while ($true) {
    if (-not $Once) { Clear-Host }
    "=== Shield-in-loop (BONUS, isolated)  $(Get-Date -Format 'HH:mm:ss') ==="
    ""
    $running = 0; $queued = 0; $done = 0; $sumFps = 0; $sumSteps = 0
    Get-ChildItem "$RUNS\*_train.log" -ErrorAction SilentlyContinue | ForEach-Object {
        $name = $_.Name -replace '_train.log',''
        $runDir = Get-ChildItem $RUNS -Directory -Filter "${name}_*" -ErrorAction SilentlyContinue | Select-Object -First 1
        $fin = $runDir -and (Test-Path (Join-Path $runDir.FullName 'final_model.zip'))
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
            "{0,-22} DONE" -f $name
        } elseif ($ageSec -gt 180) {
            $queued += 1
            "{0,-22} QUEUED / idle (log {1:N0}s; last ckpt ~{2:N0})" -f $name, $ageSec, $steps
        } else {
            $running += 1; $sumFps += $fps; $sumSteps += $steps
            $pct = [math]::Round(100.0 * $steps / $TOTAL, 1)
            $eta = if ($fps -gt 0) { [math]::Round(($TOTAL - $steps) / $fps / 3600.0, 1) } else { "?" }
            "{0,-22} {1,9:N0} /1M  {2,5}%  {3,3} fps  {4}  ~{5} h left" -f $name, $steps, $pct, $fps, $srTxt, $eta
        }
    }
    ""
    "runs: $running training, $queued queued, $done done | live fps: $sumFps | steps this exp: {0:N0}" -f $sumSteps
    $gpu = (nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader) 2>$null
    if ($gpu) { "gpu : $gpu  (temp <~85C, power <600W = healthy)" }
    if ($Once) { break }
    Start-Sleep -Seconds 30
}
