# Live status of the 6-term weighter probe. Refreshes every 30s (Ctrl+C exits;
# training is unaffected). -Once for a single snapshot.
#   powershell -NoProfile -ExecutionPolicy Bypass -File prayer_experiment\dashboard_sixterm.ps1
param([switch]$Once)
Set-Location (Join-Path $PSScriptRoot "..")
$TOTAL = 500000
while ($true) {
  if (-not $Once) { Clear-Host }
  "=== 6-term weighter probe (redesign env)  $(Get-Date -Format 'HH:mm:ss') ==="
  $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*sixterm_train*' -and $_.CommandLine -match '--agent' }
  "training procs: $(@($p).Count)  |  gpu: $(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used --format=csv,noheader)"
  ""
  $sumfps = 0; $done = 0; $running = 0
  Get-ChildItem prayer_experiment\sixterm_runs\*_train.log -EA SilentlyContinue | Sort-Object Name | ForEach-Object {
    $name = $_.Name -replace '_train.log',''
    $runDir = Get-ChildItem prayer_experiment\sixterm_runs -Directory -Filter "${name}_*" -EA SilentlyContinue | Select-Object -First 1
    $fin = $runDir -and (Test-Path (Join-Path $runDir.FullName 'final_model.zip'))
    $t = Get-Content $_.FullName -Tail 55
    $s  = ($t | Select-String 'total_timesteps\s+\|\s+(\d+)').Matches | Select-Object -Last 1
    $sr = ($t | Select-String 'success_rate\s+\|\s+([\d.]+)').Matches | Select-Object -Last 1
    $cv = ($t | Select-String '\bcvr\s+\|\s+([\d.]+)').Matches | Select-Object -Last 1
    $f  = ($t | Select-String '\bfps\s+\|\s+(\d+)').Matches | Select-Object -Last 1
    $tm = ($t | Select-String 'temperature\s+\|\s+([\d.]+)').Matches | Select-Object -Last 1
    $steps = if ($s) { [int]$s.Groups[1].Value } else { 0 }
    $fps = if ($f) { [int]$f.Groups[1].Value } else { 0 }
    $srT = if ($sr) { "{0,5:P0}" -f [double]$sr.Groups[1].Value } else { "  ?" }
    $cvT = if ($cv) { "{0:N3}" -f [double]$cv.Groups[1].Value } else { "?" }
    $tmT = if ($tm) { "{0:N2}" -f [double]$tm.Groups[1].Value } else { "-" }
    if ($fin) { $done++; "{0,-14} DONE   SR {1}  CVR {2}  temp {3}" -f $name,$srT,$cvT,$tmT }
    else {
      $running++; $sumfps += $fps
      $pct = [math]::Round(100.0*$steps/$TOTAL,1)
      $eta = if ($fps -gt 0){ [math]::Round(($TOTAL-$steps)/$fps/60.0,0) } else { "?" }
      "{0,-14} {1,7:N0}/500k {2,5}%  {3,3}fps  SR {4}  CVR {5}  temp {6}  ~{7}m" -f $name,$steps,$pct,$fps,$srT,$cvT,$tmT,$eta
    }
  }
  ""
  "$running running, $done done  |  aggregate fps: $sumfps"
  if (Test-Path prayer_experiment\sixterm_runs\sixterm_eval.json) { "*** EVAL COMPLETE -> prayer_experiment\sixterm_runs\sixterm_eval.json ***" }
  if ($Once) { break }
  Start-Sleep -Seconds 30
}
