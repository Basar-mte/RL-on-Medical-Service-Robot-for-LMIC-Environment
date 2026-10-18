# Campaign recovery: resumes every incomplete run in runs/ from its latest
# checkpoint (train_sac.py --resume_from), launches any never-started
# attention seeds fresh, waits for everything, then runs the evaluation
# chain. Safe to re-run: completed runs (final_model.zip) are skipped and
# resumes continue from the newest checkpoint each time.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\resume_campaign.ps1

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"
$env:SDL_VIDEODRIVER = "dummy"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

$py = (Resolve-Path ".venv\Scripts\python.exe").Path
$LogFile = "runs\campaign_resume_$(Get-Date -Format yyyyMMdd_HHmmss).log"
"[resume] start: $(Get-Date -Format s)" | Out-File $LogFile -Append -Encoding utf8

$ADAPTIVE = @("cwn", "attention")
$procs = @()

# --- Resume incomplete runs that have checkpoints ---
Get-ChildItem runs -Directory | Where-Object {
    $_.Name -match '^(.+)_seed(\d+)_\d+$' -and
    -not (Test-Path (Join-Path $_.FullName 'final_model.zip'))
} | ForEach-Object {
    $null = $_.Name -match '^(.+)_seed(\d+)_\d+$'
    $agent = $Matches[1]; $seed = $Matches[2]
    $has_ckpt = (Get-ChildItem (Join-Path $_.FullName 'checkpoints') -Filter 'sac_*_steps.zip' -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0
    if (-not $has_ckpt) {
        "[resume] $($_.Name): no checkpoint - skipping (relaunch fresh manually if needed)" |
            Out-File $LogFile -Append -Encoding utf8
        return
    }
    $n_envs = if ($ADAPTIVE -contains $agent) { 1 } else { 2 }
    $log = "runs\${agent}_seed${seed}_train.log"
    $cmd = "`"$py`" train\train_sac.py --agent $agent --seed $seed --timesteps 1000000 " +
           "--n_envs $n_envs --no_camera --no_check_env --resume_from `"$($_.FullName)`" >> `"$log`" 2>&1"
    "[resume] resuming $($_.Name) (n_envs=$n_envs)" | Out-File $LogFile -Append -Encoding utf8
    $procs += Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -NoNewWindow -PassThru
    Start-Sleep -Seconds 2   # stagger torch/CUDA init
}

# --- Launch never-started attention seeds fresh ---
foreach ($seed in 1, 2) {
    $exists = Get-ChildItem runs -Directory | Where-Object { $_.Name -like "attention_seed${seed}_*" }
    if (-not $exists) {
        $log = "runs\attention_seed${seed}_train.log"
        $cmd = "`"$py`" train\train_sac.py --agent attention --seed $seed --timesteps 1000000 " +
               "--n_envs 1 --no_camera --no_check_env --log_dir runs >> `"$log`" 2>&1"
        "[resume] launching attention seed$seed fresh" | Out-File $LogFile -Append -Encoding utf8
        $procs += Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $cmd -NoNewWindow -PassThru
        Start-Sleep -Seconds 2
    }
}

"[resume] $($procs.Count) training processes running; waiting..." | Out-File $LogFile -Append -Encoding utf8
if ($procs.Count -gt 0) { $procs | Wait-Process }

"[resume] training done: $(Get-Date -Format s) - starting evaluation" | Out-File $LogFile -Append -Encoding utf8

& $py eval\post_training.py --runs_dir runs --n_episodes 100 2>&1 | Out-File $LogFile -Append -Encoding utf8
& $py eval\run_ablations.py --runs_dir runs --n_episodes 50 2>&1 | Out-File $LogFile -Append -Encoding utf8
& $py eval\log_trajectories.py --runs_dir runs 2>&1 | Out-File $LogFile -Append -Encoding utf8

"[resume] all done: $(Get-Date -Format s)" | Out-File $LogFile -Append -Encoding utf8
