# Train ONE agent's seeds (default 0,1,2 concurrently), no camera, 1M steps.
# Designed for the one-agent-per-day schedule.
#
# Usage (from the repo root):
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_agent.ps1 -Agent cwn
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_agent.ps1 -Agent attention
#   ... agents: cwn | attention | fixed_equal | nav_heavy | vanilla | he2025
#
# After the LAST agent has trained, produce the paper tables/figures data:
#   .venv\Scripts\python.exe eval\post_training.py --runs_dir runs --n_episodes 100
#   .venv\Scripts\python.exe eval\run_ablations.py --runs_dir runs --n_episodes 50
#   .venv\Scripts\python.exe eval\log_trajectories.py --runs_dir runs
# (post_training can also be run after any single agent to peek at its numbers;
#  it caches per-run eval_results.json, so re-running later is cheap.)
#
# Monitor:  Get-Content runs\campaign.log -Tail 20 -Wait
#           Get-Content runs\cwn_seed0_train.log -Tail 30
# Stop:     Get-Process python* | Where-Object { $_.Path -like "*lmic-hospitalnav*" } | Stop-Process
#
# Timing measured on this machine (RTX 5090 / 24 cores, no camera, 3 seeds
# concurrent): roughly 5.5-7 h per agent for 1M steps x 3 seeds.

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("cwn", "attention", "fixed_equal", "nav_heavy", "vanilla", "he2025")]
    [string]$Agent,
    [string]$Seeds = "0,1,2",
    [int]$Timesteps = 1000000
)

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"
$env:SDL_VIDEODRIVER = "dummy"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

# Unique log per invocation: Out-File holds an exclusive handle for the whole
# run, so two concurrent launcher scripts must never share one log file.
$LogFile = "runs\agentday_${Agent}_$(Get-Date -Format yyyyMMdd_HHmmss).log"

"[agent-day] $Agent seeds=$Seeds timesteps=$Timesteps start: $(Get-Date -Format s)" |
    Out-File $LogFile -Append -Encoding utf8

& .venv\Scripts\python.exe train\run_all.py --agents $Agent --seeds $Seeds `
    --parallel --max_concurrent 3 --n_envs 2 --timesteps $Timesteps `
    --no_camera --no_check_env 2>&1 |
    Out-File $LogFile -Append -Encoding utf8

"[agent-day] $Agent done: $(Get-Date -Format s)" | Out-File $LogFile -Append -Encoding utf8
