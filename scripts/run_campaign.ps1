# Full paper-scale campaign (NO-CAMERA): 6 agents x 3 seeds x 1M steps,
# 8 runs concurrent, then evaluation (100 episodes/agent/seed), aggregation,
# density ablations, and real-trajectory logging.
#
# The camera is excluded (--no_camera): the paper's contributions (benchmark,
# CARS/CVR, CWN) do not depend on it, and dropping the NatureCNN makes every
# SAC gradient step ~3x faster. 360-degree LiDAR + Ackermann kinematics are
# kept (the defensible sim-to-real elements).
#
# Launch (detached, survives the launching shell):
#   Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File scripts\run_campaign.ps1" -WindowStyle Hidden
# Monitor:
#   Get-Content runs\campaign.log -Tail 20 -Wait          # launcher progress
#   Get-Content runs\cwn_seed0_train.log -Tail 30         # any individual run
#   .venv\Scripts\tensorboard.exe --logdir runs           # curves
# Stop everything:
#   Get-Process python* | Where-Object { $_.Path -like "*lmic-hospitalnav*" } | Stop-Process
#
# Sizing rationale (measured on this machine, RTX 5090 / 24-core / 127 GB):
#   - Without the camera the replay buffer is ~0.07 GB/run (was 4.2 GB) and
#     gradient steps are MLP-only, so concurrency rises 6 -> 8; CPU is the
#     only remaining constraint.
#   - --n_envs 2 for the fixed-weight baselines (cwn/attention are clamped
#     to 1 inside run_all.py): env parallelism adds little at 1 gradient
#     step per transition; more workers just oversubscribes the CPU.
#   - torch intra-op threads are capped at 2 per process in train_sac.py to
#     stop concurrent runs thrashing each other.

param(
    # Comma-separated agent subset (default: all 6). Example:
    #   -Agents "attention,fixed_equal,nav_heavy,vanilla,he2025"
    [string]$Agents = "cwn,fixed_equal,nav_heavy,vanilla,he2025,attention"
)

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"
$env:SDL_VIDEODRIVER = "dummy"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

# Unique log per invocation: Out-File holds an exclusive handle for the whole
# multi-hour pipeline, so two concurrently running launcher scripts sharing
# one log file would make the second silently abort on its first write.
$LogFile = "runs\campaign_$(Get-Date -Format yyyyMMdd_HHmmss).log"

"[campaign] start (no-camera, agents=$Agents): $(Get-Date -Format s)" | Out-File $LogFile -Append -Encoding utf8

& .venv\Scripts\python.exe train\run_all.py --agents $Agents --parallel --max_concurrent 8 --n_envs 2 --timesteps 1000000 --no_camera --no_check_env 2>&1 |
    Out-File $LogFile -Append -Encoding utf8

"[campaign] training done: $(Get-Date -Format s) - starting evaluation" | Out-File $LogFile -Append -Encoding utf8

& .venv\Scripts\python.exe eval\post_training.py --runs_dir runs --n_episodes 100 2>&1 |
    Out-File $LogFile -Append -Encoding utf8

& .venv\Scripts\python.exe eval\run_ablations.py --runs_dir runs --n_episodes 50 2>&1 |
    Out-File $LogFile -Append -Encoding utf8

& .venv\Scripts\python.exe eval\log_trajectories.py --runs_dir runs 2>&1 |
    Out-File $LogFile -Append -Encoding utf8

"[campaign] all done: $(Get-Date -Format s)" | Out-File $LogFile -Append -Encoding utf8
