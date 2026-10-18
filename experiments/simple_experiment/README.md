# Simplified Tier-2 experiment

Does a simpler world change the story? Same walled Tier-2 map, same robot,
tasks, cultural zones, and 6 agents × 3 seeds × 1M steps, with two changes:

| | Tier 2 (paper) | This experiment |
|---|---|---|
| Static obstacles | 92 (walls, nurse desk, 48 beds, 3 static people) | 41 (walls + nurse desk only) |
| Pedestrians (training) | 45 fixed | 10–20, drawn uniformly per episode |
| Density sweep | 23 / 45 / 74 | 10 / 10–20 / 20 |

Isolated like every other folder in `experiments/`: no file in `envs/`,
`rewards/`, `train/`, `eval/`, `baselines/` or `configs/` is modified.
Pedestrian counts are patched in memory (`patch.py`); the map is a generated
YAML (`make_config.py` → `hospital_simple.yaml`, preview `layout_preview.png`,
all floor-plan gates pass).

## Run

```bash
python experiments/simple_experiment/campaign.py    # idempotent: re-run to resume
```

Training writes to `runs/` (git-ignored); the chain then evaluates 500 eps at
10–20 peds plus 200 eps at 10 and at 20, aggregates with `summarize.py`, runs the
inference shield (`eval_shielded_simple.py`), and logs trajectories. Results:
`runs/results/{table,per_task,density_sweep,shielded}.json`.

## Monitoring and auto-restart

```bash
.venv\Scripts\python.exe experiments\simple_experiment\dashboard.py --watch
```

- **Checkpoints** every 50k steps (model + replay buffer + normalizer), so a resume continues exactly.
- **Crashed or hung run:** the orchestrator kills any leftover process for that run and relaunches it from its newest checkpoint.
- **Orchestrator dies / logoff / reboot:** scheduled task `HospitalNavSimpleWatchdog` runs `watchdog.ps1` every 5 minutes (only while you are logged in) and relaunches the orchestrator if it is missing. A restarted orchestrator adopts runs that are still training. The task stops relaunching once `runs/ALL_DONE` exists.
- Remove the task when finished: `schtasks /Delete /TN HospitalNavSimpleWatchdog /F`

Any main-repo script can be run under the patch with
`python experiments/simple_experiment/run.py <script> [args]`.

Do not use `eval/post_training.py` here: it re-renders the paper figures.
