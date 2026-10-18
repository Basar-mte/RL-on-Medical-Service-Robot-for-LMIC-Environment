# Shield weaning (Tier 2, 45 pedestrians, 1M steps)

**Question.** Can the policy learn what the reactive safety shield does, so it
no longer needs the shield at deployment, while keeping the shield's success gain?

**Why.** On the trained Tier-2 policies the inference shield alters **59–71 %**
of all actions (measured 2026-10-04 on `fixed_equal_s0` / `vanilla_s0`, 30 eps).
The policies never learned to slow down near obstacles; the shield does it for
them. 71–76 % of Tier-2 episodes end in a collision. Training naively with the
shield in the loop made it worse (−15…−23 SR vs inference shielding, Tier 1):
the policy leaned on the filter.

## Method

Four parts, all injected at runtime (`run.py`); no main-repo file changes.

1. **Executed-action relabeling.** The shield runs inside the environment
   (`ShieldWrapper`). The replay buffer stores the action that was *executed*,
   so the critic learns the value of what happened, never that an unsafe
   proposal was fine.
2. **Corrective distillation.** On intervened transitions an auxiliary loss
   pulls the actor's mean action toward the stored (shield) action.
3. **Annealed intervention budget.** The distillation weight λ is set by dual
   ascent toward a target intervention rate that decays linearly from 0.60
   (the measured reliance) to 0 at 80 % of training. Where the policy still
   relies on the shield, λ rises.
4. **Shielded exploration throughout**, so early training sees fewer collision
   terminations.

Each component exists in prior work (Alshiekh 2018 relabeling; EGPO's
Lagrangian intervention budget, Peng 2021; Guerrier 2025's decaying CBF;
intervention penalties in Bejarano 2025, Wang 2018). The combination with a
budget annealed to zero for shield-free deployment, and its application to
crowd navigation, is what is new. Do not claim the mechanisms as new.

## Runs (15, one wave)

| mode | agents | what it tests |
|---|---|---|
| `wean` | fixed_equal, attention, vanilla × 3 seeds | the method |
| `fixed` | fixed_equal × 3 | budget held at 0.60: is the anneal needed? |
| `naive` | fixed_equal × 3 | old in-loop design, matched to Tier 2 |

Baselines: existing 1M Tier-2 runs, `runs_v4/results_1M/` (fixed_equal 28.4,
vanilla 26.8, attention 24.9 % SR; best with inference shield 34.5 %).

**Success looks like:** `wean` unshielded SR above its baseline's *shielded*
SR, with CVR unchanged or lower, and `wean` > `fixed` > `naive`.

## Run / monitor

```bash
.venv\Scripts\python.exe experiments\shield_weaning\campaign.py            # idempotent; waits for the simple experiment to finish
.venv\Scripts\python.exe experiments\shield_weaning\dashboard.py --watch   # per-run progress + intervention rate / budget / lambda
```

Scheduled task `HospitalNavWeanWatchdog` relaunches the orchestrator every
5 min if it died (while logged in); stops once `runs/ALL_DONE` exists.
Remove with `schtasks /Delete /TN HospitalNavWeanWatchdog /F`.

Eval chain (automatic): 500 eps unshielded per run → `summarize.py` →
inference-shield eval per mode → `results/comparison.md`.

**Models from this experiment must be loaded with `wean` importable**
(they pickle a reference to `WeanReplayBuffer`): run any evaluation through
`python experiments/shield_weaning/run.py eval/evaluate.py ...`.

Smoke-tested 2026-10-04: relabeling exact (unit test), checkpoint → resume
restores buffer and λ, evaluation loads, `naive` mode runs.
