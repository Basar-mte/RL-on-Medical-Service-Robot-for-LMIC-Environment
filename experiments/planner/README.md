# Route planner experiment (hierarchical planner + RL local policy)

**Problem.** The stuck-anatomy analysis of the stable (pilot-4) policies showed that,
with the safety shield on, 42–48 % of episodes time out; 97–100 % of those are frozen
episodes, and in 86 % no pedestrian is within 1 m. The mapless policy drives into dead
ends (goal behind walls/beds), the shield stops it, the Ackermann chassis cannot pivot,
and the policy never learned a recovery. The specimen task is ~0 % for every policy
because its route is never found. Freezing is a *routing* problem, not a crowd problem.

**Fix.** A global route planner supplies the policy's goal observation:

* `planner.py` — cost-to-go field by Dijkstra on the robot-inflated occupancy grid,
  with a per-cell multiplier derived from the CARS reward itself:
  `m(x) = 1 + ratio · c(x) / 0.8` where `c(x)` is the per-step cultural penalty the reward
  charges at `x` (prayer room 2/3·intensity, gender wards 0.1/3) and 0.8 is the progress
  reward of one full-speed step. With `ratio = w_cult/w_nav` the planner minimises the
  same scalarised objective the reward integrates along a route ("one objective, two
  levels"). `ratio = 0` → plain shortest path (geometric planner, culture-blind agents).
* `patch.py` — in-process patch of `LMICHospitalNavEnv` (no main-code change): the goal
  bearing becomes the bearing to a waypoint `L` m along the route (pulled back to line of
  sight); for training, `info["dist_to_goal"]` becomes the remaining route length so the
  progress reward is geodesic for every agent (CARS, vanilla, he2025 all read that key).
  Controlled by env vars `PLANNER_MODE/RATIO/L/DIST/PROGRESS/LOS`;
  `experiments/stability_pilot/patch_sac.py` applies it when `PLANNER_MODE` is set.
* `plot_routes.py` — routes per task/spawn/variant → `routes_<config>.png` (sanity + figure).
  On Tier 2 the prayer shortcut exists only from the south-corridor spawn (STAT, restock:
  5.2 m inside the room); in a prayer window the cultural planner detours (+2.3 m);
  outside it keeps the shortcut, as the reward's 0.3 intensity implies.
* `eval_planner.py` — evaluation-only test on the existing pilot-4 policies (policy
  unchanged, only its goal observation is steered). `run_matrix.py` runs the matrix
  (6 policies × {no planner, geometric, cultural} × shield, + ablations on ratio, L,
  distance channel, line of sight) in parallel; `summarize.py` prints the tables.

**Results (evaluation-only, pilot-4 policies, Tier 2, 200 episodes each; `summarize.py`).**
Means over policies, with the shield:

| variant | models | SR % | coll % | timeout % | frozen % | CVR | prayer CVR | SPL |
|---|---|---|---|---|---|---|---|---|
| no planner | 4 | 41.5 | 12.2 | 46.2 | 46.1 | 0.114 | 0.052 | 0.41 |
| geometric planner | 5 | 78.7 | 10.0 | 11.3 | 11.0 | 0.117 | 0.032 | 0.77 |
| cultural planner, 1× | 6 | 82.2 | 8.5 | 9.3 | 8.9 | 0.118 | 0.026 | 0.81 |
| cultural planner, 3× | 4 | 83.6 | 8.9 | 7.5 | 6.9 | 0.093 | **0.000** | 0.81 |

Without the shield the cultural planner gives 71.6 % SR / 27.5 % collisions (timeouts
0.9 %): the planner removes the freezing, the shield removes the collisions. Per task,
the route specialisation disappears (specimen 0–2 % → 72–90 %; every policy now solves
all five tasks). Ablations on fixed_equal_s1 (cultural, shield, SR %): look-ahead 1.5 m
77.0 / **2.5 m 81.5** / 4 m 77.5; no line-of-sight pull-back 74.0; geodesic distance
channel 82.0 (harmless even for policies trained on Euclidean distance); cultural cost 3×
instead of 1×: 85.5, confirmed on attention s0/s1 and fixed_equal s0 (+2.5 to +4 pp each,
prayer-room steps 0.000 in all four). **Chosen for training:** cult (3×) / geo, L = 2.5 m,
line of sight on, geodesic distance channel, geodesic progress reward. The 1× setting
(reward-consistent) keeps the out-of-window prayer shortcut; 3× makes the detour
cost-optimal at all times, which is the conservative institutional rule.

**Campaign design (scripts/campaign_v5_stable.py, relaunched with the planner).** Per-run
planner settings travel through the process environment (`env_for(k)` in the orchestrator,
mirrored by `scripts/eval_v5.py` so every policy is evaluated with the wiring it was
trained for): `PLANNER_MODE` cult for CARS agents / geo for vanilla & he2025
(`planner="auto"`), `PLANNER_RATIO=1`, `PLANNER_L=2.5`, `PLANNER_DIST=geo`,
`PLANNER_PROGRESS=1` (0 at evaluation), `PLANNER_LOS=1`. Groups: `tier2` and `tier1`
(6 agents × 5 seeds × 2M, hierarchical), `tier2_mapless` (fixed_equal/vanilla × 5 × 2M, no
planner — the end-to-end baseline), `tier2_planner_ablation` (fixed_equal→geo, vanilla→cult
× 3 × 2M, completing the 2×2 {policy reward} × {planner cost}), `tier1_mapless`
(fixed_equal × 3), `tier2_pivot` (fixed_equal × 3 × 1M, pivot kinematics + planner).
The smoke mode (`V5_SMOKE=1`) exercises every combination end-to-end before a launch.

**Classical local drivers (scripts/eval_classical_v5.py, Tier 2, 500 eps, same planner).**
Stock planner + social force freezes (6 % — its speed law reaches v = 0 when the force
points > 90° off heading and an Ackermann chassis cannot turn at v = 0). The Ackermann-aware
variant (`sf_prior.AckermannSF`: same forces, pure-pursuit speed law with a minimum rolling
speed; gains goal 1.5 / wall 1.0 / vmin 0.4 selected on tuning episodes, seeds 5000+) reaches
**95.4 % / 4.6 % collisions** without shield (95.7 % on a third seed range), 93.0 % / 0 % with
v1, 97.8 % / 0 % with v2. The learned policies (80–85 % / 15–20 %) are 20 % faster (28 vs 34 s)
but keep people closer (SVR 0.042 vs 0.014); capping their speed does not close the gap
(collisions 19 → 14 % at 0.6×), so the deficit is distance-keeping set by the reward balance.
Audit (`audit_classical_vs_learned.png`): same inputs, same routes, no privileged information.

**Kinematic shield v2 (`arc_shield.py`).** Collision anatomy of the v1-shielded learned
policies: 79–81 % of residual contacts at the robot's side, 16–19 % behind, 0–5 % in front —
the forward cone's blind spot. v2 governs speed by the arc length to the first obstacle inside
the body's swept band along the commanded steering arc (LiDAR points + the 5 tracked
pedestrians propagated 0.6 s; lateral margin 3 cm for static points, 15 cm for people;
15 cm stopping margin along the track; DWA-style steering alternatives; reverse governed).
Identical episodes (300): fixed_equal_s1 94.3 % / **0.0 %** / 5.7 % timeouts (v1 89.0 / 6.3 /
4.7), vanilla_s0 95.3 / 0.0 / 4.7 (v1 90.7 / 5.3 / 4.0), same 31 s per delivery; classical
97.8 / 0.0 / 2.2. `scripts/eval_v5.py` reports it as the `shield2` column for every group.

**Deployment.** This reverses the "map for localization only, RL picks the route" wiring
in `deploy/MENTORPI_DEPLOYMENT.md` §1: the deployed stack becomes
SLAM map + zone labels → the same cost field (numpy only) → waypoint → `obs["goal"]`,
i.e. the planner-follower wiring that `ros2_stack_legacy/path_to_waypoint.py` already
implements, with policies now *trained* for it (the distance channel stays the true
remaining distance, so the out-of-distribution objection no longer applies).
