# Simulation v2 Changelog: 270° LiDAR/Diff-Drive → 360° LiDAR+Camera/Ackermann

> **Note (final config):** the camera sensor described below was later made
> **optional and disabled for all paper experiments** — the benchmark and the
> Yahboom A1 deployment are LiDAR-only, with a 90-dim vector observation. The
> 360° LiDAR and Ackermann kinematics are retained. This document is kept as
> the historical record of the v2 revision.

## 1. Summary

The original ask was a straightforward fix: switch the robot's LiDAR from
270° to 360° field of view. Scope expanded once the real goal became clear
— sim-to-real transfer to a physical robot (Yahboom A1 Ackermann chassis +
Raspberry Pi 5 + LDROBOT LD19 LiDAR + pan-tilt monocular camera + RRC Lite
motor controller). That target hardware motivated three simulation
changes together: matching the LD19's real 360°/12m spec, adding a camera
sensor fused with LiDAR, and switching from differential-drive to
Ackermann/bicycle-model steering kinematics to match the chassis.

## 2. Before / After Comparison

| Aspect | v1 (legacy, `runs_legacy_270fov_nocamera/`) | v2 (current, `runs/`) |
|---|---|---|
| LiDAR FOV | 270° | 360° |
| LiDAR max range | 10.0 m | 12.0 m |
| LiDAR ray count | 48 | 48 (unchanged) |
| Sensors | LiDAR only | LiDAR + synthetic egocentric camera (84×84×3 RGB, 90° forward FOV, 5m range) |
| Observation space | flat `Box(85,)` | `Dict` (`lidar`, `robot_state`, `pedestrians`, `context`, `camera`), 85-dim vector portion unchanged + camera |
| Action space | `[v, ω]` differential-drive, `ω ∈ [-π, π]` | `[v, δ]` Ackermann, `δ ∈ [-30°, 30°]` (WHEELBASE=0.20m, MAX_STEER=30° — placeholders pending physical chassis measurement) |
| Robot kinematics model | unicycle (heading changes independent of `v`, can pivot in place) | bicycle model (`heading_rate = (v/L)·tan(δ)`, cannot pivot at `v=0`) |
| Policy network | `MlpPolicy` | `MultiInputPolicy` (`CombinedExtractor`: NatureCNN on camera + MLP[256,256] on vector obs) |
| Replay buffer | 200,000, unconstrained | 100,000 — `DictReplayBuffer` doesn't support `optimize_memory_usage`, camera key roughly doubles per-transition storage |
| Single-env throughput (reference dev machine) | ~171 steps/s (documented, LiDAR-only) | ~22 steps/s with camera (~55 without, isolating the camera's cost) |
| `--n_envs` | parsed but never used (always 1 env) | actually parallelises via `SubprocVecEnv`; measured ~3× throughput recovery at `n_envs=4` |
| Training steps × seeds | 300k (demo) / 1M (paper) × up to 3 seeds | 50k × seed 0 (validation, in progress) → 1M × 3 seeds (paper target, not yet run) |
| Execution mode | `run_all.py --parallel` (6 agents concurrently) | sequential by default — 6× the camera-inflated replay buffer would exceed RAM on memory-constrained machines |
| Hardware target | none (simulation only) | Yahboom A1 + Raspberry Pi 5 + LD19 + RRC Lite (see `hardware_deploy/`) |
| `runs/` status | archived to `runs_legacy_270fov_nocamera/` | active |

## 3. Motivation

Direct sim-to-real transfer requires the simulated dynamics/sensors to
match what the real hardware can actually do and sense. Differential-drive
kinematics let the robot pivot in place — a real Ackermann-steered vehicle
cannot — so a policy trained on the old model would learn to rely on
maneuvers the physical chassis can't execute. See `hardware_deploy/README.md`
for the full hardware-side plan.

## 4. What Changed in the Environment

`envs/lmic_hospital_env.py`:
- `LIDAR_FOV`: 270.0 → 360.0; `LIDAR_MAX`: 10.0 → 12.0.
- New `lidar_ray_angles(fov_deg, n_rays)` function (shared with
  `baselines/classical.py`, preventing the two from silently diverging on
  ray geometry — this was the exact class of bug found in the old code,
  where `SocialForceController` hardcoded a separate ±135°/48-ray copy).
- New `spaces.Dict` observation space (`lidar`, `robot_state`,
  `pedestrians`, `context`, `camera`) replacing the flat `Box(85,)`. This
  also fixed a pre-existing bug: the old bounds construction gave the
  pedestrian obs block blanket `[-1,1]` bounds meant for a different slice,
  silently clipping any pedestrian more than ~1m away to exactly ±1.0.
- New `_render_camera()` — a rasterized, forward-facing, occlusion-aware
  egocentric crop built from the existing pygame renderer, rotated to the
  robot's heading. Two real bugs were found and fixed during
  implementation: a rotation-angle formula that pointed exactly backward at
  90°/270° headings (verified via dumped PNG frames at each cardinal
  heading), and floating-point rounding seams between adjacent floor cells.
- `_render_rgb()` (the `rgb_array` render mode) now also draws
  pedestrians/robot/goal — previously it only rasterized the floor, a
  pre-existing gap independent of this change.
- Ackermann/bicycle-model kinematics replacing the unicycle model in
  `step()`; new `WHEELBASE`/`MAX_STEER` constants (placeholders).
- `baselines/classical.py`'s `AStarController`/`SocialForceController` now
  output Ackermann `[v, δ]` via an inverse bicycle-model conversion
  (`atan2(ω·L, v)`), and import the shared LiDAR geometry instead of a
  second hardcoded copy.

## 5. What Changed in Training

`train/train_sac.py`:
- `MlpPolicy` → `MultiInputPolicy`.
- `VecNormalize(norm_obs_keys=...)` excludes `camera` — otherwise SB3's
  image-space auto-detection stops routing it through the CNN branch.
- `VecTransposeImage` now wraps the raw env **before** `VecNormalize`, not
  after — the opposite order crashes `_store_transition()` on the first
  terminal step (`ValueError: could not broadcast input array from shape
  (3,84,84) into shape (84,84,3)`), because SB3's off-policy training loop
  fetches VecNormalize's cached "original observation" for the replay
  buffer, and that cache reflects whatever shape VecNormalize itself
  received.
- `--n_envs` actually parallelises now (`SubprocVecEnv` when > 1) —
  previously parsed but ignored.
- `--buffer_size` default lowered 200k → 100k (see §2).

## 6. What Changed in Evaluation & Figures

- `eval/evaluate.py`: Dict-obs-safe `_batch`/`_unbatch` helpers; fixed a
  `_find_vecnormalize()` glob that never matched SB3's real checkpoint
  filenames; fixed a pre-existing double-reset bug where a second,
  unseeded `reset()` call silently discarded the intended task/density/seed
  (confirmed this made task/density conditioning never actually work
  previously); fixed a 5-value unpack of `VecEnv.step()`'s actual 4-value
  return (`obs, rewards, dones, infos`) that would have crashed on the very
  first step of any VecNormalize-wrapped evaluation.
- `eval/run_ablations.py`: fixed a call with 3 positional args against a
  2-arg function signature, and a `.pop("records")` on a dict that never
  had that key — both would raise before completing.
- `eval/aggregate.py`: default `--output` path aligned with
  `eval/post_training.py`'s (`runs/results/table.json`, not the repo-root
  `results/table.json` the two previously diverged on); `to_latex()`
  rewritten without pandas' Styler path (which needs a newer jinja2 than
  installed in this environment).
- New `eval/log_trajectories.py`: records one real deterministic rollout
  per agent, replacing `figures/make_figures.py`'s previously
  **entirely-fabricated** synthetic trajectory waypoints.
- `figures/make_figures.py`: `fig6_trajectories` now accepts real rollout
  CSVs (falls back to the old synthetic paths if absent); `fig2_architecture`'s
  observation-space diagram text updated for the Dict/camera structure.

## 7. Results Comparison (v1 vs v2)

**v2 validation-scale results** (50k steps × seed 0 only, 20 eval episodes
— NOT the 1M-step × 3-seed paper target; these numbers exist to confirm the
pipeline works end-to-end, not as research results):

| Agent | SR ↑ | CVR ↓ | SPL ↑ | MPD ↑ | SVR ↓ |
|---|---|---|---|---|---|
| CWN (ours) | 0.10 | 0.022 | 0.02 | 1.86 | 0.176 |
| B5: Attn-Weight | 0.10 | 0.031 | 0.02 | 1.31 | 0.186 |
| B2: Nav-Heavy | 0.15 | 0.018 | 0.00 | 0.89 | 0.276 |
| B1: Equal-Weight | 0.15 | 0.024 | 0.00 | 0.89 | 0.265 |
| B4: He et al. (2025) | 0.08 | 0.020 | 0.00 | 1.34 | 0.189 |
| B3: Vanilla SAC | 0.08 | 0.021 | 0.00 | 2.16 | 0.166 |

Success rates are low (8-15%) across every agent, exactly as expected at
50k/1M (5%) of the paper's target training budget — these should not be
read as a CWN-vs-baseline comparison, all agents are far from converged.
`figures/output/fig4_results.png`'s y-axis ranges are tuned for
near-converged SR/SPL values, so most bars at this scale render clipped
off-screen — a cosmetic consequence of validation-scale data, not a bug.
Real (very short, 7-10 step) rollout trajectories were successfully
recorded and plotted for all 6 agents in `fig6_trajectories`, replacing the
previous fabricated paths — confirming `eval/log_trajectories.py` and the
`_load_real_trajectories()` figure wiring both work correctly.

**v1 comparison**: not run as part of this change — see
`runs_legacy_270fov_nocamera/*/eval_results.json` for the archived 270°/
no-camera/diff-drive results if a direct before/after table is needed
later (those were themselves preliminary 300k-step runs, so any v1-vs-v2
comparison should wait until both are at the same, converged training
budget to be meaningful).

**Next step**: run the full paper-scale training (`train/run_all.py
--timesteps 1000000 --n_envs 4`, sequential, ~3 seeds) before drawing any
actual research conclusions from this table.

## 8. Backward-Compatibility Notes

- `runs_legacy_270fov_nocamera/` checkpoints cannot be loaded against the
  new Dict observation space / Ackermann action space without retraining
  (different shapes entirely — `SAC.load()` will fail against a mismatched
  space).
- `figures/output/` renders from before this change reflect the v1 sim and
  should not be cited going forward once regenerated.

## 9. Known Gaps / Follow-Ups

- Term-based CWN ablations (A2: no-prayer, A3: no-gender, A4: no-attendant)
  are structurally unwireable as a post-hoc eval trick against a frozen
  policy (reward doesn't affect inference-time action selection) — they
  need dedicated retrained CWN variants with the term masked at training
  time. `fig7_cwn_ablation`'s term-ablation panel should be treated as
  illustrative until those variants exist.
- **Unrelated pre-existing bug found during validation, not fixed here**:
  pedestrians can spawn overlapping the robot's fixed spawn point at high
  density, causing an instant spawn-collision independent of kinematics
  model — affects the v1 sim equally, out of scope for this change.
- Hardware-side gaps (RRC Lite protocol, real pedestrian detector, zone
  mapping for a real test site) are tracked in `hardware_deploy/raspberry_pi5/README.md`'s
  "implemented vs. TODO" table, not here.
- Full paper-scale retrain (1M steps × 3 seeds × 6 agents) has not yet been
  run — the 50k × seed 0 validation pass exists to confirm the pipeline
  works end-to-end first, given the substantially higher per-step cost of
  camera rendering.
