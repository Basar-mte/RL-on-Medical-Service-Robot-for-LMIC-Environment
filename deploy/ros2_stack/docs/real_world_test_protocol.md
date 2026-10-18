# Reduced-Density Real-World Test Protocol

**Purpose**: demonstrate hardware feasibility and sim-to-real transfer for
a reviewer audience — this is explicitly **not** a hospital-equivalent
benchmark. The simulator's own "low" density tier already uses 23
simulated pedestrians; a real test with a handful of people is well below
even that floor. Label any resulting writeup a "hardware feasibility and
reduced-density validation," not a replication of hospital conditions.

## 1. Test-site selection

Choose the largest available space that is *structurally* analogous to the
simulator's corridors (see `configs/dmch_hospital.yaml`: ~3m-wide main
corridor) — a clinic hallway, an unused institutional wing, or a community
space corridor. It does not need to be a hospital. Add static clutter
standing in for the sim's "overflow beds in corridors" motif — benches,
boxes, or carts narrowing the path — since that clutter, not the label
"hospital," is the core LMIC-relevant navigation challenge being tested.

## 2. Pedestrian count

Target 2-5 people for the reduced-density run. State explicitly in any
writeup that this is below the simulator's own "low" density (23 agents),
not an attempt at parity.

## 3. Task framing

1-2 simple point-to-point runs loosely modeled on the simulator's task
catalogue (`envs/lmic_hospital_env.py::TASKS`, e.g. a "nurse-station-like"
start point to a "ward-like" goal, roughly 10-20m). Use this for narrative
framing only — don't claim deadline/urgency semantics that don't
meaningfully exist on a real test day.

## 4. Metrics — what transfers honestly, what doesn't

Reuse `eval/evaluate.py::aggregate_metrics`'s field names for direct
comparability with the sim results table:

| Metric | Transfers to real test? |
|---|---|
| SR (Success Rate) | Yes |
| Collision count / rate | Yes |
| CT (Completion Time) | Yes |
| Min pedestrian distance (MPD-analogue) | Yes |
| CVR (Cultural Violation Rate) | **No** — tied to the sim's gender-zone/prayer-room constructs; only port this if the test site is deliberately set up with taped-off zone analogues as a qualitative demonstration (optional, not required) |
| SVR (Social Violation Rate) | Same caveat as CVR |

## 5. Sample size

N ≥ 10-20 runs minimum per condition, explicitly labeled a small-N
feasibility pilot, not a statistically powered study.

## 6. Evidence capture plan

1. **External video** (tripod or handheld) of every run, showing the robot
   + environment — distinct from the onboard camera, and the only way to
   independently verify collisions/near-misses.
2. **Onboard camera frames** — via `scripts/field_test_logger.py`'s
   periodic snapshots.
3. **Full sensor/action/odometry CSV log** — `scripts/field_test_logger.py --out_dir field_test_logs/run_NNN`.
4. **Annotated real trajectory plot** — overlay the logged `(x, y)`
   odometry trace on a rough sketch of the test site's layout, styled
   consistently with the paper's `figures/output/fig6_trajectories` (real
   meter-scale X/Y axes) for visual continuity.

## 7. Comparison table template

| Condition | SR | CT (s) | MPD-analogue (m) |
|---|---|---|---|
| Sim (low density, 23 agents) | | | |
| Real (reduced density, 2-5 people) | | | |

Footnote every such table with the density-gap and small-N caveats from
sections 2 and 5 — don't let the table imply parity it doesn't have.

## 8. Reviewer-facing framing text (adapt directly)

> To validate sim-to-real transfer under real hardware and sensing
> constraints, we conducted a reduced-density feasibility test on the
> physical robot in [venue], using [N] runs with 2-5 ambient pedestrians —
> substantially below our simulator's own low-density tier (23 agents).
> This test demonstrates hardware-level navigation competence and
> sim-to-real transfer of the trained policy; it is not intended to
> replicate the pedestrian density or cultural-constraint conditions of
> the simulated LMIC hospital ward, which remain the basis for our main
> results.
