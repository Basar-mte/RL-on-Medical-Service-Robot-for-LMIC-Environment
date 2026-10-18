# What to do — step by step

Work through these in order. Each step says what to run and how you know it
worked. `README.md` explains *what* the package is; this explains *what you do*.

Nothing here retrains anything. You are running already-trained policies through
a higher-fidelity simulator to see how they survive real chassis physics.

---

## Before Isaac: two measurements (15 minutes, with the robot)

These gate whether the physics means anything. Do them first — everything
downstream inherits the error.

- [ ] **Weigh the robot**, fully assembled with battery, Pi, and LiDAR.
      The URDF currently says **1.8 kg**, which I estimated, not measured. The
      CAD export said 0.27 kg, which is definitely wrong.
- [ ] **Measure the turning circle.** Full steering lock, drive a complete
      circle, measure its diameter. Halve it — that's your true minimum turning
      radius. The URDF implies 0.137 m.

Then rebuild the model with the real mass:

```bash
python deploy/isaac/build_urdf.py --mass <your_kg>
python deploy/isaac/check_urdf.py
```

> If the measured turning radius differs a lot from 0.137 m, the wheelbase is
> wrong somewhere. The URDF says 0.1371 m and the robot's own driver says
> 0.145 m; they cannot both be right. Fix whichever disagrees with the tape
> measure before going further.

---

## Step 1 — Verify the package survived the copy

```bash
python deploy/isaac/check_urdf.py
python deploy/isaac/check_world_usd.py
python deploy/isaac/isaac_core.py
```

**Expected:** three `VERDICT:` lines, all positive.

- `ready for the Isaac URDF importer`
- `stage matches the training config`
- `core matches the training environment`

**If the third one fails, stop.** It means the observation assembly no longer
matches what the policies were trained on, and every number after that would be
meaningless. It is the one check worth trusting completely — it compares against
the real training environment, field by field.

---

## Step 2 — Convert the URDF to USD (once)

In Isaac Sim: **Isaac Utils → Workflows → URDF Importer**, load
`deploy/isaac/mentorpi/mentorpi_ackermann.urdf` with

| Setting | Value |
|---|---|
| Fixed base | **off** |
| `steer_lf_Joint`, `steer_rf_Joint` | drive type **position** |
| `wheel_lb_Joint`, `wheel_rb_Joint` | drive type **velocity** |
| Self-collision | **off** |

Save as `deploy/isaac/mentorpi/mentorpi_ackermann.usd`, beside the URDF.

- [ ] **Sanity-check in the viewport before continuing.** Drag the
      `steer_*_Joint` sliders: the front wheels must visibly turn. Then press
      play with no commands — the robot should sit still on its wheels, not
      sink through the floor, jitter, or drift.

Jitter or sinking almost always means mass or inertia, which is why the weighing
step comes first.

---

## Step 3 — Open the ward

Open `deploy/isaac/mentorpi/hospital_tier2.usda` directly — no conversion
needed. You should see the walled ward: three wards along the top, a corridor
across the middle, four rooms below, beds as low boxes.

- [ ] Confirm the robot spawns inside the corridor, not inside a wall. The
      benchmark spawn is `(21.5, 15.0)`.

---

## Step 4 — Finish the two stubs

`eval_isaac.py` runs as shipped but measures very little until these are done.
Both are marked with `TODO(isaac)` in the file.

- [ ] **LiDAR.** Currently a placeholder returning uniform free space, so the
      policy drives as though the ward were empty. Attach a RayCaster (or RTX
      lidar) configured **48 rays, 360°, 12 m max**, and pass the ranges into
      `build_obs(lidar_ranges=...)`. Order matters: ray *i* must correspond to
      the same bearing the training env used — evenly spaced from −π, matching
      `scan_utils.BASE`.
- [ ] **Crowd.** `people` is empty, so the pedestrian block is zeros and both
      density terms read 0. Without pedestrians you are measuring geometry and
      chassis dynamics only.

**You can legitimately stop after the LiDAR and report that**, as long as you
say it measured navigation without a crowd. That is a real result: it isolates
whether the chassis physics alone breaks the learned behaviour.

---

## Step 5 — One episode, watched

Run non-headless and watch the whole thing before you trust any batch:

```bash
python deploy/isaac/eval_isaac.py \
    --checkpoint models/fixed_equal_seed2_best.zip \
    --vecnorm    models/fixed_equal_seed2_vecnorm.pkl \
    --episodes 1
```

`fixed_equal_seed2` is the strongest Tier-2 policy, so it is the right first
test — if it behaves badly, suspect the harness before the policy.

Watch for:

| What you see | What it means |
|---|---|
| Drives smoothly toward the goal | harness is wired correctly |
| Spins in place / drives in circles | steering sign flipped — negate `delta` in the bridge |
| Barely turns | using the real wheelbase instead of the sim's 0.20 m |
| Creeps very slowly | velocity target in rad/s vs m/s confusion |
| Ignores obstacles | LiDAR still stubbed (Step 4) |
| Jitters, tips, or slides | mass/inertia — go back and weigh it |

**Always pass the matched `--vecnorm`.** A mismatched normalizer silently
destroys performance and looks exactly like a bad policy.

---

## Step 6 — Batch and record

```bash
python deploy/isaac/eval_isaac.py \
    --checkpoint models/<agent>_best.zip \
    --vecnorm    models/<agent>_vecnorm.pkl \
    --episodes 20 --headless
```

Repeat for each of the four policies in `models/`. Results land in
`deploy/isaac/results/isaac_eval.json`, so rename between runs or they overwrite.

Record per run: agent, seed, checkpoint sha256 (in `MANIFEST.json`), episode
count, and whether the crowd was enabled.

---

## Step 7 — Report it correctly

- [ ] Use **per-policy outcomes** — "reached" / "not reached", plus the failure
      mode (halted at doorway, collision, off-route, timeout). Repeats may be
      "reached in N of M attempts".
- [ ] **Never quote a success rate** beside the benchmark tables. Different
      simulator, different sensor model, different denominator — it is the
      comparison a referee flags first.
- [ ] Report Isaac as a **separate condition**, not merged into `tab:wardb`.
- [ ] State whether the crowd was enabled. Without it, say so explicitly.

Expect **lower numbers than Tier 2**. Isaac's sensor model differs from the
benchmark's exact ray–circle LiDAR, so the observation distribution shifts even
with identical geometry. That gap is itself a finding about sim-to-real
fidelity — report it as one rather than treating it as a failure.

---

## If you get stuck

| Symptom | Likely cause |
|---|---|
| `mentorpi_ackermann.usd not found` | Step 2 not done, or saved elsewhere |
| Import errors on `isaaclab` | wrong Python — use Isaac's, not a fresh venv |
| Import errors on `envs.*` | running from the wrong directory; run from the package root |
| `check_world_usd` reports drift | `hospital_v4.yaml` changed — re-run `build_world_usd.py` |
| Robot falls through the floor | fixed base left on, or collision not imported |
| Policy output is `nan` | vecnorm missing or mismatched |

`eval_isaac.py` has never been executed — it was written on a machine without
Isaac Lab. Expect to adjust a few import paths and config field names for your
release. The parts that carry the numbers (`isaac_core.py`) are tested; the glue
is not.
