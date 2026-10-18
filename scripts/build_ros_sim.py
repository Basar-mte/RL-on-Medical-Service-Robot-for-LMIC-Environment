"""Build ros_sim/ — the self-contained ROS 2 folder for Gazebo/RViz simulation and
MentorPi deployment of the FINAL campaign's policies (best seed of EVERY trained
agent, both tiers).

Contents:
  hospitalnav_deploy/   the ROS 2 package (route planner, v2 shield, residual/classical
                        controller modes, people tracker, loggers, worlds, launch files)
  models/               best-seed checkpoints with their MATCHED normalisers
  results_summary/      the benchmark tables these models came from + the routes figure
  MANIFEST.json         sha256 of every model file + its benchmark numbers
  README.md             exact bring-up commands

    .venv\\Scripts\\python.exe scripts\\build_ros_sim.py
"""
from __future__ import annotations
import hashlib, json, shutil, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import campaign_v5_stable as C  # noqa: E402

OUT = ROOT / "ros_sim"

# folder -> (group, agent, seed, planner cultural_ratio (None = mapless, no planner),
#            sb3 algo, note). Best seed by shield2 SR from runs_v5/results/<group>/per_seed.json.
PICKS = {
    # ---- Tier 2 (walled ward — the shipped Gazebo world) ----
    "residual_sf_s2":  ("tier2_residual", "residual_sf", 2, 3.0, "sac",
                        "RECOMMENDED: residual RL over the classical prior - 99.8 % shield2 (seed), 99.4 +- 0.4 (3 seeds)"),
    "fixed_equal_s2":  ("tier2", "fixed_equal", 2, 3.0, "sac",
                        "pure learned CARS policy, equal weights - 97.0 % shield2 (seed), 95.4 +- 1.7 (5 seeds)"),
    "vanilla_s3":      ("tier2", "vanilla", 3, 0.0, "sac",
                        "culture-blind contrast agent (geo planner) - 97.6 % shield2 (seed); cuts through the prayer room"),
    "cwn_s4":          ("tier2", "cwn", 4, 3.0, "sac",
                        "context weight network (learned reward weights) - 97.4 % shield2 (seed)"),
    "nav_heavy_s4":    ("tier2", "nav_heavy", 4, 3.0, "sac",
                        "nav-weighted CARS policy - 97.0 % shield2 (seed)"),
    "attention_s4":    ("tier2", "attention", 4, 3.0, "sac",
                        "attention reward weighter - 96.2 % shield2 (seed)"),
    "risk_field_s0":   ("tier2", "he2025", 0, 0.0, "sac",
                        "risk-field baseline (He 2025 style, geo planner) - 96.2 % shield2 (seed)"),
    "safe_heavy_s2":   ("tier2_safe", "safe_heavy", 2, 3.0, "sac",
                        "safety-weighted CARS policy - 94.8 % shield2 (seed)"),
    "td3_equal_s1":    ("tier2_td3", "td3_equal", 1, 3.0, "td3",
                        "TD3 external validation (algo:=td3) - 93.2 % shield2 (best seed; 65.0 +- 11.3 across seeds)"),
    "ppo_equal_s0":    ("tier2_ppo", "ppo_equal", 0, 3.0, "ppo",
                        "PPO library defaults (algo:=ppo) - 0 % in benchmark; shipped for completeness, does not navigate"),
    "mapless_fixed_equal_s3": ("tier2_mapless", "fixed_equal", 3, None, "sac",
                        "MAPLESS (no planner - straight-line goal input) - 63.4 % shield2 (seed); the routing-bottleneck demo"),
    "mapless_vanilla_s2":     ("tier2_mapless", "vanilla", 2, None, "sac",
                        "MAPLESS culture-blind - 59.4 % shield2 (seed)"),
    # ---- Tier 1 (open tier; no dedicated Gazebo world - run in the ward world) ----
    "t1_fixed_equal_s3": ("tier1", "fixed_equal", 3, 3.0, "sac", "Tier-1 CARS policy - 96.4 % shield2 (seed)"),
    "t1_vanilla_s2":     ("tier1", "vanilla", 2, 0.0, "sac", "Tier-1 culture-blind (geo planner) - 96.8 % shield2 (seed)"),
    "t1_cwn_s2":         ("tier1", "cwn", 2, 3.0, "sac", "Tier-1 context weight network - 96.2 % shield2 (seed)"),
    "t1_nav_heavy_s4":   ("tier1", "nav_heavy", 4, 3.0, "sac", "Tier-1 nav-weighted - 96.6 % shield2 (seed)"),
    "t1_attention_s2":   ("tier1", "attention", 2, 3.0, "sac", "Tier-1 attention weighter - 96.6 % shield2 (seed)"),
    "t1_risk_field_s4":  ("tier1", "he2025", 4, 0.0, "sac", "Tier-1 risk-field baseline (geo planner) - 97.0 % shield2 (seed)"),
    "t1_mapless_fixed_equal_s0": ("tier1_mapless", "fixed_equal", 0, None, "sac",
                        "Tier-1 MAPLESS (no planner) - 93.0 % shield2 (seed; open tier is forgiving without walls)"),
}

RESULT_TABLES = ("tier2", "tier2_residual", "tier2_mapless", "tier2_safe", "tier2_td3",
                 "tier2_ppo", "tier1", "tier1_mapless")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)
    (OUT / "results_summary").mkdir()

    # 1. the ROS 2 package
    shutil.copytree(ROOT / "deploy" / "ros2_stack" / "hospitalnav_deploy", OUT / "hospitalnav_deploy",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 2. models + manifest
    manifest = dict(built=time.strftime("%Y-%m-%d %H:%M:%S"), campaign="runs_v5 (ALL_DONE 2026-10-14)",
                    planner=dict(lookahead_m=2.5, los=True, geodesic_distance_channel=True),
                    shield="v2 swept-path (shield_version: 2)", residual_scale=0.5, models={})
    table_cache = {}
    for folder, (group, agent, seed, ratio, algo, note) in PICKS.items():
        rd = C.run_dir((group, agent, seed))
        dst = OUT / "models" / folder
        dst.mkdir()
        files = {}
        for name in ("final_model.zip", "vec_normalize.pkl"):
            shutil.copy(rd / name, dst / name)
            files[name] = sha256(dst / name)
        if group not in table_cache:
            table_cache[group] = json.load(open(C.RUNS / "results" / group / "table.json"))
        t = table_cache[group][agent]
        manifest["models"][folder] = dict(
            source_run=str(rd.relative_to(ROOT)), files=files, note=note, algo=algo,
            planner_cultural_ratio=ratio,
            benchmark=dict(final_sr=round(t["final"]["sr"], 4), shield2_sr=round(t["shield2"]["sr"], 4),
                           shield2_collisions=round(t["shield2"]["collision"], 4), cvr=round(t["final"]["cvr"], 4)))

    # 3. verification: every model loads with its stock SB3 class and yields an in-box action
    from stable_baselines3 import SAC, TD3, PPO
    LOADERS = {"sac": SAC, "td3": TD3, "ppo": PPO}
    for folder, (group, agent, seed, ratio, algo, _n) in PICKS.items():
        mdir = OUT / "models" / folder
        model = LOADERS[algo].load(str(mdir / "final_model.zip"), device="cpu")
        obs = {k: np.zeros(s.shape, dtype=s.dtype)[None] for k, s in model.observation_space.spaces.items()}
        a, _ = model.predict(obs, deterministic=True)
        assert model.action_space.contains(a[0]), folder
        keys = sorted(model.observation_space.spaces.keys())
        print(f"  verified {folder} ({algo}): obs keys {keys}, action {np.round(a[0], 3)}")
        manifest["models"][folder]["observation_keys"] = keys

    # 4. results summary + routes figure
    for g in RESULT_TABLES:
        shutil.copy(C.RUNS / "results" / g / "table.json", OUT / "results_summary" / f"{g}_table.json")
    fig = ROOT / "experiments" / "planner" / "routes_hospital_v4.png"
    if fig.exists():
        shutil.copy(fig, OUT / "results_summary" / "planner_routes.png")

    json.dump(manifest, open(OUT / "MANIFEST.json", "w"), indent=1)
    (OUT / "README.md").write_text(README, encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"ros_sim/ built: {total:.1f} MB, {len(PICKS)} models")


README = """# ros_sim - Gazebo/RViz simulation and MentorPi deployment package

Best seed of EVERY trained agent from the final campaign (both tiers) with the full
hierarchical stack: route planner (cultural costs) -> local driver -> kinematic
swept-path shield (v2). Built by `scripts/build_ros_sim.py`; file hashes and benchmark
numbers in `MANIFEST.json`.

## Quick start (Ubuntu 22.04, ROS 2 Humble, Gazebo classic)

```bash
mkdir -p ~/hn_ws/src && cp -r hospitalnav_deploy ~/hn_ws/src/
cd ~/hn_ws && colcon build --symlink-install && source install/setup.bash

# recommended driver (residual RL)
ros2 launch hospitalnav_deploy gazebo_full_bringup.launch.py \\
    agent:=residual_sf cultural_ratio:=3.0 \\
    model_zip:=$PWD/../models/residual_sf_s2/final_model.zip \\
    vecnorm_pkl:=$PWD/../models/residual_sf_s2/vec_normalize.pkl

# classical driver - no model files at all
ros2 launch hospitalnav_deploy gazebo_full_bringup.launch.py agent:=classical_sf2 cultural_ratio:=3.0
```

To run ANY model in `models/`: set `agent:=` to its base name (drop the `t1_`/`mapless_`
prefix and `_sN` suffix; `risk_field` -> `agent:=he2025`), point `model_zip`/`vecnorm_pkl`
into its folder, and set `cultural_ratio` and (for td3/ppo) `algo:=` from the table below.

## Models in this package (Tier-2 benchmark: success %, v2 shield)

| folder | launch args | SR % | note |
|---|---|---|---|
| `residual_sf_s2` | `agent:=residual_sf` | **99.4** | recommended |
| (no model) | `agent:=classical_sf2` | 97.8 | tuned social force |
| `vanilla_s3` | `agent:=vanilla cultural_ratio:=0.0` | 96.1 | culture-blind - takes the prayer-room shortcut |
| `fixed_equal_s2` | `agent:=fixed_equal` | 95.4 | pure learned CARS policy |
| `cwn_s4` | `agent:=cwn` | 94.8 | learned reward weights |
| `nav_heavy_s4` | `agent:=nav_heavy` | 95.2 | nav-weighted |
| `attention_s4` | `agent:=attention` | 94.6 | attention weighter |
| `risk_field_s0` | `agent:=he2025 cultural_ratio:=0.0` | 95.7 | risk-field baseline |
| `safe_heavy_s2` | `agent:=safe_heavy` | 94.6 | safety-weighted |
| `td3_equal_s1` | `agent:=fixed_equal algo:=td3` | 93.2 (seed) | TD3 validation |
| `ppo_equal_s0` | `agent:=fixed_equal algo:=ppo` | 0 | does not navigate (negative result) |
| `mapless_fixed_equal_s3` | `agent:=fixed_equal` + goal_publisher | 63.4 (seed) | NO planner - routing-bottleneck demo |
| `mapless_vanilla_s2` | `agent:=vanilla` + goal_publisher | 59.4 (seed) | NO planner, culture-blind |

Tier-1 models (`t1_*`): same launch args as their Tier-2 namesakes. They were trained on
the OPEN tier; the shipped Gazebo world is the walled ward, so treat Tier-1 runs as
qualitative demos (the planner replans on the live map, so they do run).

Mapless models take the straight-line goal (publish the goal without the route_planner,
e.g. `ros2 run hospitalnav_deploy goal_publisher`); giving them the planner waypoint
would not match their training inputs.

## RViz

Add `/map`, `/rl_route` (check it avoids the prayer room before the robot moves),
`/rl_goal`, `/scan`, TF. Goals: RViz "2D Goal Pose", or `room_patrol` runs the
nurse-desk -> emergency-room task automatically.

## Key parameters

| param | default | meaning |
|---|---|---|
| `agent` | - | driver mode (see table) |
| `algo` | sac | SB3 class of model_zip: sac / td3 / ppo |
| `cultural_ratio` | 3.0 | planner zone costs; **0.0 for vanilla / risk_field** (trained on the geo planner) |
| `shield_version` | 2 | 2 = swept-path (zero collisions in benchmark); 1 = forward cone |
| `residual_scale` | 0.5 | residual agents only - must match training |
| `lookahead` | 2.5 | planner waypoint distance (trained value - do not change) |

**Always keep each model with its own `vec_normalize.pkl`** - a mismatched normaliser
silently destroys performance.

## Real robot (MentorPi)

Follow `deploy/MENTORPI_DEPLOYMENT.md` in the repository: manual SLAM drive -> zone-label
PNG -> localization -> autonomous runs. The command bridge keeps the TRAINING wheelbase
(0.20 m) so commanded curvature is preserved.

## Reporting convention (important)

Report physical/Gazebo runs as per-policy binary outcomes - "reached" / "not reached"
plus the failure mode, or "reached in N of M attempts". **Never synthesize a success
rate from demonstration runs.**

## Known limits (measured in the benchmark)

* A non-yielding crowd degrades every driver at this density; the shield cannot prevent
  someone walking into the robot.
* Unmapped obstacles are handled (zero collisions with the v2 shield) at the cost of
  timeouts; replanning follows the live SLAM map.
* The zero-collision result assumes clean actuation: 10 % motor noise yields
  1.4-3.0 % contacts in simulation.
"""


if __name__ == "__main__":
    main()
