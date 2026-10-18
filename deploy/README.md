# deploy/ — everything needed to run the trained system off-benchmark

Current, maintained deployment assets (October 2026, final-campaign policies). The
ready-to-use bundle — package + best checkpoints + bring-up commands — is **`ros_sim/`
at the repository root** (built by `scripts/build_ros_sim.py`); this folder holds the
sources it is built from.

| item | what it is |
|---|---|
| [`MENTORPI_DEPLOYMENT.md`](MENTORPI_DEPLOYMENT.md) | **The deployment guide.** Architecture (route planner → local driver → swept-path shield), command translation and the wheelbase rule, SLAM mapping → zone labelling → localization → autonomous runs, reporting convention, known failure modes. |
| [`ros2_stack/hospitalnav_deploy/`](ros2_stack/hospitalnav_deploy/) | The maintained ROS 2 package: `route_planner` (cultural cost-to-go on the live map), `controller` (driver modes: learned / `residual_sf` / `classical_sf2`, built-in v2 shield), `people_tracker`, loggers, Gazebo worlds, launch files. |
| [`ros2_stack/training/`](ros2_stack/training/) | `layout.py` — the single source the Tier-2 YAML, the Gazebo world and the zone map are generated from. Edit here, regenerate everywhere. |
| `ros2_stack/setup_pi5.sh`, `requirements-pi.txt` | Raspberry Pi 5 provisioning for the MentorPi. |
| [`isaac/`](isaac/README.md) | Optional Isaac Sim testbed: physics-corrected MentorPi URDF (verified), Tier-2 USD stage (verified), observation glue (verified); `eval_isaac.py` requires an Isaac install and has open TODOs (LiDAR sensor, crowd). Secondary to the Gazebo path. |
| `mentorpi_reference/` | Hiwonder's vendor source tree (reference only, git-ignored, not ours to redistribute) — the origin of the chassis constants quoted in the guide. |

Superseded material (old Yahboom-era guide, August demo packages and sim recordings,
pre-planner model drops) lives in `archive/` with its own README; git history preserves
all previously tracked versions.
