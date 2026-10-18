"""One demonstration run with figure/video capture attached.

    python3 figure_run.py --preset fixed_equal --tier walled --peds busy --out-dir DIR \\
        --video --rviz-stream --still paper_topdown.rviz:done:rviz_topdown.png

Launches gz_bringup exactly as a campaign run (same planner, policy, shield,
sequencer and budgets), in its own ROS domain / gz partition / cleanup scope,
and attaches:
  --video          record_video.py: cinematic Gazebo footage   -> DIR/gazebo.mp4
  --rviz-stream    capture_rviz.py stream of the live-SLAM view -> DIR/rviz_stream/
                   (turns on viz_slam: slam_toolbox maps onto /slam_map for display;
                   navigation still uses the prebuilt map)
  --still CFG:WHEN:NAME   a 4K RViz still (WHEN = moving | done | simtime:<s>)
The run's own record (result.json, run.csv) is written to DIR as well: these
are demonstration runs, reported - if at all - as single outcomes.
"""
from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCOPE, DOMAIN, PARTITION = "figrun", 43, "figrun"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fixed_equal")
    ap.add_argument("--tier", default="walled")
    ap.add_argument("--peds", default="busy")
    ap.add_argument("--escape", default="true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--rviz-stream", action="store_true")
    ap.add_argument("--still", action="append", default=[])
    ap.add_argument("--rtf", default="")
    ap.add_argument("--start-delay", type=float, default=20.0)
    a = ap.parse_args()
    out = os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    res = os.path.join(out, "result.json")
    if os.path.exists(res):
        os.remove(res)

    env = dict(os.environ, RMW_IMPLEMENTATION="rmw_cyclonedds_cpp", ROS_DOMAIN_ID=str(DOMAIN),
               HN_SCOPE=SCOPE, GZ_PARTITION=PARTITION)
    clean = lambda: subprocess.run([os.path.join(HERE, "clean_sim.sh"), SCOPE], capture_output=True)
    clean()
    video = a.video or a.rviz_stream
    rtf = a.rtf or ("0.3" if video else "0")       # 0.3x: headroom for 1080p30 rendering, encoding and RViz grabs
    cmd = ["ros2", "launch", "hospitalnav_deploy", "gz_bringup.launch.py",
           f"preset:={a.preset}", f"tier:={a.tier}", f"peds:={a.peds}", f"escape_enabled:={a.escape}",
           "gui:=false", "rviz:=false", "route:=direct", "goal_timeout:=300.0",
           f"patrol_start_delay:={a.start_delay}", f"rtf:={rtf}",
           f"viz_slam:={'true' if a.rviz_stream else 'false'}", "label_size:=0.9",
           f"csv_path:={out}/run.csv", f"result_path:={res}", f"run_label:=figrun_{a.tier}_{a.preset}"]
    log = open(os.path.join(out, "run.log"), "w")
    launch = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    helpers = []
    try:
        time.sleep(8.0)
        if a.video:
            helpers.append(subprocess.Popen(
                [sys.executable, "-u", os.path.join(HERE, "record_video.py"), "--out", os.path.join(out, "gazebo.mp4"),
                 "--start-sim", str(a.start_delay - 2.0), "--until-file", res],
                env=env, stdout=open(os.path.join(out, "video.log"), "w"), stderr=subprocess.STDOUT))
        if a.rviz_stream:
            helpers.append(subprocess.Popen(
                [sys.executable, "-u", os.path.join(HERE, "capture_rviz.py"), "--config", "video_mapping.rviz",
                 "--screen", "1200x930", "--domain", str(DOMAIN), "--when", "now",
                 "--stream", os.path.join(out, "rviz_stream"), "--fps", "10", "--until-file", res],
                env=env, stdout=open(os.path.join(out, "rviz_stream.log"), "w"), stderr=subprocess.STDOUT))
        stills = []
        for spec in a.still:
            cfg, when, name = spec.split(":", 2) if not spec.count(":") > 2 else (
                spec.split(":")[0], ":".join(spec.split(":")[1:3]), spec.split(":")[3])
            stills.append(subprocess.Popen(
                [sys.executable, "-u", os.path.join(HERE, "capture_rviz.py"), "--config", cfg,
                 "--domain", str(DOMAIN), "--when", when, "--late-start", "--settle", "15",
                 "--out", os.path.join(out, name)],
                env=env, stdout=open(os.path.join(out, name + ".log"), "w"), stderr=subprocess.STDOUT))
        helpers += stills
        t0 = time.time()
        while not os.path.exists(res) and time.time() - t0 < 1500 and launch.poll() is None:
            time.sleep(2.0)
        print("run finished" if os.path.exists(res) else "run produced no result", flush=True)
        for p in helpers:                               # captures finish on their own
            try:
                p.wait(timeout=240)
            except subprocess.TimeoutExpired:
                p.terminate()
    finally:
        try:
            os.killpg(os.getpgid(launch.pid), signal.SIGTERM)
            launch.wait(timeout=20)
        except Exception:
            pass
        clean()
    if os.path.exists(res):
        import json
        r = json.load(open(res))
        print(f"{a.tier}/{a.preset}/peds-{a.peds}: {r['outcome']}, {r['goals'][-1]['seconds']} s, "
              f"path {r['path_length_m']} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
