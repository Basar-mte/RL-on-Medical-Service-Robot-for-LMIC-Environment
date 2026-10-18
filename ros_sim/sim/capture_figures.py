"""Renders the Gazebo figures for the paper.

Camera rigs are INJECTED into a world at capture time rather than baked into it,
exactly as verify_scan_invariance.py injects its lidar probes. That keeps the
worlds used for the actual runs free of extra sensors (nothing to slow the runs
down or to explain away in the methods section) while letting any tier be shot
with the identical camera set.

    python3 capture_figures.py --tier real
    python3 capture_figures.py --tier furnished --peds busy --robot 30,15,3.1416
    python3 capture_figures.py --tier walled --out ../results_figures
    python3 capture_figures.py --tier open --shots floorplan,corridor

--peds splices ped_routes' walking actors in (they are mid-stride at capture
time, --settle seconds into the loop) and --robot x,y,yaw spawns the MentorPi
model at a pose so the robot appears in the shot. The capture runs in its own
gz transport partition (figcap) with HN_SCOPE=probe, so it can run beside a
campaign without either killing the other.

Each shot is saved as a PNG plus a .json noting the world, pose and lens, so a
figure can always be regenerated exactly.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "hospitalnav_deploy")
WORLDS = os.path.join(PKG, "worlds")

# name -> (x, y, z, roll, pitch, yaw, hfov_rad, width, height)
# Poses are in the world frame of the 40 x 30 m floor; the corridor runs along
# y = 15, the three north wards are y > 17, the south rooms y < 13.
SHOTS = {
    # straight down: the whole floor plan, for the layout/route figure. Yaw
    # pi/2 puts world +x to the image's right and +y up (map convention);
    # 37 m up, the 40 x 30 m floor fits a 4:3 frame with a 1 m margin.
    "floorplan":   (20.0, 15.0, 37.0, 0.0, 1.5708, 1.5708, 1.05, 2000, 1500),
    # the corridor, from the robot's end, eye height
    "corridor":    (38.5, 15.0, 1.55, 0.0, 0.06, 3.1416, 1.20, 1920, 1080),
    # looking back up the corridor towards the nurse desk
    "nursedesk":   (33.0, 14.2, 1.45, 0.0, 0.05, 0.42, 1.15, 1920, 1080),
    # inside a ward, showing the bed bays
    # along General Ward 1's west bed row (beds, bedside cabinets, chairs)
    "ward":        (22.0, 18.4, 1.85, 0.0, 0.205, 2.33, 1.25, 1920, 1080),
    # the prayer room - the culturally restricted zone the policies route around
    # (inside the room, clear of every pedestrian lane by > 1 m; from the
    # corridor the shot would be blocked by the doorway curtain)
    "prayer":      (16.3, 12.5, 2.0, 0.0, 0.219, -2.207, 1.25, 1920, 1080),
    # the emergency room, the patrol's first destination
    "emergency":   (8.4, 12.5, 2.0, 0.0, 0.195, -2.207, 1.25, 1920, 1080),
    # an oblique establishing shot over the whole ward
    "oblique":     (2.0, 2.0, 12.0, 0.0, 0.60, 0.72, 1.30, 1920, 1080),
    # chase view: just behind and above the MentorPi (place it with --robot 33,14.9,3.1416)
    "chase":       (34.6, 14.95, 0.75, 0.0, 0.22, 3.1416, 1.30, 1920, 1080),
    # the nurse desk end of the corridor at eye height, looking east
    "nursedesk_east": (31.0, 15.2, 1.55, 0.0, 0.08, 0.0, 1.20, 1920, 1080),
    # the MentorPi itself, low and close (place it with --robot 33,14.9,3.1416)
    "robot_closeup": (31.9, 14.1, 0.42, 0.0, 0.231, 0.629, 0.55, 1920, 1080),
    # doorways from the corridor: frame, lintel, double door opened against the wall
    "door_ward": (17.9, 13.9, 1.65, 0.0, 0.159, 0.975, 1.10, 1920, 1080),
    "door_er": (7.3, 15.7, 1.65, 0.0, 0.181, -2.276, 1.10, 1920, 1080),
    # low robot's-eye view along the corridor, pedestrians ahead
    "robot_eye":   (33.6, 14.95, 0.30, 0.0, -0.02, 3.1416, 1.25, 1920, 1080),
}
CEILING = """
    <model name="ceiling_slab">
      <static>true</static>
      <pose>20.0 15.0 3.03 0 0 0</pose>
      <link name="link">
        <visual name="v">
          <cast_shadows>false</cast_shadows>
          <geometry><mesh><uri>package://hospitalnav_deploy/meshes/gen/{obj}</uri>
            <scale>40.0 30.0 0.06</scale></mesh></geometry>
        </visual>
      </link>
    </model>"""

CAM = """
    <model name="figcam_{name}">
      <static>true</static>
      <pose>{x} {y} {z} {r} {p} {yw}</pose>
      <link name="link">
        <sensor name="cam" type="camera">
          <topic>figcam_{name}</topic>
          <update_rate>5</update_rate>
          <always_on>1</always_on>
          <camera>
            <horizontal_fov>{fov}</horizontal_fov>
            <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
            <clip><near>0.02</near><far>120</far></clip>
            <anti_aliasing>8</anti_aliasing>
            <visibility_mask>1073741821</visibility_mask>
          </camera>
        </sensor>
      </link>
    </model>"""


def _reap():
    """Only this script's servers (their world path carries the figcap_ prefix):
    a campaign running alongside is left alone."""
    subprocess.run(["pkill", "-9", "-f", r"figcap_.*\.world"], capture_output=True)
    for _ in range(30):
        if subprocess.run(["pgrep", "-f", r"figcap_.*\.world"],
                          capture_output=True).returncode != 0:
            return
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="real",
                    choices=["real", "furnished", "walled", "open", "reference"])
    ap.add_argument("--peds", default="none", choices=["none", "light", "busy", "crowd45"])
    ap.add_argument("--peds-log", default="",
                    help="a peds.csv-format log (crowd_log.py, or a run's peds.csv): walkers are frozen "
                         "at their SIMULATED poses at --peds-at instead of the raw schedule")
    ap.add_argument("--peds-at", type=float, default=120.0, help="log time (sim s) of the frozen crowd")
    ap.add_argument("--robot", default="", help="x,y,yaw to place the MentorPi in the scene")
    ap.add_argument("--shots", default="", help="comma-separated subset of " + ",".join(SHOTS))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results_figures"))
    ap.add_argument("--settle", type=float, default=35.0,
                    help="seconds to let lighting/texture streaming settle before capture")
    ap.add_argument("--world", default="", help="override: path of the world file to shoot")
    ap.add_argument("--scale", type=float, default=1.0, help="resolution multiplier (2 = 4K for eye-level shots)")
    ap.add_argument("--suffix", default="", help="appended to every output file name")
    ap.add_argument("--ceiling", action="store_true",
                    help="close the suspended ceiling for eye-level shots (hides top-down views); "
                         "it casts no shadow, so the lighting is unchanged")
    a = ap.parse_args()

    names = [s.strip() for s in a.shots.split(",") if s.strip()] or list(SHOTS)
    for n in names:
        if n not in SHOTS:
            raise SystemExit(f"unknown shot '{n}'. Known: {', '.join(SHOTS)}")

    world_src = a.world or os.path.join(WORLDS, f"hospital_{a.tier}.world")
    if not os.path.exists(world_src):
        raise SystemExit(f"no such world: {world_src}")
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)

    sdf = open(world_src).read()
    if a.peds != "none":
        sys.path.insert(0, os.path.join(PKG, "hospitalnav_deploy"))
        from ped_routes import inject_actors
        if a.peds_log:
            # one instant of a simulated walk, held for the whole capture: the
            # social-force walkers keep their spacing, which the raw schedule
            # (walkers on fixed loops) does not
            frozen = os.path.join(tempfile.mkdtemp(prefix="figpeds_"), "peds_frozen.csv")
            with open(a.peds_log) as f:
                head = f.readline()
                rows = [l for l in f if l.strip()]
            best = min(rows, key=lambda l: abs(float(l.split(",", 1)[0]) - a.peds_at))
            rest = best.split(",", 1)[1]
            with open(frozen, "w") as g:
                g.write(head + f"0.0,{rest}" + f"1000000.0,{rest}")
            sdf = inject_actors(sdf, a.peds, bodies=False, playback=frozen)
        else:
            sdf = inject_actors(sdf, a.peds, bodies=False)  # stills: walkers need no physics body
    if a.ceiling:
        sys.path.insert(0, HERE)
        import gen_textures as T
        obj = T.textured_box(T.floor_for("ceiling", 40.0, 30.0), ks=0.02, ns=4.0)
        import gen_worlds as G
        sdf = sdf.replace("\n  </world>", CEILING.format(obj=obj) + G.luminaires_sdf() + "\n  </world>")
    if a.robot:
        rx, ry, ryaw = [float(v) for v in a.robot.split(",")]
        urdf = os.path.abspath(os.path.join(PKG, "urdf", "mentorpi_real.urdf"))
        sdf = sdf.replace("\n  </world>", f"""
    <include>
      <uri>file://{urdf}</uri>
      <name>mentorpi</name>
      <static>true</static>
      <pose>{rx} {ry} 0.0 0 0 {ryaw}</pose>
    </include>
  </world>""")
    rig = "".join(
        CAM.format(name=n, x=S[0], y=S[1], z=S[2], r=S[3], p=S[4], yw=S[5],
                   fov=S[6], w=int(S[7] * a.scale), h=int(S[8] * a.scale))
        for n, S in ((n, SHOTS[n]) for n in names))
    tmp = tempfile.mkdtemp(prefix="figcap_")
    world = os.path.join(tmp, f"{a.tier}_cams.world")
    with open(world, "w") as f:
        f.write(sdf.replace("\n  </world>", rig + "\n  </world>"))

    env = dict(os.environ)
    env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
        [os.path.abspath(PKG), os.path.abspath(os.path.join(PKG, "..")),
         env.get("GZ_SIM_RESOURCE_PATH", "")]).strip(os.pathsep)
    env["GZ_PARTITION"] = "figcap"
    env["HN_SCOPE"] = "probe"

    _reap()
    print(f"launching {a.tier} with {len(names)} camera(s) ...", flush=True)
    gz = subprocess.Popen(["gz", "sim", "-s", "-r", "-v1", world], env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          start_new_session=True)
    saved = []
    try:
        time.sleep(a.settle)
        for n in names:
            png = os.path.join(out_dir, f"gazebo_{a.tier}_{n}{a.suffix}.png")
            # gz's image saver writes <prefix><counter>.png into a directory
            d = os.path.join(tmp, n)
            os.makedirs(d, exist_ok=True)
            r = subprocess.run(
                ["gz", "topic", "-e", "-t", f"/figcam_{n}", "-n", "1", "--json-output"],
                capture_output=True, text=True, timeout=240, env=env)
            if not r.stdout.strip():
                print(f"  {n}: no frame received", flush=True)
                continue
            try:
                msg = json.loads(r.stdout)
            except json.JSONDecodeError:
                print(f"  {n}: could not parse frame", flush=True)
                continue
            import base64
            import numpy as np
            from PIL import Image
            w, h = int(msg["width"]), int(msg["height"])
            raw = base64.b64decode(msg["data"])
            arr = np.frombuffer(raw, np.uint8)
            if arr.size < w * h * 3:
                print(f"  {n}: short frame ({arr.size} bytes for {w}x{h})", flush=True)
                continue
            Image.fromarray(arr[:w * h * 3].reshape(h, w, 3)).save(png)
            S = SHOTS[n]
            with open(png.replace(".png", ".json"), "w") as f:
                json.dump({"world": os.path.basename(world_src), "shot": n,
                           "pose_xyz": S[:3], "pose_rpy": S[3:6],
                           "horizontal_fov_rad": S[6],
                           "resolution": [int(S[7] * a.scale), int(S[8] * a.scale)],
                           "peds": a.peds, "robot": a.robot, "ceiling": a.ceiling,
                           "peds_log": os.path.abspath(a.peds_log) if a.peds_log else None,
                           "peds_at": a.peds_at if a.peds_log else None,
                           "anti_aliasing": 8,
                           "sim_stamp_s": float(msg["header"]["stamp"].get("sec", 0))
                           + float(msg["header"]["stamp"].get("nsec", 0)) * 1e-9}, f, indent=2)
            saved.append(png)
            print(f"  saved {os.path.relpath(png, HERE)}", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(gz.pid), signal.SIGTERM)
            gz.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(gz.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        _reap()
    print(f"\n{len(saved)} figure(s) -> {out_dir}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
