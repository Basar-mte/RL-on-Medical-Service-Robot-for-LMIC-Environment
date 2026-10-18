"""Re-renders a recorded run as cinematic Gazebo footage, offline.

    python3 replay_video.py --run-dir RUN --tier walled --peds busy --out gazebo.mp4

WHY REPLAY
  A camera steered live can only react: in the corridor it works, but when the
  robot turns into a doorway a camera that trails it ends up behind a wall
  (measured: blank-wall frames, and the final shot facing the corridor while the
  robot was inside the emergency room). A replay knows the whole trajectory, so
  the camera can anticipate every turn and be placed where it can actually SEE.

WHAT IS REPLAYED
  The world of the run (same tier) with the walkers played back from the run's
  own peds.csv (every walker's simulated pose and walk-cycle time, logged by
  PedestrianBodies - they yield to the robot, so the schedule alone would not
  reproduce them) and a visual-only MentorPi posed from run.csv (10 Hz,
  interpolated). The robot and the camera are posed INSIDE the simulator by
  the ReplayRig system every physics step, so each rendered frame shows
  exactly the poses of its sim time; the world free-runs as fast as the camera
  renders. Frame stamps are in the run's sim time, so compose_video.py pairs
  the footage with the run's live RViz stream unchanged. Nothing about the
  run's outcome is re-simulated - this is purely a camera.

THE CAMERA
  For every frame the planner takes the robot's motion heading over the next
  ~2 s, tries viewpoints behind the robot (then rotated away from behind, then
  higher) and keeps the first from which the robot is in clear line of sight
  past every wall and door lintel of the world AND every walker at that
  instant (from peds.csv); the chosen track is then low-pass filtered. It opens
  on an establishing shot that eases into the follow. A faint red ring on the
  floor marks the robot (a visual-only replay prop: in a crowd the small black
  robot is otherwise hard to pick out).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PKG = os.path.join(ROOT, "hospitalnav_deploy")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "hospitalnav_deploy"))
sys.path.insert(0, "/opt/ros/jazzy/opt/gz_transport_vendor/lib/python")
sys.path.insert(0, "/opt/ros/jazzy/opt/gz_msgs_vendor/lib/python")
import numpy as np                                     # noqa: E402
from gz.transport13 import Node                        # noqa: E402
from gz.msgs10.boolean_pb2 import Boolean              # noqa: E402
from gz.msgs10.clock_pb2 import Clock                  # noqa: E402
from gz.msgs10.image_pb2 import Image as GzImage       # noqa: E402
from gz.msgs10.pose_pb2 import Pose                    # noqa: E402
from gz.msgs10.world_control_pb2 import WorldControl   # noqa: E402
from record_video import CAM_SDF, quat, look, smoothstep   # noqa: E402

WORLD = "hospital"
FPS = 30


# ------------------------------------------------------------- the world --
def occluders(tier):
    """3-D boxes (x0, x1, y0, y1, z0, z1) a camera must see past."""
    import gen_worlds as G
    import layout as L
    tr = G.load_trained()
    boxes = []
    for p in tr["wall"]:
        if tier == "open" and int(p.name.rsplit("_", 1)[1]) >= 4:
            continue
        cx, cy, _ = p.pose[:3]
        sx, sy, _ = p.box
        boxes.append((cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, 0.0, G.WALL_VIS_H))
    if tier != "open":
        ht = L.WALL_T / 2
        for (name, side, w, lo, hi) in G.door_gaps():           # the lintels over the real openings
            if side in ("N", "S"):
                boxes.append((lo, hi, w - ht, w + ht, G.DOOR_TOP, G.WALL_VIS_H))
            else:
                boxes.append((w - ht, w + ht, lo, hi, G.DOOR_TOP, G.WALL_VIS_H))
            # and the room sign over each door (either face): a camera sliding
            # past one fills a third of the frame with the back of a sign
            # (generous above it, so the camera does not skim over a sign either)
            c, half, sd = (lo + hi) / 2, 1.1, 0.9
            if side in ("N", "S"):
                boxes.append((c - half, c + half, w - sd, w + sd, G.SIGN_Z - 0.25, G.SIGN_Z + 1.0))
            else:
                boxes.append((w - sd, w + sd, c - half, c + half, G.SIGN_Z - 0.25, G.SIGN_Z + 1.0))
    return np.array(boxes)


def blocked(B, p0, p1):
    """Does the segment p0 -> p1 pass through any box (slab test, vectorised)?"""
    p0 = np.asarray(p0, float); d = np.asarray(p1, float) - p0
    tmin = np.zeros(len(B)); tmax = np.ones(len(B))
    for k, (lo, hi) in enumerate(((0, 1), (2, 3), (4, 5))):
        if abs(d[k]) < 1e-9:
            out = (p0[k] < B[:, lo]) | (p0[k] > B[:, hi])
            tmin = np.where(out, 2.0, tmin)
            continue
        t1 = (B[:, lo] - p0[k]) / d[k]; t2 = (B[:, hi] - p0[k]) / d[k]
        tmin = np.maximum(tmin, np.minimum(t1, t2)); tmax = np.minimum(tmax, np.maximum(t1, t2))
    return bool(np.any(tmin <= tmax))


def inside(B, p, margin=0.25):
    x, y, z = p
    return bool(np.any((B[:, 0] - margin <= x) & (x <= B[:, 1] + margin) & (B[:, 2] - margin <= y)
                       & (y <= B[:, 3] + margin) & (z <= B[:, 5] + 0.15)))


# ------------------------------------------------------- the camera track --
def trajectory(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    T = np.array([float(r["sim_abs"]) for r in rows])
    X = np.array([[float(r["x"]), float(r["y"]), float(r["yaw"])] for r in rows])
    return T, X


def pose_at(T, X, t):
    i = int(np.clip(np.searchsorted(T, t) - 1, 0, len(T) - 2))
    f = float(np.clip((t - T[i]) / max(T[i + 1] - T[i], 1e-6), 0, 1))
    x, y = X[i, :2] + (X[i + 1, :2] - X[i, :2]) * f
    dy = (X[i + 1, 2] - X[i, 2] + math.pi) % (2 * math.pi) - math.pi
    return x, y, X[i, 2] + dy * f


AZ = [0, 25, -25, 50, -50, 75, -75, 105, -105, 140, -140, 180]
RINGS = [(3.2, 3.5), (2.6, 4.6), (2.0, 6.0), (1.3, 7.5), (0.6, 9.0)]   # (back distance, height)


PERSON_R, PERSON_H = 0.30, 1.75


def people_block(peds, eye, target):
    """Does the sight line eye -> target pass through a walker (an upright
    cylinder)? peds: (N, 2) walker positions at that instant, or None."""
    if peds is None or not len(peds):
        return False
    e, tg = np.asarray(eye, float), np.asarray(target, float)
    u = np.linspace(0.08, 0.97, 14)[:, None]
    pts = e + (tg - e) * u
    low = pts[:, 2] < PERSON_H
    if not low.any():
        return False
    pts = pts[low]
    d = np.hypot(pts[:, None, 0] - peds[None, :, 0], pts[:, None, 1] - peds[None, :, 1])
    return bool((d < PERSON_R).any())


def plan(T, X, B, t0, t1, wide_eye, wide_at, intro=5.0, crowd=None):
    """crowd: callable t -> (N, 2) walker positions, so the camera also looks
    past people (in a dense crowd a follow camera at head height sees only
    backs); None ignores them."""
    times = np.arange(t0, t1, 1.0 / FPS)
    eyes, ats = [], []
    heading = None
    for t in times:
        peds = crowd(t) if crowd else None
        x, y, yaw = pose_at(T, X, t)
        xa, ya, _ = pose_at(T, X, t + 2.0)
        xb, yb, _ = pose_at(T, X, t - 0.5)
        if math.hypot(xa - xb, ya - yb) > 0.4:
            h = math.atan2(ya - yb, xa - xb)
        else:
            h = heading if heading is not None else yaw
        heading = h
        target = (x, y, 0.20)
        choice = None
        for back, z in RINGS:
            for a in AZ:
                ang = h + math.pi + math.radians(a)
                eye = (x + back * math.cos(ang), y + back * math.sin(ang), z)
                if inside(B, eye) or blocked(B, eye, target) or blocked(B, eye, (x, y, 0.6)):
                    continue
                if people_block(peds, eye, target):
                    continue
                choice = eye
                break
            if choice:
                break
        if choice is None:
            choice = (x, y - 0.5, 10.0)
        eyes.append(choice)
        ats.append((x + 1.2 * math.cos(h), y + 1.2 * math.sin(h), 0.2))
    eyes, ats = np.array(eyes), np.array(ats)
    # low-pass the track forward and backward (zero lag), position tau 0.6 s
    k = 1 - math.exp(-1.0 / (FPS * 0.6))
    for arr in (eyes, ats):
        for rng in (range(1, len(arr)), range(len(arr) - 2, -1, -1)):
            for i in rng:
                j = i - 1 if rng.step == 1 else i + 1
                arr[i] = arr[j] + (arr[i] - arr[j]) * k
    # lift pass: wherever smoothing (or a viewpoint switch) left the robot hidden,
    # raise the camera until it is visible, spreading the lift over the
    # surrounding second so the camera climbs before the wall, not through it
    need = eyes[:, 2].copy()
    for i, t in enumerate(times):
        x, y, _ = pose_at(T, X, t)
        z = eyes[i, 2]
        peds = crowd(t) if crowd else None
        while z < 11.0 and (blocked(B, (eyes[i, 0], eyes[i, 1], z), (x, y, 0.2)) or inside(B, (eyes[i, 0], eyes[i, 1], z))
                            or people_block(peds, (eyes[i, 0], eyes[i, 1], z), (x, y, 0.2))):
            z += 0.25
        need[i] = z
    win = FPS // 2
    lifted = np.array([need[max(0, i - win):i + win + 1].max() for i in range(len(need))])
    for rng in (range(1, len(lifted)), range(len(lifted) - 2, -1, -1)):
        kz = 1 - math.exp(-1.0 / (FPS * 0.35))
        for i in rng:
            j = i - 1 if rng.step == 1 else i + 1
            lifted[i] = max(need[i], lifted[j] + (lifted[i] - lifted[j]) * kz)
    eyes[:, 2] = lifted
    # establishing shot easing into the follow
    for i, t in enumerate(times):
        u = smoothstep((t - t0) / intro)
        eyes[i] = np.array(wide_eye) + (eyes[i] - np.array(wide_eye)) * u
        ats[i] = np.array(wide_at) + (ats[i] - np.array(wide_at)) * u
    return times, eyes, ats


# ------------------------------------------------------------------ main --
def write_track(path, rows):
    with open(path, "w") as f:
        f.write("t,x,y,z,qw,qx,qy,qz\n")
        for r in rows:
            f.write(",".join(f"{v:.5f}" for v in r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tier", default="walled")
    ap.add_argument("--peds", default="busy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-sim", type=float, default=18.0)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fov", type=float, default=1.10)
    ap.add_argument("--crf", type=int, default=13)
    ap.add_argument("--rtf", type=float, default=0.0,
                    help="world real-time factor while filming (0 = as fast as the camera renders)")
    ap.add_argument("--dur", type=float, default=0.0, help="film only this many seconds (tests)")
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    T, X = trajectory(os.path.join(a.run_dir, "run.csv"))
    res = json.load(open(os.path.join(a.run_dir, "result.json")))
    t_end = min(T[-1], float(res["sim_time_s"]) + 0.3 + 2.5)
    if a.dur > 0:
        t_end = min(t_end, a.start_sim + a.dur)
    B = occluders(a.tier)
    import ped_routes as P
    peds_csv = os.path.join(a.run_dir, "peds.csv")
    crowd = None
    if os.path.exists(peds_csv) and a.peds != "none":
        pts, pd = P.load_log(peds_csv)
        PX = np.stack([pd[n][0] for n in pd], 1); PY = np.stack([pd[n][1] for n in pd], 1)

        def crowd(t):
            i = int(np.clip(np.searchsorted(pts, t), 0, len(pts) - 1))
            return np.stack([PX[i], PY[i]], 1)
    times, eyes, ats = plan(T, X, B, a.start_sim, t_end, (44.0, 6.0, 10.5), (33.0, 15.0, 0.0), crowd=crowd)
    print(f"camera track planned: {len(times)} frames, sim {times[0]:.1f}-{times[-1]:.1f} s", flush=True)

    tmp = tempfile.mkdtemp(prefix="replay_")
    # Everything that moves is played back INSIDE the simulator, from tracks:
    # the robot and the camera by ReplayRig, the walkers by PedestrianBodies'
    # playback of the run's own peds.csv. The replay world's clock is shifted
    # so filming starts at sim 1.0 s instead of rendering the run's first
    # start_sim seconds for nothing; frame stamps are written back in the
    # RUN's sim time, which is what compose_video.py pairs with RViz.
    shift = 0.0
    if os.path.exists(peds_csv) and a.peds != "none":
        shift = times[0] - 1.0
        with open(peds_csv) as f, open(os.path.join(tmp, "peds.csv"), "w") as g:
            g.write(f.readline())
            for line in f:
                t, rest = line.split(",", 1)
                g.write(f"{float(t) - shift:.3f},{rest}")
        ped_opts = dict(bodies=False, playback=os.path.join(tmp, "peds.csv"))
        print(f"walkers: playback of the run's peds.csv (replay clock shifted by {shift:.2f} s)", flush=True)
    else:
        ped_opts = dict(bodies=False)
        print("walkers: scripted schedule (run has no peds.csv)", flush=True)
    rob = []
    for t, (x, y, yaw) in zip(T, X):
        rob.append((t - shift, x, y, 0.0, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    write_track(os.path.join(tmp, "robot.csv"), rob)
    write_track(os.path.join(tmp, "halo.csv"), [(r[0], r[1], r[2], 0.006, 1.0, 0.0, 0.0, 0.0) for r in rob])
    cam_rows = []
    for tt, e, at in zip(times, eyes, ats):
        pitch, cyaw = look(tuple(e), tuple(at))
        cam_rows.append((tt - shift, *e, *quat(pitch, cyaw)))
    write_track(os.path.join(tmp, "camera.csv"), cam_rows)

    sdf = open(os.path.join(PKG, "worlds", f"hospital_{a.tier}.world")).read()
    sdf = P.inject_actors(sdf, a.peds, **ped_opts)
    # physics step 1/300 s: the 30 Hz camera then renders at exact frame times
    sdf = sdf.replace("<max_step_size>0.01</max_step_size>", f"<max_step_size>{1.0 / (10 * FPS):.10f}</max_step_size>")
    sdf = sdf.replace("<real_time_factor>0</real_time_factor>", f"<real_time_factor>{a.rtf}</real_time_factor>")
    x0, y0, yaw0 = pose_at(T, X, times[0])
    # a visual-only MentorPi: no collisions, sensors or plugins (posing a model
    # that carries mesh collisions makes physics rebuild them every step), and
    # NOT static (moving a static multi-mesh model makes ogre2 rebuild its
    # static batches, ~5 s a frame measured); nothing pushes it, gravity is off
    import re
    u = open(os.path.join(PKG, "urdf", "mentorpi_real.urdf")).read()
    u = re.sub(r"<collision>.*?</collision>", "", u, flags=re.S)
    u = re.sub(r'<gazebo( reference="[^"]*")?>.*?</gazebo>', "", u, flags=re.S)
    # gravity off on EVERY link (wheels and steering knuckles included): a link
    # left under gravity pulls on the posed model through its joint, the speed
    # builds up step after step, and ~29 s in the solver diverges and the robot
    # stops rendering (measured 2026-10-17 in both demo videos)
    links = sorted(set(re.findall(r'<link name="([^"]+)"', u)))
    u = u.replace("</robot>", "".join(f'  <gazebo reference="{ln}"><gravity>false</gravity></gazebo>\n'
                                      for ln in links) + "</robot>")
    urdf = os.path.join(tmp, "mentorpi_visual.urdf")
    open(urdf, "w").write(u)
    cam = CAM_SDF.format(x=eyes[0][0], y=eyes[0][1], z=eyes[0][2], fps=FPS, fov=a.fov, w=w, h=h)
    cam = cam[cam.index("<model"):cam.rindex("</model>") + len("</model>")]
    sdf = sdf.replace("\n  </world>", f"""
    <include><uri>file://{urdf}</uri><name>mentorpi</name>
      <pose>{x0} {y0} 0 0 0 {yaw0}</pose></include>
    {cam}
    <model name="robot_halo"><pose>{x0} {y0} 0.006 0 0 0</pose>
      <link name="l"><gravity>false</gravity>
        <inertial><mass>0.1</mass><inertia><ixx>1e-3</ixx><iyy>1e-3</iyy><izz>1e-3</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <visual name="ring"><cast_shadows>false</cast_shadows><transparency>0.5</transparency>
          <geometry><cylinder><radius>0.34</radius><length>0.004</length></cylinder></geometry>
          <material><ambient>0.86 0.08 0.10 1</ambient><diffuse>0.86 0.08 0.10 1</diffuse><emissive>0.45 0.02 0.03 1</emissive></material>
        </visual></link></model>
    <plugin filename="hospitalnav_replay_rig" name="hospitalnav::ReplayRig">
      <track><model>mentorpi</model><csv>{tmp}/robot.csv</csv></track>
      <track><model>robot_halo</model><csv>{tmp}/halo.csv</csv></track>
      <track><model>vidcam</model><csv>{tmp}/camera.csv</csv></track>
    </plugin>
  </world>""")
    world = os.path.join(tmp, "replay.world")
    open(world, "w").write(sdf)
    # a partition of its own: a second replay (or a leftover server) on the same
    # partition would interleave ITS camera frames into this recording
    part = f"replay_{os.getpid()}"
    env = dict(os.environ, GZ_SIM_RESOURCE_PATH=os.pathsep.join([os.path.abspath(PKG), os.path.abspath(ROOT)]),
               GZ_PARTITION=part, HN_SCOPE="probe")
    os.environ.update(GZ_PARTITION=part)
    # start PAUSED until the renderer is up, or physics runs ahead while the
    # rendering sensors load and the opening is never filmed
    gz = subprocess.Popen(["gz", "sim", "-s", "-v2", world], env=env, stdout=open(os.path.join(tmp, "gz.log"), "w"),
                          stderr=subprocess.STDOUT, start_new_session=True)

    import imageio_ffmpeg
    enc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                            "-threads", "4", "-preset", "veryfast", "-crf", str(a.crf), "-pix_fmt", "yuv420p", a.out],
                           stdin=subprocess.PIPE)
    stamps = open(a.out + ".stamps.csv", "w"); stamps.write("frame,sim\n")
    n_frames = len(times)
    frames = queue.Queue()
    state = {"written": 0, "clock": 0.0, "last_t": -1.0}
    lock = threading.Lock()

    def writer():
        """Writes every frame in arrival (= sim time) order with its TRUE stamp.
        The camera fires on the physics-step grid, so a frame's sim time can be
        a step or two off k/30 s; the rig posed robot, camera and walkers at
        that very instant, so the frame is exact for its stamp, and
        compose_video.py resamples the footage by these stamps."""
        try:
            while True:
                item = frames.get()
                if item is None:
                    return
                enc.stdin.write(item[1]); stamps.write(f"{state['written']},{item[0]:.4f}\n")
                state["written"] += 1
        except Exception as e:                       # never die silently
            import traceback
            traceback.print_exc()
            state["error"] = repr(e)
    wt = threading.Thread(target=writer, daemon=True); wt.start()

    node = Node()
    seen = []

    def on_image(m):
        t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9 + shift
        if os.environ.get("REPLAY_DEBUG"):
            seen.append((time.time(), t))
        if t < times[0] - 0.5 / FPS or t > times[-1] + 0.5 / FPS or len(m.data) < w * h * 3:
            return
        with lock:
            if t <= state["last_t"]:
                return
            state["last_t"] = t
        frames.put((t, bytes(m.data[:w * h * 3])))
    node.subscribe(GzImage, "/vidcam", on_image)

    cnode = Node()

    def on_clock(m):
        with lock:
            state["clock"] = m.sim.sec + m.sim.nsec * 1e-9
    cnode.subscribe(Clock, f"/world/{WORLD}/clock", on_clock)

    time.sleep(12.0)
    deadline = time.time() + 60
    while time.time() < deadline:
        ctl = WorldControl(); ctl.pause = False
        cnode.request(f"/world/{WORLD}/control", ctl, WorldControl, Boolean, 25)
        time.sleep(2.0)
        with lock:
            if state["clock"] > 0.0:
                break
    print("world running; filming", flush=True)
    t0 = time.time(); last_print = 0.0
    try:
        while gz.poll() is None and "error" not in state:
            time.sleep(0.5)
            with lock:
                past = state["clock"] + shift > times[-1] + 3.0
            if past:
                print("clock passed the last frame; stopping", flush=True)
                break
            if time.time() - last_print > 20:
                last_print = time.time()
                with lock:
                    clk = state["clock"] + shift
                print(f"  frame {state['written']}/~{n_frames}, sim {clk:.1f} s, "
                      f"{state['written'] / max(time.time() - t0, 1e-6):.1f} fps", flush=True)
    finally:
        frames.put(None); wt.join(timeout=300); stamps.close(); enc.stdin.close(); enc.wait()
        for sig in (15, 9):
            try:
                os.killpg(os.getpgid(gz.pid), sig)
            except ProcessLookupError:
                break
            time.sleep(2.0)
    st = [float(l.split(",")[1]) for l in open(a.out + ".stamps.csv").read().split()[1:]]
    gaps = np.diff(st) * FPS if len(st) > 1 else np.array([1.0])
    print(f"wrote {a.out}: {state['written']} frames for {n_frames} frame times; "
          f"largest gap {gaps.max():.2f} frames", flush=True)
    if os.environ.get("REPLAY_DEBUG"):
        with open(a.out + ".arrivals.csv", "w") as f:
            f.write("wall,sim\n" + "".join(f"{x:.3f},{y:.4f}\n" for x, y in seen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
