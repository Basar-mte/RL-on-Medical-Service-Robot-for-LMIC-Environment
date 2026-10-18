"""Cinematic Gazebo footage of a live run.

Spawns a camera model ("vidcam") into the running world and steers it every
odometry tick with gz's own set_pose service: an establishing shot over the
nurse-desk end of the corridor that eases into an elevated chase camera behind
the MentorPi (heading and position low-pass filtered in SIMULATION time, so an
Ackermann robot's small steering corrections do not shake the shot). Every
rendered frame goes straight to H.264 through ffmpeg with its simulation stamp
in <out>.stamps.csv, which compose_video.py uses to sync the RViz inset.

    python3 record_video.py --out run.mp4 --start-sim 18 --until-file result.json

Talks to gz-sim directly (gz.transport13), so it needs no ROS bridge; the
world must be named "hospital" and the robot model "mentorpi".
"""
from __future__ import annotations
import argparse
import math
import os
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/opt/ros/jazzy/opt/gz_transport_vendor/lib/python")
sys.path.insert(0, "/opt/ros/jazzy/opt/gz_msgs_vendor/lib/python")
from gz.transport13 import Node                        # noqa: E402
from gz.msgs10.boolean_pb2 import Boolean              # noqa: E402
from gz.msgs10.entity_factory_pb2 import EntityFactory  # noqa: E402
from gz.msgs10.image_pb2 import Image as GzImage       # noqa: E402
from gz.msgs10.odometry_pb2 import Odometry            # noqa: E402
from gz.msgs10.pose_pb2 import Pose                    # noqa: E402

WORLD = "hospital"
CAM_SDF = """<sdf version="1.9"><model name="vidcam"><static>true</static>
  <pose>{x} {y} {z} 0 0.6 3.14</pose>
  <link name="link"><sensor name="cam" type="camera">
    <topic>vidcam</topic><update_rate>{fps}</update_rate><always_on>1</always_on>
    <camera><horizontal_fov>{fov}</horizontal_fov>
      <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
      <clip><near>0.05</near><far>120</far></clip>
      <anti_aliasing>4</anti_aliasing><visibility_mask>1073741821</visibility_mask></camera>
  </sensor></link></model></sdf>"""


def quat(pitch, yaw):
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return cy * cp, -sy * sp, cy * sp, sy * cp            # w, x, y, z  (rpy = 0, pitch, yaw)


def look(eye, target):
    dx, dy, dz = (target[i] - eye[i] for i in range(3))
    return math.atan2(-dz, math.hypot(dx, dy)), math.atan2(dy, dx)


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Rig:
    BACK, HEIGHT, AHEAD = 3.4, 3.4, 2.0
    TAU_YAW, TAU_POS = 1.2, 0.35
    WIDE_EYE, WIDE_AT = (44.0, 6.0, 10.5), (33.0, 15.0, 0.0)
    INTRO = 5.0

    def __init__(self, start_sim):
        self.start = start_sim
        self.yaw = None
        self.eye = None
        self.t = None

    def update(self, t, x, y, yaw):
        if self.yaw is None:
            self.yaw, self.t = yaw, t
        dt = max(0.0, t - self.t)
        self.t = t
        d = (yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += d * (1 - math.exp(-dt / self.TAU_YAW))
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        chase_eye = (x - self.BACK * c, y - self.BACK * s, self.HEIGHT)
        chase_at = (x + self.AHEAD * c, y + self.AHEAD * s, 0.2)
        if self.eye is None:
            self.eye = chase_eye
        k = 1 - math.exp(-dt / self.TAU_POS)
        self.eye = tuple(self.eye[i] + (chase_eye[i] - self.eye[i]) * k for i in range(3))
        u = smoothstep((t - self.start) / self.INTRO)
        eye = tuple(self.WIDE_EYE[i] + (self.eye[i] - self.WIDE_EYE[i]) * u for i in range(3))
        at = tuple(self.WIDE_AT[i] + (chase_at[i] - self.WIDE_AT[i]) * u for i in range(3))
        return eye, look(eye, at)


def dolly(a):
    """Camera steering in its OWN process: gz transport delivers all of a
    process's subscriptions on one thread, and 6 MB camera frames queued ahead
    of the odometry made the camera trail the robot by up to 25 s (measured)."""
    node = Node()
    lock = threading.Lock()
    latest = {"odom": None}
    rig = Rig(a.start_sim)

    def on_odom(m: Odometry):
        t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        q = m.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with lock:
            latest["odom"] = (t, m.pose.position.x, m.pose.position.y, yaw)
    node.subscribe(Odometry, "/odom", on_odom)
    parent, last_t = os.getppid(), -1.0
    while os.getppid() == parent:
        with lock:
            od = latest["odom"]
        if od is not None and od[0] > last_t:
            last_t = od[0]
            eye, (pitch, cyaw) = rig.update(*od)
            p = Pose()
            p.name = "vidcam"
            p.position.x, p.position.y, p.position.z = eye
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = quat(pitch, cyaw)
            node.request(f"/world/{WORLD}/set_pose", p, Pose, Boolean, 300)
        else:
            time.sleep(0.005)
    return 0


def finish_sim(path):
    """Absolute sim time at which the run finished, from its result.json
    (sim_time_s is measured from the sequencer's first tick, ~0.2 s)."""
    import json
    for _ in range(20):
        try:
            return float(json.load(open(path))["sim_time_s"]) + 0.3
        except Exception:
            time.sleep(0.5)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fov", type=float, default=1.15)
    ap.add_argument("--start-sim", type=float, default=18.0, help="sim time at which recording starts")
    ap.add_argument("--until-file", default="", help="stop 2.5 s (sim) after the run this result file records")
    ap.add_argument("--max-sim", type=float, default=600.0)
    ap.add_argument("--crf", type=int, default=13)
    ap.add_argument("--dolly", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.dolly:
        return dolly(a)
    w, h = (int(v) for v in a.size.split("x"))

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    enc = subprocess.Popen([ff, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-threads", "4",
                            "-preset", "veryfast", "-crf", str(a.crf), "-pix_fmt", "yuv420p", a.out],
                           stdin=subprocess.PIPE)
    stamps = open(a.out + ".stamps.csv", "w")
    stamps.write("frame,sim\n")

    node = Node()
    req = EntityFactory()
    req.sdf = CAM_SDF.format(x=44, y=6, z=10.5, fps=a.fps, fov=a.fov, w=w, h=h)
    ok, rep = node.request(f"/world/{WORLD}/create", req, EntityFactory, Boolean, 5000)
    print("camera spawned" if ok and rep.data else "camera spawn FAILED", flush=True)
    steer = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--dolly",
                              "--start-sim", str(a.start_sim)])

    state = {"frames": 0, "dropped": 0, "last": 0.0, "done": False}
    frames = queue.Queue(maxsize=240)

    def writer():
        while True:
            item = frames.get()
            if item is None:
                return
            t, data = item
            enc.stdin.write(data)
            stamps.write(f"{state['frames']},{t:.3f}\n")
            state["frames"] += 1
    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    def on_image(m: GzImage):
        t = m.header.stamp.sec + m.header.stamp.nsec * 1e-9
        state["last"] = t
        if state["done"] or t < a.start_sim or len(m.data) < w * h * 3:
            return
        try:                                  # never block the transport thread
            frames.put_nowait((t, bytes(m.data[:w * h * 3])))
        except queue.Full:
            state["dropped"] += 1
    node.subscribe(GzImage, "/vidcam", on_image)

    t0, last_print, stop_at, seen_file = time.time(), 0.0, None, None
    while True:
        time.sleep(0.2)
        if a.until_file and stop_at is None and os.path.exists(a.until_file):
            fs = finish_sim(a.until_file)
            stop_at = (fs + 2.5) if fs is not None else state["last"] + 2.5
            seen_file = time.time()
        if stop_at is not None and (state["last"] >= stop_at or time.time() - seen_file > 120):
            break
        if state["last"] >= a.max_sim or time.time() - t0 > 5400:
            break
        if time.time() - last_print > 20:
            last_print = time.time()
            print(f"  {state['frames']} frames ({state['dropped']} dropped), camera sim {state['last']:.1f} s", flush=True)
    state["done"] = True
    frames.put(None)
    wt.join(timeout=300)
    stamps.close()
    enc.stdin.close()
    enc.wait()
    steer.terminate()
    print(f"wrote {a.out}: {state['frames']} frames, {state['dropped']} dropped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
