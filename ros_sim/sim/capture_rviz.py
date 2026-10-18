"""RViz captures of a live run at publication resolution.

RViz runs inside its own Xephyr X server (default 3840 x 2160, hardware GL via
glamor), so a capture is not limited to the 1920 x 1080 laptop screen and the
user's desktop is untouched except for the Xephyr window. The configs written
by make_rviz_configs.py have no panels and a sentinel background (254), so the
grab is cropped exactly to the 3D view and the background turned pure white.

  still   python3 capture_rviz.py --config paper_topdown.rviz --when done --out fig.png
  stream  python3 capture_rviz.py --config video_mapping.rviz --screen 1200x930 \\
              --stream DIR --fps 10 --until-file result.json

--when: now | moving (the patrol has a goal) | done (the patrol finished) |
        simtime:<s> (simulation clock reached s seconds). Stream frames are
        JPEGs with their simulation time in DIR/stamps.csv, which is how the
        video compositor keeps the inset in sync with the Gazebo footage.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "hospitalnav_deploy")
sys.path.insert(0, HERE)
from xwd2png import xwd_to_image   # noqa: E402

SENTINEL = 254


def clean_env(domain, display=None):
    env = dict(os.environ)
    for k in list(env):
        if "SNAP" in k or k in ("GTK_PATH", "GTK_EXE_PREFIX", "GIO_MODULE_DIR", "GDK_PIXBUF_MODULE_FILE",
                                "GSETTINGS_SCHEMA_DIR", "LOCPATH", "GDK_BACKEND", "XDG_DATA_HOME"):
            env.pop(k, None)
    for k in ("LD_LIBRARY_PATH", "PATH", "PYTHONPATH"):
        if k in env:
            env[k] = ":".join(p for p in env[k].split(":") if "/snap/" not in p)
    env.update(XDG_DATA_DIRS="/usr/local/share:/usr/share", XDG_CONFIG_DIRS="/etc/xdg",
               QT_QPA_PLATFORM="xcb", ROS_DOMAIN_ID=str(domain),
               RMW_IMPLEMENTATION="rmw_cyclonedds_cpp", HN_SCOPE="probe")
    if display:
        env["DISPLAY"] = display
    return env


def free_display(start=20):
    n = start
    while os.path.exists(f"/tmp/.X11-unix/X{n}") or os.path.exists(f"/tmp/.X{n}-lock"):
        n += 1
    return f":{n}"


def render_area(display, env, screen_wh):
    """(x0, y0, x1, y1) of RViz's 3D view: Ogre renders into a native child
    window, so its exact geometry is in the X window tree. The largest window
    smaller than the screen is the render panel."""
    import re
    out = subprocess.run(["xwininfo", "-display", display, "-root", "-tree"], capture_output=True,
                         text=True, env=env).stdout
    best = None
    for w, h, x, y in re.findall(r"\s(\d+)x(\d+)\+-?\d+\+-?\d+\s+\+(-?\d+)\+(-?\d+)", out):
        w, h, x, y = int(w), int(h), int(x), int(y)
        if w * h >= screen_wh[0] * screen_wh[1] * 0.95 or w < 200 or h < 200:
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (x, y, w, h)
    if best is None:
        return None
    x, y, w, h = best
    return (max(0, x), max(0, y), min(screen_wh[0], x + w), min(screen_wh[1], y + h))


_XDISP = {}


def _xlib_grab(display):
    """Root-window grab over the X protocol directly: ~13 ms for 1200 x 930,
    and no process spawn per frame under heavy CPU load."""
    from Xlib import display as xdisplay, X
    if display not in _XDISP:
        _XDISP[display] = xdisplay.Display(display)
    d = _XDISP[display]
    root = d.screen().root
    g = root.get_geometry()
    raw = root.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xffffffff)
    a = np.frombuffer(raw.data, np.uint8).reshape(g.height, g.width, 4)
    return a[..., [2, 1, 0]].copy()          # BGRX -> RGB


def grab(display, env, crop=None):
    try:
        a = _xlib_grab(display)
        return _crop(a, crop)
    except Exception:
        pass
    raw = b""
    for _ in range(5):                       # a grab can race a resize/expose: retry
        raw = subprocess.run(["xwd", "-display", display, "-root", "-silent"], capture_output=True, env=env).stdout
        if len(raw) > 100:
            break
        time.sleep(0.5)
    tmp = f"/tmp/.hn_grab_{os.getpid()}.xwd"
    with open(tmp, "wb") as f:
        f.write(raw)
    a = np.asarray(xwd_to_image(tmp).convert("RGB")).copy()
    os.remove(tmp)
    return _crop(a, crop)


def _crop(a, crop):
    if crop is None:
        m = (a == SENTINEL).all(-1)
        if not m.any():
            return a, None
        ys, xs = np.nonzero(m)
        crop = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    x0, y0, x1, y1 = crop
    a = a[y0:y1, x0:x1]
    a[(a == SENTINEL).all(-1)] = 255
    return a, crop


class Watch:
    """Patrol status + simulation clock, from the run's ROS graph."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from rosgraph_msgs.msg import Clock
        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("capture_rviz_watch")
        self.status, self.sim = "", 0.0
        self.node.create_subscription(String, "/patrol_status", lambda m: setattr(self, "status", m.data), 10)
        self.node.create_subscription(Clock, "/clock",
                                      lambda m: setattr(self, "sim", m.clock.sec + m.clock.nanosec * 1e-9), 10)

    def spin(self, s=0.05):
        self.rclpy.spin_once(self.node, timeout_sec=s)

    def wait(self, when, timeout=1800):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.spin(0.1)
            if when == "now":
                return True
            if when == "moving" and "->" in self.status:
                return True
            if when == "done" and self.status.startswith("done"):
                return True
            if when.startswith("simtime:") and self.sim >= float(when.split(":")[1]):
                return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="paper_topdown.rviz", help="file name in hospitalnav_deploy/rviz/ or a path")
    ap.add_argument("--screen", default="3840x2160")
    ap.add_argument("--domain", type=int, default=42)
    ap.add_argument("--when", default="done")
    ap.add_argument("--settle", type=float, default=12.0, help="seconds after RViz starts before a still is taken")
    ap.add_argument("--late-start", action="store_true",
                    help="start RViz only once --when fires (stills: trail and route are re-published, so nothing is lost)")
    ap.add_argument("--out", default="")
    ap.add_argument("--stream", default="", help="directory for a frame stream instead of a still")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--until-file", default="", help="stream until this file exists (+2 s)")
    ap.add_argument("--max-seconds", type=float, default=900.0)
    a = ap.parse_args()

    cfg = a.config if os.path.isabs(a.config) else os.path.join(PKG, "rviz", a.config)
    host = clean_env(a.domain, display=os.environ.get("DISPLAY", ":0"))
    # several captures may start at once: each tries displays from its own
    # pid-derived offset, and a display only counts once ITS Xephyr owns it
    xe, disp = None, None
    for n in range(20 + os.getpid() % 40, 120):
        if os.path.exists(f"/tmp/.X11-unix/X{n}") or os.path.exists(f"/tmp/.X{n}-lock"):
            continue
        p = subprocess.Popen(["Xephyr", f":{n}", "-screen", a.screen, "-glamor", "-noreset", "-title", "RViz capture"],
                             env=host, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(40):
            if p.poll() is not None or os.path.exists(f"/tmp/.X11-unix/X{n}"):
                break
            time.sleep(0.1)
        if p.poll() is None and os.path.exists(f"/tmp/.X11-unix/X{n}"):
            xe, disp = p, f":{n}"
            break
    if xe is None:
        print("could not start Xephyr"); return 1
    env = clean_env(a.domain, display=disp)
    os.environ["ROS_DOMAIN_ID"] = str(a.domain)
    os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    watch = Watch()
    rv = None

    def start_rviz():
        log = open((a.out or os.path.join(a.stream, "x")) + ".rviz.log", "w")
        return subprocess.Popen(["rviz2", "-d", cfg, "--ros-args", "-p", "use_sim_time:=true"],
                                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    if a.stream:
        os.makedirs(a.stream, exist_ok=True)
    try:
        if not a.late_start:
            rv = start_rviz()
        if not watch.wait(a.when):
            print(f"condition '{a.when}' never happened"); return 1
        if rv is None:
            rv = start_rviz()
        scr = tuple(int(v) for v in a.screen.split("x"))
        if not a.stream:
            t_end = time.time() + a.settle
            while time.time() < t_end:
                watch.spin(0.1)
            img, crop = grab(disp, env, render_area(disp, env, scr))
            if crop is None:
                print("RViz render area not found (sentinel background missing)"); return 1
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            Image.fromarray(img).save(a.out)
            print(f"wrote {a.out} ({img.shape[1]}x{img.shape[0]}, sim t={watch.sim:.1f} s)")
            return 0
        os.makedirs(a.stream, exist_ok=True)
        t_end = time.time() + 8.0                  # let RViz come up
        while time.time() < t_end:
            watch.spin(0.1)
        stamps = open(os.path.join(a.stream, "stamps.csv"), "w")
        stamps.write("frame,wall,sim\n")
        crop, k, t0, stop_at = render_area(disp, env, scr), 0, time.time(), None
        while time.time() - t0 < a.max_seconds:
            tick = time.time()
            watch.spin(0.0)
            img, crop = grab(disp, env, crop)
            if crop is None:
                continue
            Image.fromarray(img).save(os.path.join(a.stream, f"frame_{k:06d}.jpg"), quality=93)
            stamps.write(f"{k},{tick:.3f},{watch.sim:.3f}\n"); stamps.flush()
            k += 1
            if a.until_file and os.path.exists(a.until_file) and stop_at is None:
                stop_at = time.time() + 2.0
            if stop_at and time.time() > stop_at:
                break
            while time.time() - tick < 1.0 / a.fps:
                watch.spin(0.01)
        print(f"streamed {k} frames to {a.stream}")
        return 0
    finally:
        for p in (rv, xe):
            if p is not None:
                try:
                    os.killpg(os.getpgid(p.pid), 15)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
