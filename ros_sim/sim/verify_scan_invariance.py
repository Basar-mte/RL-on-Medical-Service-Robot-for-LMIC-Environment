"""Proves the photoreal pass did not change what the policy sees.

gz's gpu_lidar is a rendering sensor - it sees <visual>, not <collision> - so
adding scenery is only safe if the scenery stays out of the LD19's horizontal
plane at z = 0.1535 m. gen_worlds.py enforces that statically (every decorative
prop declares its z-extent and the build aborts if one enters the guard band).
This script checks it the other way round: empirically, from the sensor.

Method: drop a bank of LD19-spec probes at the same poses into
  hospital_reference.world  (trained collision geometry only, no realism) and
  hospital_walled.world      (the photoreal world the runs use)
and compare the ranges ray-for-ray, plus the 48-ray vector the policy actually
consumes (scan_utils.scan_to_lidar48).

    python3 verify_scan_invariance.py [--tier walled|open] [--poses N]

PASS CRITERION
  What the policy consumes is the 48-ray vector, so the gate is POLICY_TOL on
  that vector (5 mm, i.e. 4e-4 of the 12 m normalisation). Raw 360-ray deltas
  are reported alongside: the photoreal walls differ from the bare reference by
  up to ~5 cm on a handful of grazing hits beyond 8.5 m (measured 2026-10-16,
  deterministic - the same world twice gives 0.0000 m, and stripping every
  decorative prop changes nothing - so it is a property of gz's GPU depth
  rendering of the two wall descriptions, not of any scenery), which never
  reaches the policy's input beyond 2 mm. Writes a JSON report next to the
  worlds for the paper's appendix.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORLDS = os.path.join(HERE, "..", "hospitalnav_deploy", "worlds")
sys.path.insert(0, os.path.join(HERE, "..", "hospitalnav_deploy", "hospitalnav_deploy"))

LIDAR_Z = 0.1535
TOL = 1e-3          # metres, per raw ray - reported, informational
POLICY_TOL = 5e-3   # metres, on the 48-ray policy vector - the pass criterion

# Poses that matter: the patrol route, each room's interior, and the doorways -
# i.e. everywhere the robot actually scans during a run.
POSES = [
    (37.0, 15.0), (35.0, 15.0), (31.0, 15.0), (28.0, 15.0), (24.0, 15.0),
    (20.0, 15.0), (18.5, 15.0), (14.0, 15.0), (9.0, 15.0), (5.0, 14.0),
    (2.0, 15.0), (20.0, 16.5), (20.0, 13.5), (35.9, 16.0),
    (5.0, 12.6), (5.0, 10.0), (5.0, 7.5), (3.0, 5.0), (7.0, 4.0),
    (13.0, 12.6), (13.0, 9.0), (13.0, 7.5), (11.0, 4.0), (15.0, 5.0),
    (23.5, 12.6), (23.5, 7.5), (31.0, 12.6), (31.0, 7.5), (33.0, 4.0),
    (6.5, 17.4), (6.5, 20.0), (6.5, 23.5), (3.0, 27.0), (10.0, 26.0),
    (20.0, 17.4), (20.0, 23.5), (16.0, 27.0), (24.0, 20.0),
    (33.5, 17.4), (33.5, 23.5), (37.0, 27.0), (30.0, 21.0),
    (18.5, 10.0), (18.5, 4.0), (17.0, 14.6), (35.0, 16.4),
]

PROBE = """
    <model name="probe_{i}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <sensor name="lidar" type="gpu_lidar">
          <update_rate>10</update_rate>
          <topic>probe_{i}</topic>
          <lidar>
            <scan><horizontal><samples>360</samples><resolution>1</resolution>
              <min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle>
            </horizontal></scan>
            <range><min>0.10</min><max>12.0</max><resolution>0.01</resolution></range>
            <visibility_mask>1073741819</visibility_mask>
          </lidar>
          <always_on>1</always_on>
        </sensor>
      </link>
    </model>"""


def world_with_probes(src: str, poses) -> str:
    sdf = open(src).read()
    probes = "".join(PROBE.format(i=i, x=x, y=y, z=LIDAR_Z)
                     for i, (x, y) in enumerate(poses))
    return sdf.replace("\n  </world>", probes + "\n  </world>")


def _reap():
    """gz sim forks a gz-sim-server child; terminating the launcher leaves it
    publishing. A leftover server from the previous world would answer the next
    world's probes too, so each capture must start from a clean slate. Only
    THIS script's servers are touched (their world path carries the scaninv_
    prefix) so a campaign running alongside is never killed; its topics are
    kept apart by GZ_PARTITION as well."""
    subprocess.run(["pkill", "-9", "-f", "scaninv_.*\\.world"], capture_output=True)
    for _ in range(30):
        r = subprocess.run(["pgrep", "-f", "scaninv_.*\\.world"], capture_output=True)
        if r.returncode != 0:
            return
        time.sleep(0.5)
    raise RuntimeError("could not clear this script's gz-sim-server processes")


def capture(world_path: str, n: int, settle=12.0) -> dict[int, np.ndarray]:
    env = dict(os.environ)
    pkg = os.path.abspath(os.path.join(HERE, "..", "hospitalnav_deploy"))
    # the package dir resolves the worlds' relative texture paths; its parent
    # resolves package://hospitalnav_deploy/meshes/... (the Fuel assets)
    env["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join([pkg, os.path.dirname(pkg)])
    env["GZ_PARTITION"] = "scaninv"           # isolated from any live campaign
    env["HN_SCOPE"] = "probe"
    _reap()
    proc = subprocess.Popen(["gz", "sim", "-s", "-r", "-v0", world_path],
                            env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    out: dict[int, np.ndarray] = {}
    try:
        time.sleep(settle)
        for i in range(n):
            r = subprocess.run(["gz", "topic", "-e", "-t", f"/probe_{i}", "-n", "1"],
                               capture_output=True, text=True, timeout=40, env=env)
            vals = [float(v) for v in re.findall(r"^\s*ranges:\s*(\S+)", r.stdout, re.M)]
            if len(vals) < 360:
                raise RuntimeError(f"probe_{i}: got {len(vals)} ranges from {world_path}")
            # one publisher, one message: take the first scan if gz echoed more
            out[i] = np.array(vals[:360], dtype=float)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        _reap()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="walled", choices=["walled", "open", "real", "furnished"])
    ap.add_argument("--poses", type=int, default=len(POSES))
    a = ap.parse_args()
    poses = POSES[:a.poses]

    from scan_utils import scan_to_lidar48, LIDAR_MAX

    ref_src = os.path.join(WORLDS, "hospital_reference.world")
    tst_src = os.path.join(WORLDS, f"hospital_{a.tier}.world")
    if a.tier == "open":
        print("NOTE: the open tier intentionally removes the interior partitions, so\n"
              "      it is compared against a reference built the same way.")
        ref_src = os.path.join(WORLDS, "hospital_reference_open.world")

    tmp = tempfile.mkdtemp(prefix="scaninv_")
    results = {}
    scans = {}
    for tag, src in (("reference", ref_src), (a.tier, tst_src)):
        path = os.path.join(tmp, f"{tag}.world")
        with open(path, "w") as f:
            f.write(world_with_probes(src, poses))
        print(f"capturing {len(poses)} probes in {tag} ...", flush=True)
        scans[tag] = capture(path, len(poses))

    worst_ray = 0.0
    worst_pol = 0.0
    bad = []
    bad_pol = []
    for i, (x, y) in enumerate(poses):
        r = np.nan_to_num(scans["reference"][i], posinf=LIDAR_MAX, neginf=LIDAR_MAX)
        t = np.nan_to_num(scans[a.tier][i], posinf=LIDAR_MAX, neginf=LIDAR_MAX)
        r = np.minimum(r, LIDAR_MAX)
        t = np.minimum(t, LIDAR_MAX)
        d = float(np.max(np.abs(r - t)))
        inc = 2 * math.pi / 360
        pr = scan_to_lidar48(r, -math.pi, inc)
        pt = scan_to_lidar48(t, -math.pi, inc)
        dp = float(np.max(np.abs(pr - pt)) * LIDAR_MAX)
        worst_ray = max(worst_ray, d)
        worst_pol = max(worst_pol, dp)
        results[f"{x},{y}"] = {"max_ray_delta_m": d, "max_policy_delta_m": dp}
        if d > TOL:
            bad.append((x, y, d))
        if dp > POLICY_TOL:
            bad_pol.append((x, y, dp))

    report = {
        "tier": a.tier,
        "lidar_plane_z_m": LIDAR_Z,
        "poses_tested": len(poses),
        "ray_tolerance_m": TOL,
        "policy_tolerance_m": POLICY_TOL,
        "max_ray_delta_m": worst_ray,
        "max_policy_48ray_delta_m": worst_pol,
        "poses_with_ray_delta": len(bad),
        "passed": not bad_pol,
        "per_pose": results,
    }
    out = os.path.join(WORLDS, f"scan_invariance_{a.tier}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  poses tested          : {len(poses)}")
    print(f"  worst single-ray delta: {worst_ray:.6f} m")
    print(f"  worst policy-48 delta : {worst_pol:.6f} m")
    print(f"  report                : {out}")
    if bad:
        print(f"\n  raw rays: {len(bad)} pose(s) differ beyond {TOL} m (grazing long-range hits):")
        for x, y, d in bad[:12]:
            print(f"    ({x}, {y}): {d:.4f} m")
    if bad_pol:
        print(f"\nFAIL - the policy's 48-ray input differs beyond {POLICY_TOL} m at "
              f"{len(bad_pol)} pose(s):")
        for x, y, d in bad_pol[:12]:
            print(f"    ({x}, {y}): {d:.4f} m")
        return 1
    print(f"\nPASS - the '{a.tier}' world gives the policy the same 48-ray input as the "
          f"trained geometry at every pose tested (worst {worst_pol*1000:.2f} mm).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
