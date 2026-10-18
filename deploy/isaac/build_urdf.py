"""Build a physics-ready Ackermann URDF for the Hiwonder MentorPi.

Hiwonder's `ack.xacro` is a VISUALISATION model: all four wheels are
`continuous` roll joints and there is no steering joint anywhere, because the
real robot steers through a servo linkage that the driver handles in software
(`controller/ackermann.py` computes a servo angle directly). RViz never needs
the geometry, so it was never modelled.

Isaac Sim does need it. This script emits a self-contained URDF that:

  1. inserts revolute steering knuckles between `base_link` and each front
     wheel, re-parenting the front roll joints onto them (limits +/-45 deg,
     matching the clamp in `AckermannChassis.set_velocity`);
  2. rescales the inertial parameters, which are CAD exports totalling ~0.25 kg
     for a robot that really weighs 1.5-2 kg -- see TOTAL_MASS_KG below;
  3. rewrites `package://mentorpi_description/meshes/...` to relative paths and
     copies the meshes alongside, so the URDF imports without a ROS workspace.

The source package is left untouched.

    python deploy/isaac/build_urdf.py [--mass 1.8]
"""
from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "deploy" / "mentorpi_reference" / "simulations" / "mentorpi_description"
OUT = ROOT / "deploy" / "isaac" / "mentorpi"

# Measured on the real robot, NOT taken from the URDF: the CAD export totals
# ~0.25 kg, which is roughly 6x too light once the Pi 5, battery, LiDAR and
# chassis plate are included. Weigh the robot and pass --mass to override.
TOTAL_MASS_KG = 1.8

# Steering limit: AckermannChassis.set_velocity() hard-clamps to 45 degrees.
MAX_STEER_RAD = 0.7853981634
STEER_EFFORT = 5.0        # N*m, servo-scale
STEER_VELOCITY = 6.0      # rad/s
WHEEL_EFFORT = 2.0        # N*m
WHEEL_VELOCITY = 40.0     # rad/s

FRONT_JOINTS = {"wheel_lf_Joint": "lf", "wheel_rf_Joint": "rf"}


def _strip_xacro(text: str) -> str:
    """Turn ack.xacro into parseable URDF: drop xacro includes and the macro ns."""
    text = re.sub(r"<xacro:[^>]*/>", "", text)
    text = re.sub(r"</?xacro:[^>]*>", "", text)
    text = text.replace(' xmlns:xacro="http://ros.org/wiki/xacro"', "")
    return text


def _inertial(mass: float, ixx: float, iyy: float, izz: float) -> ET.Element:
    el = ET.Element("inertial")
    ET.SubElement(el, "origin", rpy="0 0 0", xyz="0 0 0")
    ET.SubElement(el, "mass", value=f"{mass:.6f}")
    ET.SubElement(el, "inertia", ixx=f"{ixx:.8f}", ixy="0", ixz="0",
                  iyy=f"{iyy:.8f}", iyz="0", izz=f"{izz:.8f}")
    return el


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=TOTAL_MASS_KG,
                    help="true total robot mass in kg (weigh it)")
    args = ap.parse_args()

    src_urdf = SRC / "urdf" / "ack.xacro"
    root = ET.fromstring(_strip_xacro(src_urdf.read_text(encoding="utf-8")))
    root.set("name", "mentorpi_ackermann")

    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {l.get("name"): l for l in root.findall("link")}

    # ---- 1. steering knuckles -------------------------------------------
    for jname, tag in FRONT_JOINTS.items():
        wheel_joint = joints[jname]
        origin = wheel_joint.find("origin")
        xyz, rpy = origin.get("xyz"), origin.get("rpy", "0 0 0")

        knuckle = f"steer_{tag}_Link"
        link = ET.SubElement(root, "link")
        link.set("name", knuckle)
        # A knuckle needs non-zero inertia or the solver will complain; keep it
        # small so it does not perturb the chassis dynamics.
        link.append(_inertial(0.01, 1e-5, 1e-5, 1e-5))

        steer = ET.SubElement(root, "joint")
        steer.set("name", f"steer_{tag}_Joint")
        steer.set("type", "revolute")
        ET.SubElement(steer, "origin", xyz=xyz, rpy=rpy)
        ET.SubElement(steer, "parent", link="base_link")
        ET.SubElement(steer, "child", link=knuckle)
        ET.SubElement(steer, "axis", xyz="0 0 1")          # steer about vertical
        ET.SubElement(steer, "limit", lower=f"{-MAX_STEER_RAD:.7f}",
                      upper=f"{MAX_STEER_RAD:.7f}",
                      effort=str(STEER_EFFORT), velocity=str(STEER_VELOCITY))

        # re-parent the wheel onto its knuckle, at zero offset
        wheel_joint.find("parent").set("link", knuckle)
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
        print(f"  + steer_{tag}_Joint (revolute, +/-45 deg) -> {knuckle} -> {jname}")

    # ---- 2. actuator limits on the roll joints --------------------------
    for jname, j in joints.items():
        if j.get("type") == "continuous" and jname.startswith("wheel_"):
            ET.SubElement(j, "limit", effort=str(WHEEL_EFFORT),
                          velocity=str(WHEEL_VELOCITY))
            # a clean unit axis; the CAD export carries 1e-5 noise terms
            ax = j.find("axis")
            if ax is not None:
                ax.set("xyz", "0 1 0")

    # ---- 3. rescale mass ------------------------------------------------
    masses = [(l, l.find("inertial/mass")) for l in links.values()]
    current = sum(float(m.get("value")) for _, m in masses if m is not None)
    scale = args.mass / current
    for l, m in masses:
        if m is None:
            continue
        m.set("value", f"{float(m.get('value')) * scale:.6f}")
        inert = l.find("inertial/inertia")
        for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            inert.set(k, f"{float(inert.get(k)) * scale:.10f}")
    print(f"  mass {current:.3f} kg -> {args.mass:.3f} kg (x{scale:.2f})")

    # ---- 4. relative mesh paths + copy meshes ---------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        rel = fn.split("mentorpi_description/")[-1]
        mesh.set("filename", rel)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = SRC / rel
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    out_urdf = OUT / "mentorpi_ackermann.urdf"
    ET.indent(root, space="  ")
    out_urdf.write_text('<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode"),
                        encoding="utf-8")

    n_rev = sum(1 for j in root.findall("joint") if j.get("type") == "revolute")
    n_cont = sum(1 for j in root.findall("joint") if j.get("type") == "continuous")
    print(f"\nwrote {out_urdf.relative_to(ROOT)}")
    print(f"  {len(root.findall('link'))} links, "
          f"{n_rev} revolute (steering), {n_cont} continuous (roll)")


if __name__ == "__main__":
    main()
