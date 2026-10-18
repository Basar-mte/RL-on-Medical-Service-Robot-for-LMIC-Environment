"""Validate the generated URDF before handing it to Isaac Sim's importer."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
URDF = HERE / "mentorpi" / "mentorpi_ackermann.urdf"

root = ET.parse(URDF).getroot()
links = {l.get("name") for l in root.findall("link")}
joints = root.findall("joint")

print(f"robot: {root.get('name')}   links: {len(links)}   joints: {len(joints)}")

# ---- single kinematic root ------------------------------------------------
children = {j.find("child").get("link") for j in joints}
roots = links - children
print(f"\nroot link(s): {sorted(roots)}   {'OK' if len(roots) == 1 else '<<< must be exactly 1'}")

# ---- steering chain -------------------------------------------------------
print("\nsteering chain:")
ok = True
for tag in ("lf", "rf"):
    steer = next((j for j in joints if j.get("name") == f"steer_{tag}_Joint"), None)
    wheel = next((j for j in joints if j.get("name") == f"wheel_{tag}_Joint"), None)
    if steer is None or wheel is None:
        print(f"  {tag}: MISSING"); ok = False; continue
    chain_ok = (steer.find("parent").get("link") == "base_link"
                and wheel.find("parent").get("link") == steer.find("child").get("link"))
    lim = steer.find("limit")
    deg = math.degrees(float(lim.get("upper")))
    print(f"  base_link -> {steer.find('child').get('link')} -> "
          f"{wheel.find('child').get('link')}   +/-{deg:.0f} deg   "
          f"{'OK' if chain_ok else '<<< BROKEN'}")
    ok &= chain_ok

# ---- geometry recovered from the joint origins ----------------------------
def origin(name):
    j = next(j for j in joints if j.get("name") == name)
    return [float(v) for v in j.find("origin").get("xyz").split()]

lf, lb = origin("steer_lf_Joint"), origin("wheel_lb_Joint")
rf = origin("steer_rf_Joint")
wheelbase = lf[0] - lb[0]
track = abs(lf[1] - rf[1])
print(f"\ngeometry from URDF:")
print(f"  wheelbase {wheelbase:.4f} m   track {track:.4f} m")
print(f"  driver constants (controller/ackermann.py): 0.145 / 0.133")
print(f"  -> differ by {abs(wheelbase-0.145)*1000:.0f} mm / {abs(track-0.133)*1000:.0f} mm")
print(f"  min turning radius at 45 deg: {wheelbase / math.tan(math.pi/4):.3f} m")

# ---- mass -----------------------------------------------------------------
total = sum(float(m.get("value")) for m in root.iter("mass"))
print(f"\ntotal mass: {total:.3f} kg")

# ---- meshes ---------------------------------------------------------------
missing = [m.get("filename") for m in root.iter("mesh")
           if not (HERE / "mentorpi" / m.get("filename")).exists()]
print(f"meshes: {len(list(root.iter('mesh')))} referenced, "
      f"{len(missing)} missing {missing if missing else ''}")

print("\nVERDICT:", "ready for the Isaac URDF importer"
      if ok and len(roots) == 1 and not missing else "NOT READY -- see above")
