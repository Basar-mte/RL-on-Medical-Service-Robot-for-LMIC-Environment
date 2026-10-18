"""Cross-validate the generated USD stage against the YAML it came from.

Catches the failure that matters: a stage that opens fine in Isaac but whose
geometry has silently drifted from the config the policies were trained on.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CFG = ROOT / "configs" / "hospital_v4.yaml"
USDA = ROOT / "deploy" / "isaac" / "mentorpi" / "hospital_tier2.usda"

text = USDA.read_text(encoding="utf-8")
cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
W, H = float(cfg["map"]["width_m"]), float(cfg["map"]["height_m"])

# ---- parse every Cube: name, translate, scale, collider? -------------------
PRIM = re.compile(
    r'def Cube "([^"]+)"\s*(\([^)]*\))?\s*\{'
    r'[^}]*?xformOp:translate = \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)'
    r'[^}]*?xformOp:scale = \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)',
    re.S)

prims = []
for m in PRIM.finditer(text):
    name, api = m.group(1), m.group(2) or ""
    tx, ty, tz, sx, sy, sz = (float(m.group(i)) for i in range(3, 9))
    prims.append(dict(name=name, c=(tx, ty, tz), half=(sx, sy, sz),
                      collide="PhysicsCollisionAPI" in api))

print(f"parsed {len(prims)} Cube prims from {USDA.name}")
colliders = [p for p in prims if p["collide"]]
visual = [p for p in prims if not p["collide"]]
print(f"  {len(colliders)} colliders, {len(visual)} visual-only")

# ---- counts must match the config -----------------------------------------
obs = [o for o in cfg["static_obstacles"]
       if (max(p[0] for p in o["polygon"]) - min(p[0] for p in o["polygon"])) > 0
       and (max(p[1] for p in o["polygon"]) - min(p[1] for p in o["polygon"])) > 0]
expected = len(obs) + 1                      # +1 for Ground
print(f"\ncounts: config has {len(obs)} sized obstacles (+ground) = {expected}")
print(f"        stage has {len(colliders)} colliders "
      f"{'OK' if len(colliders) == expected else '<<< MISMATCH'}")
print(f"        config rooms {len(cfg['rooms'])} vs zone slabs {len(visual)} "
      f"{'OK' if len(visual) == len(cfg['rooms']) else '<<< MISMATCH'}")

# ---- every obstacle in the config must exist in the stage at the right box -
def rect(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
            max(xs) - min(xs), max(ys) - min(ys))

# Match on (centre, size) with multiplicity: the config legitimately contains
# co-located obstacles (adjacent rooms each emit the wall they share), so a
# centre-keyed lookup would silently compare the wrong pair.
from collections import Counter
have = Counter((round(p["c"][0], 3), round(p["c"][1], 3),
                round(p["half"][0] * 2, 3), round(p["half"][1] * 2, 3))
               for p in colliders)
bad = []
for o in obs:
    cx, cy, sx, sy = rect(o["polygon"])
    key = (round(cx, 3), round(cy, 3), round(sx, 3), round(sy, 3))
    if have[key] <= 0:
        bad.append(f"no match for {o.get('label')} @ ({cx:.2f},{cy:.2f}) {sx:.2f}x{sy:.2f}")
    else:
        have[key] -= 1
print(f"\ngeometry cross-check: {len(obs) - len(bad)}/{len(obs)} obstacles match")
for b in bad[:5]:
    print("   !!", b)

# ---- nothing may escape the map -------------------------------------------
esc = [p["name"] for p in prims if p["name"] != "Ground" and (
    p["c"][0] - p["half"][0] < -0.5 or p["c"][0] + p["half"][0] > W + 0.5 or
    p["c"][1] - p["half"][1] < -0.5 or p["c"][1] + p["half"][1] > H + 0.5)]
print(f"outside the {W:.0f}x{H:.0f} m map: {len(esc)} {esc[:4] if esc else ''}")

# ---- everything must sit on or above the floor ----------------------------
sunk = [p["name"] for p in prims
        if p["name"] != "Ground" and p["c"][2] - p["half"][2] < -1e-6]
print(f"below the floor plane: {len(sunk)} {sunk[:4] if sunk else ''}")

ok = not bad and not esc and not sunk and len(colliders) == expected
print("\nVERDICT:", "stage matches the training config"
      if ok else "DRIFT DETECTED -- do not use until resolved")
sys.exit(0 if ok else 1)
