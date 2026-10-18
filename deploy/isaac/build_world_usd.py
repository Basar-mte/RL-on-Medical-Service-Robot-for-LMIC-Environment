"""Emit the Tier-2 ward as a USD stage for Isaac Sim.

Generated from `configs/hospital_v4.yaml` -- the SAME file the RL environment
rasterizes and the Gazebo world is built from. That is deliberate: the paper
claims the geometry a policy trains on and the geometry it is deployed into
cannot drift apart, and that only holds while every renderer reads one source.

Writes plain-text USDA rather than going through the `pxr` bindings, so the
stage can be produced and checked on a machine without Isaac installed. Isaac
Sim opens `.usda` directly.

Conventions follow the Gazebo generator so the two worlds agree:
    walls      2.0 m tall, static colliders
    obstacles  0.6 m tall (beds, carts, desks), static colliders
    zones      0.02 m visual slabs, NO collision -- they mark cultural regions
    ground     one thin static box spanning the map

    python deploy/isaac/build_world_usd.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CFG = ROOT / "configs" / "hospital_v4.yaml"
OUT = ROOT / "deploy" / "isaac" / "mentorpi" / "hospital_tier2.usda"

WALL_H, OBST_H, ZONE_H = 2.0, 0.6, 0.02


def _palette():
    """Zone colours from the figure palette, so the sim matches the paper."""
    try:
        from figures.make_figures import ZONE_RGB
        return {k: tuple(float(c) for c in v) for k, v in ZONE_RGB.items()}
    except Exception:
        return {}


def _box(name, cx, cy, cz, sx, sy, sz, rgb, collide=True, indent=8):
    """A USD Cube. `size = 2` spans -1..1, so scale by half-extents."""
    pad = " " * indent
    api = '(prepend apiSchemas = ["PhysicsCollisionAPI"])' if collide else ""
    r, g, b = rgb
    return (
        f'{pad}def Cube "{name}" {api}\n'
        f"{pad}{{\n"
        f"{pad}    double size = 2\n"
        f"{pad}    float3 xformOp:translate = ({cx:.4f}, {cy:.4f}, {cz:.4f})\n"
        f"{pad}    float3 xformOp:scale = ({sx/2:.4f}, {sy/2:.4f}, {sz/2:.4f})\n"
        f'{pad}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n'
        f"{pad}    color3f[] primvars:displayColor = [({r:.3f}, {g:.3f}, {b:.3f})]\n"
        f"{pad}}}\n")


def _rect(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    W = float(cfg["map"]["width_m"])
    H = float(cfg["map"]["height_m"])
    pal = _palette()
    grey = (0.55, 0.55, 0.60)

    walls, obstacles, zones = [], [], []

    for room in cfg.get("rooms", []):
        cx, cy, sx, sy = _rect(room["polygon"])
        rgb = pal.get(room["type"], (0.93, 0.93, 0.93))
        zones.append(_box(f'Zone_{room["id"]}', cx, cy, ZONE_H / 2, sx, sy,
                          ZONE_H, rgb, collide=False))

    n_w = n_o = 0
    for i, obs in enumerate(cfg.get("static_obstacles", [])):
        lbl = obs.get("label", f"obs{i}")
        cx, cy, sx, sy = _rect(obs["polygon"])
        if sx <= 0 or sy <= 0:
            continue
        is_wall = "wall" in lbl.lower()
        h = WALL_H if is_wall else OBST_H
        rgb = grey if is_wall else (0.80, 0.70, 0.40)
        safe = "".join(c if c.isalnum() else "_" for c in lbl)
        prim = _box(f"{safe}_{i}", cx, cy, h / 2, sx, sy, h, rgb)
        (walls if is_wall else obstacles).append(prim)
        n_w += is_wall
        n_o += not is_wall

    ground = _box("Ground", W / 2, H / 2, -0.05, W, H, 0.1, (0.85, 0.85, 0.85),
                  indent=4)

    stage = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "LMIC-HospitalNav Tier 2 (walled ward). Generated from
           configs/hospital_v4.yaml by deploy/isaac/build_world_usd.py --
           do not edit by hand; regenerate instead."
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravity = (0, 0, -9.81)
        float physics:gravityMagnitude = 9.81
    }}

{ground}
    def Xform "Zones"
    {{
{"".join(zones)}    }}

    def Xform "Walls"
    {{
{"".join(walls)}    }}

    def Xform "Obstacles"
    {{
{"".join(obstacles)}    }}
}}
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(stage, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  map {W:.0f} x {H:.0f} m")
    print(f"  {len(zones)} zone slabs (visual only)")
    print(f"  {n_w} wall colliders @ {WALL_H} m")
    print(f"  {n_o} obstacle colliders @ {OBST_H} m")
    print(f"  palette: {'figure palette' if pal else 'fallback (figures import failed)'}")
    print(f"  {OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
