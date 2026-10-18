"""Generates the photoreal gz-sim (Harmonic) hospital worlds, both tiers.

    python3 gen_worlds.py            # hospital_walled.world + hospital_open.world
    python3 gen_worlds.py --ceiling  # ...with the suspended ceiling closed

WHERE THE GEOMETRY COMES FROM
  Every piece of collision geometry is parsed straight out of the SHIPPED
  worlds/hospital.world - the Gazebo-Classic world the runs_v5 models were
  benchmarked against - and re-emitted at byte-identical pose and size. It is
  deliberately NOT regenerated from layout.py: the layout.py on this machine is
  an older revision carrying 28 beds where the shipped world has 48, and the
  shipped world is what the published numbers refer to. layout.py is used only
  for room metadata that carries no collision (names, colours, doorway sides).

TIERS
  walled  (Tier-2)  all 40 walls: the seven enclosed rooms.
  open    (Tier-1)  wall_0..wall_3 only - the outer shell. Rooms keep their zone
                    labels, floors and furniture but lose their partitions, which
                    is what "trained on the OPEN tier" means: the cultural zones
                    are semantic, not physical, so a culture-blind policy can cut
                    straight across the prayer room instead of routing around it.

THE FIDELITY RULE (why this file is fussy about height)
  gz-sim's gpu_lidar is a RENDERING sensor: it sees <visual> geometry and is
  blind to <collision>. Measured 2026-10-15 on this machine - a visual-only box
  returned 1.80 m, a collision-only box returned inf. So "decorative, no
  collision" does NOT hide a prop from the policy. What hides it is HEIGHT: the
  LD19 sits at z = 0.1535 m (base_footprint->base_link 0.061 + base_link->laser
  0.0925) and scans one horizontal plane there.

  Every visual emitted here is therefore one of:
    TRAINED - crosses the lidar plane; pose and size copied from the shipped
              world, only the material differs.
    FREE    - lies wholly below LIDAR_LO or above LIDAR_HI, so no ray can reach
              it at any robot pose.
  _free() records the z-extent of every decorative prop and build() refuses to
  write a world if any of them enters the band. verify_scan_invariance.py then
  confirms it empirically against the reference world.
"""
from __future__ import annotations
import argparse
import math
import os
import re

import layout as L
import gen_textures as T

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "hospitalnav_deploy")
OUT = os.path.join(PKG, "worlds")
TRAINED_WORLD = os.path.join(OUT, "hospital.world")

LIDAR_Z = 0.1535                      # see module docstring
LIDAR_LO, LIDAR_HI = 0.10, 0.30       # guard band no decorative prop may enter
WALL_VIS_H = 3.0                      # visual wall top (collision stays at 1.2)
CEIL_Z = 3.0
SIGN_W, SIGN_H = 1.60, 0.42           # nameplate board, metres (same for every room)
SIGN_Z = 2.62                         # board centre: on the lintel above the door head
SIGN_T = 0.128                        # board centre from the wall centre line: flush on the face

_violations: list[str] = []
_clutter: list[str] = []          # furnished tier: props that DO cross the band, by design


def _free(name: str, z0: float, z1: float) -> None:
    if not (z1 <= LIDAR_LO or z0 >= LIDAR_HI):
        _violations.append(f"{name}: z=[{z0:.3f},{z1:.3f}] enters the lidar band")


def _unmapped(name: str) -> None:
    """A floor-standing prop the scan WILL see and the prebuilt map does not
    contain. Only the furnished tier may register these."""
    _clutter.append(name)


# ------------------------------------------------------- the trained world --
class Prop:
    __slots__ = ("name", "pose", "box", "cyl")

    def __init__(self, name, pose, box, cyl):
        self.name, self.pose, self.box, self.cyl = name, pose, box, cyl

    @property
    def xy(self):
        return self.pose[0], self.pose[1]


def load_trained(path=TRAINED_WORLD) -> dict[str, list[Prop]]:
    src = open(path, encoding="utf-8", errors="replace").read()
    out: dict[str, list[Prop]] = {}
    for name, body in re.findall(r'<model name="([^"]+)">(.*?)</model>', src, re.S):
        pm = re.search(r"<pose>([^<]+)</pose>", body)
        if not pm:
            continue
        pose = [float(v) for v in pm.group(1).split()]
        bm = re.search(r"<box><size>([^<]+)</size></box>", body)
        cm = re.search(r"<cylinder><radius>([^<]+)</radius><length>([^<]+)</length>", body)
        box = [float(v) for v in bm.group(1).split()] if bm else None
        cyl = (float(cm.group(1)), float(cm.group(2))) if cm else None
        kind = re.sub(r"_?\d+$", "", name)
        out.setdefault(kind, []).append(Prop(name, pose, box, cyl))
    return out


# ----------------------------------------------------------------- helpers --
def _pbr(albedo: str, rough=0.65, metal=0.0) -> str:
    """A textured surface. NOT an SDF <pbr> block: on this machine every <pbr>
    material renders black (see gen_textures.textured_box), so _box() turns
    this token into a textured unit-cube OBJ scaled to the box. The roughness
    maps onto the MTL specular exponent."""
    return f"@TEX:{albedo}:{rough:.2f}"


GEN_MESH = "package://hospitalnav_deploy/meshes/gen"


def _flat(r, g, b, a=1.0, emis=0.0) -> str:
    return f"""<material>
          <ambient>{r*0.5:.3f} {g*0.5:.3f} {b*0.5:.3f} {a}</ambient>
          <diffuse>{r:.3f} {g:.3f} {b:.3f} {a}</diffuse>
          <specular>0.14 0.14 0.14 1</specular>
          <emissive>{r*emis:.3f} {g*emis:.3f} {b*emis:.3f} 1</emissive>
        </material>"""


def _box(name, cx, cy, cz, sx, sy, sz, mat, collide=False, shadows=True) -> str:
    col = (f'<collision name="c"><geometry><box><size>{sx} {sy} {sz}</size>'
           f"</box></geometry></collision>") if collide else ""
    if mat.startswith("@TEX:"):
        _tag, png, rough = mat.split(":")
        ks = max(0.02, 0.35 * (1.0 - float(rough)))
        obj = T.textured_box(png, ks=ks, ns=4.0 + 60.0 * (1.0 - float(rough)))
        geom = (f"<geometry><mesh><uri>{GEN_MESH}/{obj}</uri>"
                f"<scale>{sx} {sy} {sz}</scale></mesh></geometry>")
        mat = ""
    else:
        geom = f"<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>"
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{cx} {cy} {cz} 0 0 0</pose>
      <link name="link">
        {col}
        <visual name="v">
          <cast_shadows>{'true' if shadows else 'false'}</cast_shadows>
          {geom}
          {mat}
        </visual>
      </link>
    </model>"""


def _room_rect(name):
    for (x0, y0, x1, y1, n, _d) in L.ROOMS:
        if n == name:
            return x0, y0, x1, y1
    return None


def _room_at(x, y):
    for (x0, y0, x1, y1, n, _d) in L.ROOMS:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return n
    return None


FLOOR_KIND = {"male_ward": "ward", "general_ward_1": "ward", "female_ward": "ward",
              "general_ward_2": "ward", "general_ward_3": "ward",
              "ward_icu_emerg": "emerg", "prayer": "prayer"}


# ------------------------------------------------------------------ pieces --

def floors(tr) -> str:
    """FREE (z<=0.001): textured floors, tiled at true physical scale. Their TOP
    is at z = 0, the physics ground plane the robot's wheels and every walker
    stand on (the floors are visual only): with the slabs' tops at 0.02 m, as
    before 2026-10-17, the 32 mm wheels, the walkers' feet and every piece of
    furniture rendered sunk 2 cm into the tiles. Room floors sit 1 mm higher
    than the corridor's so their overlap cannot z-fight."""
    out = []
    trained_floors = [p for k, v in tr.items() if k.startswith("floor_") for p in v]
    for p in trained_floors:                           # trained floor tints
        name = p.name[len("floor_"):]
        cx, cy = p.xy
        sx, sy, _ = p.box
        tex = T.floor_for(FLOOR_KIND.get(name, "ward"), sx, sy)
        _free(p.name, -0.019, 0.001)
        out.append(_box(p.name, cx, cy, -0.009, sx, sy, 0.02,
                        _pbr(tex, rough=0.35), shadows=False))
    # the trained world tints only the rooms; give the circulation space a floor too
    for tag, (x0, y0, x1, y1) in (("corridor", L.CORRIDOR),
                                  ("south_corr", L.SOUTH_SIDE_CORRIDOR)):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sx, sy = x1 - x0, y1 - y0
        _free(f"floor_{tag}", -0.020, 0.0)
        out.append(_box(f"floor_{tag}", cx, cy, -0.010, sx, sy, 0.02,
                        _pbr(T.floor_for("corridor", sx, sy), rough=0.30),
                        shadows=False))
    return "".join(out)




def signage() -> str:
    """FREE: a readable nameplate over each room's main doorway.

    The board is a fixed physical size (SIGN_W x SIGN_H metres) for every room,
    and gen_textures renders its lettering at a fixed pixels-per-metre, so the
    text is the same height in metres on every sign - the labels are in
    proportion to the building rather than to the room or the texture. The three
    culturally-restricted rooms carry a second line saying so, which is the
    distinction the policies are actually being judged on.
    """
    out = []
    for (x0, y0, x1, y1, name, doors) in L.ROOMS:
        side, centre = doors[0]
        label = L.ROOM_LABEL[name]
        sub = "Cultural zone" if name in L.CULTURAL else None
        rgb = L.ROOM_COLOR[name]
        tex = T.sign_for(label, SIGN_W, SIGN_H, rgb, sub=sub)
        if side in ("N", "S"):
            off = SIGN_T if side == "N" else -SIGN_T
            cx, cy = centre, (y1 if side == "N" else y0) + off
            sx, sy = SIGN_W, 0.05
        else:
            off = SIGN_T if side == "E" else -SIGN_T
            cx, cy = (x1 if side == "E" else x0) + off, centre
            sx, sy = 0.05, SIGN_W
        _free(f"sign_{name}", SIGN_Z - SIGN_H / 2, SIGN_Z + SIGN_H / 2)
        out.append(_box(f"sign_{name}", cx, cy, SIGN_Z, sx, sy, SIGN_H,
                        _pbr(tex, rough=0.45), shadows=False))
        # a thin lit edge so the plate reads against the wall in a photo
        _free(f"sign_{name}_glow", SIGN_Z - SIGN_H / 2 - 0.05, SIGN_Z - SIGN_H / 2)
        out.append(_box(f"sign_{name}_glow", cx, cy, SIGN_Z - SIGN_H / 2 - 0.025,
                        sx * 0.98, sy * 0.98, 0.05,
                        _flat(*rgb, emis=0.85), shadows=False))
    return "".join(out)





def _reference_body(tr, tier="walled") -> str:
    """Trained geometry, gz-converted, nothing else - the invariance control.
    tier='open' drops the interior partitions exactly as the open world does."""
    out = []
    grey = _flat(0.75, 0.75, 0.75)
    for p in tr["wall"] + tr["bed"] + tr.get("nurse_desk", []):
        if tier == "open" and p.name.startswith("wall_") and int(p.name.rsplit("_", 1)[1]) >= 4:
            continue
        cx, cy, cz = p.pose[:3]
        sx, sy, sz = p.box
        out.append(_box(p.name, cx, cy, cz, sx, sy, sz, grey, collide=True))
    for p in tr["person"]:
        cx, cy, cz = p.pose[:3]
        r, h = p.cyl
        out.append(f"""
    <model name="{p.name}">
      <static>true</static>
      <pose>{cx} {cy} {cz} 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>{grey}</visual>
      </link>
    </model>""")
    # The shipped world's start-marker pole is deliberately NOT part of the
    # reference. It is a visual-only 0.8 m pole standing at the robot's own
    # spawn point, and the benchmark ran in Gazebo Classic whose CPU ray
    # sensor intersects COLLISION geometry - so the trained policies never saw
    # it. gz's gpu_lidar would see it (a phantom 2 cm post inside the robot's
    # footprint at spawn, and a stray return from anywhere east of x = 35),
    # which is why every tier lifts it above the band and the control omits
    # it. Measured 2026-10-16: with the pole in the reference, probes at
    # (35,15), (36.4,16) and (35.5,16.7) differed by 2.9-5.8 m on exactly the
    # rays through (37,15).
    return "".join(out)



# --------------------------------------------------------------- ward props --
# Everything below is FREE geometry: it lives above LIDAR_HI (0.30 m) or below
# LIDAR_LO (0.10 m), so no ray at the LD19's plane can ever reach it. That is
# also why there are no new FLOOR-STANDING items here (bedside cabinets, chairs,
# trolleys, IV stands on castors): anything standing on the floor would cut the
# scan plane and change what the policy sees. Those live in the separate
# 'furnished' tier instead, which is explicitly not scan-comparable.





# ------------------------------------------------------------ fuel meshes --
# Open-source hospital assets (OpenRobotics on Gazebo Fuel - the AWS RoboMaker
# Hospital World set), vendored into the package under meshes/fuel/.
#
# They are overlaid ON TOP of the trained collision boxes rather than replacing
# them, and each mesh is scaled so its footprint is no larger than the box it
# sits on. At the lidar plane the trained box is therefore still the outermost
# surface the scan can hit, so the realistic pass stays scan-identical - which
# verify_scan_invariance.py checks rather than assumes.
FUEL_MESH = "package://hospitalnav_deploy/meshes/fuel"
MESH_DIR = os.path.join(PKG, "meshes", "fuel")


def _obj_bbox(path):
    lo = [1e9] * 3
    hi = [-1e9] * 3
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                q = line.split()
                try:
                    v = (float(q[1]), float(q[2]), float(q[3]))
                except (ValueError, IndexError):
                    continue
                for i in range(3):
                    lo[i] = min(lo[i], v[i])
                    hi[i] = max(hi[i], v[i])
    return lo, hi


_bbox_cache: dict[str, tuple] = {}


def _bbox(model, obj):
    key = f"{model}/{obj}"
    if key not in _bbox_cache:
        _bbox_cache[key] = _obj_bbox(os.path.join(MESH_DIR, model, obj))
    return _bbox_cache[key]


def _mesh_fit(model, obj, cx, cy, cz, sx, sy, sz, yaw=0.0):
    """Pose + scale that puts an OBJ's bounding box into (sx, sy, sz) centred at
    (cx, cy), base at cz, rotated by yaw about the box's own centre. Returns
    (px, py, pz, kx, ky, kz, ox, oy): the model pose and the mesh offset in
    the model frame (so a collision box can be placed at the box centre)."""
    lo, hi = _bbox(model, obj)
    ex, ey, ez = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    kx, ky, kz = sx / ex, sy / ey, sz / ez
    # centre the mesh on the origin, then the <pose> puts it where we want
    ox = -((lo[0] + hi[0]) / 2) * kx
    oy = -((lo[1] + hi[1]) / 2) * ky
    oz = -lo[2] * kz
    ca, sa = math.cos(yaw), math.sin(yaw)
    px, py = cx + ca * ox - sa * oy, cy + sa * ox + ca * oy
    return px, py, cz + oz, kx, ky, kz, ox, oy


def _mesh_visual(name, model, obj, cx, cy, cz, sx, sy, sz, yaw=0.0):
    """Visual-only mesh fitted to a box (see _mesh_fit)."""
    px, py, pz, kx, ky, kz, _ox, _oy = _mesh_fit(model, obj, cx, cy, cz, sx, sy, sz, yaw)
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{px} {py} {pz} 0 0 {yaw}</pose>
      <link name="link">
        <visual name="v">
          <cast_shadows>true</cast_shadows>
          <geometry>
            <mesh><uri>{FUEL_MESH}/{model}/{obj}</uri>
              <scale>{kx:.5f} {ky:.5f} {kz:.5f}</scale></mesh>
          </geometry>
        </visual>
      </link>
    </model>"""


def _mesh_prop(name, model, obj, cx, cy, sx, sy, sz, yaw=0.0, cz=0.0):
    """Furnished-tier prop: the fitted mesh PLUS a box collision of the same
    footprint, so the physics stops the robot where the picture says it
    should. Registered as unmapped clutter (the scan sees it, the map does
    not)."""
    _unmapped(name)
    px, py, pz, kx, ky, kz, ox, oy = _mesh_fit(model, obj, cx, cy, cz, sx, sy, sz, yaw)
    # the collision box sits at the fitted box's centre, expressed in the model frame
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{px} {py} {pz} 0 0 {yaw}</pose>
      <link name="link">
        <collision name="c">
          <pose>{-ox:.4f} {-oy:.4f} {sz / 2 - (pz - cz):.4f} 0 0 0</pose>
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        </collision>
        <visual name="v">
          <cast_shadows>true</cast_shadows>
          <geometry>
            <mesh><uri>{FUEL_MESH}/{model}/{obj}</uri>
              <scale>{kx:.5f} {ky:.5f} {kz:.5f}</scale></mesh>
          </geometry>
        </visual>
      </link>
    </model>"""



# ------------------------------------------------- ward signage extras --

def prayer_mats() -> str:
    """FREE (z<=0.03): prayer mats laid in rows on the prayer-room floor."""
    rect = _room_rect("prayer")
    if rect is None:
        return ""
    x0, y0, x1, y1 = rect
    tex = T.prayer_mat()
    out = []
    k = 0
    for row in range(3):
        for col in range(5):
            mx = x0 + 1.6 + col * 1.25
            my = y0 + 2.2 + row * 2.4
            if not (x0 + 0.6 < mx < x1 - 0.6 and y0 + 0.6 < my < y1 - 1.5):
                continue
            _free(f"mat_{k}", 0.001, 0.007)
            out.append(_box(f"mat_{k}", mx, my, 0.004, 0.70, 1.15, 0.006,
                            _pbr(tex, rough=0.95), shadows=False))
            k += 1
    return "".join(out)


# ------------------------------------------------------------- furnished --
# Floor-standing hospital furniture WITH collision, placed where a real ward
# keeps it (against walls, between beds, inside rooms) and never in the
# corridor's walking lanes or a doorway. The prebuilt map does not know any of
# it: this tier is the unmapped-clutter stress test, and verify_scan_invariance
# is expected to report differences here.
NATURAL = {   # model -> (obj, natural x, y, z in metres as the asset is meant to stand)
    "bedsidetable":      ("BedsideTable.obj",      0.50, 0.58, 0.90),
    "chair":             ("Chair.obj",             0.58, 0.51, 0.82),
    "ivstand":           ("IVStand.obj",           0.44, 0.45, 1.41),
    "metalcabinet":      ("MetalCabinet.obj",      0.81, 0.45, 1.68),
    "storagerack":       ("StorageRack.obj",       1.26, 0.71, 1.74),
    "surgicaltrolley":   ("trolley.obj",           0.88, 0.37, 0.70),
    "xraymachine":       ("XRayMachine.obj",       2.33, 1.10, 1.80),
    "patientwheelchair": ("PatientWheelChair.obj", 1.12, 0.64, 1.78),
    "trolleybedpatient": ("TrolleyBedPatient.obj", 2.20, 1.00, 0.95),
    "nurse":             ("Nurse.obj",             0.57, 0.43, 1.62),
    "opscrubs":          ("OpScrubs.obj",          0.32, 0.50, 1.60),
    "nursedesk":         ("NurseDesk.obj",         3.00, 0.96, 1.05),
}


def _prop(name, model, cx, cy, yaw=0.0, scale=1.0):
    obj, nx, ny, nz = NATURAL[model]
    return _mesh_prop(name, model, obj, cx, cy, nx * scale, ny * scale, nz * scale, yaw)


def furniture(tr) -> str:
    out = []
    beds = tr["bed"]
    # ---- wards: bedside tables between beds, a chair and an IV stand per bay
    by_room: dict[str, list] = {}
    for p in beds:
        room = _room_at(*p.xy)
        if room:
            by_room.setdefault(room, []).append(p)
    k = 0
    for room, ps in by_room.items():
        x0, y0, x1, y1 = _room_rect(room)
        for wall_x, side in ((x0, "W"), (x1, "E")):
            col = sorted((p for p in ps if abs(p.xy[0] - wall_x) < 2.5), key=lambda p: p.xy[1])
            if len(col) < 2:
                continue
            inward = 1.0 if side == "W" else -1.0
            for a, b in zip(col, col[1:]):
                ymid = (a.xy[1] + b.xy[1]) / 2
                out.append(_prop(f"bedside_{k}", "bedsidetable", wall_x + inward * 0.45, ymid,
                                 yaw=0.0 if side == "W" else math.pi))
                out.append(_prop(f"chair_{k}", "chair", wall_x + inward * 1.35, ymid,
                                 yaw=(math.pi / 2 if side == "W" else -math.pi / 2)))
                k += 1
            # one IV stand per column, just outside the column's first or last
            # bed (never between beds - that is where the tables are)
            first, last = col[0], col[-1]
            if first.xy[1] - 0.72 - 0.25 > y0 + 0.2:
                iy = first.xy[1] - 0.72
            elif last.xy[1] + 0.72 + 0.25 < y1 - 0.2:
                iy = last.xy[1] + 0.72
            else:
                iy = None
            if iy is not None:
                out.append(_prop(f"ivstand_{k}", "ivstand", wall_x + inward * 0.65, iy))
        # cabinets along the wall opposite the door (north wards: N wall, south wards: S wall)
        if y0 > 15:
            out.append(_prop(f"cabinet_{room}", "metalcabinet", x0 + 3.6, y1 - 0.34))
            out.append(_prop(f"rack_{room}", "storagerack", x1 - 3.6, y1 - 0.46))
        else:
            out.append(_prop(f"cabinet_{room}", "metalcabinet", x0 + 3.0, y0 + 0.34))
    # ---- emergency room: trolley beds, X-ray, surgical trolleys, wheelchair
    ex0, ey0, ex1, ey1 = _room_rect("ward_icu_emerg")
    out.append(_prop("er_trolley_0", "trolleybedpatient", ex0 + 1.25, 5.0))
    out.append(_prop("er_trolley_1", "trolleybedpatient", ex0 + 1.25, 10.5))
    out.append(_prop("er_xray", "xraymachine", ex1 - 1.40, ey0 + 1.30, yaw=math.pi / 2))
    out.append(_prop("er_surg_0", "surgicaltrolley", ex1 - 0.45, 7.0, yaw=math.pi / 2))
    out.append(_prop("er_surg_1", "surgicaltrolley", ex1 - 0.45, 9.3, yaw=math.pi / 2))
    out.append(_prop("er_cabinet", "metalcabinet", 4.5, ey0 + 0.34))
    out.append(_prop("er_wheelchair", "patientwheelchair", ex0 + 2.1, 11.6, yaw=0.0))
    out.append(_prop("er_surgeon", "opscrubs", ex1 - 2.2, 5.6, yaw=math.pi))
    # ---- prayer room: a shoe rack just inside, nothing on the floor otherwise
    px0, py0, px1, py1 = _room_rect("prayer")
    out.append(_prop("prayer_rack", "storagerack", px0 + 1.1, py1 - 0.46))
    # ---- corridor: the six trained overflow beds already line the north wall
    # (x = 3.5, 10.5, 15, 23.5, 27, 30.8); the clutter goes in the gaps between
    # them and along the south wall, never in a doorway or a walking lane
    x0, y0, x1, y1 = L.CORRIDOR
    out.append(_prop("corr_wheelchair", "patientwheelchair", 12.8, y1 - 0.42, yaw=0.0))
    out.append(_prop("corr_ivstand", "ivstand", 16.35, y1 - 0.55))
    out.append(_prop("corr_cart", "surgicaltrolley", 27.0, y0 + 0.30, yaw=0.0))
    out.append(_prop("corr_rack", "storagerack", 28.9, y1 - 0.47, yaw=0.0))
    # ---- the nurse station proper: a counter spanning the corridor corner, the
    # trained desk box sits inside its footprint; a nurse standing at its end
    out.append(_prop("nurse_station", "nursedesk", 37.4, 16.40, yaw=0.0))
    out.append(_prop("nurse_standing", "nurse", 39.45, 16.55, yaw=math.pi))
    return "".join(out)




# ------------------------------------------------- realism v2 (2026-10-16) --
# Visibility split. gz renders both the cameras and the LD19 (gpu_lidar), and
# each visual can carry visibility flags that sensors filter with a mask
# (measured: a flag-2 box was invisible to a camera masked 0x3FFFFFFD and seen
# by the lidar; a flag-4 box the other way round). So:
#   LIDAR_ONLY  (2)  the trained collision shapes where they would look wrong on
#                    camera - the 0.5 m bed boxes under the real bed meshes, the
#                    standing people's cylinders inside the staff meshes
#   CAMERA_ONLY (4)  realism that must reach the floor or stand proud of a wall
#                    - door frames, open door leaves, beds, staff - which the
#                    lidar must not see, so the scan stays the trained one
# Unflagged visuals are seen by both and still obey the lidar-band rule.
# The robot's LD19 masks out CAMERA_ONLY; every figure/video camera masks out
# LIDAR_ONLY. (The gz GUI camera has no mask, so the GUI shows both.)
LIDAR_ONLY, CAMERA_ONLY = 2, 4
DOOR_TOP, FRAME_W, FRAME_D = 2.10, 0.08, 0.02      # door head, architrave width, stand-proud


def _flagged(xml: str, flag: int) -> str:
    """Add visibility flags to every <visual> of a model snippet."""
    return xml.replace('<visual name="v">', f'<visual name="v"><visibility_flags>{flag}</visibility_flags>')


def walls_v2(tr, tier) -> str:
    """TRAINED footprint, one full-height visual: each wall is a single
    3 m mesh over the trained collision box (copied verbatim), painted at
    physical scale (two-tone, dado line, skirting). The lidar sees the same
    footprint as the trained box - height does not reach the scan plane."""
    out = []
    for p in tr["wall"]:
        idx = int(p.name.rsplit("_", 1)[1])
        if tier == "open" and idx >= 4:
            continue
        cx, cy, cz = p.pose[:3]
        sx, sy, sz = p.box
        obj = T.wall_mesh(p.name, sx, sy, 0.0, WALL_VIS_H)
        out.append(f"""
    <model name="{p.name}">
      <static>true</static>
      <pose>{cx} {cy} 0 0 0 0</pose>
      <link name="link">
        <collision name="c"><pose>0 0 {cz} 0 0 0</pose><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
        <visual name="v"><cast_shadows>true</cast_shadows>
          <geometry><mesh><uri>{GEN_MESH}/{obj}</uri></mesh></geometry></visual>
      </link>
    </model>""")
    return "".join(out)


def beds_v2(tr) -> str:
    """Occupied hospital beds (OpenRobotics Fuel assets) on CAMERA; the trained
    0.5 m box, with its collision, on the LIDAR only - so the robot sees and hits
    exactly the trained bed while every image shows a real bed."""
    out = []
    for i, p in enumerate(tr["bed"]):
        cx, cy, cz = p.pose[:3]
        sx, sy, sz = p.box
        out.append(_flagged(_box(p.name, cx, cy, cz, sx, sy, sz, _flat(0.72, 0.74, 0.77), collide=True), LIDAR_ONLY))
        along_x = sx > sy
        yaw = 0.0 if along_x else math.pi / 2
        tx, ty = (sx, sy) if along_x else (sy, sx)
        room = _room_at(cx, cy)
        if room:                        # head against the wall, as the beds are laid out
            rect = _room_rect(room)
            rc = ((rect[0] + rect[2]) / 2) if along_x else ((rect[1] + rect[3]) / 2)
            if (cx if along_x else cy) < rc:
                yaw += math.pi
        model, obj = ("malepatientbed", "MalePatientBed.obj")
        if i % 4 == 3:
            model, obj = ("trolleybedpatient", "TrolleyBedPatient.obj")
        out.append(_flagged(_mesh_visual(f"{p.name}_real", model, obj, cx, cy, 0.0, tx, ty, 0.95, yaw=yaw),
                            CAMERA_ONLY))
    return "".join(out)


def bed_panels(tr, tier) -> str:
    """FREE: the medical service panel and monitor on the wall behind each
    ward bed's head (only where that wall exists)."""
    out = []
    for p in tr["bed"]:
        cx, cy, cz = p.pose[:3]
        sx, sy, sz = p.box
        room = _room_at(cx, cy)
        if room is None:                # corridor overflow beds: no wall at the head
            continue
        rect = _room_rect(room)
        along_x = sx > sy
        rcx, rcy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
        hs = (1.0 if cx > rcx else -1.0) if along_x else (1.0 if cy > rcy else -1.0)
        if along_x:
            px, py, psx, psy = cx + hs * (sx / 2 + 0.11), cy, 0.06, min(sy * 1.1, 1.0)
            head_wall = px
            exterior = abs(head_wall) < 0.3 or abs(head_wall - L.MAP_W) < 0.3
        else:
            px, py, psx, psy = cx, cy + hs * (sy / 2 + 0.11), min(sx * 1.1, 1.0), 0.06
            exterior = abs(py) < 0.3 or abs(py - L.MAP_H) < 0.3
        if tier == "open" and not exterior:
            continue
        _free(f"{p.name}_panel", 0.95, 1.35)
        out.append(_box(f"{p.name}_panel", px, py, 1.15, psx, psy, 0.40, _flat(0.92, 0.93, 0.90), shadows=False))
        _free(f"{p.name}_monitor", 1.45, 1.78)
        out.append(_box(f"{p.name}_monitor", px, py, 1.615,
                        psx * 0.55 if psx > psy else 0.05, psy * 0.55 if psy > psx else 0.05, 0.33,
                        _flat(0.12, 0.30, 0.34, emis=0.35), shadows=False))
    return "".join(out)


def desk_v2(tr, hidden=False) -> str:
    """TRAINED box (seen by both) dressed as a nurse-station counter: wood
    laminate front, a solid-surface top and a monitor - all above the band.
    hidden=True (furnished tier, where the full nurse-station counter stands
    over it): the trained box goes LIDAR_ONLY and the dressing is dropped."""
    out = []
    for p in tr.get("nurse_desk", []):
        cx, cy, cz = p.pose[:3]
        sx, sy, sz = p.box
        top = cz + sz / 2
        box = _box(p.name, cx, cy, cz, sx, sy, sz, _pbr("door.png", rough=0.55), collide=True)
        if hidden:
            out.append(_flagged(box, LIDAR_ONLY))
            continue
        out.append(box)
        _free("nurse_desk_top", top, top + 0.05)
        out.append(_box("nurse_desk_top", cx, cy, top + 0.025, sx + 0.10, sy + 0.10, 0.05, _flat(0.93, 0.93, 0.91)))
        _free("nurse_desk_monitor", top + 0.05, top + 0.42)
        out.append(_box("nurse_desk_monitor", cx, cy + 0.12, top + 0.24, 0.48, 0.03, 0.30, _flat(0.06, 0.06, 0.07)))
        out.append(_box("nurse_desk_screen", cx, cy + 0.104, top + 0.25, 0.44, 0.004, 0.26,
                        _flat(0.20, 0.42, 0.55, emis=0.6), shadows=False))
        _free("nurse_desk_screen", top + 0.12, top + 0.38)
    return "".join(out)


def people_v2(tr) -> str:
    """The trained standing people: staff meshes on CAMERA, the trained
    cylinder (visual LIDAR only, collision full height) for the robot."""
    out = []
    looks = [("nurse", math.pi), ("opscrubs", 0.0), ("nurse", -math.pi / 2)]
    for i, p in enumerate(tr["person"]):
        cx, cy = p.xy
        r, h = p.cyl
        model, yaw = looks[i % len(looks)]
        obj, nx, ny, nz = NATURAL[model]
        px, py, pz, kx, ky, kz, _ox, _oy = _mesh_fit(model, obj, cx, cy, 0.0, nx, ny, nz, yaw)
        out.append(f"""
    <model name="{p.name}">
      <static>true</static>
      <pose>{cx} {cy} 0 0 0 0</pose>
      <link name="link">
        <collision name="c"><pose>0 0 {h / 2} 0 0 0</pose>
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry></collision>
        <visual name="lidar"><visibility_flags>{LIDAR_ONLY}</visibility_flags><pose>0 0 {h / 2} 0 0 0</pose>
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>{_flat(0.20, 0.32, 0.36)}</visual>
        <visual name="v"><visibility_flags>{CAMERA_ONLY}</visibility_flags>
          <pose>{px - cx:.4f} {py - cy:.4f} {pz:.4f} 0 0 {yaw}</pose><cast_shadows>true</cast_shadows>
          <geometry><mesh><uri>{FUEL_MESH}/{model}/{obj}</uri><scale>{kx:.5f} {ky:.5f} {kz:.5f}</scale></mesh></geometry>
        </visual>
      </link>
    </model>""")
    return "".join(out)


_GAPS = None


def door_gaps():
    """[(room, side, wall coordinate, gap_lo, gap_hi)] for every doorway, from the
    TRAINED walls: the openings are 1.8 m wide there, not layout.DOOR_W (1.0) -
    drawing 1.0 m frames left a 0.4 m hollow beside each one (2026-10-17)."""
    global _GAPS
    if _GAPS is not None:
        return _GAPS
    tr = load_trained()
    walls = [(p.pose[0], p.pose[1], p.box[0], p.box[1]) for p in tr["wall"]]
    out = []
    for (x0, y0, x1, y1, name, doors) in L.ROOMS:
        for side, c in doors:
            if side in ("N", "S"):
                w = y1 if side == "N" else y0
                segs = [(cx - sx / 2, cx + sx / 2) for cx, cy, sx, sy in walls if sx > sy and abs(cy - w) < 0.15]
            else:
                w = x1 if side == "E" else x0
                segs = [(cy - sy / 2, cy + sy / 2) for cx, cy, sx, sy in walls if sy > sx and abs(cx - w) < 0.15]
            lo = max(b for a, b in segs if b <= c + 0.01)
            hi = min(a for a, b in segs if a >= c - 0.01)
            out.append((name, side, w, lo, hi))
    _GAPS = out
    return out


def doorways_v2(tier) -> str:
    """Every doorway as built, fitted to the TRAINED opening (1.8 m - a bed-width
    double door): an architrave that wraps the two wall ends from the floor to
    the head (CAMERA_ONLY - it stands 2 cm proud of the wall faces), the wall
    section above the door (the lintel, painted like the walls, above the band),
    and two door leaves opened flat against the room-side wall (CAMERA_ONLY, no
    collision: they lie against the wall). The opening the policies were trained
    on is exactly as wide as before; nothing hangs in it."""
    if tier == "open":
        return ""
    out = []
    wood, frame = _pbr("door.png", rough=0.6), _flat(0.86, 0.86, 0.84)
    d = L.WALL_T + 2 * FRAME_D
    leaf_w = 0.86
    for n, (name, side, w, lo, hi) in enumerate(door_gaps()):
        horiz = side in ("N", "S")
        into = (-1.0 if side in ("N", "E") else 1.0)           # towards the room interior
        c, span = (lo + hi) / 2, hi - lo

        def at(u, v_off, su, sv):                               # (along-wall, across) -> world box
            return (u, w + v_off, su, sv) if horiz else (w + v_off, u, sv, su)
        for k, u in enumerate((lo - FRAME_W / 2 + 0.01, hi + FRAME_W / 2 - 0.01)):
            px, py, sx, sy = at(u, 0.0, FRAME_W, d)
            out.append(_flagged(_box(f"door_{n}_jamb{k}", px, py, DOOR_TOP / 2, sx, sy, DOOR_TOP, frame), CAMERA_ONLY))
        px, py, sx, sy = at(c, 0.0, span + 2 * FRAME_W, d)
        out.append(_flagged(_box(f"door_{n}_head", px, py, DOOR_TOP + 0.05, sx, sy, 0.10, frame), CAMERA_ONLY))
        lx, ly, lsx, lsy = at(c, 0.0, span, L.WALL_T)
        obj = T.wall_mesh(f"lintel_{n}", lsx, lsy, DOOR_TOP, WALL_VIS_H)
        _free(f"door_{n}_lintel", DOOR_TOP, WALL_VIS_H)
        out.append(f"""
    <model name="door_{n}_lintel">
      <static>true</static><pose>{lx} {ly} 0 0 0 0</pose>
      <link name="link"><visual name="v"><cast_shadows>true</cast_shadows>
        <geometry><mesh><uri>{GEN_MESH}/{obj}</uri></mesh></geometry></visual></link>
    </model>""")
        off = into * (L.WALL_T / 2 + 0.035)
        for k, u in enumerate((lo - FRAME_W - leaf_w / 2, hi + FRAME_W + leaf_w / 2)):
            fx, fy, fsx, fsy = at(u, off, leaf_w, 0.04)
            out.append(_flagged(_box(f"door_{n}_leaf{k}", fx, fy, 0.01 + (DOOR_TOP - 0.02) / 2, fsx, fsy,
                                     DOOR_TOP - 0.02, wood), CAMERA_ONLY))       # 1 cm under-door gap
            gx, gy, gsx, gsy = at(u, off + into * 0.006, 0.26, 0.04)
            out.append(_flagged(_box(f"door_{n}_glass{k}", gx, gy, 1.45, gsx, gsy, 0.50,
                                     _flat(0.70, 0.82, 0.88, emis=0.2)), CAMERA_ONLY))
            kx, ky, ksx, ksy = at(u, off + into * 0.006, leaf_w - 0.06, 0.04)
            out.append(_flagged(_box(f"door_{n}_kick{k}", kx, ky, 0.14, ksx, ksy, 0.24, _flat(0.70, 0.72, 0.74)),
                                CAMERA_ONLY))
    return "".join(out)

def windows_v2(tr) -> str:
    """High windows on the inside face of the four exterior walls (FREE, above
    the band and above the bed-head service panels)."""
    out = []
    z0, z1 = 1.85, 2.60
    inward = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}
    for p in tr["wall"][:4]:
        idx = int(p.name.rsplit("_", 1)[1])
        cx, cy, _ = p.pose[:3]
        sx, sy, _ = p.box
        ix, iy = inward[idx]
        horiz = sx > sy
        span = sx if horiz else sy
        n = max(2, int(span // 5.0))
        for k in range(n):
            f = (k + 0.5) / n
            wx = (cx - sx / 2 + f * sx) if horiz else cx + ix * (L.WALL_T / 2 + 0.012)
            wy = cy + iy * (L.WALL_T / 2 + 0.012) if horiz else (cy - sy / 2 + f * sy)
            gw = min(1.8, span / n * 0.45)
            _free(f"{p.name}_win{k}", z0 - 0.05, z1 + 0.05)
            out.append(_box(f"{p.name}_winf{k}", wx, wy, (z0 + z1) / 2,
                            gw + 0.10 if horiz else 0.024, 0.024 if horiz else gw + 0.10, z1 - z0 + 0.10,
                            _flat(0.82, 0.83, 0.84), shadows=False))
            out.append(_box(f"{p.name}_win{k}", wx + ix * 0.008, wy + iy * 0.008, (z0 + z1) / 2,
                            gw if horiz else 0.012, 0.012 if horiz else gw, z1 - z0,
                            _flat(0.72, 0.85, 0.95, emis=0.65), shadows=False))
    return "".join(out)


def corridor_v2(tier) -> str:
    """Wall-mounted corridor details (FREE, all flush on a wall face): clock,
    notice boards, hand-sanitiser dispensers beside each door, fire
    extinguisher cabinets; and the painted wayfinding lines on the floor."""
    if tier == "open":
        return ""
    out = []
    x0, y0, x1, y1 = L.CORRIDOR
    face_n = y1 - L.WALL_T / 2 - 0.025            # corridor face of the north corridor wall
    face_s = y0 + L.WALL_T / 2 + 0.025
    # x positions are on solid wall: clear of every 1.8 m doorway and its frame
    # (the clock used to sit at x 20 - inside General Ward 1's opening)
    _free("clock", 1.95, 2.25)
    out.append(_box("clock", 24.0, face_n, 2.10, 0.42, 0.05, 0.30, _flat(0.96, 0.96, 0.94, emis=0.25), shadows=False))
    for k, cx in enumerate((10.0, 28.0)):
        _free(f"notice_{k}", 1.15, 1.85)
        out.append(_box(f"notice_{k}", cx, face_n, 1.50, 1.30, 0.05, 0.70, _flat(0.90, 0.88, 0.80), shadows=False))
    for n, (name, side, w, lo, hi) in enumerate(door_gaps()):
        out_dir = 1.0 if side in ("N", "E") else -1.0           # corridor side of that wall
        if side in ("N", "S"):
            dx, dy, sx_, sy_ = hi + 0.35, w + out_dir * (L.WALL_T / 2 + 0.05), 0.16, 0.10
        else:
            dx, dy, sx_, sy_ = w + out_dir * (L.WALL_T / 2 + 0.05), hi + 0.35, 0.10, 0.16
        _free(f"dispenser_{n}", 1.08, 1.38)
        out.append(_box(f"dispenser_{n}", dx, dy, 1.23, sx_, sy_, 0.30, _flat(0.94, 0.95, 0.96), shadows=False))
    for k, fx in enumerate((10.0, 26.0)):
        _free(f"extinguisher_{k}", 0.95, 1.65)
        out.append(_box(f"extinguisher_{k}", fx, y0 + L.WALL_T / 2 + 0.08, 1.30, 0.30, 0.16, 0.70,
                        _flat(0.85, 0.12, 0.10), shadows=False))
    for k, (yy, rgb) in enumerate(((14.35, (0.92, 0.26, 0.18)), (14.65, (0.20, 0.45, 0.85)))):
        _free(f"wayline_{k}", 0.001, 0.004)
        out.append(_box(f"wayline_{k}", (x0 + x1) / 2, yy, 0.0025, x1 - x0 - 1.0, 0.11, 0.003, _flat(*rgb), shadows=False))
    return "".join(out)


def art_v2(tier) -> str:
    """FREE: a framed picture on each room's far wall (walled tiers only)."""
    if tier == "open":
        return ""
    out = []
    for (x0, y0, x1, y1, name, doors) in L.ROOMS:
        side = doors[0][0]
        rx, ry = (x0 + x1) / 2, (y0 + y1) / 2
        off = L.WALL_T / 2 + 0.02
        px, py, psx, psy = {"S": (rx, y1 - off, 0.80, 0.03), "N": (rx, y0 + off, 0.80, 0.03),
                            "W": (x1 - off, ry, 0.03, 0.80), "E": (x0 + off, ry, 0.03, 0.80)}[side]
        r, g, b = L.ROOM_COLOR[name]
        _free(f"art_{name}", 1.30, 1.95)
        out.append(_box(f"art_{name}_frame", px, py, 1.62, psx + (0.08 if psx > psy else 0), psy + (0.08 if psy > psx else 0),
                        0.62, _flat(0.25, 0.20, 0.16), shadows=False))
        out.append(_box(f"art_{name}", px + (0.006 if psx < psy else 0) * (1 if side == "W" else -1),
                        py + (0.006 if psy < psx else 0) * (1 if side == "S" else -1), 1.62,
                        psx if psx > psy else 0.02, psy if psy > psx else 0.02, 0.50,
                        _flat(0.55 + 0.3 * r, 0.55 + 0.3 * g, 0.55 + 0.3 * b), shadows=False))
    return "".join(out)


LIGHT_SPOTS = ([((x0 + x1) / 2, (y0 + y1) / 2, n) for (x0, y0, x1, y1, n, _d) in L.ROOMS]
               + [(float(x), 15.0, f"corridor_{i}") for i, x in enumerate(range(3, 40, 6))]
               + [(18.5, float(y), f"south_{i}") for i, y in enumerate(range(4, 13, 4))])


def lights_v2() -> str:
    """The gz lights only. The luminaire panels belong to the ceiling, which the
    run worlds leave open for top-down views - capture_figures.py --ceiling adds
    both together, so no light fitting ever floats in mid-air."""
    return "".join(f"""
    <light type="point" name="light_{tag}">
      <pose>{cx} {cy} 2.62 0 0 0</pose>
      <diffuse>0.80 0.79 0.75 1</diffuse>
      <specular>0.16 0.16 0.15 1</specular>
      <attenuation><range>14</range><constant>0.38</constant>
        <linear>0.07</linear><quadratic>0.011</quadratic></attenuation>
      <cast_shadows>false</cast_shadows>
    </light>""" for (cx, cy, tag) in LIGHT_SPOTS)


def luminaires_sdf() -> str:
    """Recessed ceiling luminaires, flush under a ceiling at CEIL_Z (for --ceiling shots)."""
    return "".join(_box(f"lum_{tag}", cx, cy, CEIL_Z - 0.012, 1.20, 0.30, 0.024,
                        _flat(1.0, 0.98, 0.92, emis=1.0), shadows=False) for (cx, cy, tag) in LIGHT_SPOTS)


# ------------------------------------------------------------------- world --
HEADER = """<?xml version="1.0"?>
<!-- GENERATED by sim/gen_worlds.py - do not hand-edit.
     Tier: {tier} ({tier_desc})
     Collision geometry is copied verbatim from worlds/hospital.world (the world
     the runs_v5 benchmark numbers refer to). gz-sim Harmonic, SDF 1.9.
     Photoreal pass obeys the lidar-plane rule: the LD19 scans one horizontal
     plane at z={lz:.4f} m and gz's gpu_lidar sees VISUALS, not collisions, so
     every decorative visual lies wholly below {lo} m or above {hi} m. -->
<sdf version="1.9">
  <world name="hospital">
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>

    <physics type="dart">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>0</real_time_factor>
      <real_time_update_rate>0</real_time_update_rate>
    </physics>

    <scene>
      <ambient>0.52 0.53 0.56 1</ambient>
      <background>0.70 0.76 0.84 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>

    <light type="directional" name="sun">
      <pose>12 10 14 0 0 0</pose>
      <diffuse>0.70 0.69 0.66 1</diffuse>
      <specular>0.16 0.16 0.16 1</specular>
      <direction>-0.45 0.30 -0.84</direction>
      <cast_shadows>true</cast_shadows>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
        </collision>
        <visual name="visual">
          <pose>0 0 -0.03 0 0 0</pose>
          <cast_shadows>false</cast_shadows>
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          {ground_mat}
        </visual>
      </link>
    </model>
"""


def build(tier: str, ceiling: bool, tr) -> str:
    """tier: walled | open | reference.

    'reference' re-emits ONLY the trained collision geometry in gz form, with no
    realism pass at all. It is the A/B control for verify_scan_invariance.py: if
    a probe sees the same ranges in 'reference' and 'walled', the photoreal pass
    provably did not touch what the policy observes."""
    desc = {"walled": "all 40 walls - the seven enclosed rooms",
            "open": "outer shell only - rooms are semantic zones, not physical enclosures",
            "real": "identical to walled since realism v2 (every tier carries the real assets); kept as an alias",
            "furnished": "real + floor-standing furniture WITH collision that the map does not "
                         "contain (unmapped clutter), staff meshes, prayer mats",
            "reference": "CONTROL: trained geometry only, no realism pass",
            "reference_open": "CONTROL for the open tier: outer shell + furniture only"}[tier]
    ground = _flat(0.62, 0.63, 0.61)       # outside the building; floors are their own boxes
    if tier in ("reference", "reference_open"):
        body = _reference_body(tr, "open" if tier == "reference_open" else "walled")
        return (HEADER.format(tier=tier, tier_desc=desc, lz=LIDAR_Z, lo=LIDAR_LO,
                              hi=LIDAR_HI, ground_mat=_flat(0.85, 0.85, 0.85))
                + body + "\n  </world>\n</sdf>\n")
    wall_tier = "walled" if tier in ("real", "furnished") else tier
    body = (walls_v2(tr, wall_tier) + floors(tr) + beds_v2(tr) + bed_panels(tr, wall_tier)
            + desk_v2(tr, hidden=(tier == "furnished")) + people_v2(tr) + doorways_v2(wall_tier)
            + (signage() if wall_tier != "open" else "")
            + windows_v2(tr) + corridor_v2(wall_tier) + art_v2(wall_tier) + prayer_mats()
            + (furniture(tr) if tier == "furnished" else "")
            + lights_v2())
    return (HEADER.format(tier=tier, tier_desc=desc, lz=LIDAR_Z,
                          lo=LIDAR_LO, hi=LIDAR_HI, ground_mat=ground)
            + body + "\n  </world>\n</sdf>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceiling", action="store_true",
                    help="close the suspended ceiling (beauty shots; hides top-down views)")
    ap.add_argument("--out", default=OUT, help="output directory (default: the package worlds/)")
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)

    T.main()
    tr = load_trained()
    print(f"trained geometry: " + ", ".join(f"{k} x{len(v)}" for k, v in sorted(tr.items())))
    os.makedirs(out_dir, exist_ok=True)
    for tier in ("walled", "open", "real", "furnished", "reference", "reference_open"):
        _violations.clear()
        _clutter.clear()
        sdf = build(tier, a.ceiling, tr)
        if _violations:
            raise SystemExit("LIDAR-BAND VIOLATION:\n  " + "\n  ".join(_violations))
        if _clutter and tier != "furnished":
            raise SystemExit(f"{tier}: unmapped clutter is only allowed in the furnished tier: "
                             + ", ".join(_clutter))
        path = os.path.join(out_dir, f"hospital_{tier}.world")
        with open(path, "w") as f:
            f.write(sdf)
        extra = f", {len(_clutter)} unmapped props cross the scan plane BY DESIGN" if _clutter else ""
        print(f"wrote {os.path.relpath(path, HERE)}  "
              f"({sdf.count('<model name=')} models, {sdf.count('<light ')} lights{extra})")


if __name__ == "__main__":
    main()
