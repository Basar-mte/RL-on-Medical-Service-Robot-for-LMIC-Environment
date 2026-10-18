"""Generates the 45-pedestrian crowd for ped_routes (level 'crowd45'): the 8
designed walkers plus 37 generated ones, written to
hospitalnav_deploy/hospitalnav_deploy/crowd_routes.py as plain data.

Every generated loop is planned on the walled ward's own occupancy map with the
trained route planner's Dijkstra (no cultural cost), with obstacles inflated by
a pedestrian's half-width + margin, so a walker never brushes a bed, a wall or
the nurse desk, and enters rooms through their real 1.8 m doorways. A 1 m disk
around the robot's spawn is kept clear so nobody shoves the parked robot before
its run starts. Seeded: the crowd is the same every time.

    python3 gen_crowd_routes.py
"""
from __future__ import annotations
import math
import os
import random
import sys

import numpy as np
from PIL import Image
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "hospitalnav_deploy")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "hospitalnav_deploy"))
import route_planner as RP          # noqa: E402
import layout as L                  # noqa: E402
import gen_worlds as G              # noqa: E402

CLEAR = 0.32                        # inflation: body radius 0.22 + 0.10 margin
SPAWN, SPAWN_R = (37.0, 15.0), 1.0
RES = 0.1


def grid():
    meta = yaml.safe_load(open(os.path.join(PKG, "worlds", "hospital_map_walled.yaml")))
    img = np.array(Image.open(os.path.join(PKG, "worlds", meta["image"])))[::-1]
    blocked = img < 128
    f = int(round(RES / meta["resolution"]))
    blocked = RP.downsample_blocked(blocked, f)
    ox, oy = meta["origin"][:2]
    H, W = blocked.shape
    ys, xs = np.mgrid[0:H, 0:W]
    cx, cy = ox + (xs + 0.5) * RES, oy + (ys + 0.5) * RES
    spawn = (cx - SPAWN[0]) ** 2 + (cy - SPAWN[1]) ** 2 < SPAWN_R ** 2
    infl = RP.dilate(blocked, int(math.ceil(CLEAR / RES)))
    return infl | spawn, (ox, oy)


def free(B, origin, x, y):
    c, r = int((x - origin[0]) / RES), int((y - origin[1]) / RES)
    return 0 <= r < B.shape[0] and 0 <= c < B.shape[1] and not B[r, c]


def seg_free(B, origin, a, b):
    n = max(2, int(math.dist(a, b) / (RES / 2)))
    return all(free(B, origin, a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n + 1))


def simplify(B, origin, pts, tol=0.12):
    """Douglas-Peucker that only removes a point if the shortcut stays free."""
    if len(pts) < 3:
        return pts
    a, b = np.array(pts[0]), np.array(pts[-1])
    ab = b - a
    L2 = max(float(ab @ ab), 1e-9)
    d = [abs((ab[0] * (a[1] - p[1]) - ab[1] * (a[0] - p[0]))) / math.sqrt(L2) for p in pts]
    i = int(np.argmax(d))
    if d[i] <= tol and seg_free(B, origin, pts[0], pts[-1]):
        return [pts[0], pts[-1]]
    i = max(1, min(len(pts) - 2, i))
    return simplify(B, origin, pts[:i + 1], tol)[:-1] + simplify(B, origin, pts[i:], tol)


def plan(B, origin, a, b):
    pl = RP.RoutePlanner(B, RES, origin, None, 0.0, False, 2.5, False)
    if not pl.set_goal(*b):
        return None
    route = pl.route_xy(*a)
    if len(route) < 2:
        return None
    return simplify(B, origin, [tuple(a)] + route[1:-1] + [tuple(b)])


def room_points(B, origin, rng, room, k):
    x0, y0, x1, y1 = [r for r in L.ROOMS if r[4] == room][0][:4]
    out = []
    for _ in range(400):
        x, y = rng.uniform(x0 + 0.8, x1 - 0.8), rng.uniform(y0 + 0.8, y1 - 0.8)
        if free(B, origin, x, y):
            out.append((round(x, 2), round(y, 2)))
            if len(out) == k:
                break
    return out


def main():
    rng = random.Random(20261017)
    B, origin = grid()
    lanes = [13.75, 14.15, 14.55, 15.05, 15.45, 15.85]
    routes = {}
    models = ["nurse", "doctor", "visitor", "patient"]
    # 20 corridor walkers: out and back along a lane, turning at random points
    while len([n for n in routes if n.startswith("crowd_corr")]) < 20:
        i = len([n for n in routes if n.startswith("crowd_corr")])
        y1, y2 = rng.choice(lanes), rng.choice(lanes)
        xa, xb = rng.uniform(1.5, 12.0), rng.uniform(24.0, 35.5)
        if rng.random() < 0.5:
            xa, xb = rng.uniform(1.5, 20.0), rng.uniform(xa + 6.0, 35.5) if xa + 6 < 35.5 else 35.5
        a, b = (round(xa, 2), y1), (round(xb, 2), y2)
        if not (free(B, origin, *a) and free(B, origin, *b)):
            continue
        p1, p2 = plan(B, origin, a, b), plan(B, origin, b, a)
        if not p1 or not p2:
            continue
        model = models[i % 4]
        speed = round(rng.uniform(0.45, 0.65), 2) if model == "patient" else round(rng.uniform(0.75, 1.35), 2)
        routes[f"crowd_corr_{i:02d}"] = (model, speed, p1[:-1] + p2[:-1])
    # 17 room visitors: corridor -> through the doorway -> a point in the room -> back
    rooms = [r[4] for r in L.ROOMS]
    k = 0
    while k < 17:
        room = rooms[k % len(rooms)]
        inner = room_points(B, origin, rng, room, 1)
        if not inner:
            continue
        a = (round(rng.uniform(2.0, 35.0), 2), rng.choice(lanes))
        if not free(B, origin, *a):
            continue
        p1, p2 = plan(B, origin, a, inner[0]), plan(B, origin, inner[0], a)
        if not p1 or not p2:
            continue
        model = models[(k + 2) % 4]
        speed = round(rng.uniform(0.45, 0.6), 2) if model == "patient" else round(rng.uniform(0.7, 1.2), 2)
        routes[f"crowd_room_{k:02d}"] = (model, speed, p1[:-1] + p2[:-1])
        k += 1
    out = os.path.join(PKG, "hospitalnav_deploy", "crowd_routes.py")
    with open(out, "w") as f:
        f.write('"""GENERATED by sim/gen_crowd_routes.py - the 37 generated walkers of level crowd45."""\n')
        f.write("CROWD = {\n")
        for n, (m, sp, pts) in routes.items():
            f.write(f"    {n!r}: ({m!r}, {sp}, {[(round(x, 2), round(y, 2)) for x, y in pts]}),\n")
        f.write("}\n")
    print(f"wrote {os.path.relpath(out, HERE)}: {len(routes)} walkers, "
          f"{sum(len(v[2]) for v in routes.values())} waypoints")


if __name__ == "__main__":
    main()
