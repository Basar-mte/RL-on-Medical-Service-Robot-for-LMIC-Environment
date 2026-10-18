"""Minimum clearance of every pedestrian route from the walled map's obstacles
(a walker's body is 0.22 m; routes need > 0.30 m to walk without scraping).

    python3 route_clearance.py [--level crowd45] [--below 0.30]
"""
import argparse, math, os, sys
import numpy as np
import yaml
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hospitalnav_deploy", "hospitalnav_deploy"))
import ped_routes as P  # noqa: E402

WIN = 0.8


def load(tier="walled"):
    wd = os.path.join(HERE, "..", "hospitalnav_deploy", "worlds")
    meta = yaml.safe_load(open(os.path.join(wd, f"hospital_map_{tier}.yaml")))
    img = np.array(Image.open(os.path.join(wd, meta["image"])))[::-1]
    return img < 128, meta["resolution"], meta["origin"][:2]


def clearance(occ, r, origin, x, y):
    ox, oy = origin
    c, rr = int((x - ox) / r), int((y - oy) / r)
    k = int(WIN / r)
    sub = occ[max(0, rr - k):rr + k + 1, max(0, c - k):c + k + 1]
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return WIN
    cx = ox + (xs + max(0, c - k) + 0.5) * r
    cy = oy + (ys + max(0, rr - k) + 0.5) * r
    return float(np.min(np.hypot(cx - x, cy - y)) - r / 2)


def route_min(occ, r, origin, pts):
    worst = (WIN, None)
    for a, b in zip(pts, pts[1:] + pts[:1]):
        L = math.dist(a, b)
        for k in range(int(L / 0.05) + 1):
            f = k * 0.05 / max(L, 1e-9)
            x, y = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            c = clearance(occ, r, origin, x, y)
            if c < worst[0]:
                worst = (c, (round(x, 2), round(y, 2)))
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="crowd45")
    ap.add_argument("--below", type=float, default=0.30)
    a = ap.parse_args()
    occ, r, origin = load()
    bad = 0
    for n in P.LEVELS[a.level]:
        c, at = route_min(occ, r, origin, P.ROUTES[n][2])
        if c < a.below:
            bad += 1
            print(f"{n:22s} min clearance {c:.2f} m at {at}")
    print(f"{bad} of {len(P.LEVELS[a.level])} routes below {a.below} m")
