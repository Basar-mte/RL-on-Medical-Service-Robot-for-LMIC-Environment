"""Perception: LiDAR people tracker (node) + semantic-map lookup (library).

PeopleTracker: clusters /scan into leg-sized blobs, tracks them frame-to-frame
for velocity, and publishes the 5 nearest within 5 m as a flat Float32MultiArray
[dx,dy,vx,vy]*5 in the ROBOT frame (already normalized to [-1,1]).

SemanticMap: loads a colour-coded zone image aligned to the SLAM map and answers
zone_onehot(x,y) and dist_to_cultural(x,y) for the context vector.
"""
from __future__ import annotations
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

PED_R = 5.0          # PED_NEIGHBOR_RADIUS
PED_VMAX = 2.5       # PED_MAX_SPEED
K = 5                # K_NEIGHBORS


# ------------------------------- people tracker ------------------------------
class PeopleTracker(Node):
    def __init__(self):
        super().__init__("people_tracker")
        self.declare_parameter("eps", 0.25)          # cluster radius (m)
        self.declare_parameter("min_pts", 3)
        self.eps = float(self.get_parameter("eps").value)
        self.min_pts = int(self.get_parameter("min_pts").value)
        self.prev = []                               # [(x,y)] last-frame centroids
        self.create_subscription(LaserScan, "/scan", self._scan, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/people", 10)

    def _scan(self, s: LaserScan):
        r = np.asarray(s.ranges, float)
        ang = s.angle_min + np.arange(len(r)) * s.angle_increment
        ok = np.isfinite(r) & (r > 0.05) & (r < PED_R)
        pts = np.stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])], 1)  # robot frame
        clusters = self._dbscan(pts)
        cents = [c.mean(0) for c in clusters if len(c) >= self.min_pts and self._leg_sized(c)]
        # nearest-neighbour association to previous centroids -> velocity
        out = []
        for cx, cy in cents:
            vx = vy = 0.0
            if self.prev:
                px, py = min(self.prev, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
                if (px-cx)**2 + (py-cy)**2 < 0.5**2:
                    vx, vy = (cx-px)/0.1, (cy-py)/0.1     # ~10 Hz frames
            out.append((math.hypot(cx, cy), cx, cy, vx, vy))
        self.prev = cents
        out.sort(key=lambda t: t[0])
        arr = np.zeros((K, 4), np.float32)
        for i, (_, cx, cy, vx, vy) in enumerate(out[:K]):
            arr[i] = [cx/PED_R, cy/PED_R, vx/PED_VMAX, vy/PED_VMAX]
        m = Float32MultiArray(); m.data = np.clip(arr, -1, 1).flatten().tolist()
        self.pub.publish(m)

    def _leg_sized(self, c):
        span = c.max(0) - c.min(0)
        return 0.05 < max(span) < 0.8            # rough person/leg footprint

    def _dbscan(self, pts):
        n = len(pts); seen = np.zeros(n, bool); out = []
        for i in range(n):
            if seen[i]:
                continue
            d = np.linalg.norm(pts - pts[i], axis=1)
            nb = np.where(d < self.eps)[0]
            if len(nb) < self.min_pts:
                continue
            grp, stack = set(nb.tolist()), list(nb)
            while stack:
                j = stack.pop()
                if seen[j]:
                    continue
                seen[j] = True
                dj = np.where(np.linalg.norm(pts - pts[j], axis=1) < self.eps)[0]
                if len(dj) >= self.min_pts:
                    for k in dj:
                        if k not in grp:
                            grp.add(int(k)); stack.append(int(k))
            out.append(pts[list(grp)])
        return out


# ------------------------------- semantic map --------------------------------
_ZONES = ["walkable", "corridor", "general_ward_1", "general_ward_2", "ward_icu_emerg",
          "prayer", "male_ward", "female_ward", "general_ward_3"]  # matches zone_onehot(9)
_CULTURAL = {"prayer", "male_ward", "female_ward"}


class SemanticMap:
    """Loads a zone-label array aligned to the SLAM map (same origin/resolution).
    label_png: 8-bit indexed image where pixel value = index into _ZONES."""
    def __init__(self, label_png, resolution, origin_xy):
        from PIL import Image
        self.lab = np.array(Image.open(label_png))        # (H, W) uint8 zone indices
        self.res = float(resolution)
        self.ox, self.oy = float(origin_xy[0]), float(origin_xy[1])
        cult = np.isin(self.lab, [_ZONES.index(z) for z in _CULTURAL])
        self.cult_dt = _bfs_dist(cult) * self.res         # metres to nearest cultural cell

    def _cell(self, x, y):
        c = int((x - self.ox) / self.res); r = self.lab.shape[0] - 1 - int((y - self.oy) / self.res)
        return (min(max(r, 0), self.lab.shape[0]-1), min(max(c, 0), self.lab.shape[1]-1))

    def zone_onehot(self, x, y):
        r, c = self._cell(x, y)
        v = np.zeros(9, np.float32); v[int(self.lab[r, c])] = 1.0
        return v

    def cult_dist(self, x, y):
        r, c = self._cell(x, y)
        return min(float(self.cult_dt[r, c]) / 10.0, 1.0)  # /10 m, capped (matches sim)


def _bfs_dist(mask):
    from collections import deque
    H, W = mask.shape; dist = np.full((H, W), 1e9, np.float32); q = deque()
    for r, c in zip(*np.nonzero(mask)):
        dist[r, c] = 0; q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1; q.append((nr, nc))
    return dist


def main():
    rclpy.init(); rclpy.spin(PeopleTracker()); rclpy.shutdown()


if __name__ == "__main__":
    main()
