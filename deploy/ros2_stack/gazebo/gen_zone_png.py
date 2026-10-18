"""Generate the semantic zone PNG (for controller_node's SemanticMap: context
zone_onehot + cult_dist) from the same configs/dmch_hospital.yaml used by
gen_gazebo_world.py, so the policy's context input matches the Gazebo world.
"""
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "dmch_hospital.yaml"
OUT = Path(__file__).resolve().parent / "worlds" / "hospital_zones.png"
RES = 0.05

ZONES = ["walkable", "corridor", "nurse_station", "pharmacy",
         "ward_icu_emerg", "prayer", "male_ward", "female_ward"]


def bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    W, H = cfg["map"]["width_m"], cfg["map"]["height_m"]
    rooms = [(bbox(r["polygon"]), ZONES.index(r["type"])) for r in cfg.get("rooms", [])
             if r["type"] in ZONES]

    px_w, px_h = int(round(W / RES)), int(round(H / RES))
    img = np.zeros((px_h, px_w), dtype=np.uint8)
    for r in range(px_h):
        y = (px_h - 1 - r) * RES
        for c in range(px_w):
            x = c * RES
            for (x0, y0, x1, y1), idx in rooms:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    img[r, c] = idx
                    break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(OUT)
    print("wrote", OUT, img.shape)
    for name in ZONES:
        print(f"  {ZONES.index(name)}: {name} -> {(img == ZONES.index(name)).sum()} px")


if __name__ == "__main__":
    main()
