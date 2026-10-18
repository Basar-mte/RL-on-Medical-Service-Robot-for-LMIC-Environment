"""Procedural PBR textures for the hospital worlds.

Generated rather than downloaded so the worlds stay reproducible offline and
every asset in the paper has a source you can re-run. Each map is tileable:
the generators only use functions periodic in the tile index, so a <scale>d
material repeats without visible seams.

Output: hospitalnav_deploy/materials/textures/*.png
"""
from __future__ import annotations
import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = os.path.join(os.path.dirname(__file__), "..", "hospitalnav_deploy",
                   "materials", "textures")
RNG = np.random.default_rng(20261015)          # fixed: textures are deterministic


def _save(arr, name):
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
        os.path.join(OUT, name))
    print("wrote", name)


def _fbm(size, octaves=5, persistence=0.55):
    """Tileable value noise in [0,1] (bilinear upsample of periodic lattices)."""
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = 2 ** (o + 2)
        lat = RNG.random((n, n)).astype(np.float32)
        lat = np.pad(lat, ((0, 1), (0, 1)), mode="wrap")       # periodic
        img = np.asarray(Image.fromarray((lat * 255).astype(np.uint8))
                         .resize((size + size // n, size + size // n),
                                 Image.BICUBIC), np.float32)[:size, :size] / 255.0
        out += amp * img
        total += amp
        amp *= persistence
    return out / total


def floor_tile(name, base, grout, tile_px=128, size=512, speckle=14):
    """Vinyl/terrazzo hospital floor: square tiles, grout lines, fine speckle."""
    img = np.zeros((size, size, 3), np.float32)
    img[:] = base
    n = _fbm(size, octaves=6)
    img += ((n - 0.5) * speckle)[..., None]
    # per-tile tonal variation so the repeat is not obvious
    for ty in range(size // tile_px):
        for tx in range(size // tile_px):
            d = (RNG.random() - 0.5) * 7.0
            img[ty*tile_px:(ty+1)*tile_px, tx*tile_px:(tx+1)*tile_px] += d
    # grout lines on the tile boundaries
    g = np.zeros((size, size), bool)
    g[::tile_px, :] = True; g[:, ::tile_px] = True
    g[1::tile_px, :] = True; g[:, 1::tile_px] = True
    img[g] = grout
    # subtle specular-ish sheen streaks (buffed floor)
    sheen = _fbm(size, octaves=3)
    img += ((sheen - 0.5) * 6.0)[..., None]
    _save(img, name)


def wall_paint(name, base, size=512):
    """Painted plaster: low-frequency mottle + very fine roughness."""
    img = np.zeros((size, size, 3), np.float32)
    img[:] = base
    img += ((_fbm(size, octaves=6) - 0.5) * 9.0)[..., None]
    img += (RNG.normal(0, 1.4, (size, size)))[..., None]
    _save(img, name)


def wall_tiled(name, base, grout, tile_px=64, size=512):
    """Glazed wall tile for the clinical rooms (ER / wards wet wall)."""
    img = np.zeros((size, size, 3), np.float32)
    img[:] = base
    img += ((_fbm(size, octaves=5) - 0.5) * 5.0)[..., None]
    g = np.zeros((size, size), bool)
    g[::tile_px, :] = True; g[:, ::tile_px] = True
    img[g] = grout
    _save(img, name)


def ceiling_panel(name, size=512, panel_px=256):
    """Suspended acoustic ceiling: fissured mineral tile + T-bar grid."""
    img = np.zeros((size, size, 3), np.float32)
    img[:] = (236, 237, 232)
    fiss = _fbm(size, octaves=7, persistence=0.62)
    img += ((fiss - 0.5) * 20.0)[..., None]
    pin = (RNG.random((size, size)) > 0.982)
    img[pin] -= 16.0
    grid = np.zeros((size, size), bool)
    for k in (0, 1, 2):
        grid[k::panel_px, :] = True; grid[:, k::panel_px] = True
    img[grid] = (203, 205, 200)
    _save(img, name)


def linen(name, base, size=256):
    """Bed linen: woven weave + soft folds."""
    img = np.zeros((size, size, 3), np.float32)
    img[:] = base
    yy, xx = np.mgrid[0:size, 0:size]
    weave = (np.sin(xx * np.pi / 2) * np.sin(yy * np.pi / 2)) * 3.0
    img += weave[..., None]
    img += ((_fbm(size, octaves=4) - 0.5) * 12.0)[..., None]
    _save(img, name)


def door_wood(name, size=512):
    """Flush door leaf: vertical wood grain."""
    img = np.zeros((size, size, 3), np.float32)
    img[:] = (176, 142, 100)
    grain = _fbm(size, octaves=6)
    streak = np.repeat(grain[:, :1], size, axis=1) * 0.6 + grain * 0.4
    img += ((streak - 0.5) * 26.0)[..., None]
    _save(img, name)


def main():
    # corridor + walkable: warm grey terrazzo
    floor_tile("floor_corridor.png", (196, 193, 186), (168, 165, 158))
    # wards: soft blue-grey vinyl
    floor_tile("floor_ward.png",     (190, 199, 205), (164, 173, 180))
    # prayer room: warm sand (carpet-like, finer speckle)
    floor_tile("floor_prayer.png",   (205, 190, 158), (182, 166, 134), tile_px=256,
               speckle=22)
    # emergency: pale green clinical vinyl
    floor_tile("floor_emerg.png",    (186, 203, 190), (160, 178, 164))
    wall_paint("wall_paint.png",     (225, 226, 222))
    wall_paint("wall_accent.png",    (178, 199, 206))
    wall_tiled("wall_tile.png",      (214, 222, 221), (186, 195, 194))
    ceiling_panel("ceiling.png")
    linen("linen.png",  (232, 238, 242))
    linen("blanket.png", (124, 156, 178))
    door_wood("door.png")



# ---------------------------------------------------------------------------
# On-demand, physically-scaled maps.
#
# A gz <box> gets one 0..1 UV span per face, so a single tile image stretched
# over a 13 m wall reads as one absurd 13 m tile. These helpers instead bake a
# texture whose tile COUNT matches the surface's real size, so the repeat lands
# at the right physical scale with no extra geometry. Results are cached by
# (kind, tile count) and reused across surfaces.
# ---------------------------------------------------------------------------
_cache: dict[tuple, str] = {}

FLOOR_SPEC = {           # kind -> (base, grout, tile size in metres, speckle)
    "corridor": ((196, 193, 186), (168, 165, 158), 0.60, 14),
    "ward":     ((190, 199, 205), (164, 173, 180), 0.60, 14),
    "prayer":   ((205, 190, 158), (182, 166, 134), 1.20, 22),
    "emerg":    ((186, 203, 190), (160, 178, 164), 0.60, 14),
}


def _tiled_floor(base, grout, nx, ny, speckle, px=24, maxpx=2048):
    """Bake nx x ny tiles at ~px pixels each (clamped), tileable."""
    w = int(min(max(nx * px, 64), maxpx))
    h = int(min(max(ny * px, 64), maxpx))
    img = np.zeros((h, w, 3), np.float32)
    img[:] = base
    n = _fbm(max(w, h), octaves=6)[:h, :w]
    img += ((n - 0.5) * speckle)[..., None]
    # tile boundaries in pixel space
    xs = (np.arange(w) * nx / w).astype(int)
    ys = (np.arange(h) * ny / h).astype(int)
    tone = (RNG.random((int(ny) + 1, int(nx) + 1)) - 0.5) * 7.0
    img += tone[ys][:, xs][..., None]
    gx = np.zeros(w, bool); gy = np.zeros(h, bool)
    gx[1:] = xs[1:] != xs[:-1]
    gy[1:] = ys[1:] != ys[:-1]
    img[gy, :] = grout
    img[:, gx] = grout
    return img


def floor_for(kind, size_x, size_y):
    """Floor/ceiling map scaled to a surface of size_x by size_y metres.
    Returns the texture filename."""
    if kind == "ceiling":
        nx, ny = max(1, round(size_x / 0.6)), max(1, round(size_y / 0.6))
        key = ("ceiling", nx, ny)
        if key in _cache:
            return _cache[key]
        name = f"ceiling_{nx}x{ny}.png"
        w = int(min(max(nx * 16, 64), 2048)); h = int(min(max(ny * 16, 64), 2048))
        img = np.zeros((h, w, 3), np.float32); img[:] = (236, 237, 232)
        img += ((_fbm(max(w, h), octaves=7, persistence=0.62)[:h, :w] - 0.5) * 20)[..., None]
        xs = (np.arange(w) * nx / w).astype(int); ys = (np.arange(h) * ny / h).astype(int)
        gx = np.zeros(w, bool); gy = np.zeros(h, bool)
        gx[1:] = xs[1:] != xs[:-1]; gy[1:] = ys[1:] != ys[:-1]
        img[gy, :] = (203, 205, 200); img[:, gx] = (203, 205, 200)
        _save(img, name); _cache[key] = name
        return name

    base, grout, tile_m, speck = FLOOR_SPEC[kind]
    nx, ny = max(1, round(size_x / tile_m)), max(1, round(size_y / tile_m))
    key = (kind, nx, ny)
    if key in _cache:
        return _cache[key]
    name = f"floor_{kind}_{nx}x{ny}.png"
    _save(_tiled_floor(base, grout, nx, ny, speck), name)
    _cache[key] = name
    return name


def _cloth(base, size_x, size_y, weave_m=0.012, px_per_m=220, maxpx=512):
    w = int(min(max(size_x * px_per_m, 32), maxpx))
    h = int(min(max(size_y * px_per_m, 32), maxpx))
    img = np.zeros((h, w, 3), np.float32); img[:] = base
    yy, xx = np.mgrid[0:h, 0:w]
    nx = size_x / weave_m; ny = size_y / weave_m
    img += (np.sin(xx * np.pi * nx / w) * np.sin(yy * np.pi * ny / h) * 3.0)[..., None]
    img += ((_fbm(max(w, h), octaves=4)[:h, :w] - 0.5) * 12.0)[..., None]
    return img


def linen_for(size_x, size_y):
    key = ("linen", round(size_x, 1), round(size_y, 1))
    if key in _cache:
        return _cache[key]
    name = f"linen_{key[1]}x{key[2]}.png".replace(".", "_", 2).replace("_png", ".png")
    name = f"linen_{str(key[1]).replace('.','p')}x{str(key[2]).replace('.','p')}.png"
    _save(_cloth((232, 238, 242), size_x, size_y), name)
    _cache[key] = name
    return name


def blanket_for(size_x, size_y):
    key = ("blanket", round(size_x, 1), round(size_y, 1))
    if key in _cache:
        return _cache[key]
    name = f"blanket_{str(key[1]).replace('.','p')}x{str(key[2]).replace('.','p')}.png"
    _save(_cloth((124, 156, 178), size_x, size_y), name)
    _cache[key] = name
    return name


# ---------------------------------------------------------------------------
# Signage. Rendered at a fixed physical resolution (PX_PER_M) so that a sign's
# lettering is the same height in metres on every board, whatever the board's
# size - i.e. the labels are in proportion to the building, not to the texture.
# ---------------------------------------------------------------------------
from PIL import ImageFont

PX_PER_M = 256                  # texture resolution per metre of signboard
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _fit_font(draw, text, path, box_w, box_h):
    """Largest font size whose rendered text fits box_w x box_h pixels."""
    lo, hi, best = 4, int(box_h * 1.6) + 8, 4
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(path, mid)
        l, t, r, b = draw.textbbox((0, 0), text, font=f)
        if (r - l) <= box_w and (b - t) <= box_h:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return ImageFont.truetype(path, best)


def sign_for(text, size_x_m, size_y_m, rgb, vertical=False, sub=None):
    """A room nameplate sized to a board of size_x_m x size_y_m metres.

    The board's own aspect ratio is honoured, text is centred, and a contrasting
    bar carries the room's zone colour so the sign reads at a glance in a figure.
    """
    key = ("sign", text, round(size_x_m, 2), round(size_y_m, 2), vertical)
    if key in _cache:
        return _cache[key]
    w = max(64, int(round(size_x_m * PX_PER_M)))
    h = max(64, int(round(size_y_m * PX_PER_M)))
    if vertical:                       # board is tall: render landscape, rotate
        w, h = h, w
    img = Image.new("RGB", (w, h), (248, 248, 246))
    d = ImageDraw.Draw(img)
    bar = max(3, int(h * 0.14))
    col = tuple(int(255 * c) for c in rgb)
    d.rectangle([0, 0, w, bar], fill=col)
    d.rectangle([0, h - bar, w, h], fill=col)
    d.rectangle([0, 0, w - 1, h - 1], outline=(62, 66, 70), width=max(2, h // 90))
    pad = int(w * 0.05)
    tx_h = int((h - 2 * bar) * (0.52 if sub else 0.74))
    f = _fit_font(d, text, FONT_BOLD, w - 2 * pad, tx_h)
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    ty = bar + ((h - 2 * bar) - (b - t)) * (0.30 if sub else 0.5)
    d.text(((w - (r - l)) / 2 - l, ty - t), text, font=f, fill=(26, 30, 34))
    if sub:
        fs = _fit_font(d, sub, FONT_REG, w - 2 * pad, int((h - 2 * bar) * 0.26))
        l2, t2, r2, b2 = d.textbbox((0, 0), sub, font=fs)
        d.text(((w - (r2 - l2)) / 2 - l2, h - bar - (b2 - t2) - int(h * 0.07) - t2),
               sub, font=fs, fill=(92, 98, 104))
    if vertical:
        img = img.rotate(90, expand=True)
    name = ("sign_" + re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
            + f"_{key[2]}x{key[3]}".replace(".", "p") + ".png")
    img.save(os.path.join(OUT, name))
    print("wrote", name)
    _cache[key] = name
    return name


def prayer_mat():
    """A woven prayer mat: deep red field, gold border, a pale mihrab arch."""
    key = ("prayer_mat",)
    if key in _cache:
        return _cache[key]
    w, h = 256, 420
    img = Image.new("RGB", (w, h), (128, 34, 40))
    d = ImageDraw.Draw(img)
    for k, col in enumerate(((196, 160, 70), (128, 34, 40), (196, 160, 70))):
        d.rectangle([10 + 8 * k, 10 + 8 * k, w - 11 - 8 * k, h - 11 - 8 * k], outline=col, width=5)
    d.rectangle([48, 60, w - 48, h - 60], outline=(222, 196, 120), width=3)
    d.pieslice([70, 70, w - 70, 70 + (w - 140)], 180, 360, fill=(170, 60, 60), outline=(222, 196, 120), width=3)
    d.rectangle([70, 70 + (w - 140) // 2, w - 70, h - 80], fill=(170, 60, 60), outline=(222, 196, 120), width=3)
    arr = np.asarray(img, np.float32)
    arr += ((_fbm(512, octaves=4)[:h, :w] - 0.5) * 14.0)[..., None]
    name = "prayer_mat.png"
    _save(arr, name)
    _cache[key] = name
    return name


def floor_decal(text, size_m, rgb, bg=(255, 255, 255)):
    """A zone name painted on the floor, for the top-down figures. bg is the
    room's own floor colour so the lettering reads as paint, not a white mat."""
    key = ("decal", text, round(size_m, 2), bg)
    if key in _cache:
        return _cache[key]
    w = h = max(128, int(round(size_m * 96)))
    img = Image.new("RGB", (w, h), tuple(int(c) for c in bg))
    d = ImageDraw.Draw(img)
    f = _fit_font(d, text, FONT_BOLD, int(w * 0.92), int(h * 0.26))
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    col = tuple(int(200 * c) for c in rgb)
    d.text(((w - (r - l)) / 2 - l, (h - (b - t)) / 2 - t), text, font=f, fill=col)
    arr = np.asarray(img, np.float32)
    arr += ((_fbm(max(w, h), octaves=5)[:h, :w] - 0.5) * 10.0)[..., None]   # same speckle as the floor
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    name = "decal_" + re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() + ".png"
    img.save(os.path.join(OUT, name))
    print("wrote", name)
    _cache[key] = name
    return name


# ---------------------------------------------------------------------------
# Textured boxes as OBJ meshes.
#
# Measured 2026-10-16 on this machine (Intel iGPU, gz-sim Harmonic / ogre2):
# every SDF <pbr> material renders BLACK - with an albedo_map, with a bare
# <albedo> colour, from file://, package:// or an absolute path - while classic
# <ambient>/<diffuse> materials and OBJ meshes with MTL diffuse textures render
# correctly (that is how the Fuel assets show their textures). So a textured
# surface is emitted as a unit cube OBJ whose MTL references the PNG, scaled by
# the SDF <mesh><scale>. The cube's faces carry 0..1 UVs, which is exactly what
# the physically-scaled maps above are baked for.
# ---------------------------------------------------------------------------
import shutil

GEN_DIR = os.path.join(os.path.dirname(__file__), "..", "hospitalnav_deploy", "meshes", "gen")

_CUBE_OBJ = """# unit cube, centred, per-face 0..1 UVs - generated by gen_textures.py
mtllib {mtl}
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn  0  0 -1
vn  0  0  1
vn  0 -1  0
vn  1  0  0
vn  0  1  0
vn -1  0  0
usemtl tex
f 1/1/1 4/4/1 3/3/1
f 1/1/1 3/3/1 2/2/1
f 5/1/2 6/2/2 7/3/2
f 5/1/2 7/3/2 8/4/2
f 1/1/3 2/2/3 6/3/3
f 1/1/3 6/3/3 5/4/3
f 2/1/4 3/2/4 7/3/4
f 2/1/4 7/3/4 6/4/4
f 3/1/5 4/2/5 8/3/5
f 3/1/5 8/3/5 7/4/5
f 4/1/6 1/2/6 5/3/6
f 4/1/6 5/3/6 8/4/6
"""

_MTL = """newmtl tex
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks {ks} {ks} {ks}
Ns {ns}
illum 2
map_Kd {png}
"""


def textured_box(png_name, ks=0.08, ns=12.0):
    """Return the OBJ file name (inside meshes/gen/) of a unit cube textured with
    materials/textures/<png_name>; writes OBJ, MTL and a copy of the PNG."""
    stem = os.path.splitext(png_name)[0]
    obj = f"box_{stem}.obj"
    key = ("objbox", stem)
    if key in _cache:
        return obj
    os.makedirs(GEN_DIR, exist_ok=True)
    src = os.path.join(OUT, png_name)
    shutil.copyfile(src, os.path.join(GEN_DIR, png_name))
    with open(os.path.join(GEN_DIR, f"box_{stem}.mtl"), "w") as f:
        f.write(_MTL.format(ks=ks, ns=ns, png=png_name))
    with open(os.path.join(GEN_DIR, obj), "w") as f:
        f.write(_CUBE_OBJ.format(mtl=f"box_{stem}.mtl"))
    _cache[key] = obj
    return obj


# ---------------------------------------------------------------------------
# Walls at physical scale. One texture covers a 2 m wide x 3 m tall patch of
# hospital wall - two-tone paint (scuff-resistant lower third), a dado line,
# a vinyl skirting - and every wall gets its OWN mesh whose UVs are in metres,
# so the paint tiles at true scale along walls of any length and the dado and
# skirting sit at the same height on every face.
# ---------------------------------------------------------------------------
WALL_TILE_W, WALL_H = 2.0, 3.0
WALL_TEX = "wall_twotone.png"
DADO_Z, SKIRT_Z = 1.00, 0.10


def wall_texture():
    key = ("wall_twotone",)
    if key in _cache:
        return WALL_TEX
    w, h = 512, 768                                   # 256 px per metre
    img = np.zeros((h, w, 3), np.float32)
    z = (h - 1 - np.arange(h)) / (h - 1) * WALL_H     # metres above the floor, per row
    upper, lower = np.array((232, 230, 222.0)), np.array((196, 210, 202.0))
    img[:] = upper
    img[z < DADO_Z] = lower
    img[(z >= DADO_Z - 0.012) & (z < DADO_Z + 0.012)] = (150, 162, 156)   # dado line
    img[z < SKIRT_Z] = (92, 96, 102)                                       # skirting
    img[(z >= SKIRT_Z - 0.006) & (z < SKIRT_Z)] = (70, 73, 78)
    img += ((_fbm(max(w, h), octaves=6)[:h, :w] - 0.5) * 7.0)[..., None]
    img += RNG.normal(0, 1.0, (h, w))[..., None]
    _save(img, WALL_TEX)
    os.makedirs(GEN_DIR, exist_ok=True)
    shutil.copyfile(os.path.join(OUT, WALL_TEX), os.path.join(GEN_DIR, WALL_TEX))
    with open(os.path.join(GEN_DIR, "wall_twotone.mtl"), "w") as f:
        f.write(_MTL.format(ks=0.04, ns=6.0, png=WALL_TEX))
    _cache[key] = WALL_TEX
    return WALL_TEX


def wall_mesh(name, sx, sy, z0=0.0, z1=WALL_H):
    """An OBJ box of exactly sx x sy x (z1 - z0) (origin at its footprint centre,
    floor level 0), textured with the wall paint at physical scale. Returns the
    file name inside meshes/gen/."""
    wall_texture()
    hx, hy = sx / 2.0, sy / 2.0
    V = [(-hx, -hy, z0), (hx, -hy, z0), (hx, hy, z0), (-hx, hy, z0),
         (-hx, -hy, z1), (hx, -hy, z1), (hx, hy, z1), (-hx, hy, z1)]
    vt, faces = [], []

    def face(idx, uvs, n):
        base = len(vt) + 1
        vt.extend(uvs)
        faces.append((idx, [base, base + 1, base + 2, base + 3], n))
    v = lambda zz: zz / WALL_H                       # v = height on the wall (0 at the floor)
    # -y face (seen from -y: right = +x), +y face (right = -x), +x (right = +y), -x (right = -y)
    face([1, 2, 6, 5], [(0, v(z0)), (sx / WALL_TILE_W, v(z0)), (sx / WALL_TILE_W, v(z1)), (0, v(z1))], 3)
    face([3, 4, 8, 7], [(0, v(z0)), (sx / WALL_TILE_W, v(z0)), (sx / WALL_TILE_W, v(z1)), (0, v(z1))], 5)
    face([2, 3, 7, 6], [(0, v(z0)), (sy / WALL_TILE_W, v(z0)), (sy / WALL_TILE_W, v(z1)), (0, v(z1))], 4)
    face([4, 1, 5, 8], [(0, v(z0)), (sy / WALL_TILE_W, v(z0)), (sy / WALL_TILE_W, v(z1)), (0, v(z1))], 6)
    top = 0.995
    face([5, 6, 7, 8], [(0, top), (sx / WALL_TILE_W, top), (sx / WALL_TILE_W, top), (0, top)], 2)
    normals = ["0 0 -1", "0 0 1", "0 -1 0", "1 0 0", "0 1 0", "-1 0 0"]
    fname = f"wall_{name}.obj"
    with open(os.path.join(GEN_DIR, fname), "w") as f:
        f.write("# wall mesh, UVs in metres / tile - generated by gen_textures.wall_mesh\nmtllib wall_twotone.mtl\n")
        for x, y, zz in V:
            f.write(f"v {x:.4f} {y:.4f} {zz:.4f}\n")
        for uu, vv in vt:
            f.write(f"vt {uu:.4f} {vv:.4f}\n")
        for nrm in normals:
            f.write(f"vn {nrm}\n")
        f.write("usemtl tex\n")
        for idx, uv, n in faces:
            a, b, c, d = idx
            ta, tb, tc, td = uv
            f.write(f"f {a}/{ta}/{n} {b}/{tb}/{n} {c}/{tc}/{n}\nf {a}/{ta}/{n} {c}/{tc}/{n} {d}/{td}/{n}\n")
    return fname


if __name__ == "__main__":
    main()
