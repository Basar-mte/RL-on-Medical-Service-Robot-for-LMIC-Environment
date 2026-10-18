"""Composites the demo video: Gazebo footage full-frame, the live RViz mapping
view as a picture-in-picture in the corner, labelled, in sync.

    python3 compose_video.py --run-dir DIR --out hospitalnav_demo.mp4

Both sources carry SIMULATION timestamps (record_video.py's gazebo.mp4.stamps.csv,
capture_rviz.py's rviz_stream/stamps.csv), so every output frame shows the RViz
state at the same simulated instant as the Gazebo frame beneath it. Output is
resampled to an even 30 frames per simulated second, i.e. it plays in real time
whatever speed the simulator actually ran at. The closing card states the run's
recorded outcome (result.json), not a claim.
"""
from __future__ import annotations
import argparse
import bisect
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
POLICY = {"fixed_equal": "fixed_equal (SAC, CARS reward)", "residual_sf": "residual_sf (residual SAC)",
          "vanilla": "vanilla (culture-blind SAC)", "classical_sf2": "classical social force",
          "risk_field": "risk-field baseline"}


def font(path, size):
    return ImageFont.truetype(path, size)


def panel(draw, xy, text_lines, pad=14, fill=(15, 20, 28, 175)):
    x, y = xy
    fs = [(t, f) for t, f in text_lines]
    widths = [draw.textbbox((0, 0), t, font=f)[2] for t, f in fs]
    heights = [draw.textbbox((0, 0), t, font=f)[3] for t, f in fs]
    w = max(widths) + 2 * pad
    h = sum(heights) + pad * (len(fs) + 1) - 4
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fill)
    cy = y + pad
    for (t, f), hh in zip(fs, heights):
        draw.text((x + pad, cy), t, font=f, fill=(255, 255, 255, 255))
        cy += hh + pad - 4
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--inset-width", type=int, default=640)
    ap.add_argument("--policy", default="fixed_equal")
    ap.add_argument("--scene", default="Walled hospital ward · 8 walking pedestrians")
    ap.add_argument("--crf", type=int, default=17)
    a = ap.parse_args()
    import imageio_ffmpeg

    d = a.run_dir
    gst = [float(r["sim"]) for r in csv.DictReader(open(os.path.join(d, "gazebo.mp4.stamps.csv")))]
    rv = [(float(r["sim"]), int(r["frame"])) for r in csv.DictReader(open(os.path.join(d, "rviz_stream", "stamps.csv")))
          if float(r["sim"]) > 0]
    rv_t = [t for t, _ in rv]
    res = json.load(open(os.path.join(d, "result.json")))
    goal = res["goals"][-1]
    start_sim = gst[0]

    reader = imageio_ffmpeg.read_frames(os.path.join(d, "gazebo.mp4"))
    meta = reader.__next__()
    W, H = meta["size"]
    writer = imageio_ffmpeg.write_frames(a.out, (W, H), fps=a.fps, codec="libx264", quality=None,
                                         output_params=["-crf", str(a.crf), "-preset", "slow",
                                                        "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                                         macro_block_size=8)
    writer.send(None)
    f_title, f_sub, f_small, f_clock = font(FONT_B, 34), font(FONT_R, 24), font(FONT_R, 20), font(FONT_B, 26)
    f_big = font(FONT_B, 46)

    cur_idx, cur = -1, None
    rv_cache = (None, None)
    t_end = gst[-1]
    n_out = int((t_end - start_sim) * a.fps)
    hold = int(2.5 * a.fps)
    last = None
    for k in range(n_out + hold):
        tau = start_sim + min(k, n_out - 1) / a.fps
        # advance the Gazebo stream to the last frame stamped at or before tau
        while cur_idx + 1 < len(gst) and gst[cur_idx + 1] <= tau + 1e-6:
            cur = np.frombuffer(reader.__next__(), np.uint8).reshape(H, W, 3)
            cur_idx += 1
        if cur is None:
            continue
        frame = Image.fromarray(cur).convert("RGBA")
        over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(over)
        # --- RViz inset (bottom right) ---------------------------------------
        j = bisect.bisect_right(rv_t, tau) - 1
        if j >= 0:
            fid = rv[j][1]
            if rv_cache[0] != fid:
                im = Image.open(os.path.join(d, "rviz_stream", f"frame_{fid:06d}.jpg")).convert("RGB")
                s = a.inset_width / im.width
                rv_cache = (fid, im.resize((a.inset_width, int(im.height * s)), Image.LANCZOS))
            ins = rv_cache[1]
            iw, ih = ins.size
            x0, y0 = W - iw - 28, H - ih - 28
            shadow = Image.new("RGBA", (iw + 40, ih + 40), (0, 0, 0, 0))
            ImageDraw.Draw(shadow).rectangle([20, 20, iw + 20, ih + 20], fill=(0, 0, 0, 120))
            over.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(9)), (x0 - 14, y0 - 10))
            dr.rectangle([x0 - 4, y0 - 40, x0 + iw + 3, y0 + ih + 3], fill=(255, 255, 255, 255))
            dr.rectangle([x0 - 4, y0 - 40, x0 + iw + 3, y0 - 4], fill=(18, 24, 34, 235))
            dr.text((x0 + 10, y0 - 35), "RViz  ·  live SLAM mapping (slam_toolbox)", font=f_small,
                    fill=(255, 255, 255, 255))
            over.paste(ins, (x0, y0))
            # legend strip under the title bar
            lx = x0 + 10
            for col, lab in (((215, 25, 28), "path"), ((20, 100, 220), "route"),
                             ((0, 150, 190), "scan"), ((235, 140, 40), "people")):
                dr.rectangle([lx, y0 + 10, lx + 22, y0 + 18], fill=col + (255,))
                dr.text((lx + 28, y0 + 3), lab, font=font(FONT_R, 17), fill=(30, 30, 30, 255))
                lx += 28 + dr.textbbox((0, 0), lab, font=font(FONT_R, 17))[2] + 18
        # --- captions --------------------------------------------------------
        panel(dr, (28, 26), [("HospitalNav: culturally-aware navigation, Gazebo Harmonic", f_title),
                             (f"Trained policy: {POLICY.get(a.policy, a.policy)}  ·  MentorPi Ackermann robot", f_sub),
                             (f"{a.scene}  ·  task: nurse desk → Emergency Room", f_sub)])
        panel(dr, (28, H - 96), [(f"sim time  {tau - start_sim:6.1f} s", f_clock)])
        if k >= n_out:
            msg = (f"Goal reached in {goal['seconds']:.1f} s simulated time" if goal["outcome"] == "reached"
                   else f"Goal not reached within {goal['seconds']:.0f} s ({goal['outcome']})")
            bw = dr.textbbox((0, 0), msg, font=f_big)[2]
            dr.rounded_rectangle([(W - bw) / 2 - 30, H / 2 - 60, (W + bw) / 2 + 30, H / 2 + 20], radius=14,
                                 fill=(15, 20, 28, 210))
            dr.text(((W - bw) / 2, H / 2 - 48), msg, font=f_big, fill=(255, 255, 255, 255))
        frame.alpha_composite(over)
        out = np.asarray(frame.convert("RGB"))
        writer.send(np.ascontiguousarray(out))
        last = out
    writer.close()
    print(f"wrote {a.out}: {n_out + hold} frames ({(n_out + hold) / a.fps:.1f} s), {W}x{H}")


if __name__ == "__main__":
    raise SystemExit(main())
