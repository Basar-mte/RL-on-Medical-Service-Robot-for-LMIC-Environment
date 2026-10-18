"""Convert an X Window Dump (xwd) file to PNG - the screenshot path for RViz on
this Wayland desktop, where rviz2 runs as an Xwayland client and `xwd -id` can
read its window while GNOME's own screenshot API is closed to scripts.

    xwd -id <window> -silent > shot.xwd && python3 xwd2png.py shot.xwd shot.png
"""
import struct
import sys

import numpy as np
from PIL import Image


def xwd_to_image(path):
    data = open(path, "rb").read()
    hdr = struct.unpack(">25I", data[:100])
    (header_size, version, pixmap_format, depth, width, height, xoffset, byte_order,
     bitmap_unit, bitmap_bit_order, bitmap_pad, bits_per_pixel, bytes_per_line,
     visual_class, red_mask, green_mask, blue_mask, bits_per_rgb, colormap_entries,
     ncolors, window_width, window_height, window_x, window_y, border) = hdr
    off = header_size + ncolors * 12
    if bits_per_pixel not in (24, 32):
        raise SystemExit(f"unsupported bpp {bits_per_pixel}")
    bpp = bits_per_pixel // 8
    rows = np.frombuffer(data, np.uint8, count=bytes_per_line * height, offset=off)
    rows = rows.reshape(height, bytes_per_line)[:, :width * bpp].reshape(height, width, bpp)
    # masks tell us the channel order; byte_order 0 = LSBFirst
    def chan(mask):
        shift = (mask & -mask).bit_length() - 1
        return shift // 8 if byte_order == 0 else bpp - 1 - shift // 8
    r, g, b = rows[..., chan(red_mask)], rows[..., chan(green_mask)], rows[..., chan(blue_mask)]
    return Image.fromarray(np.stack([r, g, b], -1))


if __name__ == "__main__":
    xwd_to_image(sys.argv[1]).save(sys.argv[2])
    print("wrote", sys.argv[2])
