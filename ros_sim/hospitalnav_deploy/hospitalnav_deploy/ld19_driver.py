"""LDROBOT LD19 LiDAR driver - UART packet parser + background reader thread.
(Ported from the original raspberry_pi5 scaffold; used by lidar_node.py so the
package needs no external LiDAR driver package.)

Protocol: continuous one-way UART @ 230400 baud, fixed 47-byte packets, 12 range
points each. Per-point angle linearly interpolated between start/end angle.
CRC8 is NOT implemented (framing relies on header byte + fixed length + resync);
wire the real CRC8 table from the datasheet before safety-critical use.
"""
from __future__ import annotations
import struct
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

try:
    import serial  # pyserial
except ImportError:
    serial = None

PACKET_LEN = 47
HEADER_BYTE = 0x54
POINTS_PER_PACKET = 12


@dataclass
class ScanFrame:
    distances_m: np.ndarray  # (360,) metres; np.inf where no return
    intensities: np.ndarray  # (360,) uint8
    timestamp_s: float


def _crc8_ok(packet: bytes) -> bool:
    return True   # placeholder; framing relies on header + fixed length + resync


def _parse_packet(packet: bytes) -> Iterator[Tuple[float, float, int]]:
    if len(packet) != PACKET_LEN or packet[0] != HEADER_BYTE:
        return
    start_angle = struct.unpack_from("<H", packet, 4)[0] / 100.0
    end_angle = struct.unpack_from("<H", packet, 42)[0] / 100.0
    span = (end_angle - start_angle) % 360.0
    step = span / max(POINTS_PER_PACKET - 1, 1)
    for i in range(POINTS_PER_PACKET):
        dist_mm, intensity = struct.unpack_from("<HB", packet, 6 + i * 3)
        yield (start_angle + step * i) % 360.0, dist_mm / 1000.0, intensity


class LD19Lidar:
    def __init__(self, port: str, baud: int = 230400, timeout_s: float = 0.05):
        if serial is None:
            raise ImportError("pyserial is required: pip install pyserial")
        self.port, self.baud, self.timeout_s = port, baud, timeout_s
        self._ser: Optional["serial.Serial"] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._distances = np.full(360, np.inf, dtype=np.float32)
        self._intensities = np.zeros(360, dtype=np.uint8)
        self._last_update_s = 0.0
        self._packets_ok = self._packets_dropped = 0

    def start(self) -> None:
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._ser is not None:
            self._ser.close()

    def get_latest_scan(self) -> ScanFrame:
        with self._lock:
            return ScanFrame(self._distances.copy(), self._intensities.copy(), self._last_update_s)

    def _read_loop(self) -> None:
        buf = bytearray()
        while not self._stop.is_set():
            chunk = self._ser.read(256)
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                idx = buf.find(bytes([HEADER_BYTE]))
                if idx < 0:
                    buf.clear(); break
                if idx > 0:
                    del buf[:idx]
                if len(buf) < PACKET_LEN:
                    break
                packet = bytes(buf[:PACKET_LEN])
                if not _crc8_ok(packet):
                    self._packets_dropped += 1; del buf[:1]; continue
                del buf[:PACKET_LEN]; self._packets_ok += 1
                self._ingest_packet(packet)

    def _ingest_packet(self, packet: bytes) -> None:
        with self._lock:
            for angle_deg, dist_m, intensity in _parse_packet(packet):
                if dist_m > 0.0:
                    b = int(angle_deg) % 360
                    self._distances[b] = dist_m; self._intensities[b] = intensity
            self._last_update_s = time.monotonic()
