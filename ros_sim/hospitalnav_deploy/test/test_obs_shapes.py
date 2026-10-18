"""Sanity tests that run WITHOUT ROS or hardware (pure functions).
  pytest hardware_deploy/hospitalnav_deploy/test/test_obs_shapes.py
Verifies the LiDAR downsampling and shield match the 90-D observation spec.
"""
import math
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hospitalnav_deploy.scan_utils import scan_to_lidar48, shield, N_RAYS, MAX_LINEAR, MAX_STEER


def test_lidar_downsample_shape_and_range():
    ranges = np.full(360, 12.0); ranges[170:190] = 1.0
    out = scan_to_lidar48(ranges, angle_min=-math.pi, angle_increment=2 * math.pi / 360)
    assert out.shape == (N_RAYS,)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isfinite(out).all()


def test_shield_brakes_when_blocked():
    lidar = np.ones(48, np.float32)          # clear
    v, d = shield(0.8, 0.0, lidar)
    assert abs(v - 0.8) < 1e-6               # no obstacle -> unchanged
    lidar[:] = 0.03                          # 0.36 m all around -> blocked
    v, d = shield(0.8, 0.0, lidar)
    assert v < 0.1                           # speed governor brakes hard
    assert -MAX_STEER <= d <= MAX_STEER


def test_shield_never_exceeds_limits():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lidar = rng.random(48).astype(np.float32)
        v, d = shield(rng.uniform(-1, 1), rng.uniform(-1, 1), lidar)
        assert -MAX_LINEAR <= v <= MAX_LINEAR
        assert -MAX_STEER <= d <= MAX_STEER


if __name__ == "__main__":
    test_lidar_downsample_shape_and_range()
    test_shield_brakes_when_blocked()
    test_shield_never_exceeds_limits()
    print("all tests passed")
