"""Ground-truth crowd -> /people. Gazebo knows every actor's exact pose, so
there's no detector: subscribe to /pedestrian_poses (geometry_msgs/PoseArray,
map frame, bridged from gz-sim actors), track them frame-to-frame for
velocity, and publish the 5 nearest to the robot the same way perception.py's
PeopleTracker does from real LiDAR clusters — so controller_node needs no
changes to consume either source.

With no actors bridged (the common case for this demo), no PoseArray ever
arrives and /people simply stays at controller_node's zero-initialized
default ("no one nearby"), same as an empty sim episode.
"""
from __future__ import annotations
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener

PED_R = 5.0
PED_VMAX = 2.5
K = 5


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class GtPedestrians(Node):
    def __init__(self):
        super().__init__("gt_pedestrians")
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.prev = {}  # id -> (x, y, t)
        self.create_subscription(PoseArray, "/pedestrian_poses", self._poses, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/people", 10)

    def _pose(self):
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
            return (t.transform.translation.x, t.transform.translation.y,
                    _yaw(t.transform.rotation))
        except Exception:
            return None

    def _poses(self, msg: PoseArray):
        p = self._pose()
        if p is None:
            return
        rx0, ry0, yaw = p
        now = time.monotonic()
        rel = []
        for i, pose in enumerate(msg.poses):
            x, y = pose.position.x, pose.position.y
            vx = vy = 0.0
            if i in self.prev:
                px, py, pt = self.prev[i]
                dt = now - pt
                if 0 < dt < 1.0:
                    vx, vy = (x - px) / dt, (y - py) / dt
            self.prev[i] = (x, y, now)
            dx, dy = x - rx0, y - ry0
            d = math.hypot(dx, dy)
            if d > PED_R:
                continue
            rxr = dx * math.cos(-yaw) - dy * math.sin(-yaw)
            ryr = dx * math.sin(-yaw) + dy * math.cos(-yaw)
            vxr = vx * math.cos(-yaw) - vy * math.sin(-yaw)
            vyr = vx * math.sin(-yaw) + vy * math.cos(-yaw)
            rel.append((d, rxr, ryr, vxr, vyr))
        rel.sort(key=lambda t: t[0])

        import numpy as np
        arr = np.zeros((K, 4), np.float32)
        for i, (_d, rxr, ryr, vxr, vyr) in enumerate(rel[:K]):
            arr[i] = [rxr / PED_R, ryr / PED_R, vxr / PED_VMAX, vyr / PED_VMAX]
        m = Float32MultiArray()
        m.data = np.clip(arr, -1, 1).flatten().tolist()
        self.pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(GtPedestrians())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
