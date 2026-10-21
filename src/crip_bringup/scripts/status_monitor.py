#!/usr/bin/env python3
"""
CRIP live status monitor — makes the terminal READABLE.

Instead of scrolling thousands of log lines, this prints ONE self-refreshing
panel showing what every part of the robot is doing right now:

    ┌─ CRIP STATUS ───────────────────────────────┐
    │ POSE      x=1.24  y=0.31  yaw=12°           │
    │ SPEED     v=0.20 m/s   w=0.00 rad/s         │
    │ LIDAR     9.3 Hz   nearest ahead: 1.42 m    │
    │ PERCEPTION  clear ahead                     │
    │ WORLD     12 entities known                 │
    │ VLM       risk: none                        │
    │ BRAIN     idle                              │
    └─────────────────────────────────────────────┘

Run it INSIDE the container, in its own terminal:

    docker exec -it crip bash
    source /crip_ws/install/setup.bash
    python3 /crip_ws/src/crip_bringup/scripts/status_monitor.py

Press Ctrl+C to stop. It only *reads* topics; it never commands the robot.
"""
import math
import os
import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class StatusMonitor(Node):
    def __init__(self):
        super().__init__('status_monitor')
        self.pose = (0.0, 0.0, 0.0)
        self.cmd = (0.0, 0.0)
        self.scan_times = deque(maxlen=20)
        self.front = float('inf')
        self.n_beams = 0
        self.perception = '(waiting)'
        self.entities = '(waiting)'
        self.vlm = 'none'
        self.brain = 'idle'
        self.safety = 'ok'
        self.t0 = time.time()

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)
        self.create_subscription(Odometry, '/odom', self.on_odom, sensor_qos)
        self.create_subscription(LaserScan, '/scan', self.on_scan, sensor_qos)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_subscription(String, '/perception/summary',
                                 self.on_perception, 10)
        self.create_subscription(String, '/world_model/entities',
                                 self.on_entities, 10)
        self.create_subscription(String, '/vlm/alert', self.on_vlm, 10)
        self.create_subscription(String, '/brain/explanation', self.on_brain, 10)
        self.create_subscription(String, '/safety/audit', self.on_safety, 10)

        self.create_timer(0.5, self.render)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, math.degrees(yaw))

    def on_scan(self, msg):
        self.scan_times.append(time.time())
        self.n_beams = len(msg.ranges)
        n = len(msg.ranges)
        if n:
            # beams within +/- 20 deg of straight ahead
            best = float('inf')
            for i, r in enumerate(msg.ranges):
                a = msg.angle_min + i * msg.angle_increment
                if -0.35 <= a <= 0.35 and math.isfinite(r) and r > 0.0:
                    best = min(best, r)
            self.front = best

    def on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)

    def on_perception(self, msg):
        self.perception = msg.data[:44]

    def on_entities(self, msg):
        self.entities = msg.data[:44]

    def on_vlm(self, msg):
        self.vlm = msg.data[:44]

    def on_brain(self, msg):
        self.brain = msg.data[:44]

    def on_safety(self, msg):
        self.safety = msg.data[:44]

    def scan_hz(self):
        if len(self.scan_times) < 2:
            return 0.0
        span = self.scan_times[-1] - self.scan_times[0]
        return (len(self.scan_times) - 1) / span if span > 0 else 0.0

    def render(self):
        os.system('clear')
        x, y, yaw = self.pose
        v, w = self.cmd
        hz = self.scan_hz()
        front = ('%.2f m' % self.front) if math.isfinite(self.front) else '--'
        up = int(time.time() - self.t0)

        def row(label, value):
            return '│ %-11s %-45s │' % (label, str(value)[:45])

        moving = 'MOVING' if abs(v) > 0.01 or abs(w) > 0.01 else 'stopped'
        lidar_ok = 'OK' if hz > 1 else 'NO DATA'

        print('┌─ CRIP LIVE STATUS ' + '─' * 40 + '┐')
        print(row('UPTIME', '%ds' % up))
        print('├' + '─' * 59 + '┤')
        print(row('POSE', 'x=%.2f  y=%.2f  yaw=%.0f°' % (x, y, yaw)))
        print(row('COMMAND', 'v=%.2f m/s   w=%.2f rad/s   [%s]' % (v, w, moving)))
        print('├' + '─' * 59 + '┤')
        print(row('LIDAR', '%.1f Hz  %d beams  [%s]' % (hz, self.n_beams, lidar_ok)))
        print(row('AHEAD', 'nearest obstacle: %s' % front))
        print('├' + '─' * 59 + '┤')
        print(row('PERCEPTION', self.perception))
        print(row('WORLD MODEL', self.entities))
        print(row('VLM RISK', self.vlm))
        print(row('BRAIN', self.brain))
        print('├' + '─' * 59 + '┤')
        inside = abs(x) <= 8.0 and abs(y) <= 8.0
        print(row('ARENA', ('INSIDE (ok)' if inside else
                            '*** OUTSIDE ARENA! ***')))
        print(row('SAFETY', self.safety))
        print('└' + '─' * 59 + '┘')
        print()
        print('  Ctrl+C to exit.  This monitor only READS — it never drives.')


def main():
    rclpy.init()
    node = StatusMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nmonitor stopped.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
