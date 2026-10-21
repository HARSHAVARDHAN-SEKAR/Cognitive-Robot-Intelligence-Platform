#!/usr/bin/env python3
"""
crip_perception.obstacle_monitor

Milestone-1 stand-in for the Multi-Sensor Perception module.
Subscribes to /scan, segments the laser into sectors (front/left/right/rear),
finds the nearest obstacle per sector, and publishes a human-readable
semantic summary on /perception/summary plus the min front range on
/perception/front_range.

This is intentionally lightweight: it is the hook where YOLO / SAM /
Grounding DINO etc. get added in later milestones.
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Float32


class ObstacleMonitor(Node):
    def __init__(self):
        super().__init__('obstacle_monitor')
        self.declare_parameter('warn_distance', 0.8)
        self.declare_parameter('detect_distance', 1.5)
        self.declare_parameter('detect_period', 1.0)
        self.warn_distance = self.get_parameter('warn_distance').value
        self.detect_distance = self.get_parameter('detect_distance').value
        self.detect_period = self.get_parameter('detect_period').value
        self._last_detect = 0.0

        self.sub = self.create_subscription(
            LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.pub_summary = self.create_publisher(String, 'perception/summary', 10)
        self.pub_front = self.create_publisher(Float32, 'perception/front_range', 10)
        # Milestone-2: feed structured detections to the world model.
        # Format "label:category:dx:dy" (offset in robot frame, metres).
        self.pub_detect = self.create_publisher(String, 'perception/detections', 10)
        self.get_logger().info('obstacle_monitor started (warn_distance=%.2fm)'
                               % self.warn_distance)

    @staticmethod
    def _sector_argmin(ranges, angles, lo, hi):
        """Return (min_range, angle_of_min) within an angular sector."""
        best_r, best_a = float('inf'), 0.0
        for r, a in zip(ranges, angles):
            if lo <= a < hi and math.isfinite(r) and r > 0.0 and r < best_r:
                best_r, best_a = r, a
        return best_r, best_a

    @staticmethod
    def _sector_min(ranges, angles, lo, hi):
        vals = [r for r, a in zip(ranges, angles)
                if lo <= a < hi and math.isfinite(r) and r > 0.0]
        return min(vals) if vals else float('inf')

    def on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        angles = [msg.angle_min + i * msg.angle_increment for i in range(n)]
        ranges = list(msg.ranges)

        front = self._sector_min(ranges, angles, -math.pi/4, math.pi/4)
        left = self._sector_min(ranges, angles, math.pi/4, 3*math.pi/4)
        rear = self._sector_min(ranges, angles, 3*math.pi/4, math.pi)
        rear2 = self._sector_min(ranges, angles, -math.pi, -3*math.pi/4)
        right = self._sector_min(ranges, angles, -3*math.pi/4, -math.pi/4)
        rear = min(rear, rear2)

        def tag(d):
            if d == float('inf'):
                return 'clear'
            if d < self.warn_distance:
                return f'BLOCKED({d:.2f}m)'
            return f'{d:.2f}m'

        summary = (f"front={tag(front)} left={tag(left)} "
                   f"right={tag(right)} rear={tag(rear)}")
        self.pub_summary.publish(String(data=summary))
        self.pub_front.publish(Float32(data=float(front if math.isfinite(front) else -1.0)))

        # Milestone-2: emit a structured detection for the nearest obstacle
        # in the forward arc, throttled to detect_period seconds.
        now = time.time()
        if now - self._last_detect >= self.detect_period:
            d, a = self._sector_argmin(ranges, angles, -math.pi/2, math.pi/2)
            if math.isfinite(d) and d <= self.detect_distance:
                dx = d * math.cos(a)
                dy = d * math.sin(a)
                self.pub_detect.publish(
                    String(data=f'obstacle:static:{dx:.3f}:{dy:.3f}'))
                self._last_detect = now


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
