#!/usr/bin/env python3
"""
Integration test for the CRIP perception -> brain pipeline.

Runs obstacle_monitor and safety_pilot as real nodes (no Gazebo needed),
publishes synthetic LaserScan messages, and asserts:

  1. With a clear scan, the pilot commands forward motion (linear.x > 0).
  2. With an obstacle directly ahead, the pilot vetoes forward motion
     (linear.x == 0) and rotates (angular.z != 0).

This is the CI-runnable simulation test: it validates the closed loop
deterministically and fast.
"""
import math
import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


@pytest.mark.launch_test
def generate_test_description():
    obstacle_monitor = launch_ros.actions.Node(
        package='crip_perception', executable='obstacle_monitor',
        parameters=[{'warn_distance': 0.8}], output='screen')
    safety_pilot = launch_ros.actions.Node(
        package='crip_brain', executable='safety_pilot',
        parameters=[{'stop_distance': 0.7, 'cruise_speed': 0.25,
                     'turn_speed': 0.6}], output='screen')
    return (
        launch.LaunchDescription([
            obstacle_monitor,
            safety_pilot,
            launch_testing.actions.ReadyToTest(),
        ]),
        {},
    )


def make_scan(distance):
    """360-sample scan, all beams at `distance` metres."""
    msg = LaserScan()
    msg.angle_min = -math.pi
    msg.angle_max = math.pi
    n = 360
    msg.angle_increment = (msg.angle_max - msg.angle_min) / n
    msg.range_min = 0.12
    msg.range_max = 12.0
    msg.ranges = [float(distance)] * n
    return msg


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_pipeline')
        self.scan_pub = self.node.create_publisher(LaserScan, 'scan', 10)
        self.last_cmd = None
        self.node.create_subscription(Twist, 'cmd_vel', self._on_cmd, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _on_cmd(self, msg):
        self.last_cmd = msg

    def _pump(self, scan, seconds=3.0):
        """Publish `scan` repeatedly and spin until we have a cmd_vel."""
        end = time.time() + seconds
        self.last_cmd = None
        while time.time() < end:
            self.scan_pub.publish(scan)
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.last_cmd is not None and time.time() > end - seconds + 1.0:
                # keep pumping a bit so brain timer (10Hz) reacts to scan
                pass
        return self.last_cmd

    def test_clear_path_drives_forward(self):
        cmd = self._pump(make_scan(5.0), seconds=4.0)
        self.assertIsNotNone(cmd, "no cmd_vel received")
        self.assertGreater(cmd.linear.x, 0.0,
                           "robot should drive forward when path is clear")

    def test_obstacle_ahead_stops_and_turns(self):
        cmd = self._pump(make_scan(0.3), seconds=4.0)
        self.assertIsNotNone(cmd, "no cmd_vel received")
        self.assertAlmostEqual(cmd.linear.x, 0.0, places=3,
                               msg="robot must not drive into a close obstacle")
        self.assertNotEqual(cmd.angular.z, 0.0,
                            "robot should rotate to find a clear path")
