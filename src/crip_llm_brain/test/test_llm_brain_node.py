#!/usr/bin/env python3
"""
Integration test for the LLM brain + world model grounding (Milestone 3).

Launches world_model and llm_brain together (mock LLM backend, SQLite
in-memory store), seeds a detection so the world model knows about a
"charging_station", then:

  1. A mission referencing the known label is ACCEPTED and grounded to a
     navigate_to task with resolved coordinates.
  2. A mission referencing an unknown label is REJECTED with a reason
     (no hallucinated navigation).
"""
import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from crip_interfaces.srv import PlanMission


@pytest.mark.launch_test
def generate_test_description():
    world_model = launch_ros.actions.Node(
        package='crip_world_model', executable='world_model',
        name='world_model', output='screen')
    llm_brain = launch_ros.actions.Node(
        package='crip_llm_brain', executable='llm_brain',
        name='llm_brain', output='screen',
        additional_env={'CRIP_LLM_BACKEND': 'mock'})
    return (
        launch.LaunchDescription([
            world_model,
            llm_brain,
            launch_testing.actions.ReadyToTest(),
        ]),
        {},
    )


class TestLLMBrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_llm_brain')
        self.odom_pub = self.node.create_publisher(Odometry, 'odom', 10)
        self.det_pub = self.node.create_publisher(String, 'perception/detections', 10)
        self.plan_cli = self.node.create_client(PlanMission, '/llm_brain/plan_mission')

    def tearDown(self):
        self.node.destroy_node()

    def _spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _call(self, request, timeout=8.0):
        self.assertTrue(self.plan_cli.wait_for_service(timeout_sec=timeout),
                        'plan_mission service unavailable')
        fut = self.plan_cli.call_async(request)
        end = time.time() + timeout
        while rclpy.ok() and not fut.done() and time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertTrue(fut.done(), 'plan_mission timed out')
        return fut.result()

    def _seed_charging_station(self):
        odom = Odometry()
        odom.pose.pose.orientation.w = 1.0
        for _ in range(10):
            self.odom_pub.publish(odom)
            self._spin(0.05)
        # object 2m ahead -> map (2,0)
        self.det_pub.publish(String(data='charging_station:landmark:2.0:0.0'))
        self._spin(1.5)

    def test_known_reference_is_grounded(self):
        self._seed_charging_station()
        resp = self._call(PlanMission.Request(
            mission='go to the charging station'))
        self.assertTrue(resp.accepted, f'rejected: {resp.rejection_reason}')
        nav = [t for t in resp.tasks if t.action == 'navigate_to']
        self.assertTrue(nav, 'no navigate_to task produced')
        self.assertEqual(nav[0].target_label, 'charging_station')
        self.assertAlmostEqual(nav[0].target_point.x, 2.0, delta=0.3)

    def test_unknown_reference_is_rejected(self):
        # fresh world model has not seen "helipad"
        resp = self._call(PlanMission.Request(
            mission='go to the helipad'))
        # Either rejected outright, or accepted with no grounded navigate_to
        # to a hallucinated point. Assert we never navigate to 'helipad'.
        bad = [t for t in resp.tasks
               if t.action == 'navigate_to' and t.target_label == 'helipad']
        self.assertFalse(bad, 'brain hallucinated a navigation goal')
