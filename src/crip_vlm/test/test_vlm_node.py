#!/usr/bin/env python3
"""
Integration test for the CRIP VLM node (Milestone 4).

Launches the vlm node (mock backend), publishes a synthetic rgb8 Image, then
calls ~/analyze_scene and asserts a structured assessment is returned. Also
checks that a red-dominant frame yields a non-"none" risk level via the mock
heuristic, and that an alert is published for it.
"""
import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from crip_interfaces.srv import AnalyzeScene


@pytest.mark.launch_test
def generate_test_description():
    vlm = launch_ros.actions.Node(
        package='crip_vlm', executable='vlm', name='vlm', output='screen',
        additional_env={'CRIP_VLM_BACKEND': 'mock'})
    return (
        launch.LaunchDescription([
            vlm,
            launch_testing.actions.ReadyToTest(),
        ]),
        {},
    )


def make_image(r, g, b, w=32, h=32):
    msg = Image()
    msg.width = w
    msg.height = h
    msg.encoding = 'rgb8'
    msg.step = w * 3
    msg.data = bytes([r, g, b] * (w * h))
    return msg


class TestVLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_vlm')
        self.img_pub = self.node.create_publisher(Image, 'camera/image_raw', 10)
        self.cli = self.node.create_client(AnalyzeScene, '/vlm/analyze_scene')
        self.alerts = []
        self.node.create_subscription(String, 'vlm/alert',
                                      lambda m: self.alerts.append(m.data), 10)

    def tearDown(self):
        self.node.destroy_node()

    def _spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _analyze(self, timeout=8.0):
        self.assertTrue(self.cli.wait_for_service(timeout_sec=timeout),
                        'analyze_scene service unavailable')
        fut = self.cli.call_async(AnalyzeScene.Request(question=''))
        end = time.time() + timeout
        while rclpy.ok() and not fut.done() and time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertTrue(fut.done(), 'analyze_scene timed out')
        return fut.result()

    def test_assessment_returned_for_red_frame(self):
        # publish a red frame for a moment so the node latches it
        for _ in range(10):
            self.img_pub.publish(make_image(220, 30, 30))
            self._spin(0.05)
        resp = self._analyze()
        self.assertTrue(resp.ok, f'analyze failed: {resp.error}')
        self.assertIn(resp.assessment.risk_level,
                      ('none', 'low', 'medium', 'high'))
        self.assertTrue(resp.assessment.caption)
        self.assertEqual(resp.assessment.backend, 'mock')
        # red heuristic should raise risk above none and fire an alert
        self.assertEqual(resp.assessment.risk_level, 'medium')
        self._spin(0.5)
        self.assertTrue(any('MEDIUM' in a for a in self.alerts),
                        'expected a medium-risk alert to be published')

    def test_analyze_without_frame_errors_gracefully(self):
        # no image published in this test -> service returns ok=False, no crash
        resp = self._analyze()
        self.assertFalse(resp.ok)
        self.assertTrue(resp.error)
