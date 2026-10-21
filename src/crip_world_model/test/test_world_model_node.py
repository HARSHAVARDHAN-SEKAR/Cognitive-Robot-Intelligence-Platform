#!/usr/bin/env python3
"""
Integration test for the CRIP world model node.

Launches the real world_model node (SQLite in-memory backend), publishes a
robot pose and a detection, then calls the query services and asserts the
entity was ingested, persisted, and is retrievable.
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

from crip_interfaces.srv import HaveISeen, QueryEntities


@pytest.mark.launch_test
def generate_test_description():
    world_model = launch_ros.actions.Node(
        package='crip_world_model', executable='world_model',
        name='world_model', output='screen')
    return (
        launch.LaunchDescription([
            world_model,
            launch_testing.actions.ReadyToTest(),
        ]),
        {},
    )


class TestWorldModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_world_model')
        self.odom_pub = self.node.create_publisher(Odometry, 'odom', 10)
        self.det_pub = self.node.create_publisher(String, 'perception/detections', 10)
        self.seen_cli = self.node.create_client(HaveISeen, '/world_model/have_i_seen')
        self.query_cli = self.node.create_client(QueryEntities, '/world_model/query_entities')

    def tearDown(self):
        self.node.destroy_node()

    def _spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _call(self, client, request, timeout=5.0):
        self.assertTrue(client.wait_for_service(timeout_sec=timeout),
                        f'service {client.srv_name} unavailable')
        future = client.call_async(request)
        end = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertTrue(future.done(), 'service call timed out')
        return future.result()

    def test_ingest_and_query(self):
        # Robot sitting at the origin facing +x.
        odom = Odometry()
        odom.pose.pose.orientation.w = 1.0
        # publish pose for a moment so the node latches it
        for _ in range(10):
            self.odom_pub.publish(odom)
            self._spin(0.05)

        # Detect an object 1m ahead, 0m to the side -> map (1, 0).
        self.det_pub.publish(String(data='charging_station:landmark:1.0:0.0'))
        self._spin(1.0)

        # have_i_seen should now be true.
        resp = self._call(self.seen_cli, HaveISeen.Request(label='charging_station'))
        self.assertTrue(resp.seen, 'world model did not record the detection')
        self.assertGreaterEqual(resp.times_seen, 1)
        self.assertAlmostEqual(resp.closest.position.x, 1.0, delta=0.2)

        # query_entities by label returns it.
        q = QueryEntities.Request(label='charging_station')
        qresp = self._call(self.query_cli, q)
        self.assertTrue(qresp.found)
        self.assertEqual(qresp.entities[0].label, 'charging_station')

    def test_unknown_object_not_seen(self):
        resp = self._call(self.seen_cli, HaveISeen.Request(label='nonexistent_thing'))
        self.assertFalse(resp.seen)
        self.assertEqual(resp.times_seen, 0)
