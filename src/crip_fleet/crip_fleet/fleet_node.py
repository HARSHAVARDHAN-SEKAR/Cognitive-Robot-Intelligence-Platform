#!/usr/bin/env python3
"""
crip_fleet.fleet_node

Fleet-intelligence node (Module 11).

Bridges this robot's discoveries to the shared fleet and folds peers'
discoveries back in:

  * Publishes local detections (/perception/detections) and mission/model
    events to the fleet transport.
  * On receiving a *peer's* entity, injects it into this robot's world model
    via /perception/detections, so a thing one robot saw becomes navigable
    for all -- "one robot learns, all improve".
  * Publishes a /fleet/summary describing the shared map size and peers.

Transport selection:
  * CRIP_FLEET_MQTT=host:port  -> real MQTT (paho-mqtt) for a deployed fleet.
  * otherwise                  -> in-memory broker (single-host/demo). Note an
    in-memory broker only links FleetSync instances in the SAME process; for a
    multi-robot demo on one host, use MQTT (e.g. a local mosquitto).
"""
import os
import re

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from crip_fleet.sync import FleetSync, InMemoryBroker, MqttTransport


class FleetNode(Node):
    def __init__(self):
        super().__init__('fleet')
        self.declare_parameter('robot_id', 'robot_alpha')
        robot_id = self.get_parameter('robot_id').value

        transport, kind = self._make_transport()
        self.sync = FleetSync(robot_id, transport)
        self.sync.on_peer_entity(self._on_peer_entity)

        self.create_subscription(String, 'perception/detections',
                                 self.on_detection, 10)
        self.create_subscription(String, 'memory/log', self.on_mission, 10)
        self.create_subscription(String, 'learning/status', self.on_model, 10)

        # inject peer discoveries back into the local world model
        self.pub_detect = self.create_publisher(String, 'perception/detections', 10)
        self.pub_summary = self.create_publisher(String, 'fleet/summary', 10)
        self.create_timer(3.0, self._publish_summary)
        self.get_logger().info(
            f'fleet node ready (id={robot_id}, transport={kind})')

    def _make_transport(self):
        mqtt = os.environ.get('CRIP_FLEET_MQTT')
        if mqtt:
            try:
                host, _, port = mqtt.partition(':')
                t = MqttTransport(host or 'localhost', int(port or '1883'))
                return t, f'mqtt:{mqtt}'
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f'MQTT unavailable ({exc}); using in-memory broker')
        return InMemoryBroker(), 'in-memory'

    def on_detection(self, msg):
        # "label:category:dx:dy" -- we share absolute-ish positions; here the
        # offsets are small, so share as-is (a deployment would transform to map).
        try:
            label, category, x, y = msg.data.split(':')
            self.sync.share_entity(label, category, float(x), float(y), 1.0)
        except ValueError:
            return

    def on_mission(self, msg):
        outcome = 'unknown'
        mo = re.search(r'outcome:\s*(\w+)', msg.data)
        if mo:
            outcome = mo.group(1)
        self.sync.share_mission(msg.data, outcome)

    def on_model(self, msg):
        if 'PROMOTED' in msg.data:
            self.sync.announce_model(msg.data, 0.0)

    def _on_peer_entity(self, e):
        # inject a peer's discovery into our world model
        self.pub_detect.publish(String(
            data=f'{e.label}:{e.category}:{e.x:.2f}:{e.y:.2f}'))
        self.get_logger().info(
            f'learned from {e.robot_id}: {e.label} @({e.x:.1f},{e.y:.1f})')

    def _publish_summary(self):
        ents = self.sync.fleet_entities()
        peers = {e.robot_id for e in ents} | \
                {m['robot_id'] for m in self.sync.peer_missions}
        self.pub_summary.publish(String(
            data=f'fleet map: {len(ents)} entities; peers seen: '
                 f'{len(peers)}; peer missions: {len(self.sync.peer_missions)}'))


def main(args=None):
    rclpy.init(args=args)
    node = FleetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
