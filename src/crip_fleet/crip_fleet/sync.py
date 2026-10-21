#!/usr/bin/env python3
"""
crip_fleet.sync

Fleet-intelligence core (Module 11), broker-agnostic and unit-testable.

Multiple robots share what they learn: discovered semantic entities, mission
outcomes, and model-promotion announcements. Each robot publishes to topics
namespaced under `crip/fleet/...`; every robot subscribes to the others'.

The transport is abstracted so the same logic runs over:
  * InMemoryBroker -- a process-local pub/sub used in tests and single-host
    demos (no external dependency).
  * MqttTransport  -- a real MQTT client (paho-mqtt) for a deployed fleet.

`FleetSync` handles identity (each robot has a robot_id, and ignores its own
messages echoed back), serialization, and merge policy: incoming shared
entities are deduplicated by (label, rounded cell) just like the M2 world
model, so the fleet converges on one consistent map instead of double-counting.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---- transports ----
class Transport:
    """Minimal pub/sub interface."""
    def publish(self, topic: str, payload: str):
        raise NotImplementedError

    def subscribe(self, topic_prefix: str, cb: Callable[[str, str], None]):
        raise NotImplementedError


class InMemoryBroker(Transport):
    """Process-local broker. Multiple FleetSync instances can share one to
    simulate a fleet on a single host (and in tests)."""
    def __init__(self):
        self._subs: List[Tuple[str, Callable[[str, str], None]]] = []

    def publish(self, topic: str, payload: str):
        for prefix, cb in list(self._subs):
            if topic.startswith(prefix):
                cb(topic, payload)

    def subscribe(self, topic_prefix: str, cb):
        self._subs.append((topic_prefix, cb))


class MqttTransport(Transport):
    """Real MQTT transport via paho-mqtt (deployment). Imported lazily so the
    package works without paho installed."""
    def __init__(self, host: str = "localhost", port: int = 1883):
        import paho.mqtt.client as mqtt
        self.client = mqtt.Client()
        self._cbs: List[Tuple[str, Callable[[str, str], None]]] = []
        self.client.on_message = self._on_message
        self.client.connect(host, port, 60)
        self.client.loop_start()

    def _on_message(self, client, userdata, msg):
        payload = msg.payload.decode()
        for prefix, cb in list(self._cbs):
            if msg.topic.startswith(prefix):
                cb(msg.topic, payload)

    def publish(self, topic: str, payload: str):
        self.client.publish(topic, payload)

    def subscribe(self, topic_prefix: str, cb):
        self._cbs.append((topic_prefix, cb))
        self.client.subscribe(topic_prefix + "#")


# ---- fleet sync ----
FLEET_ROOT = "crip/fleet"


@dataclass
class SharedEntity:
    label: str
    category: str
    x: float
    y: float
    confidence: float
    robot_id: str
    t: float = field(default_factory=time.time)


def _cell(x: float, y: float, size: float = 0.5) -> str:
    return f"{round(x/size)}:{round(y/size)}"


class FleetSync:
    def __init__(self, robot_id: str, transport: Transport):
        self.robot_id = robot_id
        self.tx = transport
        # merged fleet view: key (label, cell) -> SharedEntity
        self.entities: Dict[Tuple[str, str], SharedEntity] = {}
        self.peer_missions: List[dict] = []
        self.peer_models: List[dict] = []
        self._on_entity: Optional[Callable[[SharedEntity], None]] = None
        self.tx.subscribe(FLEET_ROOT, self._on_message)

    # ----- outbound -----
    def share_entity(self, label, category, x, y, confidence):
        e = SharedEntity(label, category, x, y, confidence, self.robot_id)
        self._merge_entity(e)  # include our own in the merged view
        self.tx.publish(f"{FLEET_ROOT}/entity",
                        json.dumps(self._entity_to_dict(e)))

    def share_mission(self, mission, outcome):
        self.tx.publish(f"{FLEET_ROOT}/mission", json.dumps(
            {"robot_id": self.robot_id, "mission": mission,
             "outcome": outcome, "t": time.time()}))

    def announce_model(self, version, val_metric):
        self.tx.publish(f"{FLEET_ROOT}/model", json.dumps(
            {"robot_id": self.robot_id, "version": version,
             "metric": val_metric, "t": time.time()}))

    # ----- inbound -----
    def _on_message(self, topic: str, payload: str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return
        if data.get("robot_id") == self.robot_id:
            return  # ignore our own echo
        if topic.endswith("/entity"):
            self._merge_entity(self._dict_to_entity(data))
        elif topic.endswith("/mission"):
            self.peer_missions.append(data)
        elif topic.endswith("/model"):
            self.peer_models.append(data)

    def _merge_entity(self, e: SharedEntity):
        key = (e.label, _cell(e.x, e.y))
        existing = self.entities.get(key)
        if existing is None or e.confidence > existing.confidence:
            self.entities[key] = e
            if self._on_entity and e.robot_id != self.robot_id:
                self._on_entity(e)

    def on_peer_entity(self, cb: Callable[[SharedEntity], None]):
        """Register a callback fired when a *peer's* entity is merged in
        (e.g. to inject it into the local world model)."""
        self._on_entity = cb

    # ----- views -----
    def fleet_entities(self) -> List[SharedEntity]:
        return list(self.entities.values())

    @staticmethod
    def _entity_to_dict(e: SharedEntity) -> dict:
        return {"label": e.label, "category": e.category, "x": e.x, "y": e.y,
                "confidence": e.confidence, "robot_id": e.robot_id, "t": e.t}

    @staticmethod
    def _dict_to_entity(d: dict) -> SharedEntity:
        return SharedEntity(d["label"], d.get("category", ""), d["x"], d["y"],
                            d.get("confidence", 1.0), d.get("robot_id", "?"),
                            d.get("t", time.time()))
