#!/usr/bin/env python3
"""
crip_memory.memory_node

The CRIP semantic memory (Milestone 8 / Module 7).

Stores mission experiences as embedded vectors and answers "have I
encountered something like this before?" by cosine similarity -- semantic
recall, not exact-label lookup.

Services:
    ~/record_experience  (crip_interfaces/RecordExperience)
    ~/recall_experience  (crip_interfaces/RecallExperience)

Backends are auto-selected:
    embedder: CRIP_EMBED_BACKEND = hashing (default) | local | api
    store:    pgvector if CRIP_PG_DSN is reachable, else in-process NumPy
              (optionally persisted via CRIP_MEMORY_PATH)

The LLM brain (M3) records each executed mission and its outcome here, and can
recall similar past experiences before planning.
"""
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from crip_interfaces.srv import RecordExperience, RecallExperience

from crip_memory.embedder import make_embedder
from crip_memory.store import make_store


class MemoryNode(Node):
    def __init__(self):
        super().__init__('memory')
        self.embedder = make_embedder(self.get_logger())
        self.store = make_store(self.embedder.dim, self.embedder.name,
                                self.get_logger())

        self.srv_record = self.create_service(
            RecordExperience, '~/record_experience', self.handle_record)
        self.srv_recall = self.create_service(
            RecallExperience, '~/recall_experience', self.handle_recall)
        self.pub_log = self.create_publisher(String, 'memory/log', 10)
        self.get_logger().info(
            f'memory node ready (embedder={self.embedder.name}, '
            f'dim={self.embedder.dim}, store={self.store.backend}, '
            f'count={self.store.count()})')

    def handle_record(self, req, resp):
        try:
            md = json.loads(req.metadata_json) if req.metadata_json else {}
        except json.JSONDecodeError:
            md = {"raw": req.metadata_json}
        try:
            vec = self.embedder.embed(req.text)
            mid = self.store.add(req.text, vec, md)
            resp.ok = True
            resp.id = int(mid)
            self.pub_log.publish(String(
                data=f"recorded #{mid}: {req.text[:80]}"))
        except Exception as exc:  # noqa: BLE001
            resp.ok = False
            resp.id = -1
            self.get_logger().warn(f"record failed: {exc}")
        return resp

    def handle_recall(self, req, resp):
        k = req.k if req.k and req.k > 0 else 5
        try:
            vec = self.embedder.embed(req.query)
            results = self.store.search(vec, k)
            resp.ok = True
            resp.texts = [r.text for r in results]
            resp.scores = [float(r.score) for r in results]
            resp.metadata_json = [json.dumps(r.metadata) for r in results]
        except Exception as exc:  # noqa: BLE001
            resp.ok = False
            self.get_logger().warn(f"recall failed: {exc}")
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = MemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
