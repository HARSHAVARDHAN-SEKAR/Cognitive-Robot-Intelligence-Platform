#!/usr/bin/env python3
"""
crip_learning.trainer_node

Runs the self-learning pipeline on a schedule (the "nightly retraining" of
Module 8, compressed to a configurable interval for demos).

On each tick it launches the pipeline in a background thread (training can
take a while and must not block the executor), then publishes the result on
/learning/status. A promoted model becomes <registry>/current/model.onnx,
which the prediction node loads on (re)start.

Parameters:
    log_path     -- experience JSONL (matches the logger node)
    registry_dir -- model registry directory
    interval_sec -- seconds between training runs (default 3600)
    epochs       -- training epochs per run
    min_improvement -- relative val-MSE improvement required to promote
    run_on_start -- run once immediately on startup

Training deps (torch/onnx) are only imported inside the pipeline, so this node
loads even where they are absent; a run will then fail gracefully and report
the error on /learning/status.
"""
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TrainerNode(Node):
    def __init__(self):
        super().__init__('learning_trainer')
        import os
        home = os.path.expanduser('~')
        self.declare_parameter('log_path', os.path.join(home, '.crip', 'experience.jsonl'))
        self.declare_parameter('registry_dir', os.path.join(home, '.crip', 'registry'))
        self.declare_parameter('interval_sec', 3600.0)
        self.declare_parameter('epochs', 15)
        self.declare_parameter('min_improvement', 0.02)
        self.declare_parameter('run_on_start', True)

        self.log_path = self.get_parameter('log_path').value
        self.registry_dir = self.get_parameter('registry_dir').value
        self.epochs = int(self.get_parameter('epochs').value)
        self.min_improvement = float(self.get_parameter('min_improvement').value)

        self.pub = self.create_publisher(String, 'learning/status', 10)
        self._busy = threading.Lock()

        interval = float(self.get_parameter('interval_sec').value)
        self.create_timer(interval, self._kick)
        self.get_logger().info(
            f'learning_trainer ready (interval={interval}s, '
            f'registry={self.registry_dir})')
        if self.get_parameter('run_on_start').value:
            self._kick()

    def _kick(self):
        if not self._busy.acquire(blocking=False):
            self.get_logger().info('previous training run still active; skipping')
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            from crip_learning.pipeline import run_pipeline
            self.pub.publish(String(data='training run started'))
            result = run_pipeline(
                self.log_path, self.registry_dir, epochs=self.epochs,
                min_improvement=self.min_improvement,
                logger=lambda m: self.get_logger().info(m))
            status = (f"{result['version']}: "
                      f"{'PROMOTED' if result['promoted'] else 'kept'} -- "
                      f"{result['reason']}")
            self.pub.publish(String(data=status))
            self.get_logger().info(f'learning run complete: {status}')
        except Exception as exc:  # noqa: BLE001
            msg = f'training run failed: {exc}'
            self.pub.publish(String(data=msg))
            self.get_logger().warn(msg)
        finally:
            self._busy.release()


def main(args=None):
    rclpy.init(args=args)
    node = TrainerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
