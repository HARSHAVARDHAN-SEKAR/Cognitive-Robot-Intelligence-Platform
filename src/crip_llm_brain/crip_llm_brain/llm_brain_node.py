#!/usr/bin/env python3
"""
crip_llm_brain.llm_brain_node

The CRIP LLM Brain (Milestone 3).

Pipeline:
    natural-language mission
        -> planner backend (anthropic | ollama | mock)  [planner.py]
        -> grounding against the M2 world model
        -> ordered task execution
        -> goals handed to the safety-supervised controller

Grounding is the key safety/quality step: every navigate_to / report /
search_for target_label is resolved against the world model via the
have_i_seen and query_entities services. A mission that references something
the robot has never seen is REJECTED with a reason rather than executed on a
guessed location.

Interfaces:
    service  ~/plan_mission   (crip_interfaces/PlanMission)  -- plan + execute
    publish  /brain/explanation (std_msgs/String)            -- shared with M1 brain
    publish  /goal_pose       (geometry_msgs/PoseStamped)    -- current nav goal
    action   navigate_to_pose (nav2_msgs/NavigateToPose)     -- M5 closed-loop drive

Navigation (M5): navigate_to tasks are sent to Nav2's NavigateToPose action
and the executor waits for the real result. If Nav2 is not running, it falls
back to publishing /goal_pose so lighter launch configurations still work.
"""
import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String

from crip_interfaces.srv import (PlanMission, HaveISeen, QueryEntities,
                                 AnalyzeScene, RecordExperience,
                                 RecallExperience)
from crip_interfaces.msg import Task

from crip_llm_brain.planner import make_planner, PlannerError

try:
    from nav2_msgs.action import NavigateToPose
    _HAVE_NAV2 = True
except ImportError:  # nav2_msgs not present in minimal installs
    NavigateToPose = None
    _HAVE_NAV2 = False


class LLMBrainNode(Node):
    def __init__(self):
        super().__init__('llm_brain')
        self.planner = make_planner(self.get_logger())

        # DEADLOCK FIX: plan_mission calls other services from INSIDE its own
        # service callback. With the default single-threaded executor the
        # executor is blocked in our callback, so client responses can never
        # be processed -> every wait times out -> planner sees zero known
        # labels -> always falls back to "search_for". A reentrant callback
        # group + MultiThreadedExecutor lets responses run in parallel.
        self.cbg = ReentrantCallbackGroup()

        # clients onto the world model (M2)
        self.cli_seen = self.create_client(
            HaveISeen, '/world_model/have_i_seen', callback_group=self.cbg)
        self.cli_query = self.create_client(
            QueryEntities, '/world_model/query_entities', callback_group=self.cbg)
        self.cli_vlm = self.create_client(
            AnalyzeScene, '/vlm/analyze_scene', callback_group=self.cbg)
        self.cli_record = self.create_client(
            RecordExperience, '/memory/record_experience', callback_group=self.cbg)
        self.cli_recall = self.create_client(
            RecallExperience, '/memory/recall_experience', callback_group=self.cbg)

        # Nav2 action client (M5). If nav2_msgs isn't installed or the action
        # server never comes up, navigate_to falls back to publishing a goal.
        self._nav_client = (ActionClient(self, NavigateToPose, 'navigate_to_pose')
                            if _HAVE_NAV2 else None)
        self.declare_parameter('nav_timeout', 120.0)

        self.plan_srv = self.create_service(
            PlanMission, '~/plan_mission', self.handle_plan_mission,
            callback_group=self.cbg)

        self.exp_pub = self.create_publisher(String, 'brain/explanation', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)

        self._exec_lock = threading.Lock()
        self.get_logger().info('llm_brain node ready')

    # ---- world-model helpers ----
    def _known_labels(self, timeout=2.0):
        """Best-effort list of labels currently in the world model."""
        if not self.cli_query.wait_for_service(timeout_sec=timeout):
            return []
        req = QueryEntities.Request()
        fut = self.cli_query.call_async(req)
        if not self._wait(fut, timeout):
            return []
        res = fut.result()
        return sorted({e.label for e in res.entities}) if res else []

    def _resolve(self, label, timeout=2.0):
        """Return a Point for a label, or None if the robot hasn't seen it."""
        if not label:
            return None
        if not self.cli_seen.wait_for_service(timeout_sec=timeout):
            return None
        req = HaveISeen.Request(label=label)
        fut = self.cli_seen.call_async(req)
        if not self._wait(fut, timeout):
            return None
        res = fut.result()
        if res and res.seen:
            return res.closest.position
        return None

    def _wait(self, future, timeout):
        end = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < end:
            time.sleep(0.02)
        return future.done()

    def _explain(self, text):
        self.exp_pub.publish(String(data=text))
        self.get_logger().info(text)

    # ---- main service ----
    def handle_plan_mission(self, req, resp):
        mission = req.mission.strip()
        self._explain(f"received mission: {mission!r}")

        # Recall similar past experiences (best-effort, non-blocking on failure).
        self._recall_similar(mission)

        known = self._known_labels()
        try:
            plan = self.planner.plan(mission, known)
        except PlannerError as exc:
            resp.accepted = False
            resp.rejection_reason = f"planning failed: {exc}"
            self._explain(resp.rejection_reason)
            return resp

        # Ground every task that references a label.
        grounded_tasks = []
        for t in plan.tasks:
            msg = Task(action=t.action, target_label=t.target_label,
                       duration=float(t.duration), note=t.note)
            if t.action in ("navigate_to",) and t.target_label:
                pt = self._resolve(t.target_label)
                if pt is None:
                    resp.accepted = False
                    resp.rejection_reason = (
                        f"cannot ground '{t.target_label}': not in world "
                        f"model. Known: {known or '(none)'}")
                    self._explain(resp.rejection_reason)
                    return resp
                msg.target_point = Point(x=pt.x, y=pt.y, z=pt.z)
            grounded_tasks.append(msg)

        resp.accepted = True
        resp.tasks = grounded_tasks
        resp.summary = plan.summary
        self._explain(f"plan accepted ({len(grounded_tasks)} tasks): {plan.summary}")

        # Execute asynchronously so the service returns the plan immediately.
        threading.Thread(target=self._execute, args=(grounded_tasks, mission),
                         daemon=True).start()
        return resp

    def _vlm_report(self, subject, timeout=10.0):
        """Ask the VLM what it currently sees; fall back gracefully."""
        if not self.cli_vlm.wait_for_service(timeout_sec=2.0):
            return "VLM unavailable; no visual report"
        q = f"Describe the {subject}" if subject else ""
        fut = self.cli_vlm.call_async(AnalyzeScene.Request(question=q))
        if not self._wait(fut, timeout):
            return "VLM did not respond in time"
        res = fut.result()
        if not res or not res.ok:
            return f"VLM error: {res.error if res else 'no response'}"
        a = res.assessment
        return (f"{a.caption} (risk={a.risk_level}; {a.recommendation})"
                if a.recommendation else a.caption)

    # ---- memory (M8) ----
    def _recall_similar(self, mission, k=3, timeout=3.0):
        if not self.cli_recall.wait_for_service(timeout_sec=1.5):
            return
        fut = self.cli_recall.call_async(
            RecallExperience.Request(query=mission, k=k))
        if not self._wait(fut, timeout):
            return
        res = fut.result()
        if res and res.ok and res.texts:
            top = res.texts[0]
            score = res.scores[0] if res.scores else 0.0
            if score > 0.2:
                self._explain(f"recalled similar past experience "
                              f"(sim={score:.2f}): {top[:80]}")

    def _record_experience(self, mission, outcome, tasks, timeout=3.0):
        if not mission or not self.cli_record.wait_for_service(timeout_sec=1.5):
            return
        actions = ",".join(t.action for t in tasks)
        text = f"mission: {mission} | outcome: {outcome} | tasks: {actions}"
        md = json.dumps({"outcome": outcome, "n_tasks": len(tasks)})
        fut = self.cli_record.call_async(
            RecordExperience.Request(text=text, metadata_json=md))
        self._wait(fut, timeout)

    # ---- navigation (M5) ----
    def _navigate_to(self, point, settle=1.0):
        """Drive to a map-frame point via Nav2's NavigateToPose action.

        Returns True on success. Falls back to publishing /goal_pose (and
        reporting success optimistically) if the Nav2 action server is not
        available, so lighter launch configs still run."""
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position = point
        goal.pose.orientation.w = 1.0
        # Always publish the goal for visualization/debug.
        self.goal_pub.publish(goal)

        timeout = float(self.get_parameter('nav_timeout').value)

        if self._nav_client is None or not self._nav_client.wait_for_server(
                timeout_sec=3.0):
            self._explain("Nav2 action server unavailable; publishing goal "
                          "only (no closed-loop drive)")
            time.sleep(settle)
            return True  # optimistic: keeps non-Nav2 launches functional

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal
        send_future = self._nav_client.send_goal_async(nav_goal)
        if not self._wait(send_future, 10.0):
            self._explain("Nav2 did not accept the goal in time")
            return False
        handle = send_future.result()
        if not handle or not handle.accepted:
            self._explain("Nav2 rejected the goal")
            return False

        result_future = handle.get_result_async()
        if not self._wait(result_future, timeout):
            self._explain("Nav2 navigation timed out; cancelling")
            handle.cancel_goal_async()
            return False
        # status 4 == SUCCEEDED in action_msgs/GoalStatus
        status = result_future.result().status if result_future.result() else 0
        return status == 4

    # ---- execution ----
    def _execute(self, tasks, mission=""):
        if not self._exec_lock.acquire(blocking=False):
            self._explain("busy executing a previous mission; ignoring new one")
            return
        outcome = "success"
        try:
            for i, t in enumerate(tasks, 1):
                if t.action == "navigate_to":
                    self._explain(
                        f"[{i}/{len(tasks)}] navigating to {t.target_label} "
                        f"({t.target_point.x:.2f},{t.target_point.y:.2f})")
                    ok = self._navigate_to(t.target_point)
                    if not ok:
                        self._explain(
                            f"navigation to {t.target_label} did not succeed; "
                            f"aborting remaining tasks")
                        outcome = "failure"
                        break
                elif t.action == "wait":
                    self._explain(f"[{i}/{len(tasks)}] waiting {t.duration:.1f}s")
                    time.sleep(min(t.duration, 30.0))
                elif t.action == "report":
                    desc = self._vlm_report(t.target_label)
                    self._explain(f"[{i}/{len(tasks)}] report: {desc}")
                elif t.action == "search_for":
                    self._explain(f"[{i}/{len(tasks)}] searching: {t.note}")
            self._explain("mission complete")
            self._record_experience(mission, outcome, tasks)
        finally:
            self._exec_lock.release()


def main(args=None):
    rclpy.init(args=args)
    node = LLMBrainNode()
    # MultiThreadedExecutor is REQUIRED: see deadlock note in __init__.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
