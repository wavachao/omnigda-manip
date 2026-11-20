"""ROS 2 coordinator node hosting /agent/query_scene and /agent/execute_instruction."""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Point
from omnigda_interfaces.action import ExecuteGrasp, ExecuteInstruction
from omnigda_interfaces.msg import GraspPose, GraspPoseArray, SceneGraph
from omnigda_interfaces.srv import QueryScene
from omnigda_sim.conversions import ros_msg_to_scene_graph
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool

from omnigda.core.schemas import ManipulationInstruction, Object3DDetection, SceneGraph3D
from omnigda.perception_3d.scene_graph import SceneGraphBuilder3D

_SPATIAL_PATTERNS: list[tuple[str, str]] = [
    (r"\bto the left of\b|\bleft of\b", "left_of"),
    (r"\bto the right of\b|\bright of\b", "right_of"),
    (r"\bin front of\b|\bfront of\b", "front_of"),
    (r"\bbehind\b|\bback of\b", "behind"),
    (r"\bon top of\b|\babove\b", "above"),
    (r"\bnext to\b|\bnear\b|\bbeside\b|\bclose to\b", "near"),
]

_SUPERLATIVE_PATTERNS: list[tuple[str, str]] = [
    (r"\bleftmost\b", "leftmost"),
    (r"\brightmost\b", "rightmost"),
]

_STOP_WORDS = {
    "please",
    "pick",
    "up",
    "grasp",
    "grab",
    "take",
    "get",
    "fetch",
    "inspect",
    "find",
    "locate",
    "the",
    "a",
    "an",
    "object",
    "item",
    "thing",
    "that",
    "is",
    "which",
}


def parse_instruction_fallback(raw_text: str) -> ManipulationInstruction:
    """Lightweight rule-based fallback parser used when T7 SpatialGrounder3D is not attached."""
    text = raw_text.strip()
    lower = text.lower()

    action = "pick"
    if "place" in lower or "put" in lower or "move" in lower:
        action = "pick_and_place"
    elif lower.startswith("inspect") or lower.startswith("find") or lower.startswith("where"):
        action = "inspect"

    spatial_relation: str | None = None
    target_part = lower
    ref_part: str | None = None

    for pattern, rel_name in _SPATIAL_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            spatial_relation = rel_name
            target_part = lower[: m.start()]
            ref_part = lower[m.end() :]
            break

    if spatial_relation is None:
        for pattern, rel_name in _SUPERLATIVE_PATTERNS:
            if re.search(pattern, lower):
                spatial_relation = rel_name
                target_part = re.sub(pattern, " ", target_part)
                break

    def _clean_phrase(phrase: str | None) -> str:
        if not phrase:
            return ""
        # Strip trailing placement clauses
        phrase = re.split(r"\band\s+(?:place|put|move)\b|\bonto\b|\binto\b", phrase)[0]
        tokens = re.findall(r"[a-z0-9_]+", phrase)
        kept = [t for t in tokens if t not in _STOP_WORDS]
        return " ".join(kept)

    target_query = _clean_phrase(target_part)
    reference_query = _clean_phrase(ref_part) or None

    return ManipulationInstruction(
        raw_text=text,
        action=action,
        target_query=target_query,
        spatial_relation=spatial_relation,
        reference_query=reference_query,
    )


class CoordinatorNode(Node):
    """Coordinates SceneGraph grounding, NBV view switching, grasp selection, and execution."""

    def __init__(
        self,
        grounder: Any | None = None,
        agent: Any | None = None,
        node_name: str = "coordinator_node",
        **node_kwargs: Any,
    ) -> None:
        super().__init__(node_name, **node_kwargs)

        self.declare_parameter("entropy_nbv_threshold", 0.85)
        self.declare_parameter("confidence_accept_threshold", 0.60)
        self.declare_parameter("max_views", 3)
        self.declare_parameter("grasp_timeout_sec", 30.0)
        self.declare_parameter("nbv_wait_sec", 0.6)

        self._entropy_nbv_threshold = float(self.get_parameter("entropy_nbv_threshold").value)
        self._confidence_accept_threshold = float(
            self.get_parameter("confidence_accept_threshold").value
        )
        self._max_views = max(1, int(self.get_parameter("max_views").value))
        self._grasp_timeout_sec = float(self.get_parameter("grasp_timeout_sec").value)
        self._nbv_wait_sec = float(self.get_parameter("nbv_wait_sec").value)

        self._lock = threading.RLock()
        self._latest_scene_graph: SceneGraph3D | None = None
        self._scene_graph_seq: int = 0
        self._latest_grasps: GraspPoseArray | None = None
        self._sg_builder = SceneGraphBuilder3D()

        # Pluggable T7 OmniGDAAgent / SpatialGrounder3D hooks
        self._grounder: Any | None = grounder
        self._agent: Any | None = agent
        if self._grounder is None:
            self._try_attach_t7_grounder()

        self._cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            SceneGraph,
            "/perception/scene_graph",
            self._on_scene_graph,
            10,
            callback_group=self._cb_group,
        )
        self.create_subscription(
            GraspPoseArray,
            "/grasp/candidates",
            self._on_grasps,
            10,
            callback_group=self._cb_group,
        )

        self._execute_grasp_client = ActionClient(
            self,
            ExecuteGrasp,
            "/manip/execute_grasp",
            callback_group=self._cb_group,
        )
        self._switch_cam_client = self.create_client(
            SetBool,
            "/sim/switch_camera",
            callback_group=self._cb_group,
        )

        self._query_srv = self.create_service(
            QueryScene,
            "/agent/query_scene",
            self._handle_query_scene,
            callback_group=self._cb_group,
        )
        self._execute_instruction_server = ActionServer(
            self,
            ExecuteInstruction,
            "/agent/execute_instruction",
            execute_callback=self._execute_instruction_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"CoordinatorNode ready (t7_grounder={self._grounder is not None}, "
            f"t7_agent={self._agent is not None})"
        )

    def _try_attach_t7_grounder(self) -> None:
        """Auto-attach T7 SpatialGrounder3D if omnigda.agent is available in the workspace."""
        try:
            from omnigda.agent.grounder import SpatialGrounder3D  # type: ignore[import-not-found]

            self._grounder = SpatialGrounder3D()
            self.get_logger().info("Auto-attached omnigda.agent.grounder.SpatialGrounder3D")
        except Exception:
            self._grounder = None

    def set_grounder(self, grounder: Any) -> None:
        """Attach a custom 3D spatial grounder (e.g., T7 SpatialGrounder3D)."""
        with self._lock:
            self._grounder = grounder

    def set_agent(self, agent: Any) -> None:
        """Attach a custom manipulation agent (e.g., T7 OmniGDAAgent)."""
        with self._lock:
            self._agent = agent

    def _on_scene_graph(self, msg: SceneGraph) -> None:
        sg = ros_msg_to_scene_graph(msg)
        with self._lock:
            self._latest_scene_graph = sg
            self._scene_graph_seq += 1

    def _on_grasps(self, msg: GraspPoseArray) -> None:
        with self._lock:
            self._latest_grasps = msg

    def _needs_nbv(self, det: Object3DDetection) -> bool:
        return bool(
            det.is_ood
            or det.entropy > self._entropy_nbv_threshold
            or det.confidence < self._confidence_accept_threshold
        )

    def _ground_instruction(
        self,
        instruction: str,
        scene_graph: SceneGraph3D,
    ) -> tuple[Object3DDetection | None, bool, str]:
        """Ground a natural-language instruction against a SceneGraph3D."""
        with self._lock:
            grounder = self._grounder

        if grounder is not None and hasattr(grounder, "ground"):
            res = grounder.ground(instruction, scene_graph)
            if isinstance(res, Object3DDetection):
                return res, self._needs_nbv(res), f"Grounded via {type(grounder).__name__}"
            if hasattr(res, "target"):
                target = res.target
                needs_nbv = bool(
                    getattr(res, "needs_nbv", self._needs_nbv(target) if target else True)
                )
                reasoning = str(getattr(res, "reasoning", f"Grounded via {type(grounder).__name__}"))
                return target, needs_nbv, reasoning

        parsed = parse_instruction_fallback(instruction)
        target = self._sg_builder.resolve_query(
            scene_graph=scene_graph,
            target_query=parsed.target_query or None,
            spatial_relation=parsed.spatial_relation,
            reference_query=parsed.reference_query,
            min_confidence=0.15,
        )

        # Graceful token-overlap fallback if exact substring match didn't trigger
        if target is None and parsed.target_query and scene_graph.objects:
            query_tokens = set(parsed.target_query.replace("_", " ").split())
            best_obj: Object3DDetection | None = None
            best_score = -1.0
            for obj in scene_graph.objects:
                cname_tokens = set(obj.class_name.lower().replace("_", " ").split())
                overlap = len(query_tokens & cname_tokens)
                if overlap > 0:
                    score = float(overlap) + obj.confidence
                    if score > best_score:
                        best_score = score
                        best_obj = obj
            target = best_obj

        if target is None:
            available = [o.class_name for o in scene_graph.objects]
            return (
                None,
                True,
                f"No object matched {instruction!r} (parsed target={parsed.target_query!r}, "
                f"rel={parsed.spatial_relation!r}, available={available}).",
            )

        needs_nbv = self._needs_nbv(target)
        reasoning = (
            f"Resolved target={target.class_name!r} (#{target.instance_id}, "
            f"conf={target.confidence:.2f}, entropy={target.entropy:.2f}, "
            f"relation={parsed.spatial_relation}, needs_nbv={needs_nbv})."
        )
        return target, needs_nbv, reasoning

    def _select_best_grasp(
        self,
        target: Object3DDetection,
        grasp_array: GraspPoseArray | None,
    ) -> GraspPose | None:
        if grasp_array is None or not grasp_array.grasps:
            return None

        # Prefer collision-free grasps matching target_instance_id, then class_name
        exact_id = [
            g for g in grasp_array.grasps if g.collision_free and g.target_instance_id == target.instance_id
        ]
        if exact_id:
            return max(exact_id, key=lambda g: g.score)

        exact_cls = [
            g
            for g in grasp_array.grasps
            if g.collision_free and g.target_class_name.lower() == target.class_name.lower()
        ]
        if exact_cls:
            return max(exact_cls, key=lambda g: g.score)

        any_valid = [g for g in grasp_array.grasps if g.collision_free]
        if any_valid:
            return max(any_valid, key=lambda g: g.score)
        return None

    def _request_nbv_switch(self, use_nbv: bool = True) -> bool:
        if not self._switch_cam_client.wait_for_service(timeout_sec=1.0):
            return False
        req = SetBool.Request()
        req.data = bool(use_nbv)
        future = self._switch_cam_client.call_async(req)
        deadline = time.perf_counter() + 2.0
        while not future.done() and time.perf_counter() < deadline:
            time.sleep(0.02)
        if not future.done():
            return False
        resp = future.result()
        return bool(resp is not None and resp.success)

    def _handle_query_scene(
        self,
        request: QueryScene.Request,
        response: QueryScene.Response,
    ) -> QueryScene.Response:
        with self._lock:
            sg = self._latest_scene_graph

        if sg is None or not sg.objects:
            response.found = False
            response.target_instance_id = -1
            response.target_class_name = ""
            response.confidence = 0.0
            response.needs_nbv = True
            response.reasoning = "No 3D scene graph observations received yet."
            return response

        target, needs_nbv, reasoning = self._ground_instruction(request.instruction, sg)
        if target is None:
            response.found = False
            response.target_instance_id = -1
            response.target_class_name = ""
            response.confidence = 0.0
            response.needs_nbv = needs_nbv
            response.reasoning = reasoning
            return response

        response.found = True
        response.target_instance_id = int(target.instance_id)
        response.target_class_name = str(target.class_name)
        response.confidence = float(target.confidence)
        response.needs_nbv = bool(needs_nbv)
        response.reasoning = reasoning
        return response

    def _goal_callback(self, _goal_request: ExecuteInstruction.Goal) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle: Any) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_instruction_callback(self, goal_handle: Any) -> ExecuteInstruction.Result:
        instruction = str(goal_handle.request.instruction)
        feedback = ExecuteInstruction.Feedback()
        result = ExecuteInstruction.Result()
        num_views_used = 1

        def _pub_fb(stage: str, progress: float) -> None:
            feedback.stage = stage
            feedback.progress = float(progress)
            goal_handle.publish_feedback(feedback)

        _pub_fb("grounding_instruction", 0.15)
        with self._lock:
            sg = self._latest_scene_graph
            seq_before = self._scene_graph_seq

        if sg is None or not sg.objects:
            result.success = False
            result.message = "No 3D scene graph available for grounding."
            result.target_instance_id = -1
            result.target_class_name = ""
            result.num_views_used = num_views_used
            goal_handle.succeed()
            return result

        target, needs_nbv, reasoning = self._ground_instruction(instruction, sg)

        if (target is None or needs_nbv) and self._max_views > 1:
            _pub_fb("requesting_nbv_view", 0.35)
            if self._request_nbv_switch(use_nbv=True):
                num_views_used += 1
                deadline = time.perf_counter() + self._nbv_wait_sec
                while time.perf_counter() < deadline:
                    with self._lock:
                        if self._scene_graph_seq > seq_before:
                            sg = self._latest_scene_graph
                            break
                    time.sleep(0.03)
                if sg is not None:
                    target, needs_nbv, reasoning = self._ground_instruction(instruction, sg)

        if target is None:
            _pub_fb("target_not_found", 1.0)
            result.success = False
            result.message = reasoning
            result.target_instance_id = -1
            result.target_class_name = ""
            result.num_views_used = num_views_used
            goal_handle.succeed()
            return result

        _pub_fb("selecting_grasp", 0.55)
        with self._lock:
            grasp_array = self._latest_grasps
        chosen_grasp = self._select_best_grasp(target, grasp_array)

        if chosen_grasp is None:
            _pub_fb("no_valid_grasp", 1.0)
            result.success = False
            result.message = (
                f"Grounded {target.class_name!r} (#{target.instance_id}) but no collision-free "
                "grasp candidate is available."
            )
            result.target_instance_id = int(target.instance_id)
            result.target_class_name = str(target.class_name)
            result.num_views_used = num_views_used
            goal_handle.succeed()
            return result

        _pub_fb("executing_grasp", 0.75)
        if not self._execute_grasp_client.wait_for_server(timeout_sec=3.0):
            _pub_fb("grasp_server_unavailable", 1.0)
            result.success = False
            result.message = "/manip/execute_grasp action server unavailable."
            result.target_instance_id = int(target.instance_id)
            result.target_class_name = str(target.class_name)
            result.num_views_used = num_views_used
            goal_handle.succeed()
            return result

        grasp_goal = ExecuteGrasp.Goal()
        grasp_goal.grasp = chosen_grasp
        grasp_goal.place_position = Point(x=0.0, y=0.0, z=0.0)
        grasp_goal.do_place = False

        send_future = self._execute_grasp_client.send_goal_async(grasp_goal)
        deadline = time.perf_counter() + self._grasp_timeout_sec
        while not send_future.done() and time.perf_counter() < deadline:
            time.sleep(0.02)
        if not send_future.done():
            result.success = False
            result.message = "Timed out sending goal to /manip/execute_grasp."
            result.target_instance_id = int(target.instance_id)
            result.target_class_name = str(target.class_name)
            result.num_views_used = num_views_used
            goal_handle.succeed()
            return result

        grasp_goal_handle = send_future.result()
        if grasp_goal_handle is None or not grasp_goal_handle.accepted:
            result.success = False
            result.message = "Goal rejected by /manip/execute_grasp."
            result.target_instance_id = int(target.instance_id)
            result.target_class_name = str(target.class_name)
            result.num_views_used = num_views_used
            goal_handle.succeed()
            return result

        res_future = grasp_goal_handle.get_result_async()
        while not res_future.done() and time.perf_counter() < deadline:
            time.sleep(0.02)

        if not res_future.done() or res_future.result() is None:
            result.success = False
            result.message = "Timed out waiting for /manip/execute_grasp result."
            result.target_instance_id = int(target.instance_id)
            result.target_class_name = str(target.class_name)
            result.num_views_used = num_views_used
            goal_handle.succeed()
            return result

        grasp_res = res_future.result().result
        _pub_fb("completed" if grasp_res.success else "failed", 1.0)
        result.success = bool(grasp_res.success)
        result.message = str(grasp_res.message)
        result.target_instance_id = int(target.instance_id)
        result.target_class_name = str(target.class_name)
        result.num_views_used = num_views_used
        goal_handle.succeed()
        return result


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CoordinatorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
