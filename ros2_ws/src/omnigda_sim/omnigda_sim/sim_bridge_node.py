"""ROS 2 simulation bridge wrapping ManiSkill 3 ClutteredTabletopSim and FrankaGraspController."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import rclpy
from omnigda_interfaces.action import ExecuteGrasp
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster

from omnigda.sim.clutter_env import ClutteredTabletopSim
from omnigda.sim.controller import FrankaGraspController
from omnigda_sim.conversions import (
    matrix_to_transform_stamped,
    rgbd_frame_to_ros_msgs,
    ros_msg_to_grasp,
)


def _sapien_pose_to_matrix(pose: Any) -> np.ndarray:
    """Convert a SAPIEN Pose (p, q=[w,x,y,z]) into a (4, 4) float64 SE(3) matrix."""
    if hasattr(pose, "to_transformation_matrix"):
        return np.asarray(pose.to_transformation_matrix(), dtype=np.float64).reshape(4, 4)
    p = np.asarray(pose.p, dtype=np.float64).reshape(3)
    q_wxyz = np.asarray(pose.q, dtype=np.float64).reshape(4)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    norm = float(np.linalg.norm(q_xyzw))
    q_xyzw = q_xyzw / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    mat[:3, 3] = p
    return mat


class SimBridgeNode(Node):
    """Bridges ManiSkill 3 cluttered tabletop environment to ROS 2 topics, TF2, services, and actions."""

    def __init__(
        self,
        sim: ClutteredTabletopSim | None = None,
        controller: FrankaGraspController | None = None,
        node_name: str = "sim_bridge_node",
        **node_kwargs: Any,
    ) -> None:
        super().__init__(node_name, **node_kwargs)

        self.declare_parameter("seed", 42)
        self.declare_parameter("num_objects", 6)
        self.declare_parameter("include_novel", False)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("world_frame_id", "world")
        self.declare_parameter("camera_frame_id", "camera_color_optical_frame")
        self.declare_parameter("robot_base_frame_id", "panda_link0")
        self.declare_parameter("sim_backend", "cpu")
        self.declare_parameter("init_controller", True)

        self._seed = int(self.get_parameter("seed").value)
        self._num_objects = int(self.get_parameter("num_objects").value)
        self._include_novel = bool(self.get_parameter("include_novel").value)
        self._image_width = int(self.get_parameter("image_width").value)
        self._image_height = int(self.get_parameter("image_height").value)
        self._publish_rate_hz = max(0.5, float(self.get_parameter("publish_rate_hz").value))
        self._world_frame_id = str(self.get_parameter("world_frame_id").value)
        self._camera_frame_id = str(self.get_parameter("camera_frame_id").value)
        self._robot_base_frame_id = str(self.get_parameter("robot_base_frame_id").value)
        self._sim_backend = str(self.get_parameter("sim_backend").value)
        self._init_controller = bool(self.get_parameter("init_controller").value)

        self._sim_lock = threading.RLock()
        self._owns_sim = sim is None
        if sim is None:
            self.get_logger().info(
                f"Initializing ClutteredTabletopSim (seed={self._seed}, num_objects={self._num_objects}, "
                f"include_novel={self._include_novel}, {self._image_width}x{self._image_height})"
            )
            self._sim = ClutteredTabletopSim(
                num_objects=self._num_objects,
                include_novel=self._include_novel,
                image_width=self._image_width,
                image_height=self._image_height,
                seed=self._seed,
                sim_backend=self._sim_backend,
            )
            self._sim.reset(seed=self._seed)
        else:
            self._sim = sim

        self._controller: FrankaGraspController | None = controller
        if self._controller is None and self._init_controller:
            self._controller = FrankaGraspController(self._sim)

        self._active_camera: str = "base_camera"
        self._cb_group = ReentrantCallbackGroup()

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._rgb_pub = self.create_publisher(Image, "/camera/color/image_raw", sensor_qos)
        self._depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", sensor_qos)
        self._info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", sensor_qos)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._switch_cam_srv = self.create_service(
            SetBool,
            "/sim/switch_camera",
            self._handle_switch_camera,
            callback_group=self._cb_group,
        )

        self._execute_grasp_server = ActionServer(
            self,
            ExecuteGrasp,
            "/manip/execute_grasp",
            execute_callback=self._execute_grasp_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        period = 1.0 / self._publish_rate_hz
        self._timer = self.create_timer(period, self.publish_observation, callback_group=self._cb_group)
        self.get_logger().info(
            f"SimBridgeNode ready (active_camera={self._active_camera}, "
            f"objects={self._sim.get_object_names()})"
        )

    @property
    def active_camera(self) -> str:
        """Currently active ManiSkill camera UID."""
        return self._active_camera

    @property
    def sim(self) -> ClutteredTabletopSim:
        """Underlying ClutteredTabletopSim instance."""
        return self._sim

    def _ensure_controller(self) -> FrankaGraspController:
        if self._controller is None:
            self._controller = FrankaGraspController(self._sim)
        return self._controller

    def publish_observation(self) -> None:
        """Capture RGB-D from the active camera and publish images, camera_info, and TF2 frames."""
        with self._sim_lock:
            frame, _ = self._sim.capture_rgbd(camera_name=self._active_camera)
            robot_base_mat = _sapien_pose_to_matrix(self._sim.robot_base_pose())

        stamp = self.get_clock().now().to_msg()
        rgb_msg, depth_msg, info_msg = rgbd_frame_to_ros_msgs(
            frame,
            frame_id=self._camera_frame_id,
            stamp=stamp,
        )

        self._rgb_pub.publish(rgb_msg)
        self._depth_pub.publish(depth_msg)
        self._info_pub.publish(info_msg)

        cam_tf = matrix_to_transform_stamped(
            frame.camera_to_world,
            parent_frame_id=self._world_frame_id,
            child_frame_id=self._camera_frame_id,
            stamp=stamp,
        )
        robot_tf = matrix_to_transform_stamped(
            robot_base_mat,
            parent_frame_id=self._world_frame_id,
            child_frame_id=self._robot_base_frame_id,
            stamp=stamp,
        )
        self._tf_broadcaster.sendTransform([cam_tf, robot_tf])

    def _handle_switch_camera(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        target_cam = "nbv_camera" if bool(request.data) else "base_camera"
        with self._sim_lock:
            available = self._sim.get_camera_names()
            if target_cam not in available:
                response.success = False
                response.message = f"Camera {target_cam!r} not available (have {available})"
                return response
            self._active_camera = target_cam

        # Immediately broadcast an observation from the newly selected viewpoint
        self.publish_observation()
        response.success = True
        response.message = f"Switched active camera to {target_cam}"
        self.get_logger().info(response.message)
        return response

    def _goal_callback(self, _goal_request: ExecuteGrasp.Goal) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle: Any) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_grasp_callback(self, goal_handle: Any) -> ExecuteGrasp.Result:
        t0 = time.perf_counter()
        req: ExecuteGrasp.Goal = goal_handle.request
        feedback = ExecuteGrasp.Feedback()

        def _publish_feedback(stage: str, progress: float) -> None:
            feedback.stage = stage
            feedback.progress = float(progress)
            goal_handle.publish_feedback(feedback)

        _publish_feedback("validating_grasp", 0.1)
        grasp = ros_msg_to_grasp(req.grasp)
        place_pos = (
            np.array(
                [req.place_position.x, req.place_position.y, req.place_position.z],
                dtype=np.float64,
            )
            if req.do_place
            else None
        )

        result = ExecuteGrasp.Result()
        try:
            _publish_feedback("planning_motion", 0.3)
            with self._sim_lock:
                controller = self._ensure_controller()
                stage_name = "executing_pick_and_place" if req.do_place else "executing_pick"
                _publish_feedback(stage_name, 0.6)
                target_name = grasp.target_class_name or None
                if target_name is not None and target_name not in self._sim.get_object_names():
                    target_name = None

                if place_pos is not None:
                    exec_res = controller.execute_pick_and_place(
                        grasp=grasp,
                        place_position=place_pos,
                        target_object_name=target_name,
                    )
                else:
                    exec_res = controller.execute_pick(
                        grasp=grasp,
                        target_object_name=target_name,
                    )

            self.publish_observation()
            _publish_feedback("completed" if exec_res.success else "failed", 1.0)

            result.success = bool(exec_res.success)
            result.message = str(exec_res.message)
            result.execution_time_sec = float(time.perf_counter() - t0)
            goal_handle.succeed()
            return result
        except Exception as exc:
            self.get_logger().error(f"ExecuteGrasp failed with exception: {exc}")
            _publish_feedback("error", 1.0)
            result.success = False
            result.message = f"ExecuteGrasp exception: {exc}"
            result.execution_time_sec = float(time.perf_counter() - t0)
            goal_handle.abort()
            return result

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, "_timer") and self._timer is not None:
                self._timer.cancel()
        except Exception:
            pass
        if self._owns_sim and hasattr(self, "_sim") and self._sim is not None:
            with self._sim_lock:
                try:
                    self._sim.close()
                except Exception:
                    pass
                self._sim = None  # type: ignore[assignment]
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimBridgeNode()
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
