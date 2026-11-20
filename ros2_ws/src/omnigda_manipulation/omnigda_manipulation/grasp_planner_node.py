"""ROS 2 node running ConfidenceAwareGraspPlanner on 3D object detections and depth."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import rclpy
from omnigda_interfaces.msg import GraspPoseArray, Object3DArray
from omnigda_sim.conversions import (
    camera_info_to_intrinsics,
    grasp_to_ros_msg,
    grasps_to_marker_array,
    image_msg_to_numpy,
    make_header,
    ros_msg_to_object3d,
    transform_to_matrix,
)
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from omnigda.core.config import load_config
from omnigda.core.schemas import GraspCandidate, Object3DDetection
from omnigda.grasp.planner import ConfidenceAwareGraspPlanner


def _sample_obb_surface_points(
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
    points_per_face: int = 81,
) -> np.ndarray:
    """Synthesize a clean surface point cloud on an oriented bounding box (fallback when depth is sparse)."""
    ext = np.maximum(np.asarray(extent, dtype=np.float64).reshape(3), 0.01)
    half = 0.5 * ext
    side = max(4, int(round(np.sqrt(points_per_face))))
    grid = np.linspace(-1.0, 1.0, side, dtype=np.float64)
    u, v = np.meshgrid(grid, grid, indexing="ij")
    u = u.ravel()
    v = v.ravel()

    faces: list[np.ndarray] = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for sign in (-1.0, 1.0):
            pts = np.zeros((u.shape[0], 3), dtype=np.float64)
            pts[:, axis] = sign * half[axis]
            pts[:, other[0]] = u * half[other[0]]
            pts[:, other[1]] = v * half[other[1]]
            faces.append(pts)

    local_pts = np.concatenate(faces, axis=0)
    world_pts = local_pts @ np.asarray(rotation, dtype=np.float64).reshape(3, 3).T + np.asarray(
        center, dtype=np.float64
    ).reshape(1, 3)
    return world_pts.astype(np.float32)


def _backproject_depth_world(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_to_world: np.ndarray,
    depth_min_m: float = 0.15,
    depth_max_m: float = 2.0,
    stride: int = 2,
) -> np.ndarray:
    """Fast strided backprojection of a depth image into world-frame (N, 3) float32 points."""
    h, w = depth_m.shape[:2]
    ys = np.arange(0, h, stride, dtype=np.float64)
    xs = np.arange(0, w, stride, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth_m[::stride, ::stride].astype(np.float64)

    valid = np.isfinite(z) & (z >= depth_min_m) & (z <= depth_max_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    zv = z[valid]
    xv = (grid_x[valid] - cx) * zv / max(fx, 1e-6)
    yv = (grid_y[valid] - cy) * zv / max(fy, 1e-6)
    cam_pts = np.stack([xv, yv, zv], axis=-1)

    rot = camera_to_world[:3, :3]
    trans = camera_to_world[:3, 3]
    world_pts = cam_pts @ rot.T + trans[None, :]
    return world_pts.astype(np.float32)


class GraspPlannerNode(Node):
    """Subscribes to 3D detections (+ depth/camera_info/TF2) and publishes ranked 6DoF grasps."""

    def __init__(
        self,
        planner: ConfidenceAwareGraspPlanner | None = None,
        node_name: str = "grasp_planner_node",
        **node_kwargs: Any,
    ) -> None:
        super().__init__(node_name, **node_kwargs)

        self.declare_parameter("config_path", "")
        self.declare_parameter("world_frame_id", "world")
        self.declare_parameter("camera_frame_id", "camera_color_optical_frame")
        self.declare_parameter("top_k_per_object", 5)
        self.declare_parameter("table_z", 0.0)
        self.declare_parameter("depth_min_m", 0.15)
        self.declare_parameter("depth_max_m", 2.0)
        self.declare_parameter("obb_crop_margin_m", 0.015)

        cfg_path = str(self.get_parameter("config_path").value).strip()
        self._cfg = load_config(cfg_path or None)
        self._world_frame_id = str(self.get_parameter("world_frame_id").value)
        self._camera_frame_id = str(self.get_parameter("camera_frame_id").value)
        self._top_k_per_object = int(self.get_parameter("top_k_per_object").value)
        self._table_z = float(self.get_parameter("table_z").value)
        self._depth_min_m = float(self.get_parameter("depth_min_m").value)
        self._depth_max_m = float(self.get_parameter("depth_max_m").value)
        self._obb_crop_margin_m = float(self.get_parameter("obb_crop_margin_m").value)

        self._planner = (
            planner if planner is not None else ConfidenceAwareGraspPlanner.from_config(self._cfg)
        )
        self._lock = threading.RLock()
        self._latest_depth: Image | None = None
        self._latest_info: CameraInfo | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Image, "/camera/depth/image_raw", self._on_depth, sensor_qos)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, sensor_qos)
        self.create_subscription(Object3DArray, "/perception/objects_3d", self._on_objects, 10)

        self._grasps_pub = self.create_publisher(GraspPoseArray, "/grasp/candidates", 10)
        self._markers_pub = self.create_publisher(MarkerArray, "/grasp/markers", 10)

        self.get_logger().info(
            f"GraspPlannerNode ready (world_frame={self._world_frame_id}, "
            f"top_k_per_object={self._top_k_per_object})"
        )

    def _on_depth(self, msg: Image) -> None:
        with self._lock:
            self._latest_depth = msg

    def _on_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_info = msg

    def _get_scene_world_points(self) -> np.ndarray | None:
        with self._lock:
            depth_msg = self._latest_depth
            info_msg = self._latest_info

        if depth_msg is None or info_msg is None:
            return None

        cam_frame = depth_msg.header.frame_id or info_msg.header.frame_id or self._camera_frame_id
        cam_to_world: np.ndarray | None = None
        for candidate in (cam_frame, self._camera_frame_id):
            if not candidate:
                continue
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self._world_frame_id,
                    candidate,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
                cam_to_world = transform_to_matrix(tf_msg)
                break
            except Exception:
                continue

        if cam_to_world is None:
            return None

        raw_depth = image_msg_to_numpy(depth_msg, copy=False)
        depth_m = (
            raw_depth.astype(np.float32) / 1000.0
            if depth_msg.encoding == "16UC1"
            else raw_depth.astype(np.float32, copy=False)
        )
        intr = camera_info_to_intrinsics(info_msg)
        return _backproject_depth_world(
            depth_m=depth_m,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.cx,
            cy=intr.cy,
            camera_to_world=cam_to_world,
            depth_min_m=self._depth_min_m,
            depth_max_m=self._depth_max_m,
            stride=2,
        )

    def _split_object_and_obstacle_points(
        self,
        det: Object3DDetection,
        scene_xyz: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if scene_xyz is not None and scene_xyz.shape[0] > 0:
            rel = scene_xyz.astype(np.float64) - det.obb_center[None, :]
            local = rel @ det.obb_rotation
            half_ext = 0.5 * np.maximum(det.obb_extent, 0.01) + self._obb_crop_margin_m
            in_obb = np.all(np.abs(local) <= half_ext[None, :], axis=1)
            obj_pts = scene_xyz[in_obb]
            obs_pts = scene_xyz[~in_obb]
            if obj_pts.shape[0] >= 24:
                # Filter tabletop points out of obstacles (tabletop plane is checked directly)
                obs_above = obs_pts[obs_pts[:, 2] > (self._table_z + 0.008)]
                return obj_pts, (obs_above if obs_above.shape[0] > 0 else None)

        # Fallback: dense oriented bounding box surface point cloud
        synth_pts = _sample_obb_surface_points(det.obb_center, det.obb_extent, det.obb_rotation)
        return synth_pts, None

    def _on_objects(self, msg: Object3DArray) -> None:
        t0 = time.perf_counter()
        scene_xyz = self._get_scene_world_points()
        all_candidates: list[GraspCandidate] = []

        for obj_msg in msg.objects:
            det = ros_msg_to_object3d(obj_msg)
            obj_pts, obstacle_xyz = self._split_object_and_obstacle_points(det, scene_xyz)
            det.point_cloud_xyz = obj_pts

            plan_out = self._planner.plan(
                detection=det,
                obstacle_xyz=obstacle_xyz,
                table_z=self._table_z,
                top_k=self._top_k_per_object,
                include_colliding=False,
            )
            all_candidates.extend(plan_out.candidates)

        all_candidates.sort(key=lambda g: g.score, reverse=True)
        elapsed_ms = float((time.perf_counter() - t0) * 1000.0)

        frame_id = msg.header.frame_id or self._world_frame_id
        header = make_header(frame_id, stamp=msg.header.stamp)

        grasp_array = GraspPoseArray()
        grasp_array.header = header
        grasp_array.grasps = [grasp_to_ros_msg(g) for g in all_candidates]
        grasp_array.planning_latency_ms = elapsed_ms
        self._grasps_pub.publish(grasp_array)

        markers = grasps_to_marker_array(all_candidates, frame_id=frame_id, stamp=msg.header.stamp)
        self._markers_pub.publish(markers)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GraspPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
