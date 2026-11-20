"""ROS 2 node running Partial-Aware 4-Modality 3D GMDA and 3D Scene Graph construction."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import torch
from omnigda_interfaces.msg import Object3DArray, PerceptionMetrics, SceneGraph
from omnigda_sim.conversions import (
    image_msg_to_numpy,
    make_header,
    object3d_to_ros_msg,
    objects_to_marker_array,
    ros_msgs_to_rgbd_frame,
    scene_graph_to_ros_msg,
    stamp_to_sec,
    transform_to_matrix,
)
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from omnigda.core.config import load_config
from omnigda.core.schemas import Object3DDetection, RGBDFrame
from omnigda.perception_3d.multimodal_gmda import PartialAwareMultimodalGMDA
from omnigda.perception_3d.pointcloud import (
    compute_oriented_bounding_box,
    confidence_weighted_backproject,
)
from omnigda.perception_3d.scene_graph import SceneGraphBuilder3D


class Perception3DNode(Node):
    """Fuses RGB-D + TF2 + 2D masks into 3D detections, scene graph, metrics, and RViz2 markers."""

    def __init__(
        self,
        gmda3d: PartialAwareMultimodalGMDA | None = None,
        scene_graph_builder: SceneGraphBuilder3D | None = None,
        node_name: str = "perception_3d_node",
        **node_kwargs: Any,
    ) -> None:
        super().__init__(node_name, **node_kwargs)

        self.declare_parameter("memory_dir", "/tmp/omnigda_memory")
        self.declare_parameter("world_frame_id", "world")
        self.declare_parameter("camera_frame_id", "camera_color_optical_frame")
        self.declare_parameter("backend", "pytorch")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("enable_clip", True)
        self.declare_parameter("config_path", "")
        self.declare_parameter("min_area_px", 120)
        self.declare_parameter("table_z", 0.0)

        self._memory_dir = Path(str(self.get_parameter("memory_dir").value))
        self._gmda3d_path = self._memory_dir / "gmda3d_memory.npz"
        self._manifest_path = self._memory_dir / "registry_manifest.json"
        self._world_frame_id = str(self.get_parameter("world_frame_id").value)
        self._camera_frame_id = str(self.get_parameter("camera_frame_id").value)
        self._backend = str(self.get_parameter("backend").value)
        self._device = str(self.get_parameter("device").value)
        self._enable_clip = bool(self.get_parameter("enable_clip").value)
        self._min_area_px = int(self.get_parameter("min_area_px").value)
        self._table_z = float(self.get_parameter("table_z").value)

        cfg_path = str(self.get_parameter("config_path").value).strip()
        self._cfg = load_config(cfg_path or None)
        p3d_cfg = self._cfg.get("perception_3d", {})
        self._depth_min_m = float(p3d_cfg.get("depth_min_m", 0.15))
        self._depth_max_m = float(p3d_cfg.get("depth_max_m", 2.0))
        self._conf_denoise_thresh = float(p3d_cfg.get("confidence_denoise_thresh", 0.45))
        self._sor_nb_neighbors = int(p3d_cfg.get("sor_nb_neighbors", 20))
        self._sor_std_ratio = float(p3d_cfg.get("sor_std_ratio", 1.5))
        self._min_cluster_points = int(p3d_cfg.get("min_cluster_points", 20))

        self._lock = threading.RLock()
        if gmda3d is not None:
            self._gmda3d = gmda3d
        else:
            clip_enc = None
            if self._enable_clip and self._backend in {"tensorrt", "onnx"}:
                from omnigda.deploy import create_clip_encoder

                clip_enc = create_clip_encoder(self._cfg, backend=self._backend)
            self._gmda3d = PartialAwareMultimodalGMDA(
                config=self._cfg,
                enable_clip=self._enable_clip,
                device=self._device,
                clip_encoder=clip_enc,
            )
        self._sg_builder = (
            scene_graph_builder if scene_graph_builder is not None else SceneGraphBuilder3D()
        )
        self._seg2d_class_order: list[str] = []
        self._try_load_memory(self._gmda3d_path)

        # Latest cached sensor inputs
        self._latest_rgb: Image | None = None
        self._latest_depth: Image | None = None
        self._latest_info: CameraInfo | None = None
        self._latest_conf: Image | None = None
        self._latest_seg2d_latency_ms: float = 0.0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        sensor_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Image, "/camera/color/image_raw", self._on_rgb, sensor_qos)
        self.create_subscription(Image, "/camera/depth/image_raw", self._on_depth, sensor_qos)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_info, sensor_qos)
        self.create_subscription(Image, "/perception/confidence_map", self._on_conf, sensor_qos)
        self.create_subscription(
            Float32, "/perception/seg2d_latency_ms", self._on_seg2d_latency, 10
        )
        self.create_subscription(String, "/gda/memory_updated", self._on_memory_updated, 10)
        self.create_subscription(Image, "/perception/seg_mask", self._on_seg_mask, sensor_qos)

        self._objects_pub = self.create_publisher(Object3DArray, "/perception/objects_3d", 10)
        self._sg_pub = self.create_publisher(SceneGraph, "/perception/scene_graph", 10)
        self._metrics_pub = self.create_publisher(PerceptionMetrics, "/perception/metrics", 10)
        self._markers_pub = self.create_publisher(MarkerArray, "/perception/markers_3d", 10)

        self.get_logger().info(
            f"Perception3DNode ready (world_frame={self._world_frame_id}, "
            f"registered_classes={len(self._gmda3d.registered_classes)})"
        )

    def _try_load_memory(self, path: Path) -> bool:
        loaded = False
        with self._lock:
            if self._manifest_path.exists():
                try:
                    data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                    self._seg2d_class_order = [str(c) for c in data.get("seg2d_class_order", [])]
                except Exception:
                    pass
            if path.exists():
                try:
                    self._gmda3d.load_memory(path)
                    loaded = True
                    self.get_logger().info(
                        f"Loaded 3D GMDA memory from {path} "
                        f"({len(self._gmda3d.registered_classes)} classes)"
                    )
                except Exception as exc:
                    self.get_logger().warning(f"Failed to load 3D GMDA memory from {path}: {exc}")
        return loaded

    def _on_memory_updated(self, msg: String) -> None:
        target_path = self._gmda3d_path
        try:
            payload = json.loads(msg.data)
            if payload.get("gmda3d_path"):
                target_path = Path(str(payload["gmda3d_path"]))
            if payload.get("manifest_path"):
                self._manifest_path = Path(str(payload["manifest_path"]))
        except Exception:
            pass
        self._try_load_memory(target_path)

    def _on_rgb(self, msg: Image) -> None:
        self._latest_rgb = msg

    def _on_depth(self, msg: Image) -> None:
        self._latest_depth = msg

    def _on_info(self, msg: CameraInfo) -> None:
        self._latest_info = msg

    def _on_conf(self, msg: Image) -> None:
        self._latest_conf = msg

    def _on_seg2d_latency(self, msg: Float32) -> None:
        self._latest_seg2d_latency_ms = float(msg.data)

    def _lookup_camera_to_world(self, camera_frame_id: str) -> np.ndarray:
        frame_candidates = [camera_frame_id]
        if self._camera_frame_id not in frame_candidates:
            frame_candidates.append(self._camera_frame_id)

        for child in frame_candidates:
            if not child:
                continue
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self._world_frame_id,
                    child,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
                return transform_to_matrix(tf_msg)
            except Exception:
                continue
        return np.eye(4, dtype=np.float64)

    def _extract_detections(
        self,
        frame: RGBDFrame,
        seg_mask: np.ndarray,
        conf_map: np.ndarray | None,
    ) -> list[Object3DDetection]:
        detections: list[Object3DDetection] = []
        unique_labels = [int(v) for v in np.unique(seg_mask) if int(v) > 0]
        next_instance_id = 1

        for label_id in unique_labels:
            binary = (seg_mask == label_id).astype(np.uint8)
            num_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for cc_idx in range(1, num_cc):
                area = int(stats[cc_idx, cv2.CC_STAT_AREA])
                if area < self._min_area_px:
                    continue

                comp_mask = cc_labels == cc_idx
                x = int(stats[cc_idx, cv2.CC_STAT_LEFT])
                y = int(stats[cc_idx, cv2.CC_STAT_TOP])
                w = int(stats[cc_idx, cv2.CC_STAT_WIDTH])
                h = int(stats[cc_idx, cv2.CC_STAT_HEIGHT])
                bbox = (x, y, x + w, y + h)

                xyz, rgb_pts = confidence_weighted_backproject(
                    frame=frame,
                    mask=comp_mask,
                    confidence_map=conf_map,
                    conf_thresh=self._conf_denoise_thresh,
                    depth_min=self._depth_min_m,
                    depth_max=self._depth_max_m,
                    sor_nb_neighbors=self._sor_nb_neighbors,
                    sor_std_ratio=self._sor_std_ratio,
                    min_points=min(10, self._min_cluster_points),
                )
                if xyz.shape[0] < min(10, self._min_cluster_points):
                    continue

                mean_conf = (
                    float(np.mean(conf_map[comp_mask]))
                    if conf_map is not None and conf_map.shape == comp_mask.shape
                    else 0.75
                )

                with self._lock:
                    has_registered = len(self._gmda3d.registered_classes) > 0
                    seg2d_logits: dict[str, float] | None = None
                    if 1 <= label_id <= len(self._seg2d_class_order):
                        pred_2d_name = self._seg2d_class_order[label_id - 1]
                        seg2d_logits = {
                            cname: (2.5 if cname == pred_2d_name else -0.5)
                            for cname in self._gmda3d.registered_classes
                        }

                    if has_registered:
                        det = self._gmda3d.classify_instance(
                            instance_id=next_instance_id,
                            xyz=xyz,
                            rgb_points=rgb_pts,
                            rgb_image=frame.rgb,
                            mask_2d=comp_mask,
                            bbox_2d=bbox,
                            frame=frame,
                            confidence_map=conf_map,
                            seg2d_logits=seg2d_logits,
                            seg2d_reliability=float(np.clip(mean_conf, 0.2, 0.95)),
                            table_z=self._table_z,
                        )
                    else:
                        obb_center, obb_extent, obb_rot = compute_oriented_bounding_box(
                            xyz, gravity_aligned=True
                        )
                        det = Object3DDetection(
                            instance_id=next_instance_id,
                            class_name=f"unknown_{label_id}",
                            confidence=mean_conf,
                            entropy=1.0,
                            ood_score=100.0,
                            is_ood=True,
                            centroid=np.mean(xyz, axis=0).astype(np.float64),
                            obb_center=obb_center,
                            obb_extent=obb_extent,
                            obb_rotation=obb_rot,
                            point_cloud_xyz=xyz,
                            point_cloud_rgb=rgb_pts,
                            mask_2d=comp_mask,
                            bbox_2d=bbox,
                            modality_weights={"2d": 1.0, "3d": 0.0, "text": 0.0},
                        )

                detections.append(det)
                next_instance_id += 1

        return detections

    def _on_seg_mask(self, mask_msg: Image) -> None:
        if self._latest_rgb is None or self._latest_depth is None or self._latest_info is None:
            return

        t0 = time.perf_counter()
        cam_frame = (
            mask_msg.header.frame_id or self._latest_rgb.header.frame_id or self._camera_frame_id
        )
        cam_to_world = self._lookup_camera_to_world(cam_frame)

        frame = ros_msgs_to_rgbd_frame(
            self._latest_rgb,
            self._latest_depth,
            self._latest_info,
            camera_to_world=cam_to_world,
        )
        seg_mask = image_msg_to_numpy(mask_msg, copy=True)
        if seg_mask.ndim == 3:
            seg_mask = seg_mask[..., 0]

        conf_map: np.ndarray | None = None
        if self._latest_conf is not None:
            conf_arr = image_msg_to_numpy(self._latest_conf, copy=True)
            if conf_arr.shape[:2] == seg_mask.shape[:2]:
                conf_map = conf_arr.astype(np.float32, copy=False)

        detections = self._extract_detections(frame, seg_mask, conf_map)
        ts_sec = stamp_to_sec(mask_msg.header.stamp)
        scene_graph = self._sg_builder.build(detections, timestamp_sec=ts_sec)
        gmda3d_latency_ms = float((time.perf_counter() - t0) * 1000.0)

        stamp = mask_msg.header.stamp
        world_header = make_header(self._world_frame_id, stamp=stamp)

        obj_array = Object3DArray()
        obj_array.header = world_header
        obj_array.objects = [object3d_to_ros_msg(det) for det in detections]
        obj_array.latency_ms = gmda3d_latency_ms
        self._objects_pub.publish(obj_array)

        sg_msg = scene_graph_to_ros_msg(scene_graph, frame_id=self._world_frame_id, stamp=stamp)
        self._sg_pub.publish(sg_msg)

        markers = objects_to_marker_array(detections, frame_id=self._world_frame_id, stamp=stamp)
        self._markers_pub.publish(markers)

        metrics = PerceptionMetrics()
        metrics.header = world_header
        metrics.backend = self._backend
        metrics.seg2d_latency_ms = float(self._latest_seg2d_latency_ms)
        metrics.gmda3d_latency_ms = gmda3d_latency_ms
        metrics.total_latency_ms = float(self._latest_seg2d_latency_ms + gmda3d_latency_ms)
        metrics.gpu_mem_mb = (
            float(torch.cuda.memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
        )
        with self._lock:
            metrics.num_registered_classes = len(self._gmda3d.registered_classes)
        self._metrics_pub.publish(metrics)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Perception3DNode()
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
