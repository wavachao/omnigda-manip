"""ROS 2 node running SegmentGDA-Plus 2D few-shot semantic segmentation."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from omnigda_sim.conversions import image_msg_to_numpy, numpy_to_image_msg
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from omnigda.core.config import load_config
from omnigda.perception_2d.segment_gda import SegmentGDAPlus


class Perception2DNode(Node):
    """Subscribes to RGB frames and GDA memory updates, publishing 2D segmentation and confidence."""

    def __init__(
        self,
        seg2d: SegmentGDAPlus | None = None,
        node_name: str = "perception_2d_node",
        **node_kwargs: Any,
    ) -> None:
        super().__init__(node_name, **node_kwargs)

        self.declare_parameter("memory_dir", "/tmp/omnigda_memory")
        self.declare_parameter("backend", "pytorch")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("config_path", "")
        self.declare_parameter("max_rate_hz", 15.0)

        self._memory_dir = Path(str(self.get_parameter("memory_dir").value))
        self._seg2d_path = self._memory_dir / "seg2d_memory.npz"
        self._backend = str(self.get_parameter("backend").value)
        self._device = str(self.get_parameter("device").value)
        self._max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self._min_interval_sec = 1.0 / self._max_rate_hz if self._max_rate_hz > 0.0 else 0.0
        self._last_proc_time = 0.0

        cfg_path = str(self.get_parameter("config_path").value).strip()
        self._cfg = load_config(cfg_path or None)
        self._cfg.setdefault("perception_2d", {})["backend"] = self._backend
        self._cfg.setdefault("project", {})["device"] = self._device

        self._lock = threading.RLock()
        self._seg2d: SegmentGDAPlus | None = seg2d
        self._warned_unfitted = False
        self._try_load_memory(self._seg2d_path)

        sensor_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self._rgb_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self._on_rgb,
            sensor_qos,
        )
        self._mem_sub = self.create_subscription(
            String,
            "/gda/memory_updated",
            self._on_memory_updated,
            10,
        )

        self._mask_pub = self.create_publisher(Image, "/perception/seg_mask", sensor_qos)
        self._conf_pub = self.create_publisher(Image, "/perception/confidence_map", sensor_qos)
        self._latency_pub = self.create_publisher(Float32, "/perception/seg2d_latency_ms", 10)

        self.get_logger().info(
            f"Perception2DNode ready (backend={self._backend}, memory_dir={self._memory_dir})"
        )

    def _ensure_seg2d(self) -> SegmentGDAPlus:
        if self._seg2d is None:
            self._seg2d = SegmentGDAPlus.from_config(self._cfg)
        return self._seg2d

    def _try_load_memory(self, path: Path) -> bool:
        if not path.exists():
            return False
        with self._lock:
            try:
                seg2d = self._ensure_seg2d()
                seg2d.load_state(path)
                self._warned_unfitted = False
                self.get_logger().info(
                    f"Loaded 2D SegmentGDA-Plus memory from {path} ({len(seg2d.class_names)} classes)"
                )
                return True
            except Exception as exc:
                self.get_logger().warning(f"Could not load 2D GDA memory from {path}: {exc}")
                return False

    def _on_memory_updated(self, msg: String) -> None:
        target_path = self._seg2d_path
        try:
            payload = json.loads(msg.data)
            if payload.get("seg2d_path"):
                target_path = Path(str(payload["seg2d_path"]))
        except Exception:
            pass
        self._try_load_memory(target_path)

    def _on_rgb(self, msg: Image) -> None:
        now = time.perf_counter()
        if self._min_interval_sec > 0.0 and (now - self._last_proc_time) < self._min_interval_sec:
            return
        self._last_proc_time = now

        rgb = image_msg_to_numpy(msg, copy=True)
        if msg.encoding == "bgr8":
            rgb = np.ascontiguousarray(rgb[..., ::-1])

        h, w = int(rgb.shape[0]), int(rgb.shape[1])

        with self._lock:
            if self._seg2d is None or not self._seg2d.is_fitted:
                if not self._warned_unfitted:
                    self.get_logger().info(
                        "SegmentGDA-Plus has no registered classes yet; publishing empty masks."
                    )
                    self._warned_unfitted = True
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                empty_conf = np.zeros((h, w), dtype=np.float32)
                self._mask_pub.publish(numpy_to_image_msg(empty_mask, "mono8", header=msg.header))
                self._conf_pub.publish(numpy_to_image_msg(empty_conf, "32FC1", header=msg.header))
                self._latency_pub.publish(Float32(data=0.0))
                return

            seg_res = self._seg2d.segment(rgb)

        mask_u8 = np.clip(seg_res.class_map, 0, 255).astype(np.uint8)
        conf_f32 = seg_res.confidence_map.astype(np.float32, copy=False)

        self._mask_pub.publish(numpy_to_image_msg(mask_u8, "mono8", header=msg.header))
        self._conf_pub.publish(numpy_to_image_msg(conf_f32, "32FC1", header=msg.header))
        self._latency_pub.publish(Float32(data=float(seg_res.latency_ms)))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Perception2DNode()
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
