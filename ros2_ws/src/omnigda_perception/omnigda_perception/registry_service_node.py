"""ROS 2 node hosting /gda/register_object and /gda/list_objects with shared on-disk GDA memory."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from omnigda_interfaces.srv import ListObjects, RegisterObject
from omnigda_sim.conversions import camera_info_to_intrinsics, image_msg_to_numpy
from rclpy.node import Node
from std_msgs.msg import String

from omnigda.core.config import load_config
from omnigda.core.schemas import CameraIntrinsics, SupportSample
from omnigda.perception_2d.segment_gda import SegmentGDAPlus
from omnigda.perception_3d.multimodal_gmda import PartialAwareMultimodalGMDA


class RegistryServiceNode(Node):
    """Manages online few-shot & zero-shot class registration for 2D SegmentGDA-Plus and 3D GMDA."""

    def __init__(
        self,
        seg2d: SegmentGDAPlus | None = None,
        gmda3d: PartialAwareMultimodalGMDA | None = None,
        node_name: str = "registry_service_node",
        **node_kwargs: Any,
    ) -> None:
        super().__init__(node_name, **node_kwargs)

        self.declare_parameter("memory_dir", "/tmp/omnigda_memory")
        self.declare_parameter("backend", "pytorch")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("enable_2d", True)
        self.declare_parameter("enable_clip", True)
        self.declare_parameter("config_path", "")

        self._memory_dir = Path(str(self.get_parameter("memory_dir").value))
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._seg2d_path = self._memory_dir / "seg2d_memory.npz"
        self._gmda3d_path = self._memory_dir / "gmda3d_memory.npz"
        self._manifest_path = self._memory_dir / "registry_manifest.json"

        self._backend = str(self.get_parameter("backend").value)
        self._device = str(self.get_parameter("device").value)
        self._enable_2d = bool(self.get_parameter("enable_2d").value)
        self._enable_clip = bool(self.get_parameter("enable_clip").value)
        cfg_path = str(self.get_parameter("config_path").value).strip()
        self._cfg = load_config(cfg_path or None)
        self._cfg.setdefault("perception_2d", {})["backend"] = self._backend
        self._cfg.setdefault("project", {})["device"] = self._device

        self._lock = threading.RLock()
        self._seg2d: SegmentGDAPlus | None = seg2d
        self._gmda3d: PartialAwareMultimodalGMDA = (
            gmda3d
            if gmda3d is not None
            else PartialAwareMultimodalGMDA(
                config=self._cfg,
                enable_clip=self._enable_clip,
                device=self._device,
            )
        )

        self._sample_counts: dict[str, int] = {}
        self._seg2d_class_order: list[str] = []
        self._load_existing_memory()

        self._memory_pub = self.create_publisher(String, "/gda/memory_updated", 10)
        self._register_srv = self.create_service(
            RegisterObject,
            "/gda/register_object",
            self._handle_register_object,
        )
        self._list_srv = self.create_service(
            ListObjects,
            "/gda/list_objects",
            self._handle_list_objects,
        )

        self.get_logger().info(
            f"RegistryServiceNode ready (memory_dir={self._memory_dir}, "
            f"classes={list(self._sample_counts.keys())})"
        )

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    def _ensure_seg2d(self) -> SegmentGDAPlus:
        if self._seg2d is None:
            self._seg2d = SegmentGDAPlus.from_config(self._cfg)
            if self._seg2d_path.exists():
                self._seg2d.load_state(self._seg2d_path)
        return self._seg2d

    def _load_existing_memory(self) -> None:
        if self._manifest_path.exists():
            try:
                data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                self._sample_counts = {
                    str(k): int(v) for k, v in data.get("sample_counts", {}).items()
                }
                self._seg2d_class_order = [str(c) for c in data.get("seg2d_class_order", [])]
            except Exception as exc:
                self.get_logger().warning(f"Failed to read manifest {self._manifest_path}: {exc}")

        if self._gmda3d_path.exists():
            try:
                self._gmda3d.load_memory(self._gmda3d_path)
                for cname in self._gmda3d.registered_classes:
                    self._sample_counts.setdefault(cname, 1)
            except Exception as exc:
                self.get_logger().warning(f"Failed to load 3D GMDA memory: {exc}")

        if self._seg2d is not None and self._seg2d_path.exists():
            try:
                self._seg2d.load_state(self._seg2d_path)
                self._seg2d_class_order = list(self._seg2d.class_names)
                for cname in self._seg2d.class_names:
                    self._sample_counts.setdefault(cname, 1)
            except Exception as exc:
                self.get_logger().warning(f"Failed to load 2D SegmentGDA state: {exc}")

    def _save_manifest_atomic(self) -> None:
        payload = {
            "sample_counts": self._sample_counts,
            "seg2d_class_order": self._seg2d_class_order,
            "gmda3d_classes": list(self._gmda3d.registered_classes),
            "seg2d_path": str(self._seg2d_path),
            "gmda3d_path": str(self._gmda3d_path),
            "updated_at": time.time(),
        }
        tmp_path = self._memory_dir / "registry_manifest.tmp.json"
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self._manifest_path)

    def _save_seg2d_atomic(self, seg2d: SegmentGDAPlus) -> None:
        tmp_path = self._memory_dir / "seg2d_memory.tmp.npz"
        seg2d.save_state(tmp_path)
        os.replace(tmp_path, self._seg2d_path)

    def _save_gmda3d_atomic(self) -> None:
        tmp_path = self._memory_dir / "gmda3d_memory.tmp.npz"
        self._gmda3d.save_memory(tmp_path)
        os.replace(tmp_path, self._gmda3d_path)

    def _publish_memory_updated(self, class_name: str, event: str = "registered") -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "event": event,
                "class_name": class_name,
                "seg2d_path": str(self._seg2d_path) if self._seg2d_path.exists() else "",
                "gmda3d_path": str(self._gmda3d_path) if self._gmda3d_path.exists() else "",
                "manifest_path": str(self._manifest_path),
                "total_classes": len(self._sample_counts),
                "timestamp": time.time(),
            }
        )
        self._memory_pub.publish(msg)

    def _handle_register_object(
        self,
        request: RegisterObject.Request,
        response: RegisterObject.Response,
    ) -> RegisterObject.Response:
        t0 = time.perf_counter()
        class_name = str(request.class_name).strip()
        text_prompt = str(request.text_prompt).strip()

        if not class_name:
            response.success = False
            response.message = "class_name must be non-empty"
            response.registration_time_ms = 0.0
            response.total_classes = len(self._sample_counts)
            return response

        try:
            rgb: np.ndarray | None = None
            depth: np.ndarray | None = None
            mask: np.ndarray | None = None
            intrinsics: CameraIntrinsics | None = None

            if (
                request.rgb_image.width > 0
                and request.rgb_image.height > 0
                and len(request.rgb_image.data) > 0
            ):
                rgb = image_msg_to_numpy(request.rgb_image, copy=True)
                if request.rgb_image.encoding == "bgr8":
                    rgb = np.ascontiguousarray(rgb[..., ::-1])

            if (
                request.depth_image.width > 0
                and request.depth_image.height > 0
                and len(request.depth_image.data) > 0
            ):
                raw_depth = image_msg_to_numpy(request.depth_image, copy=True)
                depth = (
                    raw_depth.astype(np.float32) / 1000.0
                    if request.depth_image.encoding == "16UC1"
                    else raw_depth.astype(np.float32, copy=False)
                )

            if (
                request.mask_image.width > 0
                and request.mask_image.height > 0
                and len(request.mask_image.data) > 0
            ):
                raw_mask = image_msg_to_numpy(request.mask_image, copy=True)
                if raw_mask.ndim == 3:
                    raw_mask = raw_mask[..., 0]
                mask = raw_mask > 0

            if request.camera_info.width > 0 and request.camera_info.height > 0:
                intrinsics = camera_info_to_intrinsics(request.camera_info)

            has_visual_sample = rgb is not None and mask is not None and bool(np.any(mask))
            has_text = bool(text_prompt)
            if not has_visual_sample and not has_text:
                response.success = False
                response.message = (
                    "Provide either non-empty (rgb_image + mask_image) for few-shot registration "
                    "or text_prompt for zero-shot registration."
                )
                response.registration_time_ms = float((time.perf_counter() - t0) * 1000.0)
                response.total_classes = len(self._sample_counts)
                return response

            with self._lock:
                sample = SupportSample(
                    class_name=class_name,
                    rgb=rgb,
                    depth=depth,
                    mask=mask,
                    intrinsics=intrinsics,
                    camera_to_world=np.eye(4, dtype=np.float64),
                    text_prompt=text_prompt or None,
                )

                updated_parts: list[str] = []
                if has_visual_sample and self._enable_2d:
                    seg2d = self._ensure_seg2d()
                    seg2d.register_object(class_name, [sample])
                    self._seg2d_class_order = list(seg2d.class_names)
                    self._save_seg2d_atomic(seg2d)
                    updated_parts.append("seg2d")

                if has_visual_sample:
                    self._gmda3d.register_few_shot(
                        class_name,
                        [sample],
                        register_text=has_text and self._enable_clip,
                    )
                    updated_parts.append("gmda3d_fewshot")

                if has_text and (not has_visual_sample or not self._enable_clip):
                    self._gmda3d.register_zero_shot_text(class_name, text_prompt=text_prompt)
                    updated_parts.append("gmda3d_text")
                elif has_text and has_visual_sample and self._enable_clip:
                    updated_parts.append("gmda3d_text")

                self._save_gmda3d_atomic()
                self._sample_counts[class_name] = self._sample_counts.get(class_name, 0) + 1
                self._save_manifest_atomic()
                self._publish_memory_updated(class_name)

            elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
            response.success = True
            response.message = (
                f"Registered {class_name!r} ({', '.join(updated_parts)}) "
                f"in {elapsed_ms:.1f} ms"
            )
            response.registration_time_ms = elapsed_ms
            response.total_classes = len(self._sample_counts)
            self.get_logger().info(response.message)
            return response
        except Exception as exc:
            elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
            self.get_logger().error(f"Failed to register {class_name!r}: {exc}")
            response.success = False
            response.message = f"Registration failed: {exc}"
            response.registration_time_ms = elapsed_ms
            response.total_classes = len(self._sample_counts)
            return response

    def _handle_list_objects(
        self,
        _request: ListObjects.Request,
        response: ListObjects.Response,
    ) -> ListObjects.Response:
        with self._lock:
            class_names = list(self._sample_counts.keys())
            sample_counts = [int(self._sample_counts[c]) for c in class_names]
        response.class_names = class_names
        response.sample_counts = sample_counts
        return response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RegistryServiceNode()
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
