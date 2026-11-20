"""Shared typed data contracts across all OmniGDA-Manip modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(slots=True)
class RGBDFrame:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 in meters
    intrinsics: CameraIntrinsics
    camera_to_world: np.ndarray  # (4, 4) float64 homogeneous transform
    timestamp_sec: float = 0.0
    frame_id: str = "camera_color_optical_frame"
    gt_segmentation: np.ndarray | None = None  # optional (H, W) int32 sim GT


@dataclass(slots=True)
class SupportSample:
    class_name: str
    rgb: np.ndarray | None = None  # (H, W, 3) uint8
    depth: np.ndarray | None = None  # (H, W) float32 in meters
    mask: np.ndarray | None = None  # (H, W) bool
    intrinsics: CameraIntrinsics | None = None
    camera_to_world: np.ndarray | None = None
    point_cloud_xyz: np.ndarray | None = None  # (N, 3) float32
    text_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentationResult:
    class_names: list[str]
    class_map: np.ndarray  # (H, W) int32, 0 = background, 1..K = class index
    masks: dict[str, np.ndarray]  # class_name -> (H, W) bool
    confidence_map: np.ndarray  # (H, W) float32 in [0, 1]
    log_likelihoods: np.ndarray  # (C, H, W) float32 (index 0 = background)
    entropy_map: np.ndarray  # (H, W) float32
    latency_ms: float = 0.0


@dataclass(slots=True)
class Object3DDetection:
    instance_id: int
    class_name: str
    confidence: float
    entropy: float
    ood_score: float
    is_ood: bool
    centroid: np.ndarray  # (3,) float64 in world frame
    obb_center: np.ndarray  # (3,) float64
    obb_extent: np.ndarray  # (3,) float64 (dx, dy, dz)
    obb_rotation: np.ndarray  # (3, 3) float64
    point_cloud_xyz: np.ndarray  # (N, 3) float32
    point_cloud_rgb: np.ndarray | None = None  # (N, 3) float32 in [0, 1]
    mask_2d: np.ndarray | None = None  # (H, W) bool
    bbox_2d: tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    modality_weights: dict[str, float] = field(default_factory=dict)
    posterior_probs: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SceneRelation:
    subject_id: int
    relation: str  # left_of | right_of | front_of | behind | above | near
    object_id: int
    confidence: float
    distance_m: float


@dataclass(slots=True)
class SceneGraph3D:
    objects: list[Object3DDetection] = field(default_factory=list)
    relations: list[SceneRelation] = field(default_factory=list)
    timestamp_sec: float = 0.0

    def get_by_id(self, instance_id: int) -> Object3DDetection | None:
        for obj in self.objects:
            if obj.instance_id == instance_id:
                return obj
        return None


@dataclass(slots=True)
class GraspCandidate:
    pose_4x4: np.ndarray  # (4, 4) float64 gripper TCP pose in world frame
    position: np.ndarray  # (3,) float64
    quaternion_xyzw: np.ndarray  # (4,) float64
    width_m: float
    score: float
    force_closure_score: float
    antipodal_score: float
    collision_free: bool
    approach_vector: np.ndarray  # (3,) float64 unit vector
    target_instance_id: int = -1
    target_class_name: str = ""


@dataclass(slots=True)
class ManipulationInstruction:
    raw_text: str
    action: str = "pick"  # pick | pick_and_place | inspect
    target_query: str = ""
    spatial_relation: str | None = None  # left_of | right_of | near | leftmost | rightmost
    reference_query: str | None = None
    destination_query: str | None = None


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    message: str
    target_instance_id: int = -1
    target_class_name: str = ""
    grasp_candidate: GraspCandidate | None = None
    num_views_used: int = 1
    perception_latency_ms: float = 0.0
    planning_latency_ms: float = 0.0
