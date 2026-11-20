"""Confidence-aware 6DoF grasp planner with collision checking and 6DoF NMS.

TCP Coordinate Frame Convention (Franka Panda Standard)
-------------------------------------------------------
All planned ``GraspCandidate`` poses adhere to the Franka Panda TCP convention:
- ``+Z`` axis: unit approach vector pointing into the target object (``(0, 0, -1)`` in
  world frame for pure top-down grasps).
- ``+Y`` axis: unit jaw closing/opening axis between the two parallel fingers.
- ``+X`` axis: right-hand orthogonal completion ``X = Y x Z``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omnigda.core.schemas import GraspCandidate, Object3DDetection
from omnigda.grasp.collision import FrankaGripperCollisionChecker
from omnigda.grasp.sampler import AntipodalGraspSampler


@dataclass(slots=True)
class GraspPlanOutput:
    """Output of ``ConfidenceAwareGraspPlanner.plan``."""

    candidates: list[GraspCandidate] = field(default_factory=list)
    planning_latency_ms: float = 0.0
    needs_nbv: bool = False
    num_sampled: int = 0
    num_collision_free: int = 0
    best_score: float = 0.0


def _symmetric_so3_distance_rad(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Geodesic SO(3) distance between two gripper frames modulo 180-deg Z-axis jaw symmetry."""
    rel = rot_a.T @ rot_b
    tr_direct = float(np.trace(rel))
    cos_direct = float(np.clip(0.5 * (tr_direct - 1.0), -1.0, 1.0))
    ang_direct = float(np.arccos(cos_direct))

    # 180-deg flip about local +Z (parallel-jaw gripper symmetry): diag(-1, -1, +1)
    tr_flipped = float(-rel[0, 0] - rel[1, 1] + rel[2, 2])
    cos_flipped = float(np.clip(0.5 * (tr_flipped - 1.0), -1.0, 1.0))
    ang_flipped = float(np.arccos(cos_flipped))

    return min(ang_direct, ang_flipped)


class ConfidenceAwareGraspPlanner:
    """End-to-end 6DoF grasp planner combining antipodal sampling, collision checking, and NMS.

    Pipeline:
      1. Sample diverse top-down, angled, and side 6DoF grasps via ``AntipodalGraspSampler``.
      2. Evaluate tabletop and cluttered-scene obstacle collisions in batch via
         ``FrankaGripperCollisionChecker``.
      3. Rank candidates by a combined quality score integrating force-closure stability,
         antipodal alignment, approach preference, and object perception confidence.
      4. Apply 6DoF spatial/angular Non-Maximum Suppression (NMS) modulo parallel-jaw
         symmetry to return diverse top-K grasp candidates.
      5. Set ``needs_nbv=True`` whenever no collision-free grasp exceeds ``min_grasp_score``,
         signaling the Active Perception agent to acquire a Next-Best-View.
    """

    def __init__(
        self,
        sampler: AntipodalGraspSampler | None = None,
        collision_checker: FrankaGripperCollisionChecker | None = None,
        min_grasp_score: float = 0.35,
        nms_translation_thresh_m: float = 0.015,
        nms_rotation_thresh_deg: float = 20.0,
        weight_force_closure: float = 0.55,
        weight_antipodal: float = 0.35,
        weight_approach: float = 0.10,
        confidence_blend: float = 0.35,
    ) -> None:
        self.sampler = sampler if sampler is not None else AntipodalGraspSampler()
        self.collision_checker = (
            collision_checker
            if collision_checker is not None
            else FrankaGripperCollisionChecker()
        )
        self.min_grasp_score = float(min_grasp_score)
        self.nms_translation_thresh_m = float(nms_translation_thresh_m)
        self.nms_rotation_thresh_rad = float(np.deg2rad(nms_rotation_thresh_deg))
        self.weight_force_closure = float(weight_force_closure)
        self.weight_antipodal = float(weight_antipodal)
        self.weight_approach = float(weight_approach)
        self.confidence_blend = float(np.clip(confidence_blend, 0.0, 1.0))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ConfidenceAwareGraspPlanner:
        """Construct planner, sampler, and collision checker from YAML config dict."""
        grasp_cfg = cfg.get("grasp", cfg)
        return cls(
            sampler=AntipodalGraspSampler.from_config(cfg),
            collision_checker=FrankaGripperCollisionChecker.from_config(cfg),
            min_grasp_score=float(grasp_cfg.get("min_grasp_score", 0.35)),
            nms_translation_thresh_m=float(grasp_cfg.get("nms_translation_thresh_m", 0.015)),
            nms_rotation_thresh_deg=float(grasp_cfg.get("nms_rotation_thresh_deg", 20.0)),
        )

    def _compute_combined_score(
        self,
        candidate: GraspCandidate,
        detection_confidence: float,
    ) -> float:
        """Combine geometric grasp quality with perception confidence."""
        app_pref = 0.5 + 0.5 * float(np.clip(-candidate.approach_vector[2], 0.0, 1.0))
        geom_score = (
            self.weight_force_closure * candidate.force_closure_score
            + self.weight_antipodal * candidate.antipodal_score
            + self.weight_approach * app_pref
        )
        w_sum = self.weight_force_closure + self.weight_antipodal + self.weight_approach
        if w_sum > 1e-9:
            geom_score /= w_sum

        conf = float(np.clip(detection_confidence, 0.0, 1.0))
        conf_factor = (1.0 - self.confidence_blend) + self.confidence_blend * conf
        return float(np.clip(geom_score * conf_factor, 0.0, 1.0))

    def _apply_nms(
        self,
        sorted_candidates: list[GraspCandidate],
        top_k: int,
    ) -> list[GraspCandidate]:
        """6DoF spatial + symmetric SO(3) angular Non-Maximum Suppression."""
        if not sorted_candidates or top_k <= 0:
            return []

        selected: list[GraspCandidate] = []
        for cand in sorted_candidates:
            suppressed = False
            for kept in selected:
                d_pos = float(np.linalg.norm(cand.position - kept.position))
                if d_pos < self.nms_translation_thresh_m:
                    d_ang = _symmetric_so3_distance_rad(
                        cand.pose_4x4[:3, :3],
                        kept.pose_4x4[:3, :3],
                    )
                    if d_ang < self.nms_rotation_thresh_rad:
                        suppressed = True
                        break
            if not suppressed:
                selected.append(cand)
                if len(selected) >= top_k:
                    break
        return selected

    def plan(
        self,
        detection: Object3DDetection,
        obstacle_xyz: np.ndarray | None = None,
        table_z: float | None = 0.0,
        top_k: int = 10,
        num_candidates: int | None = None,
        include_colliding: bool = False,
    ) -> GraspPlanOutput:
        """Plan ranked, collision-free, diverse 6DoF grasps for a target 3D detection.

        Args:
            detection: Target ``Object3DDetection`` in world coordinates.
            obstacle_xyz: Optional ``(M, 3)`` obstacle point cloud (excluding target object).
            table_z: Height of the supporting tabletop plane in world Z, or ``None`` to disable
                tabletop collision checking.
            top_k: Maximum number of diverse grasps returned after 6DoF NMS.
            num_candidates: Number of raw hypotheses to sample before collision checking
                (defaults to ``self.sampler.num_candidates``).
            include_colliding: If ``True``, append colliding candidates (with
                ``collision_free=False``) after collision-free candidates for diagnostics.

        Returns:
            ``GraspPlanOutput`` with sorted ``candidates``, ``planning_latency_ms``, and
            ``needs_nbv`` flag.
        """
        t0 = time.perf_counter()

        raw_candidates = self.sampler.sample(
            detection=detection,
            num_candidates=num_candidates,
            table_z=table_z,
        )
        num_sampled = len(raw_candidates)

        if num_sampled == 0:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return GraspPlanOutput(
                candidates=[],
                planning_latency_ms=latency_ms,
                needs_nbv=True,
                num_sampled=0,
                num_collision_free=0,
                best_score=0.0,
            )

        collision_free_mask = self.collision_checker.check_batch(
            raw_candidates,
            obstacle_xyz=obstacle_xyz,
            table_z=table_z,
        )

        cf_candidates: list[GraspCandidate] = []
        colliding_candidates: list[GraspCandidate] = []

        for cand, is_free in zip(raw_candidates, collision_free_mask, strict=True):
            cand.collision_free = bool(is_free)
            cand.score = self._compute_combined_score(cand, detection.confidence)
            if cand.collision_free:
                cf_candidates.append(cand)
            else:
                colliding_candidates.append(cand)

        cf_candidates.sort(key=lambda c: c.score, reverse=True)
        selected = self._apply_nms(cf_candidates, top_k=top_k)

        best_cf_score = selected[0].score if selected else 0.0
        needs_nbv = (len(selected) == 0) or (best_cf_score < self.min_grasp_score)

        if include_colliding and len(selected) < top_k:
            colliding_candidates.sort(key=lambda c: c.score, reverse=True)
            extra = self._apply_nms(colliding_candidates, top_k=top_k - len(selected))
            selected.extend(extra)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return GraspPlanOutput(
            candidates=selected,
            planning_latency_ms=latency_ms,
            needs_nbv=needs_nbv,
            num_sampled=num_sampled,
            num_collision_free=len(cf_candidates),
            best_score=best_cf_score,
        )
