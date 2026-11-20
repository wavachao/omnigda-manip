"""End-to-end closed-loop OmniGDA multimodal perception, NBV, and manipulation agent.

Orchestrates all Wave-1 OmniGDA modules into a single closed-loop brain:
1. ``SegmentGDAPlus`` (2D DINOv2 few-shot dense segmentation + instance proposals)
2. ``PartialAwareMultimodalGMDA`` (confidence-weighted RGB-D backprojection + Open3D SOR +
   gravity-aligned OBB + 4-modality 3D GMDA fused with 2D SegmentGDA logits)
3. ``SceneGraphBuilder3D`` (3D spatial relation graph)
4. ``InstructionParserProtocol`` & ``SpatialGrounder3D`` (bilingual EN/ZH parsing +
   open-vocabulary OpenCLIP & 3D spatial grounding)
5. ``ConfidenceAwareGraspPlanner`` (6DoF antipodal sampling, collision checking, NMS)
6. Active Perception / Multi-View Bayesian NBV (``perceive_and_fuse_nbv``): automatically
   switches to ``nbv_camera``, matches 3D instances by centroid proximity, fuses multi-view
   point clouds via Open3D ICP, performs inverse-entropy Bayesian log-posterior fusion, and
   replans 6DoF grasps on the denser point cloud before physical execution via
   ``FrankaGraspController``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omnigda.agent.instruction_parser import (
    InstructionParserProtocol,
    RuleBasedMultimodalParser,
)
from omnigda.agent.spatial_grounding import GroundingDecision, SpatialGrounder3D
from omnigda.core.config import load_config
from omnigda.core.gda_math import normalized_entropy, softmax
from omnigda.core.schemas import (
    ExecutionResult,
    Object3DDetection,
    RGBDFrame,
    SceneGraph3D,
    SegmentationResult,
    SupportSample,
)
from omnigda.grasp.planner import ConfidenceAwareGraspPlanner, GraspPlanOutput
from omnigda.perception_2d.segment_gda import (
    InstanceProposal,
    SegmentGDAPlus,
    extract_instance_masks,
)
from omnigda.perception_3d.multimodal_gmda import PartialAwareMultimodalGMDA
from omnigda.perception_3d.pointcloud import (
    compute_oriented_bounding_box,
    confidence_weighted_backproject,
    fuse_multiview_pointclouds,
)
from omnigda.perception_3d.scene_graph import SceneGraphBuilder3D


@dataclass(slots=True)
class PerceptionOutput:
    """Structured output of single-view or multi-view fused perception."""

    segmentation: SegmentationResult
    proposals: list[InstanceProposal] = field(default_factory=list)
    detections: list[Object3DDetection] = field(default_factory=list)
    scene_graph: SceneGraph3D = field(default_factory=SceneGraph3D)
    seg2d_ms: float = 0.0
    gmda3d_ms: float = 0.0
    scene_graph_ms: float = 0.0
    total_ms: float = 0.0
    num_views_used: int = 1
    frame: RGBDFrame | None = None


def _collect_obstacle_points(
    scene_graph: SceneGraph3D,
    exclude_instance_id: int,
) -> np.ndarray | None:
    """Concatenate 3D point clouds of all non-target objects for gripper collision checking."""
    clouds = [
        obj.point_cloud_xyz
        for obj in scene_graph.objects
        if obj.instance_id != exclude_instance_id and obj.point_cloud_xyz.shape[0] > 0
    ]
    if not clouds:
        return None
    return np.concatenate(clouds, axis=0).astype(np.float32, copy=False)


class OmniGDAAgent:
    """Closed-loop multimodal perception, active NBV, and 6DoF manipulation agent."""

    def __init__(
        self,
        segmenter: SegmentGDAPlus | None = None,
        classifier_3d: PartialAwareMultimodalGMDA | None = None,
        scene_graph_builder: SceneGraphBuilder3D | None = None,
        grasp_planner: ConfidenceAwareGraspPlanner | None = None,
        instruction_parser: InstructionParserProtocol | None = None,
        spatial_grounder: SpatialGrounder3D | None = None,
        config: dict[str, Any] | None = None,
        entropy_nbv_threshold: float | None = None,
        confidence_accept_threshold: float | None = None,
        max_views: int | None = None,
        min_area_px: int = 150,
        min_cluster_points: int | None = None,
        nbv_match_radius_m: float = 0.08,
        table_z: float = 0.0,
        primary_camera: str = "base_camera",
        nbv_camera: str = "nbv_camera",
        fg_llr_threshold: float = 0.70,
        table_surface_margin_m: float = 0.004,
        same_frame_nms_radius_m: float = 0.045,
        robot_base_exclude_radius_m: float = 0.18,
        workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        | None = None,
    ) -> None:
        cfg = config if config is not None else load_config()
        agent_cfg = cfg.get("agent", {})
        cfg3d = cfg.get("perception_3d", {})

        self.config = cfg
        self.segmenter = segmenter if segmenter is not None else SegmentGDAPlus.from_config(cfg)
        self.classifier_3d = (
            classifier_3d if classifier_3d is not None else PartialAwareMultimodalGMDA(config=cfg)
        )
        self.scene_graph_builder = (
            scene_graph_builder if scene_graph_builder is not None else SceneGraphBuilder3D()
        )
        self.grasp_planner = (
            grasp_planner
            if grasp_planner is not None
            else ConfidenceAwareGraspPlanner.from_config(cfg)
        )
        self.instruction_parser: InstructionParserProtocol = (
            instruction_parser if instruction_parser is not None else RuleBasedMultimodalParser()
        )

        ent_thresh = (
            float(entropy_nbv_threshold)
            if entropy_nbv_threshold is not None
            else float(agent_cfg.get("entropy_nbv_threshold", 0.85))
        )
        conf_thresh = (
            float(confidence_accept_threshold)
            if confidence_accept_threshold is not None
            else float(agent_cfg.get("confidence_accept_threshold", 0.60))
        )

        if spatial_grounder is not None:
            self.spatial_grounder = spatial_grounder
        else:
            clip_enc = self.classifier_3d.clip_encoder if self.classifier_3d.enable_clip else None
            self.spatial_grounder = SpatialGrounder3D(
                clip_encoder=clip_enc,
                scene_graph_builder=self.scene_graph_builder,
                entropy_nbv_threshold=ent_thresh,
                confidence_accept_threshold=conf_thresh,
            )

        self._entropy_nbv_threshold = ent_thresh
        self._confidence_accept_threshold = conf_thresh
        self.spatial_grounder.entropy_nbv_threshold = ent_thresh
        self.spatial_grounder.confidence_accept_threshold = conf_thresh

        self.max_views = (
            int(max_views) if max_views is not None else int(agent_cfg.get("max_views", 3))
        )
        self.min_area_px = max(1, int(min_area_px))
        self.min_cluster_points = (
            int(min_cluster_points)
            if min_cluster_points is not None
            else int(cfg3d.get("min_cluster_points", 40))
        )
        self.nbv_match_radius_m = float(nbv_match_radius_m)
        self.table_z = float(table_z)
        self.primary_camera = str(primary_camera)
        self.nbv_camera = str(nbv_camera)
        self.fg_llr_threshold = float(fg_llr_threshold)
        self.table_surface_margin_m = float(table_surface_margin_m)
        self.same_frame_nms_radius_m = float(same_frame_nms_radius_m)
        self.robot_base_exclude_radius_m = float(robot_base_exclude_radius_m)
        self.workspace_bounds = workspace_bounds
        self._robot_base_xy: np.ndarray | None = None

        # Diagnostics retained after each perception / execution cycle
        self.last_perception: PerceptionOutput | None = None
        self.last_nbv_pre_perception: PerceptionOutput | None = None
        self.last_grounding: GroundingDecision | None = None
        self.last_grasp_plan: GraspPlanOutput | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OmniGDAAgent:
        """Instantiate an ``OmniGDAAgent`` from a YAML config dictionary."""
        return cls(config=config, **kwargs)

    @property
    def entropy_nbv_threshold(self) -> float:
        return self._entropy_nbv_threshold

    @entropy_nbv_threshold.setter
    def entropy_nbv_threshold(self, value: float) -> None:
        self._entropy_nbv_threshold = float(value)
        self.spatial_grounder.entropy_nbv_threshold = float(value)

    @property
    def confidence_accept_threshold(self) -> float:
        return self._confidence_accept_threshold

    @confidence_accept_threshold.setter
    def confidence_accept_threshold(self, value: float) -> None:
        self._confidence_accept_threshold = float(value)
        self.spatial_grounder.confidence_accept_threshold = float(value)

    def register_object_few_shot(
        self,
        class_name: str,
        samples: Sequence[SupportSample],
        text_prompt: str | None = None,
    ) -> float:
        """Register an object class into both 2D SegmentGDA+ and 3D Multimodal GMDA.

        Args:
            class_name: Unique object class identifier (e.g., ``"cyan_t_tool"``).
            samples: One or more few-shot ``SupportSample`` templates.
            text_prompt: Optional open-vocabulary text description.

        Returns:
            Wall-clock registration latency in milliseconds.
        """
        sample_list = list(samples)
        if not sample_list:
            raise ValueError(f"samples cannot be empty for class {class_name!r}")

        t0 = time.perf_counter()

        inferred_prompt = text_prompt
        if inferred_prompt is None:
            for s in sample_list:
                if s.text_prompt:
                    inferred_prompt = s.text_prompt
                    break
        effective_prompt = (inferred_prompt or class_name.replace("_", " ")).strip()

        # 1. Register in 2D SegmentGDA+ whenever RGB + mask templates are available
        samples_2d = [s for s in sample_list if s.rgb is not None and s.mask is not None]
        if samples_2d:
            self.segmenter.register_object(class_name, samples_2d)

        # 2. Register in 3D Partial-Aware Multimodal GMDA (geom_3d + size_3d + visual_clip + text)
        self.classifier_3d.register_few_shot(
            class_name=class_name,
            samples=sample_list,
            register_text=True,
            table_z=self.table_z,
        )
        if text_prompt is not None and self.classifier_3d.clip_encoder is not None:
            self.classifier_3d.register_zero_shot_text(class_name, effective_prompt)

        # 3. Sync prompt to SpatialGrounder3D for open-vocabulary text grounding
        self.spatial_grounder.register_class_prompt(class_name, effective_prompt)

        return (time.perf_counter() - t0) * 1000.0

    def register_object_zero_shot(
        self,
        class_name: str,
        text_prompt: str,
    ) -> float:
        """Zero-shot open-vocabulary text prototype registration (CLIP text branch + grounder)."""
        t0 = time.perf_counter()
        prompt = (text_prompt or class_name.replace("_", " ")).strip()
        self.classifier_3d.register_zero_shot_text(class_name, prompt)
        self.spatial_grounder.register_class_prompt(class_name, prompt)
        return (time.perf_counter() - t0) * 1000.0

    def _in_workspace(self, centroid: np.ndarray) -> bool:
        if self._robot_base_xy is not None and self.robot_base_exclude_radius_m > 0.0:
            if float(np.linalg.norm(centroid[:2] - self._robot_base_xy)) < self.robot_base_exclude_radius_m:
                return False
        if self.workspace_bounds is None:
            return True
        (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = self.workspace_bounds
        return (
            x_lo <= float(centroid[0]) <= x_hi
            and y_lo <= float(centroid[1]) <= y_hi
            and z_lo <= float(centroid[2]) <= z_hi
        )

    def _build_foreground_proposal_seg(self, seg_result: SegmentationResult) -> SegmentationResult:
        """Recover high-objectness foreground proposals via binary fg-vs-bg log-likelihood ratio.

        When many classes are registered in 2D SegmentGDA+, single-class softmax confidence can
        drop below ``confidence_threshold`` even when foreground decisively beats background
        (``max_k ll_k - ll_bg > 0``). Using the class-count-invariant binary LLR preserves all
        tabletop object proposals for downstream 4-modality 3D GMDA disambiguation.
        """
        if seg_result.log_likelihoods.shape[0] < 2:
            return seg_result

        ll_bg = seg_result.log_likelihoods[0].astype(np.float64)
        fg_lls = seg_result.log_likelihoods[1:].astype(np.float64)
        best_fg_idx = (1 + np.argmax(fg_lls, axis=0)).astype(np.int32)
        max_fg_ll = np.max(fg_lls, axis=0)

        margin = np.clip(max_fg_ll - ll_bg, -30.0, 30.0)
        fg_vs_bg = (1.0 / (1.0 + np.exp(-margin))).astype(np.float32)

        keep = (seg_result.class_map > 0) | (fg_vs_bg >= self.fg_llr_threshold)
        fg_class_map = np.where(
            keep,
            np.where(seg_result.class_map > 0, seg_result.class_map, best_fg_idx),
            0,
        ).astype(np.int32)
        fg_masks = {
            cname: (fg_class_map == idx)
            for idx, cname in enumerate(seg_result.class_names, start=1)
        }
        fg_conf = np.where(
            fg_class_map > 0,
            np.maximum(seg_result.confidence_map, fg_vs_bg),
            1.0 - fg_vs_bg,
        ).astype(np.float32)

        return SegmentationResult(
            class_names=seg_result.class_names,
            class_map=fg_class_map,
            masks=fg_masks,
            confidence_map=fg_conf,
            log_likelihoods=seg_result.log_likelihoods,
            entropy_map=seg_result.entropy_map,
            latency_ms=seg_result.latency_ms,
        )

    def _deduplicate_frame_detections(
        self,
        detections: list[Object3DDetection],
    ) -> list[Object3DDetection]:
        """Suppress small overlapping 3D boundary fragments on the same physical object."""
        if len(detections) <= 1 or self.same_frame_nms_radius_m <= 0.0:
            return detections

        ranked = sorted(
            detections,
            key=lambda d: d.confidence * float(np.log1p(max(d.point_cloud_xyz.shape[0], 1))),
            reverse=True,
        )
        kept: list[Object3DDetection] = []
        for det in ranked:
            if any(
                float(np.linalg.norm(det.centroid - k.centroid)) < self.same_frame_nms_radius_m
                for k in kept
            ):
                continue
            kept.append(det)

        for new_id, det in enumerate(kept, start=1):
            det.instance_id = new_id
        return kept

    def perceive_frame(self, frame: RGBDFrame) -> PerceptionOutput:
        """Run full 2D SegmentGDA+ -> 3D RGB-D backprojection -> 4-modality GMDA -> SceneGraph3D."""
        t_total = time.perf_counter()

        # Stage 1: 2D Few-Shot Dense Segmentation + Foreground-Recovered Instance Proposals
        t_2d = time.perf_counter()
        seg_result = self.segmenter.segment(frame.rgb)
        proposal_seg = self._build_foreground_proposal_seg(seg_result)
        proposals = extract_instance_masks(proposal_seg, min_area_px=self.min_area_px)
        seg2d_ms = (time.perf_counter() - t_2d) * 1000.0

        # Stage 2: Per-instance 3D backprojection, SOR denoising, OBB, and 4-modality GMDA fusion
        t_3d = time.perf_counter()
        raw_detections: list[Object3DDetection] = []
        min_pts_hard = min(self.min_cluster_points, 12)
        next_inst_id = 1

        for prop in proposals:
            xyz, rgb_pts = confidence_weighted_backproject(
                frame=frame,
                mask=prop.mask,
                confidence_map=proposal_seg.confidence_map,
                conf_thresh=self.classifier_3d.confidence_denoise_thresh,
                depth_min=self.classifier_3d.depth_min_m,
                depth_max=self.classifier_3d.depth_max_m,
                sor_nb_neighbors=self.classifier_3d.sor_nb_neighbors,
                sor_std_ratio=self.classifier_3d.sor_std_ratio,
                voxel_size=0.003,
                min_points=min_pts_hard,
                return_colors=True,
            )

            # Fallback to unthresholded instance mask if confidence denoising stripped a thin object
            if xyz.shape[0] < self.min_cluster_points:
                xyz_fb, rgb_fb = confidence_weighted_backproject(
                    frame=frame,
                    mask=prop.mask,
                    confidence_map=None,
                    depth_min=self.classifier_3d.depth_min_m,
                    depth_max=self.classifier_3d.depth_max_m,
                    sor_nb_neighbors=self.classifier_3d.sor_nb_neighbors,
                    sor_std_ratio=self.classifier_3d.sor_std_ratio,
                    voxel_size=0.003,
                    min_points=min_pts_hard,
                    return_colors=True,
                )
                if xyz_fb.shape[0] > xyz.shape[0]:
                    xyz, rgb_pts = xyz_fb, rgb_fb

            if xyz.shape[0] < min_pts_hard:
                continue

            # Strip stray flat tabletop surface skirt pixels and reject flat table patches
            if self.table_surface_margin_m > 0.0:
                above = xyz[:, 2] >= (self.table_z + self.table_surface_margin_m)
                if int(np.count_nonzero(above)) < min_pts_hard:
                    continue
                xyz = xyz[above]
                if rgb_pts is not None:
                    rgb_pts = rgb_pts[above]

            seg2d_dict = {
                cname: float(prop.fg_log_likelihoods[idx])
                for idx, cname in enumerate(seg_result.class_names)
            }
            seg2d_rel = float(np.clip(prop.mean_confidence, 0.25, 0.85))

            det = self.classifier_3d.classify_instance(
                instance_id=next_inst_id,
                xyz=xyz,
                rgb_points=rgb_pts,
                rgb_image=frame.rgb,
                mask_2d=prop.mask,
                bbox_2d=prop.bbox_2d,
                seg2d_logits=seg2d_dict,
                seg2d_reliability=seg2d_rel,
                table_z=self.table_z,
            )
            if not self._in_workspace(det.centroid):
                continue

            raw_detections.append(det)
            next_inst_id += 1

        detections = self._deduplicate_frame_detections(raw_detections)
        gmda3d_ms = (time.perf_counter() - t_3d) * 1000.0

        # Stage 3: 3D Spatial Scene Graph Construction
        t_sg = time.perf_counter()
        scene_graph = self.scene_graph_builder.build(
            detections=detections,
            timestamp_sec=frame.timestamp_sec,
        )
        scene_graph_ms = (time.perf_counter() - t_sg) * 1000.0
        total_ms = (time.perf_counter() - t_total) * 1000.0

        output = PerceptionOutput(
            segmentation=seg_result,
            proposals=proposals,
            detections=detections,
            scene_graph=scene_graph,
            seg2d_ms=seg2d_ms,
            gmda3d_ms=gmda3d_ms,
            scene_graph_ms=scene_graph_ms,
            total_ms=total_ms,
            num_views_used=1,
            frame=frame,
        )
        self.last_perception = output
        return output

    def fuse_multiview_detections(
        self,
        view1_dets: list[Object3DDetection],
        view2_dets: list[Object3DDetection],
        match_radius_m: float | None = None,
    ) -> list[Object3DDetection]:
        """Match 3D instances across two views and fuse point clouds + Bayesian posteriors.

        Matched instances (centroid distance < ``match_radius_m``, default 0.08m) undergo:
        1. Multi-view point cloud fusion via ``fuse_multiview_pointclouds`` (Open3D ICP + SOR).
        2. Re-estimated gravity-aligned OBB & centroid on the denser fused point cloud.
        3. Inverse-entropy weighted multi-view Bayesian log-posterior fusion:
           ``log p_fused(y) = log_softmax(w1 * log p1(y) + w2 * log p2(y))``,
           where ``w_v = (1 / (H_v + 0.05)) / mean(1 / (H + 0.05))``.
        """
        radius = float(self.nbv_match_radius_m if match_radius_m is None else match_radius_m)

        # Compute all valid pairwise centroid distances and greedily match closest pairs first
        pair_candidates: list[tuple[float, int, int]] = []
        for i, d1 in enumerate(view1_dets):
            for j, d2 in enumerate(view2_dets):
                dist = float(np.linalg.norm(d1.centroid - d2.centroid))
                if dist < radius:
                    pair_candidates.append((dist, i, j))
        pair_candidates.sort(key=lambda item: item[0])

        matched_v1: dict[int, int] = {}
        matched_v2: set[int] = set()
        for _, i, j in pair_candidates:
            if i not in matched_v1 and j not in matched_v2:
                matched_v1[i] = j
                matched_v2.add(j)

        fused_list: list[Object3DDetection] = []
        max_id = max((d.instance_id for d in view1_dets), default=0)

        for i, d1 in enumerate(view1_dets):
            if i not in matched_v1:
                fused_list.append(d1)
                continue

            d2 = view2_dets[matched_v1[i]]

            # 1. Fuse 3D point clouds with fine voxel resolution to increase surface density
            fused_xyz = fuse_multiview_pointclouds(
                clouds=[d1.point_cloud_xyz, d2.point_cloud_xyz],
                voxel_size=0.002,
                refine_icp=True,
                sor_nb_neighbors=self.classifier_3d.sor_nb_neighbors,
                sor_std_ratio=self.classifier_3d.sor_std_ratio,
            )
            # Ensure multi-view fusion preserves or increases cloud density over single views
            max_single = max(d1.point_cloud_xyz.shape[0], d2.point_cloud_xyz.shape[0])
            if fused_xyz.shape[0] < max_single:
                fused_fine = fuse_multiview_pointclouds(
                    clouds=[d1.point_cloud_xyz, d2.point_cloud_xyz],
                    voxel_size=0.001,
                    refine_icp=True,
                    sor_nb_neighbors=self.classifier_3d.sor_nb_neighbors,
                    sor_std_ratio=self.classifier_3d.sor_std_ratio,
                )
                if fused_fine.shape[0] > fused_xyz.shape[0]:
                    fused_xyz = fused_fine

            # 2. Recompute gravity-aligned OBB & centroid on the denser fused point cloud
            obb_center, obb_extent, obb_rotation = compute_oriented_bounding_box(fused_xyz)
            centroid = (
                np.mean(fused_xyz.astype(np.float64), axis=0)
                if fused_xyz.shape[0] > 0
                else obb_center.copy()
            )

            # 3. Inverse-entropy weighted multi-view Bayesian log-posterior fusion
            classes = list(
                dict.fromkeys(list(d1.posterior_probs.keys()) + list(d2.posterior_probs.keys()))
            )
            if not classes:
                classes = list(dict.fromkeys([d1.class_name, d2.class_name]))

            p1 = np.array(
                [max(float(d1.posterior_probs.get(c, 1e-6)), 1e-8) for c in classes],
                dtype=np.float64,
            )
            p2 = np.array(
                [max(float(d2.posterior_probs.get(c, 1e-6)), 1e-8) for c in classes],
                dtype=np.float64,
            )
            p1 /= float(np.sum(p1))
            p2 /= float(np.sum(p2))

            inv_h1 = 1.0 / (max(float(d1.entropy), 0.0) + 0.05)
            inv_h2 = 1.0 / (max(float(d2.entropy), 0.0) + 0.05)
            inv_h_sum = inv_h1 + inv_h2
            mean_inv_h = 0.5 * inv_h_sum

            # Scaled so average weight per view is 1.0 (sum = 2.0) -> true Bayesian sharpening
            w1 = inv_h1 / mean_inv_h
            w2 = inv_h2 / mean_inv_h
            fused_ll = w1 * np.log(p1) + w2 * np.log(p2)
            fused_probs = softmax(fused_ll)
            fused_ent = float(normalized_entropy(fused_probs))
            best_idx = int(np.argmax(fused_probs))
            best_class = classes[best_idx]
            fused_conf = float(fused_probs[best_idx])

            w1_norm = inv_h1 / inv_h_sum
            w2_norm = inv_h2 / inv_h_sum
            all_mods = set(d1.modality_weights.keys()) | set(d2.modality_weights.keys())
            fused_mod_weights = {
                m: float(
                    w1_norm * d1.modality_weights.get(m, 0.0)
                    + w2_norm * d2.modality_weights.get(m, 0.0)
                )
                for m in sorted(all_mods)
            }
            fused_mod_weights["nbv_view1_weight"] = float(w1_norm)
            fused_mod_weights["nbv_view2_weight"] = float(w2_norm)

            fused_det = Object3DDetection(
                instance_id=d1.instance_id,
                class_name=best_class,
                confidence=fused_conf,
                entropy=fused_ent,
                ood_score=float(min(d1.ood_score, d2.ood_score)),
                is_ood=bool(d1.is_ood and d2.is_ood),
                centroid=centroid,
                obb_center=obb_center,
                obb_extent=obb_extent,
                obb_rotation=obb_rotation,
                point_cloud_xyz=fused_xyz,
                point_cloud_rgb=d1.point_cloud_rgb,
                mask_2d=d1.mask_2d,
                bbox_2d=d1.bbox_2d,
                modality_weights=fused_mod_weights,
                posterior_probs={c: float(fused_probs[idx]) for idx, c in enumerate(classes)},
            )
            fused_list.append(fused_det)

        # Append newly discovered view-2 objects (occluded in view-1)
        for j, d2 in enumerate(view2_dets):
            if j in matched_v2:
                continue
            max_id += 1
            fused_list.append(
                Object3DDetection(
                    instance_id=max_id,
                    class_name=d2.class_name,
                    confidence=d2.confidence,
                    entropy=d2.entropy,
                    ood_score=d2.ood_score,
                    is_ood=d2.is_ood,
                    centroid=d2.centroid.copy(),
                    obb_center=d2.obb_center.copy(),
                    obb_extent=d2.obb_extent.copy(),
                    obb_rotation=d2.obb_rotation.copy(),
                    point_cloud_xyz=d2.point_cloud_xyz.copy(),
                    point_cloud_rgb=(
                        d2.point_cloud_rgb.copy() if d2.point_cloud_rgb is not None else None
                    ),
                    mask_2d=d2.mask_2d.copy() if d2.mask_2d is not None else None,
                    bbox_2d=d2.bbox_2d,
                    modality_weights=dict(d2.modality_weights),
                    posterior_probs=dict(d2.posterior_probs),
                )
            )

        return fused_list

    def perceive_and_fuse_nbv(
        self,
        sim: Any,
        prev_perception: PerceptionOutput,
        nbv_camera: str | None = None,
    ) -> PerceptionOutput:
        """Acquire a Next-Best-View RGB-D frame and perform multi-view 3D + Bayesian fusion."""
        t0 = time.perf_counter()
        self.last_nbv_pre_perception = prev_perception

        cam_name = nbv_camera or self.nbv_camera
        frame2, _ = sim.capture_rgbd(camera_name=cam_name)
        percep2 = self.perceive_frame(frame2)

        t_fuse = time.perf_counter()
        fused_dets = self.fuse_multiview_detections(
            view1_dets=prev_perception.detections,
            view2_dets=percep2.detections,
        )
        scene_graph = self.scene_graph_builder.build(
            detections=fused_dets,
            timestamp_sec=frame2.timestamp_sec,
        )
        fuse_sg_ms = (time.perf_counter() - t_fuse) * 1000.0
        extra_total_ms = (time.perf_counter() - t0) * 1000.0

        fused_output = PerceptionOutput(
            segmentation=percep2.segmentation,
            proposals=percep2.proposals,
            detections=fused_dets,
            scene_graph=scene_graph,
            seg2d_ms=prev_perception.seg2d_ms + percep2.seg2d_ms,
            gmda3d_ms=prev_perception.gmda3d_ms + percep2.gmda3d_ms,
            scene_graph_ms=prev_perception.scene_graph_ms + fuse_sg_ms,
            total_ms=prev_perception.total_ms + extra_total_ms,
            num_views_used=prev_perception.num_views_used + 1,
            frame=frame2,
        )
        self.last_perception = fused_output
        return fused_output

    def _plan_for_target(
        self,
        scene_graph: SceneGraph3D,
        target: Object3DDetection,
    ) -> GraspPlanOutput:
        obstacles = _collect_obstacle_points(scene_graph, exclude_instance_id=target.instance_id)
        return self.grasp_planner.plan(
            detection=target,
            obstacle_xyz=obstacles,
            table_z=self.table_z,
        )

    def execute_instruction(
        self,
        sim: Any,
        controller: Any,
        instruction_text: str,
        allow_nbv: bool = True,
        record_frames: bool = False,
    ) -> ExecutionResult:
        """Closed-loop parse -> perceive -> ground -> plan -> (optional NBV) -> physical execution."""
        self.last_nbv_pre_perception = None
        if hasattr(sim, "robot_base_pose"):
            self._robot_base_xy = np.asarray(sim.robot_base_pose.p[:2], dtype=np.float64)
        instruction = self.instruction_parser.parse(instruction_text)

        # 1. Primary camera RGB-D perception
        frame1, _ = sim.capture_rgbd(camera_name=self.primary_camera)
        percep = self.perceive_frame(frame1)
        grounding = self.spatial_grounder.ground(instruction, percep.scene_graph)
        self.last_grounding = grounding

        can_use_nbv = (
            allow_nbv
            and percep.num_views_used < self.max_views
            and (self.nbv_camera in sim.get_camera_names())
        )

        # 2. Handle pure inspection commands
        if instruction.action == "inspect":
            if can_use_nbv and ((not grounding.found) or grounding.needs_nbv):
                percep = self.perceive_and_fuse_nbv(sim, percep)
                grounding = self.spatial_grounder.ground(instruction, percep.scene_graph)
                self.last_grounding = grounding

            tgt = grounding.target_object
            return ExecutionResult(
                success=grounding.found,
                message=f"Inspect: {grounding.reasoning}",
                target_instance_id=tgt.instance_id if tgt is not None else -1,
                target_class_name=tgt.class_name if tgt is not None else "",
                grasp_candidate=None,
                num_views_used=percep.num_views_used,
                perception_latency_ms=percep.total_ms,
                planning_latency_ms=0.0,
            )

        # 3. Plan 6DoF grasps on view-1 if target was grounded
        total_grasp_plan_ms = 0.0
        grasp_out: GraspPlanOutput | None = None
        if grounding.found and grounding.target_object is not None:
            grasp_out = self._plan_for_target(percep.scene_graph, grounding.target_object)
            total_grasp_plan_ms += grasp_out.planning_latency_ms
        self.last_grasp_plan = grasp_out

        # 4. Active Perception / Multi-View Bayesian NBV trigger
        should_nbv = can_use_nbv and (
            (not grounding.found)
            or grounding.needs_nbv
            or (grasp_out is None)
            or grasp_out.needs_nbv
        )
        if should_nbv:
            percep = self.perceive_and_fuse_nbv(sim, percep)
            grounding = self.spatial_grounder.ground(instruction, percep.scene_graph)
            self.last_grounding = grounding
            if grounding.found and grounding.target_object is not None:
                grasp_out = self._plan_for_target(percep.scene_graph, grounding.target_object)
                total_grasp_plan_ms += grasp_out.planning_latency_ms
                self.last_grasp_plan = grasp_out

        # 5. Validate grounded target and planned grasp candidates
        if not grounding.found or grounding.target_object is None:
            return ExecutionResult(
                success=False,
                message=f"Grounding failed: {grounding.reasoning}",
                target_instance_id=-1,
                target_class_name="",
                grasp_candidate=None,
                num_views_used=percep.num_views_used,
                perception_latency_ms=percep.total_ms,
                planning_latency_ms=total_grasp_plan_ms,
            )

        target = grounding.target_object
        if grasp_out is None or not grasp_out.candidates:
            # One dense retry with extra candidates before declaring grasp failure
            obstacles = _collect_obstacle_points(
                percep.scene_graph, exclude_instance_id=target.instance_id
            )
            grasp_out = self.grasp_planner.plan(
                detection=target,
                obstacle_xyz=obstacles,
                table_z=self.table_z,
                num_candidates=96,
            )
            total_grasp_plan_ms += grasp_out.planning_latency_ms
            self.last_grasp_plan = grasp_out

        if not grasp_out.candidates:
            return ExecutionResult(
                success=False,
                message=f"No collision-free 6DoF grasp found for {target.class_name}.",
                target_instance_id=target.instance_id,
                target_class_name=target.class_name,
                grasp_candidate=None,
                num_views_used=percep.num_views_used,
                perception_latency_ms=percep.total_ms,
                planning_latency_ms=total_grasp_plan_ms,
            )

        # 6. Physical execution via FrankaGraspController (retry diverse NMS grasps on IK miss)
        sim_obj_names = set(sim.get_object_names())
        resolved_sim_name = target.class_name if target.class_name in sim_obj_names else None
        place_pos = (
            grounding.destination_position
            if instruction.action == "pick_and_place"
            else None
        )

        exec_res: ExecutionResult | None = None
        for cand in grasp_out.candidates[:3]:
            if place_pos is not None:
                exec_res = controller.execute_pick_and_place(
                    grasp=cand,
                    place_position=place_pos,
                    target_object_name=resolved_sim_name,
                    record_frames=record_frames,
                )
            else:
                exec_res = controller.execute_pick(
                    grasp=cand,
                    target_object_name=resolved_sim_name,
                    record_frames=record_frames,
                )

            if exec_res.success or "Motion planning failed for pre-grasp" not in exec_res.message:
                break

        assert exec_res is not None
        exec_res.target_instance_id = target.instance_id
        exec_res.num_views_used = percep.num_views_used
        exec_res.perception_latency_ms = percep.total_ms
        exec_res.planning_latency_ms = total_grasp_plan_ms + exec_res.planning_latency_ms
        return exec_res
