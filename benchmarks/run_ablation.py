"""Multi-Scene Cluttered Tabletop Ablation & Closed-Loop Franka Grasp Benchmark.

Evaluates 6 real perception/manipulation variants on RTX 3060 across N=20 random seeds
in `ClutteredTabletopSim(num_objects=6, include_novel=True)` (160 object instances:
120 base + 40 held-out novel) plus open-set OOD evaluation (120 ID + 40 OOD) and
real physical closed-loop Franka Panda grasping trials:

  1. `zero_shot_text`        : Zero-Shot OpenCLIP Text Prototypes Only (depth DBSCAN proposals)
  2. `pure_2d_no_refine`     : Pure 2D SegmentGDA (No Guided Refine, 2D logits only)
  3. `pure_2d_guided_refine` : Pure 2D SegmentGDA-Plus (Color-Guided Refine, 2D logits only)
  4. `pure_3d_geom_size`     : Pure 3D Geometric + Size GMDA (no 2D visual, depth DBSCAN proposals)
  5. `naive_equal_fusion`    : 2D + 3D Naive Equal-Weight Fusion (uniform weights, no NBV)
  6. `omnigda_full`          : OmniGDA-Manip Full (Entropy-Gated Bayesian 4-Modality + TensorRT FP16 + NBV)

Writes structured JSON results to `docs/results/ablation_results.json`.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
from sklearn.metrics import roc_auc_score

from omnigda.agent.instruction_parser import RuleBasedMultimodalParser
from omnigda.agent.pipeline import OmniGDAAgent, PerceptionOutput
from omnigda.agent.spatial_grounding import SpatialGrounder3D
from omnigda.core.config import load_config
from omnigda.core.gda_math import backproject_depth, log_softmax, normalized_entropy, softmax
from omnigda.core.schemas import (
    ExecutionResult,
    Object3DDetection,
    RGBDFrame,
    SegmentationResult,
    SupportSample,
)
from omnigda.deploy.runtime import create_clip_encoder, create_dinov2_extractor
from omnigda.grasp.planner import ConfidenceAwareGraspPlanner
from omnigda.perception_2d.backbone import DINOv2MultiScaleExtractor
from omnigda.perception_2d.segment_gda import SegmentGDAPlus, extract_instance_masks
from omnigda.perception_3d.encoders import OpenCLIPMultimodalEncoder
from omnigda.perception_3d.multimodal_gmda import PartialAwareMultimodalGMDA
from omnigda.perception_3d.pointcloud import (
    compute_oriented_bounding_box,
    confidence_weighted_backproject,
    estimate_pointcloud_completeness,
)
from omnigda.perception_3d.scene_graph import SceneGraphBuilder3D
from omnigda.sim.clutter_env import (
    DEFAULT_BASE_OBJECTS,
    HELD_OUT_NOVEL_OBJECTS,
    OBJECT_CATALOG,
    PANDA_BASE_POSITION,
    ClutteredTabletopSim,
)
from omnigda.sim.controller import FrankaGraspController
from omnigda.sim.templates import capture_few_shot_templates

ROBOT_BASE_XY = np.asarray(PANDA_BASE_POSITION[:2], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers: Class-Agnostic 3D Depth Clustering & Ablation Classifier Wrappers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DepthClusterProposal:
    """Class-agnostic tabletop object proposal extracted purely from depth."""

    xyz: np.ndarray
    rgb_points: np.ndarray
    mask_2d: np.ndarray
    bbox_2d: tuple[int, int, int, int]


def extract_depth_dbscan_proposals(
    frame: RGBDFrame,
    table_z: float = 0.0,
    z_margin: float = 0.005,
    voxel_size: float = 0.004,
    dbscan_eps: float = 0.022,
    dbscan_min_pts: int = 12,
    min_cluster_points: int = 20,
    robot_base_exclude_radius: float = 0.18,
) -> list[DepthClusterProposal]:
    """Extract class-agnostic 3D object clusters purely from tabletop RGB-D depth."""
    h, w = frame.depth.shape
    depth = frame.depth
    valid = np.isfinite(depth) & (depth >= 0.15) & (depth <= 2.0)
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return []

    z_cam = depth[ys, xs].astype(np.float64)
    x_cam = (xs.astype(np.float64) - frame.intrinsics.cx) * z_cam / frame.intrinsics.fx
    y_cam = (ys.astype(np.float64) - frame.intrinsics.cy) * z_cam / frame.intrinsics.fy
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    rot = frame.camera_to_world[:3, :3]
    trans = frame.camera_to_world[:3, 3]
    pts_world = pts_cam @ rot.T + trans[None, :]

    # Tabletop workspace filter (no color / semantic prior)
    in_ws = (
        (pts_world[:, 2] >= table_z + z_margin)
        & (pts_world[:, 2] <= 0.25)
        & (pts_world[:, 0] >= -0.32)
        & (pts_world[:, 0] <= 0.26)
        & (pts_world[:, 1] >= -0.36)
        & (pts_world[:, 1] <= 0.36)
    )
    dist_robot = np.linalg.norm(pts_world[:, :2] - ROBOT_BASE_XY[None, :], axis=-1)
    in_ws &= dist_robot >= robot_base_exclude_radius

    if int(np.count_nonzero(in_ws)) < min_cluster_points:
        return []

    pts_fg = pts_world[in_ws]
    ys_fg = ys[in_ws]
    xs_fg = xs[in_ws]

    # Voxel-grid deduplication for fast Open3D DBSCAN + exact pixel-to-cluster mapping
    vox_coords = np.floor(pts_fg / voxel_size).astype(np.int32)
    _, first_idx, inv_idx = np.unique(
        vox_coords, axis=0, return_index=True, return_inverse=True
    )
    vox_pts = pts_fg[first_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vox_pts)
    labels_vox = np.asarray(
        pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_pts, print_progress=False)
    )
    if labels_vox.size == 0:
        return []

    pix_labels = labels_vox[inv_idx]
    proposals: list[DepthClusterProposal] = []
    for cid in range(int(labels_vox.max()) + 1):
        sel = pix_labels == cid
        if int(np.count_nonzero(sel)) < min_cluster_points:
            continue

        c_ys = ys_fg[sel]
        c_xs = xs_fg[sel]
        mask = np.zeros((h, w), dtype=bool)
        mask[c_ys, c_xs] = True
        bbox = (int(c_xs.min()), int(c_ys.min()), int(c_xs.max()) + 1, int(c_ys.max()) + 1)

        # Voxel-downsampled clean 3D cloud for geometry & grasp planning
        sel_vox = labels_vox == cid
        c_xyz = vox_pts[sel_vox].astype(np.float32)
        c_rgb = frame.rgb[ys_fg[first_idx[sel_vox]], xs_fg[first_idx[sel_vox]]].astype(np.float32)
        if c_rgb.max() > 1.0:
            c_rgb = c_rgb / 255.0

        proposals.append(
            DepthClusterProposal(
                xyz=c_xyz,
                rgb_points=c_rgb,
                mask_2d=mask,
                bbox_2d=bbox,
            )
        )

    return proposals


class NaiveUniformFusionGMDA(PartialAwareMultimodalGMDA):
    """Ablation variant: fixed equal-weight fusion (no inverse-entropy or completeness gating)."""

    def classify_instance(
        self,
        instance_id: int = 0,
        xyz: np.ndarray | None = None,
        rgb_points: np.ndarray | None = None,
        rgb_image: np.ndarray | None = None,
        mask_2d: np.ndarray | None = None,
        bbox_2d: tuple[int, int, int, int] | None = None,
        frame: RGBDFrame | None = None,
        confidence_map: np.ndarray | None = None,
        seg2d_logits: dict[str, float] | np.ndarray | None = None,
        seg2d_reliability: float = 0.85,
        table_z: float = 0.0,
    ) -> Object3DDetection:
        if not self.class_names:
            raise RuntimeError("No classes registered.")

        pts = (
            np.asarray(xyz, dtype=np.float32)
            if xyz is not None
            else np.zeros((0, 3), dtype=np.float32)
        )
        obb_center, obb_extent, obb_rotation = compute_oriented_bounding_box(pts)
        centroid = (
            np.mean(pts.astype(np.float64), axis=0) if pts.shape[0] > 0 else obb_center.copy()
        )
        resolved_bbox = bbox_2d if bbox_2d is not None else self._bbox_from_mask(mask_2d)

        log_likelihoods: dict[str, np.ndarray] = {}
        modality_d2: dict[str, dict[str, float]] = {}

        if seg2d_logits is not None:
            aligned_2d = self._align_external_logits(seg2d_logits)
            if aligned_2d is not None:
                log_likelihoods["seg2d"] = aligned_2d

        if pts.shape[0] >= 6:
            if self._models["geom_3d"].is_fitted:
                g_feat = self.geom_encoder.encode(pts)
                ll_g, d2_g = self._align_gmda_log_likelihood(self._models["geom_3d"], g_feat)
                log_likelihoods["geom_3d"] = ll_g
                modality_d2["geom_3d"] = d2_g
            if self._models["size_3d"].is_fitted:
                s_feat = self.size_encoder.encode(
                    obb_extent=obb_extent, obb_center=obb_center, table_z=table_z
                )
                ll_s, d2_s = self._align_gmda_log_likelihood(self._models["size_3d"], s_feat)
                log_likelihoods["size_3d"] = ll_s
                modality_d2["size_3d"] = d2_s

        if rgb_image is not None and self.clip_encoder is not None:
            if self._models["visual_clip"].is_fitted or self._text_prototypes:
                img_emb = self.clip_encoder.encode_image_crop(
                    rgb=rgb_image,
                    mask=mask_2d,
                    bbox=resolved_bbox if resolved_bbox != (0, 0, 0, 0) else None,
                )
                if self._models["visual_clip"].is_fitted:
                    ll_v, d2_v = self._align_gmda_log_likelihood(
                        self._models["visual_clip"], img_emb
                    )
                    log_likelihoods["visual_clip"] = ll_v
                    modality_d2["visual_clip"] = d2_v
                if self._text_prototypes:
                    self._ensure_complete_text_prototypes()
                    proto_mat = np.stack(
                        [self._text_prototypes[c].astype(np.float64) for c in self.class_names],
                        axis=0,
                    )
                    sims = proto_mat @ img_emb.astype(np.float64)
                    log_likelihoods["text_clip"] = sims * self.clip_logit_scale

        # Naive equal-weight fusion: uniform 1/M average of per-modality log-posteriors
        m_count = len(log_likelihoods)
        uniform_w = 1.0 / float(max(m_count, 1))
        fused_ll = np.zeros(len(self.class_names), dtype=np.float64)
        mod_weights: dict[str, float] = {}
        for mod_name, ll in log_likelihoods.items():
            fused_ll += uniform_w * log_softmax(ll)
            mod_weights[mod_name] = uniform_w

        fused_probs = softmax(fused_ll)
        fused_ent = float(normalized_entropy(fused_probs))
        best_idx = int(np.argmax(fused_probs))
        pred_class = self.class_names[best_idx]
        confidence = float(fused_probs[best_idx])

        # Uniform unweighted average Mahalanobis d2 for OOD
        d2_vals = [
            d2_map[pred_class] if pred_class in d2_map else min(d2_map.values())
            for d2_map in modality_d2.values()
            if d2_map
        ]
        ood_score = float(np.mean(d2_vals)) if d2_vals else 0.0

        return Object3DDetection(
            instance_id=int(instance_id),
            class_name=pred_class,
            confidence=confidence,
            entropy=fused_ent,
            ood_score=ood_score,
            is_ood=bool(ood_score > self.ood_mahalanobis_threshold),
            centroid=centroid.astype(np.float64),
            obb_center=obb_center.astype(np.float64),
            obb_extent=obb_extent.astype(np.float64),
            obb_rotation=obb_rotation.astype(np.float64),
            point_cloud_xyz=pts,
            point_cloud_rgb=(
                np.asarray(rgb_points, dtype=np.float32) if rgb_points is not None else None
            ),
            mask_2d=mask_2d.astype(bool) if mask_2d is not None else None,
            bbox_2d=resolved_bbox,
            modality_weights=mod_weights,
            posterior_probs={c: float(fused_probs[i]) for i, c in enumerate(self.class_names)},
        )


# ---------------------------------------------------------------------------
# Unified Ablation Runner Wrapper (for Perception + Closed-Loop Grasping)
# ---------------------------------------------------------------------------


class AblationSystem:
    """Thin wrapper presenting a uniform perceive + execute interface for each ablation."""

    def __init__(
        self,
        name: str,
        display_name: str,
        class_names: list[str],
        cfg: dict[str, Any],
        segmenter: SegmentGDAPlus | None = None,
        classifier_3d: PartialAwareMultimodalGMDA | None = None,
        full_agent: OmniGDAAgent | None = None,
        use_depth_proposals: bool = False,
        pure_2d_only: bool = False,
        allow_nbv: bool = False,
    ) -> None:
        self.name = name
        self.display_name = display_name
        self.class_names = list(class_names)
        self.segmenter = segmenter
        self.classifier_3d = classifier_3d
        self.full_agent = full_agent
        self.use_depth_proposals = use_depth_proposals
        self.pure_2d_only = pure_2d_only
        self.allow_nbv = allow_nbv

        self.scene_graph_builder = SceneGraphBuilder3D()
        self.grasp_planner = ConfidenceAwareGraspPlanner.from_config(cfg)
        self.parser = RuleBasedMultimodalParser()
        clip_enc = (
            full_agent.classifier_3d.clip_encoder
            if full_agent is not None
            else (classifier_3d.clip_encoder if classifier_3d is not None else None)
        )
        self.grounder = SpatialGrounder3D(
            clip_encoder=clip_enc,
            scene_graph_builder=self.scene_graph_builder,
            entropy_nbv_threshold=float(cfg["agent"]["entropy_nbv_threshold"]),
            confidence_accept_threshold=float(cfg["agent"]["confidence_accept_threshold"]),
        )
        for cname in self.class_names:
            self.grounder.register_class_prompt(cname, OBJECT_CATALOG[cname].description)

    def perceive_frame(self, frame: RGBDFrame) -> tuple[PerceptionOutput, dict[str, np.ndarray]]:
        """Run perception and return `(PerceptionOutput, pred_2d_masks_by_class)`."""
        h, w = frame.rgb.shape[:2]
        if self.full_agent is not None:
            out = self.full_agent.perceive_frame(frame)
            fused_masks: dict[str, np.ndarray] = {
                c: np.zeros((h, w), dtype=bool) for c in self.class_names
            }
            for det in out.detections:
                if det.mask_2d is not None and det.class_name in fused_masks:
                    fused_masks[det.class_name] |= det.mask_2d
            return out, fused_masks

        t0 = time.perf_counter()
        pred_masks: dict[str, np.ndarray] = {c: np.zeros((h, w), dtype=bool) for c in self.class_names}
        detections: list[Object3DDetection] = []
        seg2d_ms = 0.0
        gmda3d_ms = 0.0

        if self.pure_2d_only:
            assert self.segmenter is not None
            t_2d = time.perf_counter()
            seg_res = self.segmenter.segment(frame.rgb)
            ll_bg = seg_res.log_likelihoods[0].astype(np.float64)
            fg_lls = seg_res.log_likelihoods[1:].astype(np.float64)
            best_fg_idx = (1 + np.argmax(fg_lls, axis=0)).astype(np.int32)
            max_fg_ll = np.max(fg_lls, axis=0)
            margin = np.clip(max_fg_ll - ll_bg, -30.0, 30.0)
            fg_vs_bg = (1.0 / (1.0 + np.exp(-margin))).astype(np.float32)
            keep = (seg_res.class_map > 0) | (fg_vs_bg >= 0.70)
            fg_cmap = np.where(
                keep, np.where(seg_res.class_map > 0, seg_res.class_map, best_fg_idx), 0
            ).astype(np.int32)
            fg_conf = np.where(
                fg_cmap > 0, np.maximum(seg_res.confidence_map, fg_vs_bg), 1.0 - fg_vs_bg
            ).astype(np.float32)
            prop_seg = SegmentationResult(
                class_names=seg_res.class_names,
                class_map=fg_cmap,
                masks={c: (fg_cmap == i) for i, c in enumerate(seg_res.class_names, start=1)},
                confidence_map=fg_conf,
                log_likelihoods=seg_res.log_likelihoods,
                entropy_map=seg_res.entropy_map,
                latency_ms=seg_res.latency_ms,
            )
            proposals = extract_instance_masks(prop_seg, min_area_px=150)
            seg2d_ms = (time.perf_counter() - t_2d) * 1000.0
            pred_masks = prop_seg.masks

            t_3d = time.perf_counter()
            next_id = 1
            for prop in proposals:
                xyz, rgb_pts = confidence_weighted_backproject(
                    frame=frame,
                    mask=prop.mask,
                    confidence_map=prop_seg.confidence_map,
                    conf_thresh=0.45,
                    depth_min=0.15,
                    depth_max=2.0,
                    sor_nb_neighbors=20,
                    sor_std_ratio=1.5,
                    voxel_size=0.003,
                    min_points=12,
                    return_colors=True,
                )
                if xyz.shape[0] < 12:
                    continue
                above = xyz[:, 2] >= 0.004
                if int(np.count_nonzero(above)) < 12:
                    continue
                xyz = xyz[above]
                if rgb_pts is not None:
                    rgb_pts = rgb_pts[above]
                obb_c, obb_e, obb_r = compute_oriented_bounding_box(xyz)
                centroid = np.mean(xyz.astype(np.float64), axis=0)
                if float(np.linalg.norm(centroid[:2] - ROBOT_BASE_XY)) < 0.18:
                    continue

                probs = softmax(prop.fg_log_likelihoods.astype(np.float64))
                ent = float(normalized_entropy(probs))
                best_idx = int(np.argmax(probs))
                cname = seg_res.class_names[best_idx]
                ood_score = float(-np.max(prop.fg_log_likelihoods))

                detections.append(
                    Object3DDetection(
                        instance_id=next_id,
                        class_name=cname,
                        confidence=float(probs[best_idx]),
                        entropy=ent,
                        ood_score=ood_score,
                        is_ood=False,
                        centroid=centroid,
                        obb_center=obb_c,
                        obb_extent=obb_e,
                        obb_rotation=obb_r,
                        point_cloud_xyz=xyz,
                        point_cloud_rgb=rgb_pts,
                        mask_2d=prop.mask,
                        bbox_2d=prop.bbox_2d,
                        modality_weights={"seg2d": 1.0},
                        posterior_probs={
                            c: float(probs[i]) for i, c in enumerate(seg_res.class_names)
                        },
                    )
                )
                next_id += 1
            detections = _nms_detections(detections)
            gmda3d_ms = (time.perf_counter() - t_3d) * 1000.0
            dummy_seg = seg_res

        elif self.use_depth_proposals:
            assert self.classifier_3d is not None
            t_3d = time.perf_counter()
            clusters = extract_depth_dbscan_proposals(frame)
            for idx, cl in enumerate(clusters, start=1):
                det = self.classifier_3d.classify_instance(
                    instance_id=idx,
                    xyz=cl.xyz,
                    rgb_points=cl.rgb_points,
                    rgb_image=frame.rgb,
                    mask_2d=cl.mask_2d,
                    bbox_2d=cl.bbox_2d,
                    seg2d_logits=None,
                    table_z=0.0,
                )
                detections.append(det)
                if det.class_name in pred_masks:
                    pred_masks[det.class_name] |= cl.mask_2d
            gmda3d_ms = (time.perf_counter() - t_3d) * 1000.0
            dummy_seg = SegmentationResult(
                class_names=list(self.class_names),
                class_map=np.zeros((h, w), dtype=np.int32),
                masks=pred_masks,
                confidence_map=np.zeros((h, w), dtype=np.float32),
                log_likelihoods=np.zeros((len(self.class_names) + 1, h, w), dtype=np.float32),
                entropy_map=np.zeros((h, w), dtype=np.float32),
                latency_ms=0.0,
            )

        else:
            assert self.segmenter is not None and self.classifier_3d is not None
            helper_agent = OmniGDAAgent(
                segmenter=self.segmenter,
                classifier_3d=self.classifier_3d,
                scene_graph_builder=self.scene_graph_builder,
                grasp_planner=self.grasp_planner,
                spatial_grounder=self.grounder,
            )
            out = helper_agent.perceive_frame(frame)
            fused_masks = {c: np.zeros((h, w), dtype=bool) for c in self.class_names}
            for det in out.detections:
                if det.mask_2d is not None and det.class_name in fused_masks:
                    fused_masks[det.class_name] |= det.mask_2d
            return out, fused_masks

        t_sg = time.perf_counter()
        sg = self.scene_graph_builder.build(detections=detections, timestamp_sec=frame.timestamp_sec)
        sg_ms = (time.perf_counter() - t_sg) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        return (
            PerceptionOutput(
                segmentation=dummy_seg,
                detections=detections,
                scene_graph=sg,
                seg2d_ms=seg2d_ms,
                gmda3d_ms=gmda3d_ms,
                scene_graph_ms=sg_ms,
                total_ms=total_ms,
                num_views_used=1,
                frame=frame,
            ),
            pred_masks,
        )

    def execute_pick_trial(
        self,
        sim: ClutteredTabletopSim,
        controller: FrankaGraspController,
        instruction_text: str,
    ) -> ExecutionResult:
        """Execute a closed-loop physical pick instruction and return ExecutionResult."""
        if self.full_agent is not None:
            return self.full_agent.execute_instruction(
                sim=sim,
                controller=controller,
                instruction_text=instruction_text,
                allow_nbv=self.allow_nbv,
                record_frames=False,
            )

        inst = self.parser.parse(instruction_text)
        frame, _ = sim.capture_rgbd("base_camera")
        percep, _ = self.perceive_frame(frame)
        grounding = self.grounder.ground(inst, percep.scene_graph)
        if not grounding.found or grounding.target_object is None:
            return ExecutionResult(
                success=False,
                message=f"Grounding failed: {grounding.reasoning}",
                target_instance_id=-1,
                target_class_name="",
                num_views_used=1,
                perception_latency_ms=percep.total_ms,
            )

        target = grounding.target_object
        obs_clouds = [
            o.point_cloud_xyz
            for o in percep.scene_graph.objects
            if o.instance_id != target.instance_id and o.point_cloud_xyz.shape[0] > 0
        ]
        obstacles = np.concatenate(obs_clouds, axis=0) if obs_clouds else None
        grasp_out = self.grasp_planner.plan(detection=target, obstacle_xyz=obstacles, table_z=0.0)
        if not grasp_out.candidates:
            grasp_out = self.grasp_planner.plan(
                detection=target, obstacle_xyz=obstacles, table_z=0.0, num_candidates=96
            )
        if not grasp_out.candidates:
            return ExecutionResult(
                success=False,
                message=f"No collision-free grasp for {target.class_name}",
                target_instance_id=target.instance_id,
                target_class_name=target.class_name,
                num_views_used=1,
                perception_latency_ms=percep.total_ms,
                planning_latency_ms=grasp_out.planning_latency_ms,
            )

        sim_names = set(sim.get_object_names())
        resolved_name = target.class_name if target.class_name in sim_names else None
        exec_res: ExecutionResult | None = None
        for cand in grasp_out.candidates[:3]:
            exec_res = controller.execute_pick(
                grasp=cand,
                target_object_name=resolved_name,
                record_frames=False,
            )
            if exec_res.success or "Motion planning failed for pre-grasp" not in exec_res.message:
                break
        assert exec_res is not None
        exec_res.target_instance_id = target.instance_id
        exec_res.target_class_name = target.class_name
        exec_res.num_views_used = 1
        exec_res.perception_latency_ms = percep.total_ms
        exec_res.planning_latency_ms = grasp_out.planning_latency_ms + exec_res.planning_latency_ms
        return exec_res


def _nms_detections(
    detections: list[Object3DDetection],
    radius_m: float = 0.045,
) -> list[Object3DDetection]:
    if len(detections) <= 1:
        return detections
    ranked = sorted(
        detections,
        key=lambda d: d.confidence * float(np.log1p(max(d.point_cloud_xyz.shape[0], 1))),
        reverse=True,
    )
    kept: list[Object3DDetection] = []
    for det in ranked:
        if any(float(np.linalg.norm(det.centroid - k.centroid)) < radius_m for k in kept):
            continue
        kept.append(det)
    for new_id, det in enumerate(kept, start=1):
        det.instance_id = new_id
    return kept


# ---------------------------------------------------------------------------
# Evaluation Metrics Helpers
# ---------------------------------------------------------------------------


def _gt_masks_from_frame(
    frame: RGBDFrame,
    seg_map: dict[int, str],
    object_names: list[str],
) -> dict[str, np.ndarray]:
    assert frame.gt_segmentation is not None
    gt_masks: dict[str, np.ndarray] = {}
    for name in object_names:
        ids = [sid for sid, ename in seg_map.items() if ename == name]
        gt_masks[name] = np.isin(frame.gt_segmentation, ids)
    return gt_masks


def _compute_scene_iou(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    class_names: list[str],
) -> dict[str, float]:
    ious: dict[str, float] = {}
    for cname in class_names:
        gt = gt_masks[cname]
        if not np.any(gt):
            continue
        pred = pred_masks.get(cname, np.zeros_like(gt, dtype=bool))
        inter = int(np.count_nonzero(pred & gt))
        union = int(np.count_nonzero(pred | gt))
        ious[cname] = float(inter / max(union, 1))
    return ious


def _match_detections_to_gt(
    detections: list[Object3DDetection],
    gt_positions: dict[str, np.ndarray],
    match_radius_m: float = 0.085,
) -> dict[str, Object3DDetection | None]:
    """Greedy closest-pair 3D centroid matching between detections and ground-truth objects."""
    pairs: list[tuple[float, str, int]] = []
    for gt_name, gt_pos in gt_positions.items():
        for idx, det in enumerate(detections):
            dist = float(np.linalg.norm(det.centroid - gt_pos))
            if dist < match_radius_m:
                pairs.append((dist, gt_name, idx))
    pairs.sort(key=lambda x: x[0])

    matched_gt: dict[str, Object3DDetection | None] = {k: None for k in gt_positions}
    used_det: set[int] = set()
    for _, gt_name, idx in pairs:
        if matched_gt[gt_name] is None and idx not in used_det:
            matched_gt[gt_name] = detections[idx]
            used_det.add(idx)
    return matched_gt


def _compute_ood_auroc(
    systems_ood: dict[str, AblationSystem],
    scene_cache: list[tuple[int, RGBDFrame, dict[int, str], dict[str, np.ndarray]]],
    base_names: list[str],
    novel_names: list[str],
) -> dict[str, dict[str, float]]:
    """Evaluate open-set OOD detection AUROC (6 registered base ID vs 2 unregistered novel OOD)."""
    all_names = base_names + novel_names
    scores_by_sys: dict[str, list[float]] = {k: [] for k in systems_ood}
    y_true: list[int] = []

    for _seed, frame, seg_map, _gt_pos in scene_cache:
        gt_masks = _gt_masks_from_frame(frame, seg_map, all_names)

        seg_cache: dict[str, SegmentationResult] = {}
        for sname, sys_obj in systems_ood.items():
            seg_obj = (
                sys_obj.full_agent.segmenter
                if sys_obj.full_agent is not None
                else sys_obj.segmenter
            )
            if seg_obj is not None:
                seg_cache[sname] = seg_obj.segment(frame.rgb)

        for cname in all_names:
            mask = gt_masks[cname]
            if int(np.count_nonzero(mask)) < 25:
                continue
            xyz, rgb_pts = backproject_depth(
                depth=frame.depth,
                fx=frame.intrinsics.fx,
                fy=frame.intrinsics.fy,
                cx=frame.intrinsics.cx,
                cy=frame.intrinsics.cy,
                camera_to_world=frame.camera_to_world,
                mask=mask,
                depth_min=0.15,
                depth_max=2.0,
            )
            if xyz.shape[0] < 15:
                continue
            ys, xs = np.nonzero(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

            is_ood_gt = 1 if cname in novel_names else 0
            y_true.append(is_ood_gt)

            for sname, sys_obj in systems_ood.items():
                if sys_obj.pure_2d_only:
                    seg_res = seg_cache[sname]
                    fg_lls = seg_res.log_likelihoods[1:, mask]
                    mean_fg_ll = np.mean(fg_lls, axis=1)
                    score = float(-np.max(mean_fg_ll))
                elif sys_obj.use_depth_proposals:
                    assert sys_obj.classifier_3d is not None
                    det = sys_obj.classifier_3d.classify_instance(
                        instance_id=1,
                        xyz=xyz,
                        rgb_points=rgb_pts,
                        rgb_image=frame.rgb,
                        mask_2d=mask,
                        bbox_2d=bbox,
                        seg2d_logits=None,
                        table_z=0.0,
                    )
                    score = float(det.ood_score)
                else:
                    seg_res = seg_cache[sname]
                    fg_lls = seg_res.log_likelihoods[1:, mask]
                    mean_fg_ll = np.mean(fg_lls, axis=1)
                    seg2d_dict = {
                        c: float(mean_fg_ll[i]) for i, c in enumerate(seg_res.class_names)
                    }
                    clf = (
                        sys_obj.full_agent.classifier_3d
                        if sys_obj.full_agent is not None
                        else sys_obj.classifier_3d
                    )
                    assert clf is not None
                    det = clf.classify_instance(
                        instance_id=1,
                        xyz=xyz,
                        rgb_points=rgb_pts,
                        rgb_image=frame.rgb,
                        mask_2d=mask,
                        bbox_2d=bbox,
                        seg2d_logits=seg2d_dict,
                        table_z=0.0,
                    )
                    score = float(det.ood_score)
                scores_by_sys[sname].append(score)

    y_arr = np.asarray(y_true, dtype=np.int32)
    out: dict[str, dict[str, float]] = {}
    for sname, sc in scores_by_sys.items():
        s_arr = np.asarray(sc, dtype=np.float64)
        auroc = float(roc_auc_score(y_arr, s_arr))
        id_mean = float(np.mean(s_arr[y_arr == 0]))
        ood_mean = float(np.mean(s_arr[y_arr == 1]))
        out[sname] = {
            "ood_auroc": auroc,
            "id_score_mean": id_mean,
            "ood_score_mean": ood_mean,
            "num_id_instances": int(np.sum(y_arr == 0)),
            "num_ood_instances": int(np.sum(y_arr == 1)),
        }
    return out


# ---------------------------------------------------------------------------
# Build All 6 Ablation Variants (sharing GPU encoders cleanly)
# ---------------------------------------------------------------------------


def _build_ablation_systems(
    cfg: dict[str, Any],
    templates: dict[str, list[SupportSample]],
    register_names: list[str],
    dinov2_pt: DINOv2MultiScaleExtractor,
    clip_pt: OpenCLIPMultimodalEncoder,
    dinov2_trt: Any,
    clip_trt: Any,
    measure_online_reg_names: list[str] | None = None,
) -> tuple[dict[str, AblationSystem], dict[str, float]]:
    """Instantiate & register all 6 ablation systems; return `(systems, online_reg_ms)`."""
    online_set = set(measure_online_reg_names or [])
    base_reg = [n for n in register_names if n not in online_set]
    online_reg = [n for n in register_names if n in online_set]
    reg_latencies_ms: dict[str, float] = {}

    # 1. Zero-Shot Text Only (OpenCLIP)
    clf_zs = PartialAwareMultimodalGMDA(config=cfg, clip_encoder=clip_pt, enable_clip=True)
    for cname in register_names:
        clf_zs.register_zero_shot_text(cname, OBJECT_CATALOG[cname].description)
    sys_zs = AblationSystem(
        name="zero_shot_text",
        display_name="Zero-Shot Text Only (CLIP)",
        class_names=register_names,
        cfg=cfg,
        classifier_3d=clf_zs,
        use_depth_proposals=True,
    )

    # 2a. Pure 2D SegmentGDA (No Guided Refine)
    cfg_nr = copy.deepcopy(cfg)
    cfg_nr["perception_2d"]["spatial_refine"] = False
    seg_nr = SegmentGDAPlus.from_config(cfg_nr, extractor=dinov2_pt)
    for cname in register_names:
        seg_nr.register_object(cname, templates[cname])
    sys_2d_nr = AblationSystem(
        name="pure_2d_no_refine",
        display_name="Pure 2D SegmentGDA (No Refine)",
        class_names=register_names,
        cfg=cfg_nr,
        segmenter=seg_nr,
        pure_2d_only=True,
    )

    # 2b. Pure 2D SegmentGDA-Plus (Color-Guided Refine)
    cfg_gr = copy.deepcopy(cfg)
    cfg_gr["perception_2d"]["spatial_refine"] = True
    seg_gr = SegmentGDAPlus.from_config(cfg_gr, extractor=dinov2_pt)
    for cname in register_names:
        seg_gr.register_object(cname, templates[cname])
    sys_2d_gr = AblationSystem(
        name="pure_2d_guided_refine",
        display_name="SegmentGDA-Plus (Guided Refine)",
        class_names=register_names,
        cfg=cfg_gr,
        segmenter=seg_gr,
        pure_2d_only=True,
    )

    # 3. Pure 3D Geometric + Size GMDA (no 2D visual)
    clf_3d = PartialAwareMultimodalGMDA(config=cfg, clip_encoder=None, enable_clip=False)
    for cname in register_names:
        geom_only_samples = [
            SupportSample(
                class_name=s.class_name,
                point_cloud_xyz=s.point_cloud_xyz,
                depth=s.depth,
                mask=s.mask,
                intrinsics=s.intrinsics,
                camera_to_world=s.camera_to_world,
                text_prompt=None,
                metadata=dict(s.metadata),
            )
            for s in templates[cname]
        ]
        clf_3d.register_few_shot(cname, geom_only_samples, register_text=False, table_z=0.0)
    sys_3d = AblationSystem(
        name="pure_3d_geom_size",
        display_name="Pure 3D Geom+Size GMDA",
        class_names=register_names,
        cfg=cfg,
        classifier_3d=clf_3d,
        use_depth_proposals=True,
    )

    # 4. 2D + 3D Naive Equal-Weight Fusion (no entropy gating, no NBV)
    clf_naive = NaiveUniformFusionGMDA(config=cfg, clip_encoder=clip_pt, enable_clip=True)
    for cname in register_names:
        clf_naive.register_few_shot(cname, templates[cname], register_text=True, table_z=0.0)
        clf_naive.register_zero_shot_text(cname, OBJECT_CATALOG[cname].description)
    sys_naive = AblationSystem(
        name="naive_equal_fusion",
        display_name="2D+3D Naive Equal-Weight Fusion",
        class_names=register_names,
        cfg=cfg,
        segmenter=seg_gr,
        classifier_3d=clf_naive,
    )

    # 5. OmniGDA-Manip (Full: Entropy-Gated Bayesian 4-Modality + TensorRT FP16 + NBV)
    cfg_trt = copy.deepcopy(cfg)
    cfg_trt["perception_2d"]["backend"] = "tensorrt"
    cfg_trt["perception_2d"]["spatial_refine"] = True
    seg_trt = SegmentGDAPlus.from_config(cfg_trt, extractor=dinov2_trt)
    clf_trt = PartialAwareMultimodalGMDA(config=cfg_trt, clip_encoder=clip_trt, enable_clip=True)
    full_agent = OmniGDAAgent(
        segmenter=seg_trt,
        classifier_3d=clf_trt,
        config=cfg_trt,
    )
    for cname in base_reg:
        full_agent.register_object_few_shot(
            cname, templates[cname], text_prompt=OBJECT_CATALOG[cname].description
        )
    for cname in online_reg:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = full_agent.register_object_few_shot(
            cname, templates[cname], text_prompt=OBJECT_CATALOG[cname].description
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        reg_latencies_ms[cname] = float(ms)

    sys_full = AblationSystem(
        name="omnigda_full",
        display_name="OmniGDA-Manip (Full + TRT + NBV)",
        class_names=register_names,
        cfg=cfg_trt,
        full_agent=full_agent,
        allow_nbv=True,
    )

    systems = {
        "zero_shot_text": sys_zs,
        "pure_2d_no_refine": sys_2d_nr,
        "pure_2d_guided_refine": sys_2d_gr,
        "pure_3d_geom_size": sys_3d,
        "naive_equal_fusion": sys_naive,
        "omnigda_full": sys_full,
    }
    return systems, reg_latencies_ms


# ---------------------------------------------------------------------------
# Closed-Loop Franka Grasping Suite
# ---------------------------------------------------------------------------


def _build_grasp_trials(
    seeds: list[int],
    base_names: list[str],
    novel_names: list[str],
) -> list[dict[str, Any]]:
    """Construct 18 diverse deterministic pick trials (6 base + 6 novel + 6 spatial/bilingual)."""
    spatial_specs = [
        ("pick up the leftmost object", "leftmost"),
        ("抓起最右边的物体", "rightmost"),
        ("pick up the frontmost object", "frontmost"),
        ("抓起最左边的物体", "leftmost"),
        ("pick up the rightmost object", "rightmost"),
        ("抓起最前面的物体", "frontmost"),
    ]
    trials: list[dict[str, Any]] = []
    for idx, seed in enumerate(seeds[:6]):
        base_obj = base_names[idx % len(base_names)]
        novel_obj = novel_names[idx % len(novel_names)]
        sp_instr, sp_kind = spatial_specs[idx % len(spatial_specs)]
        trials.append(
            {
                "seed": seed,
                "category": "base",
                "instruction": f"pick up the {base_obj.replace('_', ' ')}",
                "expected_target": base_obj,
                "spatial_kind": None,
            }
        )
        trials.append(
            {
                "seed": seed,
                "category": "novel",
                "instruction": f"pick up the {novel_obj.replace('_', ' ')}",
                "expected_target": novel_obj,
                "spatial_kind": None,
            }
        )
        trials.append(
            {
                "seed": seed,
                "category": "spatial",
                "instruction": sp_instr,
                "expected_target": None,
                "spatial_kind": sp_kind,
            }
        )
    return trials


def _resolve_gt_spatial_target(gt_positions: dict[str, np.ndarray], kind: str) -> str:
    if kind == "leftmost":
        return max(gt_positions, key=lambda k: float(gt_positions[k][1]))
    if kind == "rightmost":
        return min(gt_positions, key=lambda k: float(gt_positions[k][1]))
    if kind == "frontmost":
        # SceneGraphBuilder3D forward_axis is +X (away from robot), so frontmost is min(x)
        return min(gt_positions, key=lambda k: float(gt_positions[k][0]))
    raise ValueError(f"Unknown spatial kind: {kind}")


def run_closed_loop_grasp_eval(
    systems: dict[str, AblationSystem],
    trials: list[dict[str, Any]],
    num_objects: int = 6,
) -> dict[str, dict[str, Any]]:
    """Run real closed-loop Franka Panda physical pick trials for every ablation method."""
    sim = ClutteredTabletopSim(
        num_objects=num_objects,
        include_novel=True,
        seed=trials[0]["seed"],
        sim_backend="cpu",
    )
    controller = FrankaGraspController(sim, record_frames=False)
    grasp_results: dict[str, dict[str, Any]] = {}

    try:
        for sname, sys_obj in systems.items():
            print(f"  [GSR] Running {len(trials)} physical Franka pick trials for `{sname}`...")
            records: list[dict[str, Any]] = []
            for t in trials:
                sim.reset(seed=int(t["seed"]))
                gt_positions = sim.get_object_positions()
                expected = (
                    t["expected_target"]
                    if t["expected_target"] is not None
                    else _resolve_gt_spatial_target(gt_positions, t["spatial_kind"])
                )
                pre_z = float(gt_positions[expected][2])

                res = sys_obj.execute_pick_trial(
                    sim=sim,
                    controller=controller,
                    instruction_text=t["instruction"],
                )
                post_z = float(sim.get_object_positions()[expected][2])
                lifted_expected = (post_z - pre_z) >= 0.045
                correct_target = res.target_class_name == expected
                trial_ok = bool(res.success and correct_target and lifted_expected)

                records.append(
                    {
                        "seed": int(t["seed"]),
                        "category": t["category"],
                        "instruction": t["instruction"],
                        "expected_target": expected,
                        "predicted_target": res.target_class_name,
                        "exec_success": bool(res.success),
                        "lifted_expected": bool(lifted_expected),
                        "correct_target": bool(correct_target),
                        "trial_success": trial_ok,
                        "num_views_used": int(res.num_views_used),
                        "message": res.message,
                    }
                )

            def _rate(cat: str | None = None, recs: list[dict[str, Any]] = records) -> float:
                sub = recs if cat is None else [r for r in recs if r["category"] == cat]
                return float(sum(1 for r in sub if r["trial_success"]) / max(len(sub), 1))

            grasp_results[sname] = {
                "num_trials": len(records),
                "num_success": sum(1 for r in records if r["trial_success"]),
                "gsr_overall": _rate(None),
                "gsr_base": _rate("base"),
                "gsr_novel": _rate("novel"),
                "gsr_spatial": _rate("spatial"),
                "trials": records,
            }
            print(
                f"    -> `{sname}`: overall GSR = {grasp_results[sname]['num_success']}/{len(records)} "
                f"({grasp_results[sname]['gsr_overall'] * 100:.1f}%) "
                f"[base={grasp_results[sname]['gsr_base'] * 100:.1f}%, "
                f"novel={grasp_results[sname]['gsr_novel'] * 100:.1f}%, "
                f"spatial={grasp_results[sname]['gsr_spatial'] * 100:.1f}%]"
            )
    finally:
        sim.close()

    return grasp_results


# ---------------------------------------------------------------------------
# Main Ablation Benchmark Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OmniGDA-Manip N=20 multi-scene ablation & closed-loop Franka benchmark."
    )
    parser.add_argument("--num-scenes", type=int, default=20, help="Number of random scenes")
    parser.add_argument("--base-seed", type=int, default=100, help="Starting random seed for scenes")
    parser.add_argument("--num-shots", type=int, default=3, help="Few-shot support shots per class")
    parser.add_argument(
        "--output",
        type=str,
        default="docs/results/ablation_results.json",
        help="Path to write structured JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    base_names = list(DEFAULT_BASE_OBJECTS[:6])
    novel_names = list(HELD_OUT_NOVEL_OBJECTS)
    all_names = base_names + novel_names
    scene_seeds = [args.base_seed + i for i in range(args.num_scenes)]

    print(
        f"[1/5] Capturing {args.num_shots}-shot support templates for {len(all_names)} classes (seed=42)..."
    )
    templates = capture_few_shot_templates(
        object_names=all_names,
        num_shots=args.num_shots,
        seed=42,
    )

    print(
        f"[2/5] Capturing {args.num_scenes} cluttered tabletop scenes "
        f"({len(all_names)} objects/scene = {args.num_scenes * len(all_names)} instances)..."
    )
    scene_cache: list[tuple[int, RGBDFrame, dict[int, str], dict[str, np.ndarray]]] = []
    sim = ClutteredTabletopSim(
        num_objects=6,
        include_novel=True,
        seed=scene_seeds[0],
        sim_backend="cpu",
    )
    try:
        for seed in scene_seeds:
            sim.reset(seed=seed)
            frame, seg_map = sim.capture_rgbd("base_camera")
            gt_pos = sim.get_object_positions()
            scene_cache.append((seed, frame, seg_map, gt_pos))
    finally:
        sim.close()

    print("[3/5] Initializing PyTorch & TensorRT 10.9 backbones and registering ablation models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dinov2_pt = DINOv2MultiScaleExtractor(
        model_name=cfg["perception_2d"]["backbone"],
        input_size=int(cfg["perception_2d"]["input_size"]),
        feature_layers=tuple(cfg["perception_2d"]["feature_layers"]),
        device=device,
        fp16=(device == "cuda"),
    )
    clip_pt = OpenCLIPMultimodalEncoder(
        model_name=cfg["perception_3d"]["clip_model"],
        pretrained=cfg["perception_3d"]["clip_pretrained"],
        device=device,
    )
    dinov2_trt = create_dinov2_extractor(cfg, backend="tensorrt")
    clip_trt = create_clip_encoder(cfg, backend="tensorrt")

    systems_full, online_reg_ms = _build_ablation_systems(
        cfg=cfg,
        templates=templates,
        register_names=all_names,
        dinov2_pt=dinov2_pt,
        clip_pt=clip_pt,
        dinov2_trt=dinov2_trt,
        clip_trt=clip_trt,
        measure_online_reg_names=novel_names,
    )
    print(f"  -> Online few-shot registration latency (3-shot, TRT): {online_reg_ms}")

    # Warmup GPU models on first frame
    for sys_obj in systems_full.values():
        _ = sys_obj.perceive_frame(scene_cache[0][1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(
        f"[4/5] Evaluating 2D mIoU, Base/Novel Top-1 Acc, Latency, and OOD AUROC across {args.num_scenes} scenes..."
    )
    percep_metrics: dict[str, dict[str, Any]] = {}
    for sname, sys_obj in systems_full.items():
        all_ious: list[float] = []
        base_ious: list[float] = []
        novel_ious: list[float] = []
        base_correct = 0
        base_total = 0
        novel_correct = 0
        novel_total = 0
        matched_total = 0
        latencies_ms: list[float] = []
        completeness_vals: list[float] = []

        for _seed, frame, seg_map, gt_pos in scene_cache:
            gt_masks = _gt_masks_from_frame(frame, seg_map, all_names)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, pred_masks = sys_obj.perceive_frame(frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            scene_ious = _compute_scene_iou(pred_masks, gt_masks, all_names)
            for cname, iou_val in scene_ious.items():
                all_ious.append(iou_val)
                if cname in base_names:
                    base_ious.append(iou_val)
                else:
                    novel_ious.append(iou_val)

            matches = _match_detections_to_gt(out.detections, gt_pos, match_radius_m=0.085)
            for cname in all_names:
                det = matches.get(cname)
                if det is not None:
                    matched_total += 1
                    completeness_vals.append(
                        estimate_pointcloud_completeness(det.point_cloud_xyz, det.obb_extent)
                    )
                is_ok = det is not None and det.class_name == cname
                if cname in base_names:
                    base_total += 1
                    base_correct += int(is_ok)
                else:
                    novel_total += 1
                    novel_correct += int(is_ok)

        total_inst = base_total + novel_total
        percep_metrics[sname] = {
            "display_name": sys_obj.display_name,
            "miou_2d_all": float(np.mean(all_ious)) if all_ious else 0.0,
            "miou_2d_base": float(np.mean(base_ious)) if base_ious else 0.0,
            "miou_2d_novel": float(np.mean(novel_ious)) if novel_ious else 0.0,
            "base_top1_acc": float(base_correct / max(base_total, 1)),
            "novel_top1_acc": float(novel_correct / max(novel_total, 1)),
            "overall_top1_acc": float((base_correct + novel_correct) / max(total_inst, 1)),
            "detection_recall": float(matched_total / max(total_inst, 1)),
            "base_correct": int(base_correct),
            "base_total": int(base_total),
            "novel_correct": int(novel_correct),
            "novel_total": int(novel_total),
            "latency_mean_ms": float(np.mean(latencies_ms)),
            "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
            "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
            "mean_cloud_completeness": float(np.mean(completeness_vals)) if completeness_vals else 0.0,
        }
        pm = percep_metrics[sname]
        print(
            f"  -> `{sname:22s}`: 2D mIoU={pm['miou_2d_all']*100:5.2f}% | "
            f"Base Acc={pm['base_top1_acc']*100:5.1f}% | "
            f"Novel Acc={pm['novel_top1_acc']*100:5.1f}% | "
            f"Recall={pm['detection_recall']*100:5.1f}% | "
            f"Lat={pm['latency_mean_ms']:6.1f} ms"
        )

    # Open-set OOD evaluation (register only 6 base classes; novel 2 classes are unregistered OOD)
    print("  [OOD] Computing Open-Set OOD Detection AUROC (6 registered ID vs 2 unregistered OOD)...")
    systems_ood, _ = _build_ablation_systems(
        cfg=cfg,
        templates=templates,
        register_names=base_names,
        dinov2_pt=dinov2_pt,
        clip_pt=clip_pt,
        dinov2_trt=dinov2_trt,
        clip_trt=clip_trt,
    )
    ood_metrics = _compute_ood_auroc(systems_ood, scene_cache, base_names, novel_names)
    for sname, om in ood_metrics.items():
        percep_metrics[sname]["ood_auroc"] = om["ood_auroc"]
        percep_metrics[sname]["ood_id_score_mean"] = om["id_score_mean"]
        percep_metrics[sname]["ood_novel_score_mean"] = om["ood_score_mean"]
        print(
            f"  -> `{sname:22s}`: OOD AUROC={om['ood_auroc']*100:5.2f}% "
            f"(ID mean={om['id_score_mean']:.2f}, OOD mean={om['ood_score_mean']:.2f})"
        )
    del systems_ood
    gc.collect()

    # Step 5: Closed-Loop Physical Franka Grasping Evaluation
    print("[5/5] Running Closed-Loop Franka Panda Physical Grasping Evaluation (18 trials/method)...")
    trials = _build_grasp_trials(scene_seeds, base_names, novel_names)
    grasp_metrics = run_closed_loop_grasp_eval(systems_full, trials, num_objects=6)

    for sname, gm in grasp_metrics.items():
        percep_metrics[sname]["gsr_overall"] = gm["gsr_overall"]
        percep_metrics[sname]["gsr_base"] = gm["gsr_base"]
        percep_metrics[sname]["gsr_novel"] = gm["gsr_novel"]
        percep_metrics[sname]["gsr_spatial"] = gm["gsr_spatial"]
        percep_metrics[sname]["gsr_num_success"] = gm["num_success"]
        percep_metrics[sname]["gsr_num_trials"] = gm["num_trials"]

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "num_scenes": int(args.num_scenes),
            "scene_seeds": scene_seeds,
            "num_shots": int(args.num_shots),
            "base_objects": base_names,
            "novel_objects": novel_names,
            "total_instances_evaluated": int(args.num_scenes * len(all_names)),
            "grasp_trials_per_method": len(trials),
            "online_few_shot_registration_ms": online_reg_ms,
            "online_few_shot_registration_mean_ms": float(np.mean(list(online_reg_ms.values()))),
        },
        "methods": percep_metrics,
        "ood_details": ood_metrics,
        "grasp_trials": grasp_metrics,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== OmniGDA-Manip Multi-Scene Ablation Summary ===")
    print("| Method | 2D mIoU | Base Acc | Novel Acc | OOD AUROC | Franka GSR | Latency (ms) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for _sname, pm in percep_metrics.items():
        print(
            f"| {pm['display_name']} | {pm['miou_2d_all']*100:.2f}% "
            f"| {pm['base_top1_acc']*100:.1f}% | {pm['novel_top1_acc']*100:.1f}% "
            f"| {pm['ood_auroc']*100:.2f}% | {pm['gsr_overall']*100:.1f}% "
            f"({pm['gsr_num_success']}/{pm['gsr_num_trials']}) | {pm['latency_mean_ms']:.1f} |"
        )
    print(
        f"\nOnline Few-Shot Registration Latency (3-shot, TensorRT FP16): "
        f"{payload['metadata']['online_few_shot_registration_mean_ms']:.1f} ms"
    )
    print(f"Saved ablation benchmark JSON to: {out_path}")


if __name__ == "__main__":
    main()
