"""Antipodal 6DoF grasp hypothesis sampler for tabletop manipulation.

TCP Coordinate Frame Convention (Franka Panda Standard)
-------------------------------------------------------
Every sampled ``GraspCandidate`` uses the Franka Panda TCP convention:
- ``+Z`` axis: unit approach vector pointing into the target object (e.g., ``(0, 0, -1)``
  in world coordinates for a top-down tabletop grasp).
- ``+Y`` axis: unit jaw closing axis connecting the two opposing finger pads.
- ``+X`` axis: right-hand orthogonal completion ``X = Y x Z`` so ``R = [X, Y, Z]`` is a
  proper rotation matrix in ``SO(3)`` (``R^T R = I``, ``det(R) = +1``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from omnigda.core.gda_math import rotation_matrix_to_quat_xyzw
from omnigda.core.schemas import GraspCandidate, Object3DDetection


def _build_tcp_rotation(
    closing_axis_y: np.ndarray,
    approach_axis_z: np.ndarray,
) -> np.ndarray:
    """Build an orthonormal Franka Panda TCP rotation matrix ``R = [X, Y, Z]`` in SO(3)."""
    zx, zy, zz = float(approach_axis_z[0]), float(approach_axis_z[1]), float(approach_axis_z[2])
    zn = math.sqrt(zx * zx + zy * zy + zz * zz)
    if zn < 1e-9:
        zx, zy, zz = 0.0, 0.0, -1.0
    else:
        inv_zn = 1.0 / zn
        zx, zy, zz = zx * inv_zn, zy * inv_zn, zz * inv_zn

    yx, yy, yz = float(closing_axis_y[0]), float(closing_axis_y[1]), float(closing_axis_y[2])
    dot_yz = yx * zx + yy * zy + yz * zz
    yx -= dot_yz * zx
    yy -= dot_yz * zy
    yz -= dot_yz * zz
    yn = math.sqrt(yx * yx + yy * yy + yz * yz)
    if yn < 1e-9:
        rx, ry, rz = (1.0, 0.0, 0.0) if abs(zy) > 0.9 else (0.0, 1.0, 0.0)
        dot_rz = rx * zx + ry * zy + rz * zz
        yx, yy, yz = rx - dot_rz * zx, ry - dot_rz * zy, rz - dot_rz * zz
        yn = math.sqrt(yx * yx + yy * yy + yz * yz)
    inv_yn = 1.0 / yn
    yx, yy, yz = yx * inv_yn, yy * inv_yn, yz * inv_yn

    # X = Y x Z
    xx = yy * zz - yz * zy
    xy = yz * zx - yx * zz
    xz = yx * zy - yy * zx
    xn = math.sqrt(xx * xx + xy * xy + xz * xz)
    inv_xn = 1.0 / xn
    xx, xy, xz = xx * inv_xn, xy * inv_xn, xz * inv_xn

    # Re-orthogonalize Y = Z x X to guarantee machine-precision SO(3)
    yx = zy * xz - zz * xy
    yy = zz * xx - zx * xz
    yz = zx * xy - zy * xx
    inv_yn2 = 1.0 / math.sqrt(yx * yx + yy * yy + yz * yz)
    yx, yy, yz = yx * inv_yn2, yy * inv_yn2, yz * inv_yn2

    return np.array(
        [[xx, yx, zx], [xy, yy, zy], [xz, yz, zz]],
        dtype=np.float64,
    )


def build_tcp_pose(
    position: np.ndarray,
    closing_axis_y: np.ndarray,
    approach_axis_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct an orthonormal Franka Panda TCP pose from position, jaw Y, and approach Z.

    Convention:
      - ``Z`` = unit approach vector pointing into the object
      - ``Y`` = unit jaw closing axis (orthogonalized against ``Z``)
      - ``X = Y x Z`` = right-hand orthogonal completion

    Returns:
        Tuple ``(pose_4x4, position_f64, quaternion_xyzw, approach_vector)``.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    rot = _build_tcp_rotation(closing_axis_y, approach_axis_z)
    pose_4x4 = np.eye(4, dtype=np.float64)
    pose_4x4[:3, :3] = rot
    pose_4x4[:3, 3] = pos
    quat_xyzw = rotation_matrix_to_quat_xyzw(rot)
    approach_vector = rot[:, 2].copy()
    return pose_4x4, pos.copy(), quat_xyzw, approach_vector


def _orthonormalize_rotation(rot: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix onto SO(3) via SVD."""
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=np.float64).reshape(3, 3))
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _materialize_candidates(
    raw_items: list[tuple[np.ndarray, np.ndarray, float, float, float, float]],
    target_instance_id: int,
    target_class_name: str,
) -> list[GraspCandidate]:
    """Batch-convert ``(rot, pos, width_m, score, fc, anti)`` tuples to ``GraspCandidate``s."""
    if not raw_items:
        return []

    rots = np.stack([item[0] for item in raw_items], axis=0)  # (N, 3, 3)
    quats = Rotation.from_matrix(rots).as_quat().astype(np.float64)  # (N, 4) xyzw

    candidates: list[GraspCandidate] = []
    for idx, (rot, pos, width_m, score, fc, anti) in enumerate(raw_items):
        pose_4x4 = np.eye(4, dtype=np.float64)
        pose_4x4[:3, :3] = rot
        pose_4x4[:3, 3] = pos
        candidates.append(
            GraspCandidate(
                pose_4x4=pose_4x4,
                position=pos.copy(),
                quaternion_xyzw=quats[idx].copy(),
                width_m=width_m,
                score=score,
                force_closure_score=fc,
                antipodal_score=anti,
                collision_free=True,
                approach_vector=rot[:, 2].copy(),
                target_instance_id=target_instance_id,
                target_class_name=target_class_name,
            )
        )
    return candidates


class AntipodalGraspSampler:
    """Generates diverse 6DoF tabletop & antipodal grasp hypotheses for an ``Object3DDetection``.

    Combines two complementary grasp families:
      1. **Top-down OBB-aligned grasps** across graspable horizontal principal axes with yaw
         perturbations, lateral offsets along elongated axes, and slight approach tilts.
      2. **Angled & side antipodal contact-pair grasps** sampled from opposing surface point
         pairs whose separation fits ``[gripper_min_width_m, gripper_max_width_m]``.

    Each candidate is scored using:
      - **Antipodal score**: alignment of outward surface normals ``n1, n2`` with the closing
        axis ``d_hat = (p2 - p1) / ||p2 - p1||``.
      - **Force-closure score**: Coulomb friction cone condition (``tan(theta) <= mu``),
        center-of-mass torque-balance penalty, and gripper jaw clearance margin.
    """

    def __init__(
        self,
        gripper_max_width_m: float = 0.08,
        gripper_min_width_m: float = 0.012,
        finger_depth_m: float = 0.045,
        finger_tip_extension_m: float = 0.010,
        finger_width_x_m: float = 0.020,
        num_candidates: int = 64,
        friction_coeff: float = 0.5,
        table_z_margin_m: float = 0.008,
        jaw_clearance_m: float = 0.008,
        max_surface_points: int = 384,
        max_pair_points: int = 128,
        normal_max_nn: int = 20,
        seed: int | None = 42,
    ) -> None:
        self.gripper_max_width_m = float(gripper_max_width_m)
        self.gripper_min_width_m = float(gripper_min_width_m)
        self.finger_depth_m = float(finger_depth_m)
        self.finger_tip_extension_m = float(finger_tip_extension_m)
        self.finger_width_x_m = float(finger_width_x_m)
        self.num_candidates = max(1, int(num_candidates))
        self.friction_coeff = max(1e-3, float(friction_coeff))
        self.table_z_margin_m = float(table_z_margin_m)
        self.jaw_clearance_m = max(0.0, float(jaw_clearance_m))
        self.max_surface_points = max(32, int(max_surface_points))
        self.max_pair_points = max(16, int(max_pair_points))
        self.normal_max_nn = max(6, int(normal_max_nn))
        self.seed = seed

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> AntipodalGraspSampler:
        """Construct sampler from either full YAML config or the ``grasp`` sub-dict."""
        grasp_cfg = cfg.get("grasp", cfg)
        project_cfg = cfg.get("project", {})
        return cls(
            gripper_max_width_m=float(grasp_cfg.get("gripper_max_width_m", 0.08)),
            gripper_min_width_m=float(grasp_cfg.get("gripper_min_width_m", 0.012)),
            finger_depth_m=float(grasp_cfg.get("finger_depth_m", 0.045)),
            finger_tip_extension_m=float(grasp_cfg.get("finger_tip_extension_m", 0.010)),
            finger_width_x_m=float(grasp_cfg.get("finger_width_x_m", 0.020)),
            num_candidates=int(grasp_cfg.get("num_candidates", 64)),
            friction_coeff=float(grasp_cfg.get("friction_coeff", 0.5)),
            table_z_margin_m=float(grasp_cfg.get("table_z_margin_m", 0.008)),
            jaw_clearance_m=float(grasp_cfg.get("jaw_clearance_m", 0.008)),
            seed=project_cfg.get("seed", 42),
        )

    def estimate_outward_normals(
        self,
        points_xyz: np.ndarray,
        centroid: np.ndarray,
        obb_extent: np.ndarray,
    ) -> np.ndarray:
        """Estimate smooth outward-oriented unit surface normals via Open3D."""
        pts = np.asarray(points_xyz, dtype=np.float64)
        n_pts = pts.shape[0]
        if n_pts == 0:
            return np.zeros((0, 3), dtype=np.float64)

        outward = pts - centroid[None, :]
        outward_norm = np.linalg.norm(outward, axis=1, keepdims=True)
        outward_unit = np.divide(outward, np.maximum(outward_norm, 1e-9))

        if n_pts < 6:
            return outward_unit

        min_ext = float(np.min(np.maximum(obb_extent, 0.005)))
        radius = float(np.clip(0.35 * min_ext, 0.012, 0.045))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=min(self.normal_max_nn, n_pts - 1),
            )
        )
        normals = np.asarray(pcd.normals, dtype=np.float64).copy()

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        bad = (norms[:, 0] < 1e-8) | ~np.all(np.isfinite(normals), axis=1)
        normals[~bad] /= norms[~bad]
        normals[bad] = outward_unit[bad]

        dots = np.sum(normals * outward, axis=1)
        normals[dots < 0.0] *= -1.0
        return normals

    def score_contact_pairs_batch(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        n1: np.ndarray,
        n2: np.ndarray,
        centroid: np.ndarray,
        obb_extent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute vectorized antipodal and force-closure scores for ``(K, 3)`` contact pairs."""
        diff = p2 - p1
        dist = np.linalg.norm(diff, axis=1)
        safe_dist = np.maximum(dist, 1e-9)
        d_hat = diff / safe_dist[:, None]

        c1 = np.clip(-np.sum(n1 * d_hat, axis=1), 0.0, 1.0)
        c2 = np.clip(np.sum(n2 * d_hat, axis=1), 0.0, 1.0)
        antipodal = c1 * c2

        sin1 = np.sqrt(np.maximum(0.0, 1.0 - c1 * c1))
        sin2 = np.sqrt(np.maximum(0.0, 1.0 - c2 * c2))
        tan1 = sin1 / np.maximum(c1, 1e-6)
        tan2 = sin2 / np.maximum(c2, 1e-6)
        tan_max = np.maximum(tan1, tan2)

        mu = self.friction_coeff
        in_cone = (c1 > 1e-4) & (c2 > 1e-4) & (tan_max <= mu)
        cone_score = np.where(
            in_cone,
            1.0 - 0.45 * (tan_max / mu),
            0.55 * np.exp(-2.5 * np.maximum(0.0, tan_max - mu) / mu),
        )
        cone_score = np.where((c1 > 1e-4) & (c2 > 1e-4), cone_score, 0.0)

        midpoint = 0.5 * (p1 + p2)
        d_com = np.linalg.norm(midpoint - centroid[None, :], axis=1)
        char_radius = max(0.5 * float(np.linalg.norm(obb_extent)), 0.025)
        com_score = np.exp(-((d_com / char_radius) ** 2))

        clearance = np.maximum(0.0, self.gripper_max_width_m - dist)
        clearance_score = np.clip(
            clearance / max(0.25 * self.gripper_max_width_m, 1e-4),
            0.15,
            1.0,
        )

        force_closure = np.clip(
            cone_score * (0.65 + 0.25 * com_score + 0.10 * clearance_score),
            0.0,
            1.0,
        )
        return antipodal, force_closure

    def _subsample_points(
        self,
        points_xyz: np.ndarray,
        max_points: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return np.zeros((0, 3), dtype=np.float64)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] <= max_points:
            return pts
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        return pts[idx]

    def _evaluate_obb_grasp_contacts(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        pts: np.ndarray,
        normals: np.ndarray,
        centroid: np.ndarray,
        obb_extent: np.ndarray,
        fallback_span_m: float,
    ) -> tuple[np.ndarray, float, float, float] | None:
        """Refine TCP Y-center, jaw width, and contact scores from local finger-pad slice."""
        if pts.shape[0] < 4:
            span = float(fallback_span_m)
            if span > self.gripper_max_width_m:
                return None
            width_m = float(
                np.clip(
                    span + self.jaw_clearance_m,
                    self.gripper_min_width_m,
                    self.gripper_max_width_m,
                )
            )
            d_com = float(np.linalg.norm(pos - centroid))
            char_r = max(0.5 * float(np.linalg.norm(obb_extent)), 0.025)
            com_s = float(np.exp(-((d_com / char_r) ** 2)))
            clear_s = float(
                np.clip(
                    max(0.0, self.gripper_max_width_m - span)
                    / max(0.25 * self.gripper_max_width_m, 1e-4),
                    0.15,
                    1.0,
                )
            )
            fc = float(np.clip(0.75 * (0.65 + 0.25 * com_s + 0.10 * clear_s), 0.0, 1.0))
            return pos, width_m, 0.78, fc

        p_local = (pts - pos[None, :]) @ rot
        finger_z_min = self.finger_tip_extension_m - self.finger_depth_m
        finger_z_max = self.finger_tip_extension_m

        in_slice = (
            (np.abs(p_local[:, 0]) <= self.finger_width_x_m * 0.5 + 0.012)
            & (p_local[:, 2] >= finger_z_min - 0.008)
            & (p_local[:, 2] <= finger_z_max + 0.008)
        )
        slice_idx = np.nonzero(in_slice)[0]

        if slice_idx.size < 4:
            in_slice = np.abs(p_local[:, 0]) <= self.finger_width_x_m * 0.5 + 0.018
            slice_idx = np.nonzero(in_slice)[0]

        if slice_idx.size < 4:
            span = float(fallback_span_m)
            if span > self.gripper_max_width_m:
                return None
            width_m = float(
                np.clip(
                    span + self.jaw_clearance_m,
                    self.gripper_min_width_m,
                    self.gripper_max_width_m,
                )
            )
            return pos, width_m, 0.75, 0.70

        y_vals = p_local[slice_idx, 1]
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        span = y_max - y_min

        if span > self.gripper_max_width_m:
            return None
        if span < 0.004:
            span = min(float(fallback_span_m), self.gripper_max_width_m)

        y_center_shift = 0.5 * (y_min + y_max)
        refined_pos = pos + y_center_shift * rot[:, 1]

        band = max(0.004, 0.15 * span)
        left_ids = slice_idx[y_vals <= (y_min + band)]
        right_ids = slice_idx[y_vals >= (y_max - band)]

        p1 = pts[left_ids].mean(axis=0, keepdims=True)
        p2 = pts[right_ids].mean(axis=0, keepdims=True)
        n1_raw = normals[left_ids].mean(axis=0, keepdims=True)
        n2_raw = normals[right_ids].mean(axis=0, keepdims=True)

        y_axis = rot[:, 1]
        n1_norm = float(np.linalg.norm(n1_raw))
        n2_norm = float(np.linalg.norm(n2_raw))
        n1 = n1_raw / n1_norm if n1_norm > 1e-6 else (-y_axis)[None, :]
        n2 = n2_raw / n2_norm if n2_norm > 1e-6 else y_axis[None, :]

        anti_arr, fc_arr = self.score_contact_pairs_batch(
            p1, p2, n1, n2, centroid, obb_extent
        )
        anti = float(anti_arr[0])
        fc = float(fc_arr[0])

        # Partial top-surface-only view fallback when edge normals point nearly straight up
        if anti < 0.25 and abs(float(n1[0, 2])) > 0.75 and abs(float(n2[0, 2])) > 0.75:
            p1_wall = (refined_pos - 0.5 * span * y_axis)[None, :]
            p2_wall = (refined_pos + 0.5 * span * y_axis)[None, :]
            anti_w, fc_w = self.score_contact_pairs_batch(
                p1_wall, p2_wall, (-y_axis)[None, :], y_axis[None, :], centroid, obb_extent
            )
            anti = 0.75 * float(anti_w[0])
            fc = 0.75 * float(fc_w[0])

        width_m = float(
            np.clip(
                span + self.jaw_clearance_m,
                self.gripper_min_width_m,
                self.gripper_max_width_m,
            )
        )
        return refined_pos, width_m, anti, fc

    def _sample_top_down_obb_raw(
        self,
        detection: Object3DDetection,
        pts: np.ndarray,
        normals: np.ndarray,
        centroid: np.ndarray,
        budget: int,
        table_z: float | None,
    ) -> list[tuple[np.ndarray, np.ndarray, float, float, float, float]]:
        """Generate top-down and slightly tilted OBB-aligned raw grasp tuples."""
        if budget <= 0:
            return []

        rot_obb = _orthonormalize_rotation(detection.obb_rotation)
        extent = np.maximum(np.asarray(detection.obb_extent, dtype=np.float64).reshape(3), 1e-3)
        center = np.asarray(detection.obb_center, dtype=np.float64).reshape(3)

        v_idx = int(np.argmax(np.abs(rot_obb[2, :])))
        h_indices = [i for i in range(3) if i != v_idx]
        h_indices.sort(key=lambda idx: float(extent[idx]))

        if pts.shape[0] > 0:
            z_min = float(np.min(pts[:, 2]))
            z_max = float(np.max(pts[:, 2]))
        else:
            z_min = float(center[2] - 0.5 * extent[v_idx])
            z_max = float(center[2] + 0.5 * extent[v_idx])

        ref_table_z = float(table_z) if table_z is not None else z_min
        min_safe_tcp_z = ref_table_z + self.table_z_margin_m + self.finger_tip_extension_m + 0.002
        z_levels: list[float] = []
        z_primary = float(np.clip(max(center[2], min_safe_tcp_z), z_min, z_max + 0.004))
        z_levels.append(z_primary)
        if abs(center[2] - z_primary) > 0.004:
            z_levels.append(float(center[2]))
        elif (z_max - z_min) > 0.035:
            z_upper = float(z_min + 0.72 * (z_max - z_min))
            if abs(z_upper - z_primary) > 0.006:
                z_levels.append(z_upper)

        graspable_axes = [
            idx for idx in h_indices if float(extent[idx]) <= self.gripper_max_width_m + 0.006
        ]
        if not graspable_axes:
            graspable_axes = [h_indices[0]]

        yaw_degs = [0.0, -12.0, 12.0, -24.0, 24.0]
        tilt_degs = [0.0, -15.0, 15.0]
        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        raw_items: list[tuple[np.ndarray, np.ndarray, float, float, float, float]] = []
        for h_idx in graspable_axes:
            other_idx = h_indices[1] if h_idx == h_indices[0] else h_indices[0]
            ext_h = float(extent[h_idx])
            ext_other = float(extent[other_idx])

            axis_xy = rot_obb[:, h_idx].copy()
            axis_xy[2] = 0.0
            norm_xy = float(np.linalg.norm(axis_xy))
            if norm_xy < 1e-6:
                axis_xy = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                axis_xy /= norm_xy

            long_xy = np.array([-axis_xy[1], axis_xy[0], 0.0], dtype=np.float64)
            lat_offsets = [0.0, -0.22 * ext_other, 0.22 * ext_other] if ext_other > 0.05 else [0.0]

            for z_lvl in z_levels:
                for lat in lat_offsets:
                    base_pos = center.copy()
                    base_pos[2] = z_lvl
                    base_pos += lat * long_xy

                    for yaw_deg in yaw_degs:
                        rad = np.deg2rad(yaw_deg)
                        cy, sy = float(np.cos(rad)), float(np.sin(rad))
                        y_dir = np.array(
                            [
                                cy * axis_xy[0] - sy * axis_xy[1],
                                sy * axis_xy[0] + cy * axis_xy[1],
                                0.0,
                            ],
                            dtype=np.float64,
                        )
                        x_horiz = np.array([-y_dir[1], y_dir[0], 0.0], dtype=np.float64)
                        local_tilts = tilt_degs if (yaw_deg == 0.0 and lat == 0.0) else [0.0]

                        for tilt_deg in local_tilts:
                            trad = np.deg2rad(tilt_deg)
                            approach = float(np.cos(trad)) * down + float(np.sin(trad)) * x_horiz
                            rot = _build_tcp_rotation(y_dir, approach)
                            eval_res = self._evaluate_obb_grasp_contacts(
                                base_pos,
                                rot,
                                pts,
                                normals,
                                centroid,
                                extent,
                                fallback_span_m=ext_h,
                            )
                            if eval_res is None:
                                continue
                            ref_pos, width_m, anti, fc = eval_res
                            score = 0.55 * fc + 0.35 * anti + 0.10 * max(0.0, -float(rot[2, 2]))
                            raw_items.append(
                                (rot, ref_pos, width_m, float(np.clip(score, 0.0, 1.0)), fc, anti)
                            )

        raw_items.sort(key=lambda item: item[3], reverse=True)
        return raw_items[:budget]

    def _sample_antipodal_pair_raw(
        self,
        detection: Object3DDetection,
        pts: np.ndarray,
        normals: np.ndarray,
        centroid: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> list[tuple[np.ndarray, np.ndarray, float, float, float, float]]:
        """Generate angled and side grasps from opposing antipodal surface point pairs."""
        if budget <= 0 or pts.shape[0] < 4:
            return []

        n_pts = pts.shape[0]
        if n_pts > self.max_pair_points:
            sub_idx = rng.choice(n_pts, size=self.max_pair_points, replace=False)
            p_sub = pts[sub_idx]
            n_sub = normals[sub_idx]
        else:
            p_sub = pts
            n_sub = normals

        m = p_sub.shape[0]
        idx_i, idx_j = np.triu_indices(m, k=1)
        diff = p_sub[idx_j] - p_sub[idx_i]
        dists = np.linalg.norm(diff, axis=1)

        min_sep = max(0.005, self.gripper_min_width_m * 0.5)
        max_sep = max(min_sep + 1e-4, self.gripper_max_width_m - 0.001)
        valid_dist = (dists >= min_sep) & (dists <= max_sep)
        if not np.any(valid_dist):
            return []

        idx_i = idx_i[valid_dist]
        idx_j = idx_j[valid_dist]
        dists = dists[valid_dist]

        p1 = p_sub[idx_i]
        p2 = p_sub[idx_j]
        n1 = n_sub[idx_i]
        n2 = n_sub[idx_j]

        extent = np.maximum(np.asarray(detection.obb_extent, dtype=np.float64).reshape(3), 1e-3)
        anti_scores, fc_scores = self.score_contact_pairs_batch(
            p1, p2, n1, n2, centroid, extent
        )

        pair_quality = 0.60 * fc_scores + 0.40 * anti_scores
        good_mask = anti_scores >= 0.30
        keep_ids = np.nonzero(good_mask)[0] if np.any(good_mask) else np.arange(pair_quality.size)
        order = keep_ids[np.argsort(pair_quality[keep_ids])[::-1]]

        approach_angles_deg = [-30.0, 30.0, -55.0, 55.0, -80.0, 80.0, 0.0]
        max_pairs = max(6, int(np.ceil(budget / 3.0)))
        selected_pairs: list[int] = []
        selected_mids: list[np.ndarray] = []
        selected_axes: list[np.ndarray] = []

        for k in order[: min(order.size, 600)]:
            mid = 0.5 * (p1[k] + p2[k])
            axis = (p2[k] - p1[k]) / dists[k]
            too_close = False
            mx, my, mz = float(mid[0]), float(mid[1]), float(mid[2])
            ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
            for prev_mid, prev_axis in zip(selected_mids, selected_axes, strict=True):
                dx = mx - float(prev_mid[0])
                dy = my - float(prev_mid[1])
                dz = mz - float(prev_mid[2])
                if (
                    (dx * dx + dy * dy + dz * dz) < 1e-4
                    and abs(ax * float(prev_axis[0]) + ay * float(prev_axis[1]) + az * float(prev_axis[2])) > 0.94
                ):
                    too_close = True
                    break
            if not too_close:
                selected_pairs.append(int(k))
                selected_mids.append(mid)
                selected_axes.append(axis)
                if len(selected_pairs) >= max_pairs:
                    break

        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        raw_items: list[tuple[np.ndarray, np.ndarray, float, float, float, float]] = []

        for k in selected_pairs:
            mid = 0.5 * (p1[k] + p2[k])
            d_hat = (p2[k] - p1[k]) / dists[k]
            width_m = float(
                np.clip(
                    float(dists[k]) + self.jaw_clearance_m,
                    self.gripper_min_width_m,
                    self.gripper_max_width_m,
                )
            )
            anti = float(anti_scores[k])
            fc = float(fc_scores[k])

            u = down - float(np.dot(down, d_hat)) * d_hat
            u_norm = float(np.linalg.norm(u))
            if u_norm < 1e-4:
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                u = ref - float(np.dot(ref, d_hat)) * d_hat
                u /= float(np.linalg.norm(u))
            else:
                u /= u_norm
            vx = float(d_hat[1] * u[2] - d_hat[2] * u[1])
            vy = float(d_hat[2] * u[0] - d_hat[0] * u[2])
            vz = float(d_hat[0] * u[1] - d_hat[1] * u[0])
            inv_vn = 1.0 / math.sqrt(vx * vx + vy * vy + vz * vz)
            v = np.array([vx * inv_vn, vy * inv_vn, vz * inv_vn], dtype=np.float64)

            for ang_deg in approach_angles_deg:
                ang = np.deg2rad(ang_deg)
                approach = float(np.cos(ang)) * u + float(np.sin(ang)) * v
                if approach[2] > 0.15:
                    continue

                rot = _build_tcp_rotation(d_hat, approach)
                app_pref = 0.5 + 0.5 * float(np.clip(-rot[2, 2], 0.0, 1.0))
                score = 0.55 * fc + 0.35 * anti + 0.10 * app_pref
                raw_items.append(
                    (rot, mid, width_m, float(np.clip(score, 0.0, 1.0)), fc, anti)
                )

        raw_items.sort(key=lambda item: item[3], reverse=True)
        return raw_items[:budget]

    def sample(
        self,
        detection: Object3DDetection,
        num_candidates: int | None = None,
        table_z: float | None = None,
    ) -> list[GraspCandidate]:
        """Sample diverse 6DoF grasp hypotheses for a detected 3D object."""
        total_budget = self.num_candidates if num_candidates is None else max(1, int(num_candidates))
        rng = np.random.default_rng(self.seed)

        pts = self._subsample_points(detection.point_cloud_xyz, self.max_surface_points, rng)
        if pts.shape[0] > 0:
            centroid = (
                np.asarray(detection.centroid, dtype=np.float64).reshape(3)
                if np.all(np.isfinite(detection.centroid))
                else pts.mean(axis=0)
            )
        else:
            centroid = np.asarray(detection.obb_center, dtype=np.float64).reshape(3)

        extent = np.maximum(np.asarray(detection.obb_extent, dtype=np.float64).reshape(3), 1e-3)
        normals = self.estimate_outward_normals(pts, centroid, extent)

        top_down_budget = max(1, int(round(0.40 * total_budget)))
        pair_budget = max(0, total_budget - top_down_budget)

        top_down_raw = self._sample_top_down_obb_raw(
            detection=detection,
            pts=pts,
            normals=normals,
            centroid=centroid,
            budget=top_down_budget,
            table_z=table_z,
        )
        pair_raw = self._sample_antipodal_pair_raw(
            detection=detection,
            pts=pts,
            normals=normals,
            centroid=centroid,
            budget=total_budget - len(top_down_raw),
            rng=rng,
        )

        combined = top_down_raw + pair_raw
        if len(combined) < total_budget and len(top_down_raw) == top_down_budget and pair_budget > 0:
            extra_top = self._sample_top_down_obb_raw(
                detection=detection,
                pts=pts,
                normals=normals,
                centroid=centroid,
                budget=total_budget - len(pair_raw),
                table_z=table_z,
            )
            combined = extra_top + pair_raw

        return _materialize_candidates(
            combined[:total_budget],
            target_instance_id=detection.instance_id,
            target_class_name=detection.class_name,
        )
