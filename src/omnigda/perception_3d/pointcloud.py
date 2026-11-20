"""3D point cloud processing for partial-view robotic manipulation.

Implements:
- Confidence-weighted RGB-D backprojection with Open3D Statistical Outlier Removal (SOR)
  and optional voxel downsampling to eliminate flying silhouette depth pixels
- Gravity-aligned (Z-up) and general 3D PCA Oriented Bounding Box (OBB) estimation
- Multi-view point cloud fusion with conservative Open3D ICP refinement for NBV active perception
- Partial-view point cloud completeness reliability scoring in [0.2, 1.0]
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from omnigda.core.gda_math import backproject_depth
from omnigda.core.schemas import RGBDFrame


def confidence_weighted_backproject(
    frame: RGBDFrame,
    mask: np.ndarray,
    confidence_map: np.ndarray | None = None,
    conf_thresh: float = 0.45,
    depth_min: float = 0.15,
    depth_max: float = 2.0,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 1.5,
    voxel_size: float | None = 0.003,
    min_points: int = 10,
    return_colors: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Backproject masked RGB-D pixels into world-frame 3D points with confidence denoising.

    Filters pixels by ``mask & (confidence_map >= conf_thresh)`` before backprojection to remove
    ambiguous silhouette pixels where flying depth artifacts typically occur, then applies Open3D
    Statistical Outlier Removal (SOR) and optional voxel grid downsampling.

    Args:
        frame: Input RGB-D frame with intrinsics and camera_to_world extrinsics.
        mask: Boolean (H, W) foreground instance mask.
        confidence_map: Optional (H, W) float confidence map in [0, 1] from 2D segmentation.
        conf_thresh: Minimum pixel confidence required to retain a depth measurement.
        depth_min: Minimum valid depth in meters.
        depth_max: Maximum valid depth in meters.
        sor_nb_neighbors: Number of neighbors for Open3D Statistical Outlier Removal (0 disables).
        sor_std_ratio: Standard deviation multiplier threshold for SOR.
        voxel_size: Optional voxel downsampling size in meters (None or <= 0 disables).
        min_points: Minimum points required before accepting SOR filtering.
        return_colors: Whether to also return per-point RGB colors in [0, 1].

    Returns:
        Tuple ``(xyz, colors)`` where ``xyz`` is ``(N, 3)`` float32 in world frame and ``colors``
        is ``(N, 3)`` float32 in ``[0, 1]`` (or ``None`` if ``return_colors=False``).
    """
    eff_mask = np.asarray(mask, dtype=bool)
    if eff_mask.shape != frame.depth.shape:
        raise ValueError(
            f"Mask shape {eff_mask.shape} does not match depth shape {frame.depth.shape}"
        )

    if confidence_map is not None:
        conf = np.asarray(confidence_map, dtype=np.float32)
        if conf.shape != frame.depth.shape:
            raise ValueError(
                f"confidence_map shape {conf.shape} does not match depth shape {frame.depth.shape}"
            )
        eff_mask = eff_mask & (conf >= float(conf_thresh))

    intr = frame.intrinsics
    xyz, colors = backproject_depth(
        depth=frame.depth,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.cx,
        cy=intr.cy,
        camera_to_world=frame.camera_to_world,
        mask=eff_mask,
        rgb=frame.rgb if return_colors else None,
        depth_min=depth_min,
        depth_max=depth_max,
    )

    if xyz.shape[0] == 0:
        return xyz, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    has_colors = colors is not None and colors.shape[0] == xyz.shape[0]
    if has_colors:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # Mild pre-downsampling only for very dense clouds to keep SOR fast
    if len(pcd.points) > 25000 and voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size) * 0.75)

    # Open3D Statistical Outlier Removal
    if sor_nb_neighbors > 0 and len(pcd.points) >= max(sor_nb_neighbors + 1, 5):
        pcd_clean, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(sor_nb_neighbors),
            std_ratio=float(sor_std_ratio),
        )
        if len(pcd_clean.points) >= min_points or (
            len(pcd_clean.points) > 0 and len(pcd.points) < min_points
        ):
            pcd = pcd_clean

    # Optional voxel downsampling
    if voxel_size is not None and voxel_size > 0 and len(pcd.points) > 0:
        pcd_ds = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        if len(pcd_ds.points) > 0:
            pcd = pcd_ds

    out_xyz = np.asarray(pcd.points, dtype=np.float32)
    out_rgb: np.ndarray | None = None
    if return_colors:
        if pcd.has_colors():
            out_rgb = np.asarray(pcd.colors, dtype=np.float32)
        else:
            out_rgb = np.zeros((out_xyz.shape[0], 3), dtype=np.float32)

    return out_xyz, out_rgb


def compute_oriented_bounding_box(
    xyz: np.ndarray,
    gravity_aligned: bool = True,
    min_extent: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a physically consistent Oriented Bounding Box (OBB) for a 3D point cloud.

    In gravity-aligned mode (default for tabletop manipulation), the local Z axis is locked to
    world +Z and horizontal axes are determined by 2D XY PCA with canonical ordering ``dx >= dy``,
    ensuring ``obb_extent = (dx, dy, dz)`` has a stable vertical height ``dz`` even under partial
    single-view occlusion.

    Args:
        xyz: ``(N, 3)`` array of 3D points in world frame.
        gravity_aligned: If True, align OBB Z-axis with world +Z (gravity direction).
        min_extent: Minimum extent floor in meters to prevent degenerate zero-width boxes.

    Returns:
        ``(obb_center, obb_extent, obb_rotation)`` as float64 arrays of shapes
        ``(3,)``, ``(3,)``, and ``(3, 3)`` respectively, with ``det(obb_rotation) == +1``.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) point cloud, got shape {pts.shape}")

    if pts.shape[0] == 0:
        return (
            np.zeros(3, dtype=np.float64),
            np.full(3, float(min_extent), dtype=np.float64),
            np.eye(3, dtype=np.float64),
        )

    if pts.shape[0] < 4:
        lo = np.min(pts, axis=0)
        hi = np.max(pts, axis=0)
        center = 0.5 * (lo + hi)
        extent = np.maximum(hi - lo, float(min_extent))
        return center.astype(np.float64), extent.astype(np.float64), np.eye(3, dtype=np.float64)

    if gravity_aligned:
        xy = pts[:, :2]
        xy_centered = xy - np.mean(xy, axis=0, keepdims=True)
        cov_xy = (xy_centered.T @ xy_centered) / float(pts.shape[0])
        cov_xy = 0.5 * (cov_xy + cov_xy.T)
        evals, evecs = np.linalg.eigh(cov_xy)
        # Principal horizontal direction (largest variance)
        major_2d = evecs[:, int(np.argmax(evals))]
        norm = float(np.linalg.norm(major_2d))
        if norm < 1e-12:
            major_2d = np.array([1.0, 0.0], dtype=np.float64)
        else:
            major_2d = major_2d / norm

        # Canonical sign to prevent arbitrary 180-degree flips
        if major_2d[0] < 0.0 or (abs(major_2d[0]) < 1e-8 and major_2d[1] < 0.0):
            major_2d = -major_2d

        x_axis = np.array([major_2d[0], major_2d[1], 0.0], dtype=np.float64)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_axis)
        rot = np.column_stack([x_axis, y_axis, z_axis])

        local = pts @ rot
        lo = np.min(local, axis=0)
        hi = np.max(local, axis=0)
        extent = np.maximum(hi - lo, float(min_extent))

        # Ensure canonical horizontal ordering dx >= dy while preserving right-handedness & +Z
        if extent[1] > extent[0]:
            x_new = rot[:, 1].copy()
            if x_new[0] < 0.0 or (abs(x_new[0]) < 1e-8 and x_new[1] < 0.0):
                x_new = -x_new
            y_new = np.cross(z_axis, x_new)
            rot = np.column_stack([x_new, y_new, z_axis])
            local = pts @ rot
            lo = np.min(local, axis=0)
            hi = np.max(local, axis=0)
            extent = np.maximum(hi - lo, float(min_extent))

        local_center = 0.5 * (lo + hi)
        obb_center = rot @ local_center
        return obb_center.astype(np.float64), extent.astype(np.float64), rot.astype(np.float64)

    # General 3D PCA OBB
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = (centered.T @ centered) / float(pts.shape[0])
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    rot = evecs[:, order]
    if np.linalg.det(rot) < 0.0:
        rot[:, 2] = -rot[:, 2]

    local = pts @ rot
    lo = np.min(local, axis=0)
    hi = np.max(local, axis=0)
    extent = np.maximum(hi - lo, float(min_extent))
    obb_center = rot @ (0.5 * (lo + hi))
    return obb_center.astype(np.float64), extent.astype(np.float64), rot.astype(np.float64)


def fuse_multiview_pointclouds(
    clouds: list[np.ndarray],
    voxel_size: float = 0.004,
    refine_icp: bool = True,
    icp_max_correspondence_dist: float = 0.015,
    min_icp_fitness: float = 0.35,
    max_icp_translation_m: float = 0.05,
    max_icp_rotation_deg: float = 15.0,
    sor_nb_neighbors: int = 16,
    sor_std_ratio: float = 2.0,
) -> np.ndarray:
    """Fuse multiple world-frame viewpoint point clouds into a unified 3D surface.

    Designed for Next-Best-View (NBV) active perception: merges partial clouds across camera
    viewpoints, optionally refining slight calibration/pose drift via conservative Open3D
    point-to-point ICP whenever sufficient spatial overlap exists.

    Args:
        clouds: List of ``(N_i, 3)`` world-frame point clouds.
        voxel_size: Output voxel downsampling resolution in meters.
        refine_icp: Whether to attempt Open3D ICP alignment on overlapping views.
        icp_max_correspondence_dist: Max correspondence radius in meters for ICP.
        min_icp_fitness: Minimum overlap fitness ratio to accept an ICP refinement.
        max_icp_translation_m: Maximum accepted ICP translation drift in meters.
        max_icp_rotation_deg: Maximum accepted ICP rotation correction in degrees.
        sor_nb_neighbors: Neighbors for light post-fusion SOR cleanup (0 disables).
        sor_std_ratio: Std ratio for post-fusion SOR cleanup.

    Returns:
        Fused ``(M, 3)`` float32 point cloud in world frame.
    """
    valid_clouds = [
        np.asarray(c, dtype=np.float64)
        for c in clouds
        if c is not None and np.asarray(c).ndim == 2 and np.asarray(c).shape[0] > 0
    ]
    if not valid_clouds:
        return np.zeros((0, 3), dtype=np.float32)

    fused = o3d.geometry.PointCloud()
    fused.points = o3d.utility.Vector3dVector(valid_clouds[0])
    if voxel_size > 0 and len(fused.points) > 0:
        fused = fused.voxel_down_sample(voxel_size=float(voxel_size))

    max_rot_rad = np.deg2rad(float(max_icp_rotation_deg))
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25)

    for src_arr in valid_clouds[1:]:
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_arr)
        if voxel_size > 0 and len(src_pcd.points) > 0:
            src_pcd = src_pcd.voxel_down_sample(voxel_size=float(voxel_size))

        if refine_icp and len(src_pcd.points) >= 30 and len(fused.points) >= 30:
            reg = o3d.pipelines.registration.registration_icp(
                source=src_pcd,
                target=fused,
                max_correspondence_distance=float(icp_max_correspondence_dist),
                init=np.eye(4, dtype=np.float64),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=criteria,
            )
            tf = np.asarray(reg.transformation, dtype=np.float64)
            trans_norm = float(np.linalg.norm(tf[:3, 3]))
            cos_angle = float(np.clip((np.trace(tf[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
            rot_angle = float(np.arccos(cos_angle))
            if (
                reg.fitness >= min_icp_fitness
                and trans_norm <= max_icp_translation_m
                and rot_angle <= max_rot_rad
            ):
                src_pcd.transform(tf)

        fused += src_pcd
        if voxel_size > 0 and len(fused.points) > 0:
            fused = fused.voxel_down_sample(voxel_size=float(voxel_size))

    if sor_nb_neighbors > 0 and len(fused.points) >= max(sor_nb_neighbors + 1, 20):
        cleaned, _ = fused.remove_statistical_outlier(
            nb_neighbors=int(sor_nb_neighbors),
            std_ratio=float(sor_std_ratio),
        )
        if len(cleaned.points) > 0:
            fused = cleaned

    return np.asarray(fused.points, dtype=np.float32)


def estimate_pointcloud_completeness(
    xyz: np.ndarray,
    obb_extent: np.ndarray,
    min_score: float = 0.2,
    max_score: float = 1.0,
) -> float:
    """Estimate partial-view 3D point cloud completeness reliability in ``[0.2, 1.0]``.

    Combines four geometric cues that distinguish clean multi-view / full object clouds from
    sparse, occluded, or single-sliver partial views:
    1. Absolute point count saturation.
    2. Surface point density relative to the OBB surface area.
    3. 3D octant occupancy coverage inside the bounding volume.
    4. Azimuthal angular coverage around the object's vertical centroid axis.

    Args:
        xyz: ``(N, 3)`` point cloud array.
        obb_extent: ``(3,)`` oriented bounding box extents ``(dx, dy, dz)`` in meters.
        min_score: Lower bound of returned reliability score (default ``0.2``).
        max_score: Upper bound of returned reliability score (default ``1.0``).

    Returns:
        Reliability score in ``[min_score, max_score]``.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    n_pts = pts.shape[0] if pts.ndim == 2 else 0
    if n_pts < 10:
        return float(min_score)

    ext = np.maximum(np.asarray(obb_extent, dtype=np.float64), 1e-3)
    dx, dy, dz = float(ext[0]), float(ext[1]), float(ext[2])

    # 1. Point count saturation
    count_score = min(float(n_pts) / 250.0, 1.0)

    # 2. Surface density (points per m^2 of OBB surface area)
    surface_area = 2.0 * (dx * dy + dy * dz + dz * dx)
    density = float(n_pts) / max(surface_area, 1e-5)
    density_score = float(1.0 - np.exp(-density / 15000.0))

    # 3. 3D octant coverage with per-octant support saturation (prevents sparse noise saturation)
    centroid = np.mean(pts, axis=0)
    rel = pts - centroid[None, :]
    octant_bits = (
        (rel[:, 0] >= 0.0).astype(np.int32) * 4
        + (rel[:, 1] >= 0.0).astype(np.int32) * 2
        + (rel[:, 2] >= 0.0).astype(np.int32)
    )
    octant_counts = np.bincount(octant_bits, minlength=8).astype(np.float64)
    octant_cov = float(np.mean(np.minimum(octant_counts / 8.0, 1.0)))

    # 4. Azimuthal angular coverage around vertical Z axis (8 sectors of 45 deg)
    r_xy = np.hypot(rel[:, 0], rel[:, 1])
    radial_mask = r_xy > max(0.05 * max(dx, dy), 1e-4)
    if np.any(radial_mask):
        angles = np.arctan2(rel[radial_mask, 1], rel[radial_mask, 0])
        sectors = np.floor(((angles + np.pi) / (2.0 * np.pi)) * 8.0).astype(np.int32)
        sectors = np.clip(sectors, 0, 7)
        sector_counts = np.bincount(sectors, minlength=8).astype(np.float64)
        azimuth_cov = float(np.mean(np.minimum(sector_counts / 8.0, 1.0)))
    else:
        azimuth_cov = 0.0

    raw = 0.35 * count_score + 0.25 * density_score + 0.20 * octant_cov + 0.20 * azimuth_cov
    score = min_score + (max_score - min_score) * float(np.clip(raw, 0.0, 1.0))
    return float(np.clip(score, min_score, max_score))


def subsample_pointcloud(
    xyz: np.ndarray,
    num_samples: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Deterministically subsample a point cloud to at most ``num_samples`` points."""
    pts = np.asarray(xyz, dtype=np.float32)
    n = pts.shape[0]
    if n <= num_samples or num_samples <= 0:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=num_samples, replace=False)
    return pts[idx]
