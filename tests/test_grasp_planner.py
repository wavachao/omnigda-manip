"""Unit, geometry, collision, and latency tests for the OmniGDA grasp planner."""

from __future__ import annotations

import time

import numpy as np
import pytest

from omnigda.core.config import load_config
from omnigda.core.gda_math import quat_xyzw_to_rotation_matrix
from omnigda.core.schemas import GraspCandidate, Object3DDetection
from omnigda.grasp import (
    AntipodalGraspSampler,
    ConfidenceAwareGraspPlanner,
    FrankaGripperCollisionChecker,
    GraspPlanOutput,
    build_tcp_pose,
)
from omnigda.grasp.planner import _symmetric_so3_distance_rad


def _make_box_detection(
    center: tuple[float, float, float],
    extent: tuple[float, float, float],
    rotation: np.ndarray | None = None,
    instance_id: int = 1,
    class_name: str = "box",
    confidence: float = 0.95,
    num_points: int = 600,
    seed: int = 0,
) -> Object3DDetection:
    rng = np.random.default_rng(seed)
    dx, dy, dz = extent
    rot = np.eye(3, dtype=np.float64) if rotation is None else np.asarray(rotation, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)

    face_pts: list[np.ndarray] = []
    per_face = max(20, num_points // 6)
    for axis in range(3):
        for sign in (-0.5, 0.5):
            local = rng.uniform(-0.5, 0.5, size=(per_face, 3))
            local[:, axis] = sign
            local[:, 0] *= dx
            local[:, 1] *= dy
            local[:, 2] *= dz
            world = local @ rot.T + c[None, :]
            face_pts.append(world)

    pts = np.vstack(face_pts).astype(np.float32)
    return Object3DDetection(
        instance_id=instance_id,
        class_name=class_name,
        confidence=confidence,
        entropy=0.1,
        ood_score=0.0,
        is_ood=False,
        centroid=c.copy(),
        obb_center=c.copy(),
        obb_extent=np.asarray(extent, dtype=np.float64),
        obb_rotation=rot.copy(),
        point_cloud_xyz=pts,
    )


def _make_cylinder_detection(
    center: tuple[float, float, float],
    radius: float,
    height: float,
    instance_id: int = 2,
    class_name: str = "bottle",
    confidence: float = 0.92,
    num_points: int = 600,
    seed: int = 1,
) -> Object3DDetection:
    rng = np.random.default_rng(seed)
    cx, cy, cz = center
    n_side = int(num_points * 0.8)
    n_cap = num_points - n_side

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_side)
    z_side = rng.uniform(-0.5 * height, 0.5 * height, size=n_side)
    side_pts = np.column_stack(
        [cx + radius * np.cos(theta), cy + radius * np.sin(theta), cz + z_side]
    )

    # Top cap points
    r_cap = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_cap))
    t_cap = rng.uniform(0.0, 2.0 * np.pi, size=n_cap)
    cap_pts = np.column_stack(
        [cx + r_cap * np.cos(t_cap), cy + r_cap * np.sin(t_cap), np.full(n_cap, cz + 0.5 * height)]
    )

    pts = np.vstack([side_pts, cap_pts]).astype(np.float32)
    c = np.asarray(center, dtype=np.float64)
    return Object3DDetection(
        instance_id=instance_id,
        class_name=class_name,
        confidence=confidence,
        entropy=0.12,
        ood_score=0.0,
        is_ood=False,
        centroid=c.copy(),
        obb_center=c.copy(),
        obb_extent=np.array([2.0 * radius, 2.0 * radius, height], dtype=np.float64),
        obb_rotation=np.eye(3, dtype=np.float64),
        point_cloud_xyz=pts,
    )


def _assert_valid_candidate_geometry(
    cand: GraspCandidate,
    min_width_m: float = 0.012,
    max_width_m: float = 0.080,
) -> None:
    assert cand.pose_4x4.shape == (4, 4)
    rot = cand.pose_4x4[:3, :3]
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-6)
    np.testing.assert_allclose(cand.pose_4x4[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-9)

    np.testing.assert_allclose(cand.position, cand.pose_4x4[:3, 3], atol=1e-9)
    assert cand.quaternion_xyzw.shape == (4,)
    assert np.isclose(np.linalg.norm(cand.quaternion_xyzw), 1.0, atol=1e-6)
    rot_from_quat = quat_xyzw_to_rotation_matrix(cand.quaternion_xyzw)
    np.testing.assert_allclose(rot_from_quat, rot, atol=1e-6)

    assert cand.approach_vector.shape == (3,)
    assert np.isclose(np.linalg.norm(cand.approach_vector), 1.0, atol=1e-6)
    np.testing.assert_allclose(cand.approach_vector, rot[:, 2], atol=1e-6)

    assert min_width_m - 1e-9 <= cand.width_m <= max_width_m + 1e-9
    assert 0.0 <= cand.antipodal_score <= 1.0
    assert 0.0 <= cand.force_closure_score <= 1.0
    assert 0.0 <= cand.score <= 1.0


def test_grasp_generation_box_cylinder_thin_bar() -> None:
    """Planner generates valid top-down and angled/side grasps on box, bottle, and thin bar."""
    cfg = load_config()
    planner = ConfidenceAwareGraspPlanner.from_config(cfg)
    min_w = float(cfg["grasp"]["gripper_min_width_m"])
    max_w = float(cfg["grasp"]["gripper_max_width_m"])

    # 1. Standard tabletop box
    box = _make_box_detection((0.50, 0.00, 0.03), (0.05, 0.06, 0.06), instance_id=1, class_name="box")
    out_box = planner.plan(box, table_z=0.0, num_candidates=64, top_k=12)
    assert isinstance(out_box, GraspPlanOutput)
    assert not out_box.needs_nbv
    assert len(out_box.candidates) >= 3
    scores = [c.score for c in out_box.candidates]
    assert scores == sorted(scores, reverse=True)
    for c in out_box.candidates:
        _assert_valid_candidate_geometry(c, min_w, max_w)
        assert c.collision_free
        assert c.target_instance_id == 1
        assert c.target_class_name == "box"
    # Both top-down and angled/side grasps present
    assert any(-c.approach_vector[2] > 0.95 for c in out_box.candidates)
    assert any(-c.approach_vector[2] < 0.92 for c in out_box.candidates)

    # 2. Tall cylinder / bottle
    bottle = _make_cylinder_detection((0.50, 0.12, 0.07), radius=0.028, height=0.14)
    out_bottle = planner.plan(bottle, table_z=0.0, num_candidates=64, top_k=12)
    assert not out_bottle.needs_nbv
    assert len(out_bottle.candidates) >= 4
    for c in out_bottle.candidates:
        _assert_valid_candidate_geometry(c, min_w, max_w)
        assert c.collision_free
    assert any(-c.approach_vector[2] > 0.95 for c in out_bottle.candidates)
    assert any(-c.approach_vector[2] < 0.92 for c in out_bottle.candidates)

    # 3. Long thin bar (0.16m > max jaw width along X, 0.024m graspable along Y)
    bar = _make_box_detection(
        (0.50, -0.10, 0.012),
        (0.16, 0.024, 0.024),
        instance_id=3,
        class_name="thin_bar",
    )
    out_bar = planner.plan(bar, table_z=0.0, num_candidates=64, top_k=10)
    assert not out_bar.needs_nbv
    assert len(out_bar.candidates) >= 1
    for c in out_bar.candidates:
        _assert_valid_candidate_geometry(c, min_w, max_w)
        assert c.collision_free
        # Jaw closing axis (+Y_tcp) must align mostly across the narrow Y dimension, not long X
        jaw_y = c.pose_4x4[:3, 1]
        assert abs(jaw_y[1]) > abs(jaw_y[0])


def test_rotated_obb_box_grasp_alignment() -> None:
    """Planner handles rotated OBBs and produces valid orthonormal grasps."""
    theta = np.deg2rad(35.0)
    rot_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    det = _make_box_detection(
        center=(0.45, 0.05, 0.03),
        extent=(0.14, 0.036, 0.06),
        rotation=rot_z,
        instance_id=4,
        class_name="rotated_bar",
    )
    planner = ConfidenceAwareGraspPlanner()
    out = planner.plan(det, table_z=0.0, num_candidates=64)
    assert not out.needs_nbv
    assert len(out.candidates) >= 2
    for c in out.candidates:
        _assert_valid_candidate_geometry(c)
        assert c.collision_free


def test_antipodal_and_friction_cone_scoring_physics() -> None:
    """Force-closure score obeys Coulomb friction cone and CoM torque-balance physics."""
    sampler = AntipodalGraspSampler(friction_coeff=0.5, gripper_max_width_m=0.08)
    centroid = np.array([0.0, 0.0, 0.03], dtype=np.float64)
    extent = np.array([0.05, 0.05, 0.06], dtype=np.float64)

    # Pair 0: Perfect antipodal at CoM (theta = 0 deg)
    # Pair 1: Inside friction cone (theta = 15 deg, tan(15)=0.268 < 0.5) at CoM
    # Pair 2: Outside friction cone (theta = 45 deg, tan(45)=1.0 > 0.5) at CoM
    # Pair 3: Perfect antipodal but displaced far from CoM
    p1 = np.array(
        [
            [0.0, -0.02, 0.03],
            [0.0, -0.02, 0.03],
            [0.0, -0.02, 0.03],
            [0.06, -0.02, 0.03],
        ],
        dtype=np.float64,
    )
    p2 = np.array(
        [
            [0.0, 0.02, 0.03],
            [0.0, 0.02, 0.03],
            [0.0, 0.02, 0.03],
            [0.06, 0.02, 0.03],
        ],
        dtype=np.float64,
    )
    a15 = np.deg2rad(15.0)
    a45 = np.deg2rad(45.0)
    n1 = np.array(
        [
            [0.0, -1.0, 0.0],
            [np.sin(a15), -np.cos(a15), 0.0],
            [np.sin(a45), -np.cos(a45), 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    n2 = np.array(
        [
            [0.0, 1.0, 0.0],
            [np.sin(a15), np.cos(a15), 0.0],
            [np.sin(a45), np.cos(a45), 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    anti, fc = sampler.score_contact_pairs_batch(p1, p2, n1, n2, centroid, extent)

    assert np.isclose(anti[0], 1.0, atol=1e-6)
    assert fc[0] > 0.90
    assert fc[0] > fc[1] > fc[2]
    assert fc[1] > 0.65  # inside friction cone
    assert fc[2] < 0.35  # outside friction cone
    assert fc[0] > fc[3]  # CoM grasp beats off-center grasp


def test_collision_checker_rejects_table_plane_collision() -> None:
    """Collision checker rejects grasps whose fingertips or body violate table_z + margin."""
    checker = FrankaGripperCollisionChecker(table_z_margin_m=0.008)
    g = checker.geometry
    table_z = 0.0

    # Top-down grasp: fingertips sit at tcp_z - finger_tip_extension_m
    # Low TCP: fingertip world Z = 0.012 - 0.010 = 0.002 < 0.008 -> MUST REJECT
    pose_low, pos_low, quat_low, app_low = build_tcp_pose(
        position=np.array([0.5, 0.0, 0.012]),
        closing_axis_y=np.array([0.0, 1.0, 0.0]),
        approach_axis_z=np.array([0.0, 0.0, -1.0]),
    )
    # High TCP: fingertip world Z = 0.030 - 0.010 = 0.020 > 0.008 -> MUST ACCEPT
    pose_high, pos_high, quat_high, app_high = build_tcp_pose(
        position=np.array([0.5, 0.0, 0.030]),
        closing_axis_y=np.array([0.0, 1.0, 0.0]),
        approach_axis_z=np.array([0.0, 0.0, -1.0]),
    )
    # Horizontal side grasp at low Z where palm half-width (0.10m) dips below table -> MUST REJECT
    pose_side_low, pos_sl, quat_sl, app_sl = build_tcp_pose(
        position=np.array([0.5, 0.0, 0.035]),
        closing_axis_y=np.array([0.0, 0.0, 1.0]),  # fingers stacked vertically
        approach_axis_z=np.array([1.0, 0.0, 0.0]),
    )

    cands = [
        GraspCandidate(pose_low, pos_low, quat_low, 0.05, 0.9, 0.9, 0.9, True, app_low),
        GraspCandidate(pose_high, pos_high, quat_high, 0.05, 0.9, 0.9, 0.9, True, app_high),
        GraspCandidate(pose_side_low, pos_sl, quat_sl, 0.05, 0.9, 0.9, 0.9, True, app_sl),
    ]
    res = checker.check_batch_detailed(cands, obstacle_xyz=None, table_z=table_z)

    assert not res.collision_free[0]
    assert res.table_collision[0]
    assert res.min_world_z[0] <= table_z + checker.table_z_margin_m

    assert res.collision_free[1]
    assert not res.table_collision[1]
    assert np.isclose(res.min_world_z[1], 0.030 - g.finger_tip_extension_m, atol=1e-6)

    assert not res.collision_free[2]
    assert res.table_collision[2]


def test_collision_checker_rejects_adjacent_obstacle_components() -> None:
    """Collision checker rejects obstacles hitting left finger, right finger, palm, or wrist."""
    checker = FrankaGripperCollisionChecker(table_z_margin_m=0.008)
    width_m = 0.04
    pose, pos, quat, app = build_tcp_pose(
        position=np.array([0.5, 0.0, 0.05]),
        closing_axis_y=np.array([0.0, 1.0, 0.0]),
        approach_axis_z=np.array([0.0, 0.0, -1.0]),
    )
    cand = GraspCandidate(pose, pos, quat, width_m, 0.9, 0.9, 0.9, True, app)
    rot = pose[:3, :3]

    def local_to_world(local_pt: list[float]) -> np.ndarray:
        p = np.asarray(local_pt, dtype=np.float64)
        return (rot @ p + pos).reshape(1, 3).astype(np.float32)

    # 1. Left finger (-Y) hit: y = -(width/2 + finger_thickness/2)
    obs_left = local_to_world([0.0, -(width_m * 0.5 + 0.008), -0.01])
    assert not checker.check_single(cand, obstacle_xyz=obs_left, table_z=0.0)

    # 2. Right finger (+Y) hit: y = +(width/2 + finger_thickness/2)
    obs_right = local_to_world([0.0, +(width_m * 0.5 + 0.008), -0.01])
    assert not checker.check_single(cand, obstacle_xyz=obs_right, table_z=0.0)

    # 3. Palm hit: behind fingers at z_local = -0.06
    obs_palm = local_to_world([0.0, 0.07, -0.06])
    assert not checker.check_single(cand, obstacle_xyz=obs_palm, table_z=0.0)

    # 4. Wrist approach shaft hit: z_local = -0.14
    obs_wrist = local_to_world([0.01, 0.01, -0.14])
    assert not checker.check_single(cand, obstacle_xyz=obs_wrist, table_z=0.0)

    # 5. Point strictly inside jaw cavity (|y| < width/2) -> does NOT trigger self-collision
    obs_inside_gap = local_to_world([0.0, 0.0, 0.0])
    assert checker.check_single(cand, obstacle_xyz=obs_inside_gap, table_z=0.0)

    # 6. Far obstacle -> collision-free
    obs_far = np.array([[0.9, 0.5, 0.1]], dtype=np.float32)
    assert checker.check_single(cand, obstacle_xyz=obs_far, table_z=0.0)


def test_needs_nbv_when_object_completely_surrounded_by_obstacles() -> None:
    """Planner sets needs_nbv=True when surrounding obstacles block all grasp approaches."""
    planner = ConfidenceAwareGraspPlanner.from_config(load_config())
    center = (0.50, 0.00, 0.03)
    target = _make_box_detection(center, (0.04, 0.04, 0.06), instance_id=5, class_name="blocked_box")

    # Build a dense obstacle enclosure tightly surrounding all 4 sides and directly overhead
    rng = np.random.default_rng(123)
    cx, cy, cz = center
    obs_blocks: list[np.ndarray] = []

    # 4 surrounding walls placed just outside the target box (blocking fingers & palm)
    for x_lo, x_hi in ((cx - 0.055, cx - 0.025), (cx + 0.025, cx + 0.055)):
        wall = np.column_stack(
            [
                rng.uniform(x_lo, x_hi, size=250),
                rng.uniform(cy - 0.08, cy + 0.08, size=250),
                rng.uniform(0.005, 0.16, size=250),
            ]
        )
        obs_blocks.append(wall)
    for y_lo, y_hi in ((cy - 0.055, cy - 0.025), (cy + 0.025, cy + 0.055)):
        wall = np.column_stack(
            [
                rng.uniform(cx - 0.08, cx + 0.08, size=250),
                rng.uniform(y_lo, y_hi, size=250),
                rng.uniform(0.005, 0.16, size=250),
            ]
        )
        obs_blocks.append(wall)
    # Overhead lid blocking top-down palm and wrist approach shaft
    roof = np.column_stack(
        [
            rng.uniform(cx - 0.09, cx + 0.09, size=300),
            rng.uniform(cy - 0.09, cy + 0.09, size=300),
            rng.uniform(0.065, 0.16, size=300),
        ]
    )
    obs_blocks.append(roof)
    obstacles = np.vstack(obs_blocks).astype(np.float32)

    out = planner.plan(target, obstacle_xyz=obstacles, table_z=0.0, num_candidates=64)
    assert out.needs_nbv is True
    assert out.num_collision_free == 0
    assert len(out.candidates) == 0

    # With include_colliding=True, diagnostic candidates are returned with collision_free=False
    out_diag = planner.plan(
        target, obstacle_xyz=obstacles, table_z=0.0, num_candidates=64, include_colliding=True
    )
    assert out_diag.needs_nbv is True
    assert len(out_diag.candidates) > 0
    assert all(not c.collision_free for c in out_diag.candidates)


def test_nms_enforces_6dof_diversity() -> None:
    """Returned grasps are diverse in 6DoF space according to NMS thresholds."""
    planner = ConfidenceAwareGraspPlanner(
        nms_translation_thresh_m=0.015,
        nms_rotation_thresh_deg=20.0,
    )
    box = _make_box_detection((0.50, 0.00, 0.03), (0.05, 0.06, 0.06))
    out = planner.plan(box, table_z=0.0, num_candidates=64, top_k=10)
    assert len(out.candidates) >= 3

    for i in range(len(out.candidates)):
        for j in range(i + 1, len(out.candidates)):
            ci = out.candidates[i]
            cj = out.candidates[j]
            d_pos = float(np.linalg.norm(ci.position - cj.position))
            d_ang = _symmetric_so3_distance_rad(ci.pose_4x4[:3, :3], cj.pose_4x4[:3, :3])
            assert (d_pos >= planner.nms_translation_thresh_m - 1e-9) or (
                d_ang >= planner.nms_rotation_thresh_rad - 1e-9
            )


def test_latency_budgets_collision_and_full_planner() -> None:
    """Collision batch check < 5ms and full 64-candidate planning < 30ms."""
    cfg = load_config()
    planner = ConfidenceAwareGraspPlanner.from_config(cfg)
    checker = planner.collision_checker

    box = _make_box_detection((0.50, 0.00, 0.03), (0.05, 0.06, 0.06), num_points=800)
    rng = np.random.default_rng(99)
    # Cluttered tabletop scene obstacle cloud (1500 points outside a 0.14m XY corridor)
    obs = rng.uniform([0.2, -0.35, 0.0], [0.8, 0.35, 0.14], size=(1800, 3)).astype(np.float32)
    xy_dist = np.hypot(obs[:, 0] - box.centroid[0], obs[:, 1] - box.centroid[1])
    obs = obs[xy_dist > 0.14][:1500]

    # Warmup (Open3D KDTree + NumPy BLAS)
    _ = planner.plan(box, obstacle_xyz=obs, table_z=0.0, num_candidates=64)

    # 1. Benchmark collision checker alone on 64 candidates
    raw_cands = planner.sampler.sample(box, num_candidates=64, table_z=0.0)
    assert len(raw_cands) == 64
    coll_times_ms: list[float] = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = checker.check_batch(raw_cands, obstacle_xyz=obs, table_z=0.0)
        coll_times_ms.append((time.perf_counter() - t0) * 1000.0)
    median_coll_ms = float(np.median(coll_times_ms))
    assert median_coll_ms < 5.0, f"Collision check too slow: {median_coll_ms:.2f} ms"

    # 2. Benchmark full end-to-end grasp planning on 64 candidates
    plan_times_ms: list[float] = []
    for _ in range(5):
        out = planner.plan(box, obstacle_xyz=obs, table_z=0.0, num_candidates=64)
        assert not out.needs_nbv
        plan_times_ms.append(out.planning_latency_ms)
    median_plan_ms = float(np.median(plan_times_ms))
    assert median_plan_ms < 30.0, f"Planning latency too slow: {median_plan_ms:.2f} ms"


def test_empty_point_cloud_triggers_nbv() -> None:
    """Empty point cloud with ungraspable OBB degrades gracefully without crashing."""
    planner = ConfidenceAwareGraspPlanner()
    empty_det = Object3DDetection(
        instance_id=9,
        class_name="empty",
        confidence=0.1,
        entropy=1.0,
        ood_score=0.0,
        is_ood=False,
        centroid=np.array([0.5, 0.0, 0.03], dtype=np.float64),
        obb_center=np.array([0.5, 0.0, 0.03], dtype=np.float64),
        obb_extent=np.array([0.25, 0.25, 0.25], dtype=np.float64),  # exceeds max gripper width
        obb_rotation=np.eye(3, dtype=np.float64),
        point_cloud_xyz=np.zeros((0, 3), dtype=np.float32),
    )
    out = planner.plan(empty_det, table_z=0.0, num_candidates=16)
    assert out.needs_nbv is True
    assert len(out.candidates) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
