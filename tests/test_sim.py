"""Unit and integration tests for ManiSkill 3 cluttered tabletop simulation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omnigda.core.gda_math import backproject_depth
from omnigda.core.schemas import ExecutionResult, RGBDFrame, SupportSample
from omnigda.sim import (
    DEFAULT_BASE_OBJECTS,
    HELD_OUT_NOVEL_OBJECTS,
    OBJECT_CATALOG,
    ClutteredTabletopSim,
    FrankaGraspController,
    build_top_down_grasp,
    capture_few_shot_templates,
)


@pytest.fixture(scope="module")
def clutter_sim() -> ClutteredTabletopSim:
    """Shared cluttered tabletop environment including both base and held-out novel objects."""
    sim = ClutteredTabletopSim(
        num_objects=7,
        include_novel=True,
        image_width=640,
        image_height=480,
        seed=42,
    )
    yield sim
    sim.close()


def test_env_reset_determinism_and_spacing(clutter_sim: ClutteredTabletopSim) -> None:
    """Resetting with the same seed reproduces identical poses & RGB-D with zero interpenetration."""
    poses_a = clutter_sim.reset(seed=42)
    frame_a, _ = clutter_sim.capture_rgbd("base_camera")

    # Verify all base and held-out novel objects are present
    expected_names = set(DEFAULT_BASE_OBJECTS) | set(HELD_OUT_NOVEL_OBJECTS)
    assert set(poses_a.keys()) == expected_names

    # Verify realistic resting heights and non-overlapping pairwise spacing
    names = list(poses_a.keys())
    for name in names:
        pose = poses_a[name]
        assert pose.shape == (4, 4)
        spec = OBJECT_CATALOG[name]
        assert abs(pose[2, 3] - spec.z_offset) < 0.015

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            dist_xy = float(np.linalg.norm(poses_a[ni][:2, 3] - poses_a[nj][:2, 3]))
            min_sep = OBJECT_CATALOG[ni].footprint_radius + OBJECT_CATALOG[nj].footprint_radius
            assert dist_xy >= min_sep * 0.95, f"{ni} and {nj} overlap: dist={dist_xy:.4f}m"

    # Reset with a different seed should alter placements
    poses_other = clutter_sim.reset(seed=123)
    diff_other = max(np.linalg.norm(poses_a[n][:3, 3] - poses_other[n][:3, 3]) for n in names)
    assert diff_other > 0.05

    # Reset back to seed=42 must match poses_a and RGB frame deterministically
    poses_b = clutter_sim.reset(seed=42)
    frame_b, _ = clutter_sim.capture_rgbd("base_camera")
    for name in names:
        np.testing.assert_allclose(poses_a[name], poses_b[name], atol=1e-4)
    np.testing.assert_array_equal(frame_a.rgb, frame_b.rgb)
    np.testing.assert_allclose(frame_a.depth, frame_b.depth, atol=1e-5)

    # Verify overview rendering for demo video capture
    overview = clutter_sim.render_overview()
    assert overview.shape == (480, 640, 3)
    assert overview.dtype == np.uint8
    assert float(overview.std()) > 5.0


def test_dual_camera_rgbd_capture(clutter_sim: ClutteredTabletopSim) -> None:
    """Both base_camera and nbv_camera produce valid OpenCV RGBDFrames and segmentation maps."""
    clutter_sim.reset(seed=42)
    cam_names = clutter_sim.get_camera_names()
    assert "base_camera" in cam_names
    assert "nbv_camera" in cam_names

    frames: dict[str, RGBDFrame] = {}
    for cam in ("base_camera", "nbv_camera"):
        frame, seg_map = clutter_sim.capture_rgbd(cam)
        frames[cam] = frame

        assert isinstance(frame, RGBDFrame)
        assert frame.rgb.shape == (480, 640, 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.depth.shape == (480, 640)
        assert frame.depth.dtype == np.float32
        assert frame.gt_segmentation is not None
        assert frame.gt_segmentation.shape == (480, 640)
        assert frame.gt_segmentation.dtype == np.int32

        # Intrinsics sanity
        intr = frame.intrinsics
        assert intr.width == 640 and intr.height == 480
        assert intr.fx > 100.0 and intr.fy > 100.0
        assert 0.0 < intr.cx < intr.width and 0.0 < intr.cy < intr.height

        # Extrinsics must be a valid SE(3) matrix
        c2w = frame.camera_to_world
        assert c2w.shape == (4, 4)
        rot = c2w[:3, :3]
        np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-5)
        np.testing.assert_allclose(c2w[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6)

        # Depth in meters should contain valid tabletop points
        valid_d = frame.depth[(frame.depth > 0.1) & (frame.depth < 2.5)]
        assert valid_d.size > 10_000

        # Segmentation map should map IDs to every spawned tabletop object
        mapped_names = set(seg_map.values())
        for obj_name in clutter_sim.get_object_names():
            assert obj_name in mapped_names

    # Verify base_camera and nbv_camera are genuinely distinct viewpoints
    cam_sep = float(
        np.linalg.norm(
            frames["base_camera"].camera_to_world[:3, 3]
            - frames["nbv_camera"].camera_to_world[:3, 3]
        )
    )
    assert cam_sep > 0.35


def test_backprojection_geometry_accuracy(clutter_sim: ClutteredTabletopSim) -> None:
    """CRITICAL GEOMETRY TEST: backprojected median 3D point matches SAPIEN GT within 2.5 cm."""
    gt_poses = clutter_sim.reset(seed=42)

    for cam in ("base_camera", "nbv_camera"):
        frame, seg_map = clutter_sim.capture_rgbd(cam)
        assert frame.gt_segmentation is not None
        name_to_id = {name: sid for sid, name in seg_map.items()}

        for obj_name, pose_4x4 in gt_poses.items():
            seg_id = name_to_id[obj_name]
            mask = frame.gt_segmentation == seg_id
            assert int(mask.sum()) >= 200, f"{obj_name} poorly visible in {cam}"

            pts_world, colors = backproject_depth(
                depth=frame.depth,
                fx=frame.intrinsics.fx,
                fy=frame.intrinsics.fy,
                cx=frame.intrinsics.cx,
                cy=frame.intrinsics.cy,
                camera_to_world=frame.camera_to_world,
                mask=mask,
                rgb=frame.rgb,
                depth_min=0.1,
                depth_max=2.5,
            )
            assert pts_world.shape[0] >= 200
            assert colors is not None and colors.shape == pts_world.shape

            median_xyz = np.median(pts_world, axis=0)
            gt_xyz = pose_4x4[:3, 3]
            err_m = float(np.linalg.norm(median_xyz - gt_xyz))
            assert err_m < 0.025, (
                f"{cam} backprojection error for {obj_name} is {err_m:.4f}m "
                f"(median={median_xyz}, gt={gt_xyz}); expected < 0.025m"
            )


def test_capture_few_shot_templates() -> None:
    """Few-shot template generator produces clean multi-view RGB-D + masks for base & novel objects."""
    target_objects = ["red_mug", "blue_bottle", "cyan_t_tool", "magenta_cube"]
    templates = capture_few_shot_templates(
        object_names=target_objects,
        num_shots=3,
        seed=42,
        image_width=320,
        image_height=240,
    )

    assert set(templates.keys()) == set(target_objects)
    for name in target_objects:
        samples = templates[name]
        assert len(samples) == 3

        masks = []
        for idx, sample in enumerate(samples):
            assert isinstance(sample, SupportSample)
            assert sample.class_name == name
            assert sample.rgb is not None and sample.rgb.shape == (240, 320, 3)
            assert sample.rgb.dtype == np.uint8
            assert sample.depth is not None and sample.depth.shape == (240, 320)
            assert sample.depth.dtype == np.float32
            assert sample.mask is not None and sample.mask.shape == (240, 320)
            assert sample.mask.dtype == bool
            assert int(sample.mask.sum()) > 150

            masked_depths = sample.depth[sample.mask]
            assert float(np.min(masked_depths)) > 0.10
            assert float(np.max(masked_depths)) < 2.00

            assert sample.intrinsics is not None
            assert sample.camera_to_world is not None and sample.camera_to_world.shape == (4, 4)
            assert sample.point_cloud_xyz is not None and sample.point_cloud_xyz.shape[0] > 150
            assert sample.text_prompt is not None and len(sample.text_prompt) > 0
            assert sample.metadata["shot_idx"] == idx
            masks.append(sample.mask)

        # Distinct shots should produce genuinely different views
        assert not np.array_equal(masks[0], masks[1])


def test_franka_grasp_controller_physical_pick_and_place(
    clutter_sim: ClutteredTabletopSim,
) -> None:
    """FrankaGraspController physically picks and lifts tabletop objects by > 0.05m and places."""
    controller = FrankaGraspController(
        clutter_sim,
        lift_height_m=0.16,
        record_frames=True,
        record_stride=12,
    )

    # 1. Pick and lift yellow_block and red_mug
    for obj_name in ("yellow_block", "red_mug"):
        poses = clutter_sim.reset(seed=42)
        pose = poses[obj_name]
        z_before = float(pose[2, 3])
        yaw = float(Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")[2])

        grasp = build_top_down_grasp(
            position=pose[:3, 3],
            yaw_rad=yaw,
            width_m=OBJECT_CATALOG[obj_name].grasp_width_m,
            target_class_name=obj_name,
        )
        controller.clear_recorded_frames()
        result = controller.execute_pick(grasp, target_object_name=obj_name)

        assert isinstance(result, ExecutionResult)
        assert result.success, f"Pick failed for {obj_name}: {result.message}"
        assert result.target_class_name == obj_name
        assert result.planning_latency_ms > 0.0

        z_after = float(clutter_sim.get_object_positions()[obj_name][2])
        assert (z_after - z_before) > 0.05, (
            f"Expected {obj_name} lifted by >0.05m, got {z_after - z_before:.4f}m"
        )

        frames = controller.get_recorded_frames()
        assert len(frames) >= 3
        assert frames[0].shape == (480, 640, 3) and frames[0].dtype == np.uint8

    # 2. Full Pick-and-Place on yellow_block
    poses = clutter_sim.reset(seed=42)
    pose = poses["yellow_block"]
    yaw = float(Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")[2])
    grasp = build_top_down_grasp(
        position=pose[:3, 3],
        yaw_rad=yaw,
        width_m=OBJECT_CATALOG["yellow_block"].grasp_width_m,
        target_class_name="yellow_block",
    )
    place_target = np.array([0.0, 0.0, 0.025], dtype=np.float64)
    res_place = controller.execute_pick_and_place(
        grasp=grasp,
        place_position=place_target,
        target_object_name="yellow_block",
        record_frames=False,
    )
    assert res_place.success, f"Pick-and-Place failed: {res_place.message}"
    final_xy = clutter_sim.get_object_positions()["yellow_block"][:2]
    assert float(np.linalg.norm(final_xy - place_target[:2])) < 0.08
