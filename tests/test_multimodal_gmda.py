"""Comprehensive tests for 3D point cloud processing, 4-modality encoders,
PartialAwareMultimodalGMDA, and SceneGraphBuilder3D.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from omnigda.core.schemas import (
    CameraIntrinsics,
    Object3DDetection,
    RGBDFrame,
    SupportSample,
)
from omnigda.perception_3d import (
    Geometric3DEncoder,
    OpenCLIPMultimodalEncoder,
    PartialAwareMultimodalGMDA,
    PhysicalSizeEncoder,
    SceneGraphBuilder3D,
    compute_oriented_bounding_box,
    confidence_weighted_backproject,
    estimate_pointcloud_completeness,
    fuse_multiview_pointclouds,
)

# ---------------------------------------------------------------------------
# Synthetic 3D point cloud & RGB helpers
# ---------------------------------------------------------------------------


def _sample_box_surface(
    extent: tuple[float, float, float],
    center: tuple[float, float, float] = (0.0, 0.0, 0.05),
    n_points: int = 600,
    yaw_rad: float = 0.0,
    noise_std: float = 0.001,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dx, dy, dz = extent
    n_per_face = max(n_points // 6, 10)
    faces = []
    # +/- X faces
    for sgn in (-0.5, 0.5):
        pts = np.column_stack(
            [
                np.full(n_per_face, sgn * dx),
                rng.uniform(-0.5 * dy, 0.5 * dy, n_per_face),
                rng.uniform(-0.5 * dz, 0.5 * dz, n_per_face),
            ]
        )
        faces.append(pts)
    # +/- Y faces
    for sgn in (-0.5, 0.5):
        pts = np.column_stack(
            [
                rng.uniform(-0.5 * dx, 0.5 * dx, n_per_face),
                np.full(n_per_face, sgn * dy),
                rng.uniform(-0.5 * dz, 0.5 * dz, n_per_face),
            ]
        )
        faces.append(pts)
    # +/- Z faces
    for sgn in (-0.5, 0.5):
        pts = np.column_stack(
            [
                rng.uniform(-0.5 * dx, 0.5 * dx, n_per_face),
                rng.uniform(-0.5 * dy, 0.5 * dy, n_per_face),
                np.full(n_per_face, sgn * dz),
            ]
        )
        faces.append(pts)
    cloud = np.vstack(faces)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    cloud = cloud @ rot_z.T + np.asarray(center, dtype=np.float64)[None, :]
    if noise_std > 0:
        cloud += rng.normal(0.0, noise_std, size=cloud.shape)
    return cloud.astype(np.float32)


def _sample_cylinder_surface(
    radius: float,
    height: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.05),
    n_points: int = 600,
    angle_range: tuple[float, float] = (0.0, 2.0 * np.pi),
    noise_std: float = 0.001,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_side = int(0.75 * n_points)
    n_cap = max((n_points - n_side) // 2, 10)
    theta = rng.uniform(angle_range[0], angle_range[1], n_side)
    z = rng.uniform(-0.5 * height, 0.5 * height, n_side)
    side = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])

    caps = []
    for sgn in (-0.5, 0.5):
        r = radius * np.sqrt(rng.uniform(0.0, 1.0, n_cap))
        th = rng.uniform(angle_range[0], angle_range[1], n_cap)
        caps.append(np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n_cap, sgn * height)]))

    cloud = np.vstack([side, *caps]) + np.asarray(center, dtype=np.float64)[None, :]
    if noise_std > 0:
        cloud += rng.normal(0.0, noise_std, size=cloud.shape)
    return cloud.astype(np.float32)


def _sample_sphere_surface(
    radius: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.05),
    n_points: int = 600,
    noise_std: float = 0.001,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(n_points, 3))
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8)
    cloud = radius * vec + np.asarray(center, dtype=np.float64)[None, :]
    if noise_std > 0:
        cloud += rng.normal(0.0, noise_std, size=cloud.shape)
    return cloud.astype(np.float32)


def _render_synthetic_object_rgb(
    shape: str,
    color_rgb: tuple[int, int, int],
    size: int = 224,
    bg_rgb: tuple[int, int, int] = (210, 210, 210),
) -> tuple[np.ndarray, np.ndarray]:
    """Render a clean synthetic tabletop object image + foreground mask."""
    img = np.full((size, size, 3), bg_rgb, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    cx, cy = size // 2, size // 2

    if shape == "apple":
        cv2.circle(img, (cx, cy + 6), 58, color_rgb, -1, lineType=cv2.LINE_AA)
        # Highlight & stem/leaf for unmistakable CLIP semantics
        cv2.circle(img, (cx - 20, cy - 12), 14, (255, 160, 160), -1, lineType=cv2.LINE_AA)
        cv2.line(img, (cx, cy - 48), (cx + 4, cy - 72), (90, 60, 30), 6, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (cx + 18, cy - 64), (16, 8), -25, 0, 360, (40, 160, 50), -1)
        m = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(m, (cx, cy + 6), 60, 1, -1)
        cv2.ellipse(m, (cx + 18, cy - 64), (20, 12), -25, 0, 360, 1, -1)
        mask = m.astype(bool)
    elif shape == "banana":
        # Yellow crescent banana
        m = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(img, (cx, cy + 10), (75, 55), 0, 195, 345, color_rgb, 26, lineType=cv2.LINE_AA)
        cv2.ellipse(m, (cx, cy + 10), (75, 55), 0, 195, 345, 1, 30)
        # Dark tips
        cv2.circle(img, (cx - 72, cy - 6), 6, (95, 75, 25), -1)
        cv2.circle(img, (cx + 72, cy - 6), 6, (95, 75, 25), -1)
        mask = m.astype(bool)
    elif shape == "box":
        x1, y1, x2, y2 = cx - 55, cy - 55, cx + 55, cy + 55
        cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb, -1)
        # Subtle 3D top bevel shading
        lighter = tuple(min(255, c + 40) for c in color_rgb)
        pts_top = np.array([[x1, y1], [x2, y1], [x2 - 14, y1 - 18], [x1 + 14, y1 - 18]])
        cv2.fillPoly(img, [pts_top], lighter)
        mask[y1 - 18 : y2, x1:x2] = True
    elif shape == "cylinder":
        x1, y1, x2, y2 = cx - 42, cy - 55, cx + 42, cy + 55
        cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb, -1)
        lighter = tuple(min(255, c + 35) for c in color_rgb)
        cv2.ellipse(img, (cx, y1), (42, 14), 0, 0, 360, lighter, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (cx, y2), (42, 14), 0, 0, 180, color_rgb, -1, lineType=cv2.LINE_AA)
        mask[max(0, y1 - 14) : min(size, y2 + 14), x1:x2] = True
    else:
        raise ValueError(f"Unknown synthetic shape: {shape}")

    return img, mask


@pytest.fixture(scope="module")
def clip_encoder() -> OpenCLIPMultimodalEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return OpenCLIPMultimodalEncoder(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
        fp16=(device == "cuda"),
    )


# ---------------------------------------------------------------------------
# 1. Point cloud backprojection, OBB, multiview fusion, and completeness tests
# ---------------------------------------------------------------------------


def test_confidence_weighted_backproject_removes_flying_pixels_and_improves_obb():
    h, w = 240, 320
    intr = CameraIntrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0, width=w, height=h)

    # Camera looking along +Y in world frame so image Y maps to world -Z (gravity aligned!)
    # cam_x -> world +X, cam_y -> world -Z, cam_z -> world +Y
    c2w = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.15],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Synthesize a front face of an object at depth=0.60m:
    # 40x50 pixels at depth 0.60m -> width = 40*0.6/300 = 0.08m, height = 50*0.6/300 = 0.10m
    depth = np.full((h, w), 1.50, dtype=np.float32)  # background wall
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    conf = np.zeros((h, w), dtype=np.float32)

    y1, y2, x1, x2 = 95, 145, 140, 180
    # Interior clean surface
    depth[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2] = 0.60
    rgb[y1:y2, x1:x2] = (200, 40, 40)
    mask[y1:y2, x1:x2] = True
    conf[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2] = 0.94

    # Noisy flying depth pixels along the 2-pixel silhouette boundary (low confidence!)
    border = mask.copy()
    border[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2] = False
    depth[border] = 1.48  # background bleed flying pixels (+0.88m jump!)
    conf[border] = 0.22

    frame = RGBDFrame(rgb=rgb, depth=depth, intrinsics=intr, camera_to_world=c2w)

    # Naive backprojection without confidence filtering or SOR
    xyz_naive, _ = confidence_weighted_backproject(
        frame=frame,
        mask=mask,
        confidence_map=None,
        sor_nb_neighbors=0,
        voxel_size=None,
    )
    _, ext_naive, _ = compute_oriented_bounding_box(xyz_naive, gravity_aligned=True)

    # Confidence-weighted + SOR backprojection
    xyz_clean, rgb_clean = confidence_weighted_backproject(
        frame=frame,
        mask=mask,
        confidence_map=conf,
        conf_thresh=0.45,
        sor_nb_neighbors=20,
        sor_std_ratio=1.5,
        voxel_size=0.002,
    )
    center_clean, ext_clean, rot_clean = compute_oriented_bounding_box(
        xyz_clean, gravity_aligned=True
    )

    assert xyz_clean.shape[0] > 100
    assert rgb_clean is not None and rgb_clean.shape == xyz_clean.shape
    # Naive cloud includes 0.88m flying depth tail; denoised cloud has tight depth extent!
    assert ext_naive.max() > 0.75
    assert ext_clean.max() < 0.14
    assert np.isclose(ext_clean[2], 0.092, atol=0.015)  # physical vertical height dz
    assert np.allclose(rot_clean[:, 2], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.isclose(np.linalg.det(rot_clean), 1.0, atol=1e-6)
    assert np.all(np.isfinite(center_clean))


def test_multiview_fusion_and_completeness_scoring():
    # Two complementary partial views of a cylinder (front half + back half with overlap)
    view_a = _sample_cylinder_surface(
        radius=0.04, height=0.10, angle_range=(-0.15, np.pi + 0.15), n_points=220, seed=1
    )
    # Slight 3mm calibration offset on second view
    view_b = _sample_cylinder_surface(
        radius=0.04, height=0.10, angle_range=(np.pi - 0.15, 2.0 * np.pi + 0.15), n_points=220, seed=2
    ) + np.array([0.003, -0.002, 0.0], dtype=np.float32)

    fused = fuse_multiview_pointclouds([view_a, view_b], voxel_size=0.004, refine_icp=True)
    _, ext_a, _ = compute_oriented_bounding_box(view_a)
    _, ext_fused, _ = compute_oriented_bounding_box(fused)

    comp_sparse = estimate_pointcloud_completeness(view_a[:25], ext_a)
    comp_single = estimate_pointcloud_completeness(view_a, ext_a)
    comp_fused = estimate_pointcloud_completeness(fused, ext_fused)

    assert 0.2 <= comp_sparse < comp_single < comp_fused <= 1.0
    assert np.isclose(ext_fused[2], 0.10, atol=0.012)


# ---------------------------------------------------------------------------
# 2. 4-Modality Encoder tests (including real OpenCLIP on RTX 3060 GPU)
# ---------------------------------------------------------------------------


def test_geometric_and_physical_size_encoders_and_adapter_hook(tmp_path: Path):
    geom = Geometric3DEncoder(max_points=512, seed=42)
    size_enc = PhysicalSizeEncoder()

    box_0 = _sample_box_surface((0.08, 0.04, 0.05), yaw_rad=0.0, seed=10)
    box_rot = _sample_box_surface((0.08, 0.04, 0.05), yaw_rad=1.25, seed=11)
    sphere = _sample_sphere_surface(radius=0.045, seed=12)

    f_box0 = geom.encode(box_0)
    f_boxrot = geom.encode(box_rot)
    f_sphere = geom.encode(sphere)

    assert f_box0.shape == (geom.feature_dim,)
    assert np.isclose(np.linalg.norm(f_box0), 1.0, atol=1e-5)

    # Rotation invariance: rotated box matches unrotated box much better than a sphere
    sim_same = float(np.dot(f_box0, f_boxrot))
    sim_diff = float(np.dot(f_box0, f_sphere))
    assert sim_same > 0.90
    assert sim_same > sim_diff + 0.10

    # Test pluggable linear projection adapter hook (for T8 distilled weights)
    rng = np.random.default_rng(7)
    w_proj = rng.normal(0.0, 0.2, size=(64, geom.RAW_DIM)).astype(np.float32)
    b_proj = rng.normal(0.0, 0.05, size=(64,)).astype(np.float32)
    adapter_path = tmp_path / "distilled_3d_adapter.npz"
    np.savez(adapter_path, weight=w_proj, bias=b_proj)

    geom.load_projection_adapter(adapter_path)
    assert geom.feature_dim == 64
    f_proj = geom.encode(box_0)
    assert f_proj.shape == (64,)
    assert np.isclose(np.linalg.norm(f_proj), 1.0, atol=1e-5)

    # PhysicalSizeEncoder separates identical shapes of different metric scales
    small_cube = _sample_box_surface((0.04, 0.04, 0.04), center=(0.0, 0.0, 0.02), seed=1)
    large_cube = _sample_box_surface((0.10, 0.10, 0.10), center=(0.0, 0.0, 0.05), seed=2)
    s_small = size_enc.encode(xyz=small_cube, table_z=0.0)
    s_large = size_enc.encode(xyz=large_cube, table_z=0.0)
    assert s_small.shape == (size_enc.feature_dim,)
    assert float(np.linalg.norm(s_large - s_small)) > 1.2


def test_openclip_multimodal_encoder_on_gpu(clip_encoder: OpenCLIPMultimodalEncoder):
    if torch.cuda.is_available():
        assert clip_encoder.device.type == "cuda"

    img_apple, mask_apple = _render_synthetic_object_rgb("apple", (210, 25, 25))
    img_banana, mask_banana = _render_synthetic_object_rgb("banana", (240, 210, 30))
    img_blue_box, mask_blue_box = _render_synthetic_object_rgb("box", (30, 80, 220))

    emb_apple = clip_encoder.encode_image_crop(img_apple, mask=mask_apple)
    emb_banana = clip_encoder.encode_image_crop(img_banana, mask=mask_banana)
    emb_box = clip_encoder.encode_image_crop(img_blue_box, mask=mask_blue_box)

    text_embs = clip_encoder.encode_text(["red apple", "yellow banana", "blue cube box"])
    assert text_embs.shape == (3, clip_encoder.embed_dim)
    assert np.allclose(np.linalg.norm(text_embs, axis=1), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(emb_apple), 1.0, atol=1e-4)

    # Zero-shot cosine similarity sanity check: diagonal dominates off-diagonal
    sim_mat = np.stack([emb_apple, emb_banana, emb_box], axis=0) @ text_embs.T
    assert int(np.argmax(sim_mat[0])) == 0  # red apple
    assert int(np.argmax(sim_mat[1])) == 1  # yellow banana
    assert int(np.argmax(sim_mat[2])) == 2  # blue cube box

    # Verify pluggable TensorRT/ONNX image encoder hook (T5 interface)
    called = {"count": 0}

    def fake_trt_hook(batch_nchw: np.ndarray) -> np.ndarray:
        called["count"] += 1
        assert batch_nchw.shape == (1, 3, 224, 224)
        vec = np.ones((1, clip_encoder.embed_dim), dtype=np.float32)
        return vec

    clip_encoder.set_image_encoder_hook(fake_trt_hook)
    try:
        emb_hook = clip_encoder.encode_image_crop(img_apple, mask=mask_apple)
        assert called["count"] == 1
        assert np.isclose(np.linalg.norm(emb_hook), 1.0, atol=1e-5)
    finally:
        clip_encoder.set_image_encoder_hook(None)


# ---------------------------------------------------------------------------
# 3. PartialAwareMultimodalGMDA: fusion, zero-shot, OOD, persistence
# ---------------------------------------------------------------------------


def test_gmda_few_shot_and_zero_shot_text_registration(clip_encoder: OpenCLIPMultimodalEncoder):
    clf = PartialAwareMultimodalGMDA(clip_encoder=clip_encoder)

    img_apple, mask_apple = _render_synthetic_object_rgb("apple", (215, 25, 25))
    img_box, mask_box = _render_synthetic_object_rgb("box", (30, 80, 220))
    img_banana, mask_banana = _render_synthetic_object_rgb("banana", (240, 210, 30))

    apple_samples = [
        SupportSample(
            class_name="red_apple",
            rgb=img_apple,
            mask=mask_apple,
            point_cloud_xyz=_sample_sphere_surface(0.04, center=(0.0, 0.0, 0.04), seed=s),
        )
        for s in range(3)
    ]
    box_samples = [
        SupportSample(
            class_name="blue_box",
            rgb=img_box,
            mask=mask_box,
            point_cloud_xyz=_sample_box_surface((0.08, 0.08, 0.08), center=(0.0, 0.0, 0.04), seed=s),
        )
        for s in range(3)
    ]
    clf.register_few_shot("red_apple", apple_samples)
    clf.register_few_shot("blue_box", box_samples)
    # Brand-new class registered purely via natural language!
    clf.register_zero_shot_text("yellow_banana", "yellow banana")

    # Classify few-shot apple
    det_apple = clf.classify_instance(
        instance_id=1,
        xyz=_sample_sphere_surface(0.04, center=(0.1, 0.0, 0.04), seed=99),
        rgb_image=img_apple,
        mask_2d=mask_apple,
    )
    assert det_apple.class_name == "red_apple"
    assert det_apple.confidence > 0.75
    assert np.isclose(sum(det_apple.modality_weights.values()), 1.0, atol=1e-5)
    assert np.isclose(sum(det_apple.posterior_probs.values()), 1.0, atol=1e-5)

    # Classify zero-shot banana (unseen in 3D geometry/size, recognized via OpenCLIP text!)
    det_banana = clf.classify_instance(
        instance_id=2,
        xyz=_sample_box_surface((0.16, 0.035, 0.035), center=(0.2, 0.1, 0.02), seed=77),
        rgb_image=img_banana,
        mask_2d=mask_banana,
    )
    assert det_banana.class_name == "yellow_banana"
    assert "text_clip" in det_banana.modality_weights


def test_entropy_gated_fusion_resolves_ambiguous_2d_with_3d_size(
    clip_encoder: OpenCLIPMultimodalEncoder,
):
    """Two identically colored red cylinders of different heights: 2D appearance is 50/50
    ambiguous, so entropy-gated Bayesian fusion must dynamically upweight 3D size/geometry."""
    clf = PartialAwareMultimodalGMDA(clip_encoder=clip_encoder)
    img_red_cyl, mask_red_cyl = _render_synthetic_object_rgb("cylinder", (215, 30, 30))

    short_samples = [
        SupportSample(
            class_name="short_red_cylinder",
            rgb=img_red_cyl,
            mask=mask_red_cyl,
            point_cloud_xyz=_sample_cylinder_surface(
                radius=0.035, height=0.045, center=(0.0, 0.0, 0.0225), seed=10 + s
            ),
        )
        for s in range(4)
    ]
    tall_samples = [
        SupportSample(
            class_name="tall_red_cylinder",
            rgb=img_red_cyl,
            mask=mask_red_cyl,
            point_cloud_xyz=_sample_cylinder_surface(
                radius=0.035, height=0.145, center=(0.0, 0.0, 0.0725), seed=20 + s
            ),
        )
        for s in range(4)
    ]

    clf.register_few_shot("short_red_cylinder", short_samples)
    clf.register_few_shot("tall_red_cylinder", tall_samples)

    # Query tall red cylinder with completely ambiguous 2D appearance + ambiguous seg2d logits
    det_tall = clf.classify_instance(
        instance_id=10,
        xyz=_sample_cylinder_surface(
            radius=0.035, height=0.145, center=(0.15, -0.05, 0.0725), seed=101
        ),
        rgb_image=img_red_cyl,
        mask_2d=mask_red_cyl,
        seg2d_logits={"short_red_cylinder": 0.0, "tall_red_cylinder": 0.0},
    )
    assert det_tall.class_name == "tall_red_cylinder"
    assert det_tall.confidence > 0.85
    assert det_tall.modality_weights["size_3d"] > det_tall.modality_weights["visual_clip"]
    assert det_tall.modality_weights["size_3d"] > det_tall.modality_weights["seg2d"]

    # Query short red cylinder as well
    det_short = clf.classify_instance(
        instance_id=11,
        xyz=_sample_cylinder_surface(
            radius=0.035, height=0.045, center=(-0.10, 0.05, 0.0225), seed=102
        ),
        rgb_image=img_red_cyl,
        mask_2d=mask_red_cyl,
        seg2d_logits={"short_red_cylinder": 0.0, "tall_red_cylinder": 0.0},
    )
    assert det_short.class_name == "short_red_cylinder"
    assert det_short.confidence > 0.85


def test_entropy_gated_fusion_resolves_noisy_partial_3d_with_visual_cues(
    clip_encoder: OpenCLIPMultimodalEncoder,
):
    """Two geometrically identical cubes of different colors queried with a sparse, noisy,
    partial 3D cloud: completeness reliability & entropy gating must trust visual/text cues."""
    clf = PartialAwareMultimodalGMDA(clip_encoder=clip_encoder)

    img_red_box, mask_red_box = _render_synthetic_object_rgb("box", (220, 30, 30))
    img_blue_box, mask_blue_box = _render_synthetic_object_rgb("box", (30, 80, 225))

    red_samples = [
        SupportSample(
            class_name="red_cube",
            rgb=img_red_box,
            mask=mask_red_box,
            point_cloud_xyz=_sample_box_surface((0.07, 0.07, 0.07), center=(0.0, 0.0, 0.035), seed=s),
            text_prompt="red cube",
        )
        for s in range(4)
    ]
    blue_samples = [
        SupportSample(
            class_name="blue_cube",
            rgb=img_blue_box,
            mask=mask_blue_box,
            point_cloud_xyz=_sample_box_surface(
                (0.07, 0.07, 0.07), center=(0.0, 0.0, 0.035), seed=50 + s
            ),
            text_prompt="blue cube",
        )
        for s in range(4)
    ]
    clf.register_few_shot("red_cube", red_samples)
    clf.register_few_shot("blue_cube", blue_samples)

    # Severely partial, sparse, noisy 3D sliver (only 24 noisy points on one thin front corner)
    rng = np.random.default_rng(303)
    partial_noisy_xyz = np.column_stack(
        [
            rng.uniform(0.0, 0.025, 24),
            rng.uniform(0.0, 0.010, 24),
            rng.uniform(0.01, 0.045, 24),
        ]
    ).astype(np.float32) + rng.normal(0.0, 0.004, size=(24, 3)).astype(np.float32)

    _, ext_partial, _ = compute_oriented_bounding_box(partial_noisy_xyz)
    assert estimate_pointcloud_completeness(partial_noisy_xyz, ext_partial) < 0.55

    det = clf.classify_instance(
        instance_id=20,
        xyz=partial_noisy_xyz,
        rgb_image=img_blue_box,
        mask_2d=mask_blue_box,
    )
    assert det.class_name == "blue_cube"
    assert det.confidence > 0.85
    assert det.modality_weights["visual_clip"] > det.modality_weights["geom_3d"]
    assert det.modality_weights["visual_clip"] > det.modality_weights["size_3d"]


def test_ood_detection_and_memory_persistence(
    clip_encoder: OpenCLIPMultimodalEncoder,
    tmp_path: Path,
):
    clf = PartialAwareMultimodalGMDA(clip_encoder=clip_encoder)

    img_red_box, mask_red_box = _render_synthetic_object_rgb("box", (220, 30, 30))
    img_blue_cyl, mask_blue_cyl = _render_synthetic_object_rgb("cylinder", (30, 80, 225))
    img_green_banana, mask_green_banana = _render_synthetic_object_rgb("banana", (35, 195, 55))

    clf.register_few_shot(
        "small_red_cube",
        [
            SupportSample(
                class_name="small_red_cube",
                rgb=img_red_box,
                mask=mask_red_box,
                point_cloud_xyz=_sample_box_surface(
                    (0.05, 0.05, 0.05), center=(0.0, 0.0, 0.025), seed=s
                ),
            )
            for s in range(4)
        ],
    )
    clf.register_few_shot(
        "small_blue_cylinder",
        [
            SupportSample(
                class_name="small_blue_cylinder",
                rgb=img_blue_cyl,
                mask=mask_blue_cyl,
                point_cloud_xyz=_sample_cylinder_surface(
                    radius=0.025, height=0.06, center=(0.0, 0.0, 0.03), seed=20 + s
                ),
            )
            for s in range(4)
        ],
    )

    # In-distribution query
    id_xyz = _sample_box_surface((0.05, 0.05, 0.05), center=(0.1, 0.0, 0.025), seed=88)
    det_id = clf.classify_instance(
        instance_id=1,
        xyz=id_xyz,
        rgb_image=img_red_box,
        mask_2d=mask_red_box,
    )
    assert det_id.class_name == "small_red_cube"
    assert not det_id.is_ood

    # Out-of-distribution distractor: huge flat slab with completely novel green banana appearance
    ood_xyz = _sample_box_surface((0.24, 0.14, 0.02), center=(0.0, 0.0, 0.01), seed=99)
    det_ood = clf.classify_instance(
        instance_id=2,
        xyz=ood_xyz,
        rgb_image=img_green_banana,
        mask_2d=mask_green_banana,
    )
    assert det_ood.ood_score > det_id.ood_score * 3.0
    assert det_ood.is_ood is True

    # Save & reload memory into a fresh classifier instance
    mem_file = clf.save_memory(tmp_path / "gmda_memory.npz")
    assert mem_file.exists()

    clf_reloaded = PartialAwareMultimodalGMDA(clip_encoder=clip_encoder)
    clf_reloaded.load_memory(mem_file)
    assert clf_reloaded.registered_classes == clf.registered_classes

    det_id_reloaded = clf_reloaded.classify_instance(
        instance_id=1,
        xyz=id_xyz,
        rgb_image=img_red_box,
        mask_2d=mask_red_box,
    )
    assert det_id_reloaded.class_name == det_id.class_name
    assert np.isclose(det_id_reloaded.confidence, det_id.confidence, atol=1e-4)
    assert np.isclose(det_id_reloaded.ood_score, det_id.ood_score, atol=1e-4)
    for cname, prob in det_id.posterior_probs.items():
        assert np.isclose(det_id_reloaded.posterior_probs[cname], prob, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. SceneGraphBuilder3D tests
# ---------------------------------------------------------------------------


def _make_dummy_det(
    instance_id: int,
    class_name: str,
    centroid: tuple[float, float, float],
    extent: tuple[float, float, float] = (0.06, 0.06, 0.06),
    confidence: float = 0.9,
) -> Object3DDetection:
    c = np.asarray(centroid, dtype=np.float64)
    return Object3DDetection(
        instance_id=instance_id,
        class_name=class_name,
        confidence=confidence,
        entropy=0.1,
        ood_score=5.0,
        is_ood=False,
        centroid=c,
        obb_center=c.copy(),
        obb_extent=np.asarray(extent, dtype=np.float64),
        obb_rotation=np.eye(3, dtype=np.float64),
        point_cloud_xyz=np.zeros((16, 3), dtype=np.float32),
    )


def test_scene_graph_builder_spatial_relations_and_queries():
    # ROS / ManiSkill world frame: +X forward (away from robot), +Y left, +Z up
    mug_left = _make_dummy_det(1, "mug", (0.30, 0.20, 0.05))
    bottle_center = _make_dummy_det(2, "bottle", (0.30, 0.00, 0.06))
    mug_right = _make_dummy_det(3, "mug", (0.30, -0.20, 0.05))
    box_behind = _make_dummy_det(4, "box", (0.52, 0.00, 0.05))
    lid_above = _make_dummy_det(5, "lid", (0.30, 0.00, 0.19))

    builder = SceneGraphBuilder3D(near_threshold_m=0.25)
    sg = builder.build([mug_left, bottle_center, mug_right, box_behind, lid_above], timestamp_sec=1.5)

    assert sg.timestamp_sec == 1.5
    assert len(sg.objects) == 5

    edge_set = {(r.subject_id, r.relation, r.object_id) for r in sg.relations}
    assert (1, "left_of", 2) in edge_set
    assert (3, "right_of", 2) in edge_set
    assert (2, "front_of", 4) in edge_set
    assert (4, "behind", 2) in edge_set
    assert (5, "above", 2) in edge_set
    assert (1, "near", 2) in edge_set
    assert (5, "near", 2) in edge_set

    for rel in sg.relations:
        assert 0.0 < rel.confidence <= 1.0
        assert rel.distance_m >= 0.0

    # Pairwise relation query: "mug left_of bottle" -> mug_left (id=1)
    q_left = builder.query_relation(sg, subject_class="mug", relation="left_of", reference_class="bottle")
    assert [o.instance_id for o in q_left] == [1]

    # Pairwise relation query: "mug right_of bottle" -> mug_right (id=3)
    q_right = builder.query_relation(sg, subject_class="mug", relation="right_of", reference_class="bottle")
    assert [o.instance_id for o in q_right] == [3]

    # Superlatives: leftmost, rightmost, highest, nearest
    assert builder.resolve_query(sg, target_query="mug", spatial_relation="leftmost").instance_id == 1
    assert builder.resolve_query(sg, target_query="mug", spatial_relation="rightmost").instance_id == 3
    assert builder.resolve_query(sg, spatial_relation="highest").instance_id == 5
    nearest_to_box = builder.resolve_query(
        sg, target_query="bottle", spatial_relation="nearest", reference_query="box"
    )
    assert nearest_to_box is not None and nearest_to_box.instance_id == 2

