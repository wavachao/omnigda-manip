"""Unit and GPU integration tests for SegmentGDA+ 2D perception module."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from omnigda.core.config import load_config
from omnigda.core.schemas import SegmentationResult, SupportSample
from omnigda.perception_2d import (
    DINOv2MultiScaleExtractor,
    LightweightConvFeatureExtractor,
    SegmentGDAPlus,
    create_feature_extractor,
    extract_instance_masks,
    fast_color_guided_filter,
)


def _make_tabletop_bg(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Render a textured wood-grain tabletop with subtle lighting gradient and dark back wall."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grain = np.sin(yy * 0.12 + np.sin(xx * 0.025) * 2.5) * 12.0
    grad = (xx / w) * 18.0 - (yy / h) * 14.0
    noise = rng.normal(0.0, 3.0, size=(h, w, 3)).astype(np.float32)

    base = np.zeros((h, w, 3), dtype=np.float32)
    base[..., 0] = 175.0 + grain + grad
    base[..., 1] = 152.0 + grain * 0.8 + grad * 0.8
    base[..., 2] = 126.0 + grain * 0.6 + grad * 0.6

    wall_h = max(1, h // 8)
    base[:wall_h, :, 0] = 65.0 + (xx[:wall_h] / w) * 15.0
    base[:wall_h, :, 1] = 70.0 + (xx[:wall_h] / w) * 15.0
    base[:wall_h, :, 2] = 78.0 + (xx[:wall_h] / w) * 15.0
    return np.clip(base + noise, 0.0, 255.0).astype(np.uint8)


def _draw_textured_object(
    img: np.ndarray,
    kind: str,
    cx: float,
    cy: float,
    scale: float = 1.0,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Composite a shaded, textured synthetic tabletop object and return its boolean mask."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    th = np.deg2rad(angle_deg)
    rx = dx * np.cos(th) + dy * np.sin(th)
    ry = -dx * np.sin(th) + dy * np.cos(th)

    if kind == "red_mug":
        r_x, r_y = 48.0 * scale, 56.0 * scale
        inside = (np.abs(rx) <= r_x) & (np.abs(ry) <= r_y)
        shade = np.clip(1.0 - 0.35 * (rx / max(r_x, 1.0)), 0.55, 1.15)
        stripe = (np.abs(ry) < 10.0 * scale).astype(np.float32)
        col = np.zeros_like(img, dtype=np.float32)
        col[..., 0] = (205.0 * (1.0 - stripe) + 240.0 * stripe) * shade
        col[..., 1] = (38.0 * (1.0 - stripe) + 235.0 * stripe) * shade
        col[..., 2] = (42.0 * (1.0 - stripe) + 230.0 * stripe) * shade
        img[inside] = np.clip(col[inside], 0.0, 255.0).astype(np.uint8)
        mask[inside] = True
    elif kind == "blue_box":
        r_x, r_y = 60.0 * scale, 42.0 * scale
        inside = (np.abs(rx) <= r_x) & (np.abs(ry) <= r_y)
        checker = (((np.floor(rx / 14.0) + np.floor(ry / 14.0)) % 2) == 0).astype(np.float32)
        col = np.zeros_like(img, dtype=np.float32)
        col[..., 0] = 30.0 + 40.0 * checker
        col[..., 1] = 85.0 + 55.0 * checker
        col[..., 2] = 195.0 + 45.0 * checker
        img[inside] = np.clip(col[inside], 0.0, 255.0).astype(np.uint8)
        mask[inside] = True
    elif kind == "green_bottle":
        rad = 46.0 * scale
        dist = np.sqrt(rx * rx + ry * ry)
        inside = dist <= rad
        cap = dist <= 16.0 * scale
        radial = np.clip(1.1 - 0.45 * (dist / max(rad, 1.0)), 0.5, 1.1)
        col = np.zeros_like(img, dtype=np.float32)
        col[..., 0] = np.where(cap, 235.0, 38.0 * radial)
        col[..., 1] = np.where(cap, 210.0, 178.0 * radial)
        col[..., 2] = np.where(cap, 45.0, 62.0 * radial)
        img[inside] = np.clip(col[inside], 0.0, 255.0).astype(np.uint8)
        mask[inside] = True
    else:
        raise ValueError(f"Unknown synthetic object kind: {kind}")

    return mask


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return float(inter) / float(max(union, 1))


def _compute_boundary_iou(pred: np.ndarray, gt: np.ndarray, band_px: int = 11) -> float:
    kernel = np.ones((band_px, band_px), dtype=np.uint8)
    gt_u8 = gt.astype(np.uint8)
    band = (cv2.dilate(gt_u8, kernel) - cv2.erode(gt_u8, kernel)) > 0
    return _compute_iou(pred & band, gt & band)


@pytest.fixture(scope="module")
def dinov2_extractor() -> DINOv2MultiScaleExtractor:
    """Module-scoped real DINOv2 multi-scale extractor on RTX 3060 GPU (with CPU fallback)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return DINOv2MultiScaleExtractor(
        model_name="vit_small_patch14_dinov2.lvd142m",
        input_size=518,
        feature_layers=(8, 10, 11),
        fusion_mode="mean",
        device=device,
        fp16=(device == "cuda"),
        pretrained=True,
    )


def test_lightweight_extractor_and_factory() -> None:
    ext = create_feature_extractor(backend="lightweight")
    assert isinstance(ext, LightweightConvFeatureExtractor)
    assert ext.backend_name == "lightweight-cpu-fp32"
    assert ext.feature_dim == 48

    img = _make_tabletop_bg(h=240, w=320, seed=1)
    feat1 = ext.extract(img)
    feat2 = ext.extract(img)
    assert feat1.shape == (48, 32, 32)
    assert feat1.dtype == np.float32
    norms = np.linalg.norm(feat1, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-4)
    assert np.array_equal(feat1, feat2)

    # Verify SegmentGDAPlus works end-to-end with lightweight backend
    sup_img = _make_tabletop_bg(h=240, w=320, seed=2)
    m_mug = _draw_textured_object(sup_img, "red_mug", 95, 130, scale=0.6)
    m_box = _draw_textured_object(sup_img, "blue_box", 230, 130, scale=0.6)
    seg = SegmentGDAPlus(backend="lightweight", spatial_refine=True)
    seg.fit(
        [
            SupportSample(class_name="red_mug", rgb=sup_img, mask=m_mug),
            SupportSample(class_name="blue_box", rgb=sup_img, mask=m_box),
        ]
    )
    res = seg.segment(sup_img)
    assert isinstance(res, SegmentationResult)
    assert _compute_iou(res.masks["red_mug"], m_mug) > 0.75
    assert _compute_iou(res.masks["blue_box"], m_box) > 0.75


def test_dinov2_multiscale_extractor_gpu(dinov2_extractor: DINOv2MultiScaleExtractor) -> None:
    img = _make_tabletop_bg(h=480, w=640, seed=7)
    _draw_textured_object(img, "red_mug", 220, 260)

    assert dinov2_extractor.feature_dim == 384
    if torch.cuda.is_available():
        assert dinov2_extractor.backend_name == "pytorch-cuda-fp16"

    # Warmup
    feat = dinov2_extractor.extract(img)
    assert feat.shape == (384, 37, 37)
    assert feat.dtype == np.float32
    norms = np.linalg.norm(feat, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Measure warm FP16 inference latency
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        feat_run = dinov2_extractor.extract(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) * 1000.0 / n_runs
    assert feat_run.shape == (384, 37, 37)
    if torch.cuda.is_available():
        assert avg_ms < 100.0, f"Expected GPU FP16 DINOv2 latency < 100ms, got {avg_ms:.2f}ms"

    # Verify batch extraction consistency (allow FP16 cuBLAS batch-size kernel tolerance)
    batch_feats = dinov2_extractor.extract_batch([img, img])
    assert len(batch_feats) == 2
    for bf in batch_feats:
        cos_sim = float(np.sum(bf * feat, axis=0).mean())
        assert cos_sim > 0.9999, f"Batch feature cosine similarity too low: {cos_sim:.6f}"
        assert np.allclose(bf, feat, atol=5e-3)

    # Verify concat fusion mode exposes concatenated multi-scale channels
    concat_ext = DINOv2MultiScaleExtractor(
        model_name="vit_small_patch14_dinov2.lvd142m",
        input_size=224,
        feature_layers=(8, 10, 11),
        fusion_mode="concat",
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=torch.cuda.is_available(),
        pretrained=False,
    )
    assert concat_ext.feature_dim == 384 * 3
    concat_feat = concat_ext.extract(img)
    assert concat_feat.shape == (384 * 3, 16, 16)
    assert np.allclose(np.linalg.norm(concat_feat, axis=0), 1.0, atol=1e-4)


def test_fast_color_guided_filter_sharpens_edges() -> None:
    h, w = 240, 320
    rgb = _make_tabletop_bg(h=h, w=w, seed=3)
    gt = _draw_textured_object(rgb, "blue_box", 160, 120, scale=0.8, angle_deg=15.0)

    # Simulate coarse ViT-grid logit upsampling (blurred boundary)
    coarse_fg = cv2.resize(gt.astype(np.float32) * 4.0 - 2.0, (24, 18), interpolation=cv2.INTER_AREA)
    up_fg = cv2.resize(coarse_fg, (w, h), interpolation=cv2.INTER_LINEAR)
    src = np.stack([-up_fg, up_fg], axis=0).astype(np.float32)

    refined = fast_color_guided_filter(rgb, src, radius=8, eps=1e-3, subsample=2)
    assert refined.shape == src.shape
    assert refined.dtype == np.float32

    raw_pred = src[1] > src[0]
    ref_pred = refined[1] > refined[0]
    assert _compute_boundary_iou(ref_pred, gt) >= _compute_boundary_iou(raw_pred, gt)
    assert _compute_iou(ref_pred, gt) > _compute_iou(raw_pred, gt)


def test_segment_gda_few_shot_and_online_registration_gpu(
    dinov2_extractor: DINOv2MultiScaleExtractor,
    tmp_path: Path,
) -> None:
    # 1. Build 2-shot support set for 2 initial classes
    support_initial: list[SupportSample] = []
    for shot, (p_mug, p_box, ang) in enumerate(
        [((180, 250), (450, 260), 10.0), ((240, 310), (470, 190), -18.0)]
    ):
        img = _make_tabletop_bg(seed=10 + shot)
        m_mug = _draw_textured_object(img, "red_mug", *p_mug, scale=1.0, angle_deg=ang)
        m_box = _draw_textured_object(img, "blue_box", *p_box, scale=0.95, angle_deg=-ang)
        support_initial.append(SupportSample(class_name="red_mug", rgb=img, mask=m_mug))
        support_initial.append(SupportSample(class_name="blue_box", rgb=img, mask=m_box))

    model = SegmentGDAPlus(
        extractor=dinov2_extractor,
        shrinkage_method="oas",
        num_foreground_components=2,
        num_background_components=4,
        confidence_threshold=0.55,
        spatial_refine=True,
    )
    model.fit(support_initial)
    assert model.is_fitted
    assert model.class_names == ["red_mug", "blue_box"]

    # 2. Evaluate on unseen query scene with novel object positions, scales & rotations
    query_2 = _make_tabletop_bg(seed=99)
    gt_mug_2 = _draw_textured_object(query_2, "red_mug", 430, 310, scale=1.05, angle_deg=25.0)
    gt_box_2 = _draw_textured_object(query_2, "blue_box", 185, 220, scale=0.92, angle_deg=30.0)

    res_ref = model.segment(query_2)
    assert res_ref.class_names == ["red_mug", "blue_box"]
    assert res_ref.class_map.shape == (480, 640)
    assert res_ref.class_map.dtype == np.int32
    assert res_ref.log_likelihoods.shape == (3, 480, 640)
    assert res_ref.log_likelihoods.dtype == np.float32
    assert res_ref.confidence_map.shape == (480, 640)
    assert res_ref.entropy_map.shape == (480, 640)
    assert res_ref.latency_ms > 0.0
    assert float(res_ref.confidence_map.min()) >= 0.0
    assert float(res_ref.confidence_map.max()) <= 1.0
    assert float(res_ref.entropy_map.min()) >= 0.0
    assert float(res_ref.entropy_map.max()) <= 1.0

    iou_mug_ref = _compute_iou(res_ref.masks["red_mug"], gt_mug_2)
    iou_box_ref = _compute_iou(res_ref.masks["blue_box"], gt_box_2)
    miou_2 = 0.5 * (iou_mug_ref + iou_box_ref)
    assert miou_2 > 0.75, f"Expected 2-class mIoU > 0.75, got {miou_2:.4f}"

    # Verify spatial refinement improves or preserves boundary IoU
    model.spatial_refine = False
    res_raw = model.segment(query_2)
    model.spatial_refine = True
    bnd_ref = 0.5 * (
        _compute_boundary_iou(res_ref.masks["red_mug"], gt_mug_2)
        + _compute_boundary_iou(res_ref.masks["blue_box"], gt_box_2)
    )
    bnd_raw = 0.5 * (
        _compute_boundary_iou(res_raw.masks["red_mug"], gt_mug_2)
        + _compute_boundary_iou(res_raw.masks["blue_box"], gt_box_2)
    )
    assert bnd_ref >= bnd_raw, (
        f"Expected guided refinement boundary IoU ({bnd_ref:.4f}) >= raw ({bnd_raw:.4f})"
    )

    # Verify entropy is meaningfully higher near object boundaries than in interiors
    interior = cv2.erode(gt_mug_2.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)) > 0
    bnd_band = (
        cv2.dilate(gt_mug_2.astype(np.uint8), np.ones((9, 9), dtype=np.uint8))
        - cv2.erode(gt_mug_2.astype(np.uint8), np.ones((9, 9), dtype=np.uint8))
    ) > 0
    assert float(res_ref.entropy_map[bnd_band].mean()) > float(res_ref.entropy_map[interior].mean())

    # 3. Online incremental registration of a 3rd novel unseen class without refitting from scratch
    support_novel: list[SupportSample] = []
    for shot, (pos, ang) in enumerate([((310, 260), 0.0), ((200, 320), 22.0)]):
        img3 = _make_tabletop_bg(seed=50 + shot)
        m_bottle = _draw_textured_object(img3, "green_bottle", *pos, scale=1.0, angle_deg=ang)
        support_novel.append(SupportSample(class_name="green_bottle", rgb=img3, mask=m_bottle))

    reg_latency_ms = model.register_object("green_bottle", support_novel)
    assert reg_latency_ms < 2000.0, f"Online registration took {reg_latency_ms:.1f}ms (>= 2000ms)"
    assert model.class_names == ["red_mug", "blue_box", "green_bottle"]

    # 4. Evaluate all 3 classes together to verify novel class accuracy & zero catastrophic forgetting
    query_3 = _make_tabletop_bg(seed=123)
    gt_mug_3 = _draw_textured_object(query_3, "red_mug", 145, 290, scale=0.95, angle_deg=-15.0)
    gt_box_3 = _draw_textured_object(query_3, "blue_box", 330, 195, scale=1.0, angle_deg=20.0)
    gt_bot_3 = _draw_textured_object(query_3, "green_bottle", 500, 315, scale=1.05, angle_deg=-10.0)

    res_3 = model.segment(query_3)
    assert res_3.log_likelihoods.shape == (4, 480, 640)
    iou_mug_3 = _compute_iou(res_3.masks["red_mug"], gt_mug_3)
    iou_box_3 = _compute_iou(res_3.masks["blue_box"], gt_box_3)
    iou_bot_3 = _compute_iou(res_3.masks["green_bottle"], gt_bot_3)
    miou_3 = (iou_mug_3 + iou_box_3 + iou_bot_3) / 3.0
    assert iou_mug_3 > 0.75, f"Old class red_mug degraded: IoU={iou_mug_3:.4f}"
    assert iou_box_3 > 0.75, f"Old class blue_box degraded: IoU={iou_box_3:.4f}"
    assert iou_bot_3 > 0.75, f"Novel class green_bottle IoU too low: {iou_bot_3:.4f}"
    assert miou_3 > 0.80, f"Expected 3-class mIoU > 0.80, got {miou_3:.4f}"

    # Verify instance proposal extraction on 3-object segmentation result
    proposals = extract_instance_masks(res_3, min_area_px=150)
    assert len(proposals) == 3
    assert {p.class_name for p in proposals} == {"red_mug", "blue_box", "green_bottle"}
    for prop in proposals:
        assert prop.mean_log_likelihoods.shape == (4,)
        assert prop.fg_log_likelihoods.shape == (3,)
        assert int(np.argmax(prop.fg_log_likelihoods)) + 1 == prop.class_index
        x1, y1, x2, y2 = prop.bbox_2d
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480

    # 5. Verify save_state / load_state round-trips identically
    state_file = model.save_state(tmp_path / "segment_gda_memory.npz")
    assert state_file.exists()

    reloaded = SegmentGDAPlus(extractor=dinov2_extractor)
    reloaded.load_state(state_file)
    assert reloaded.is_fitted
    assert reloaded.class_names == model.class_names

    res_reloaded = reloaded.segment(query_3)
    assert np.array_equal(res_reloaded.class_map, res_3.class_map)
    for cname in model.class_names:
        assert np.array_equal(res_reloaded.masks[cname], res_3.masks[cname])
    assert np.allclose(res_reloaded.confidence_map, res_3.confidence_map, atol=1e-6)
    assert np.allclose(res_reloaded.log_likelihoods, res_3.log_likelihoods, atol=1e-5)
    assert np.allclose(res_reloaded.entropy_map, res_3.entropy_map, atol=1e-6)


def test_extract_instance_masks_multi_instance_and_noise_filtering() -> None:
    h, w = 200, 300
    class_map = np.zeros((h, w), dtype=np.int32)
    # Two separate instances of class 1 ("red_mug")
    class_map[20:70, 20:70] = 1  # 2500 px
    class_map[110:170, 180:250] = 1  # 4200 px
    # Tiny noise speck of class 1 (< 150 px)
    class_map[5:10, 280:290] = 1  # 50 px
    # One instance of class 2 ("blue_box")
    class_map[120:170, 30:90] = 2  # 3000 px

    log_ll = np.zeros((3, h, w), dtype=np.float32)
    log_ll[0] = 0.0
    log_ll[1, class_map == 1] = 5.0
    log_ll[2, class_map == 2] = 6.0

    conf = np.full((h, w), 0.9, dtype=np.float32)
    ent = np.full((h, w), 0.1, dtype=np.float32)
    seg_res = SegmentationResult(
        class_names=["red_mug", "blue_box"],
        class_map=class_map,
        masks={"red_mug": class_map == 1, "blue_box": class_map == 2},
        confidence_map=conf,
        log_likelihoods=log_ll,
        entropy_map=ent,
    )

    proposals = extract_instance_masks(seg_res, min_area_px=150)
    assert len(proposals) == 3
    mug_props = [p for p in proposals if p.class_name == "red_mug"]
    box_props = [p for p in proposals if p.class_name == "blue_box"]
    assert len(mug_props) == 2
    assert len(box_props) == 1
    assert sorted(p.area_px for p in mug_props) == [2500, 4200]
    assert box_props[0].bbox_2d == (30, 120, 90, 170)
    assert np.allclose(box_props[0].fg_log_likelihoods, [0.0, 6.0], atol=1e-5)


def test_from_config_with_custom_extractor() -> None:
    cfg = load_config()
    lw = LightweightConvFeatureExtractor()
    seg = SegmentGDAPlus.from_config(cfg, extractor=lw)
    assert seg.extractor is lw
    assert seg.shrinkage_method == cfg["perception_2d"]["shrinkage_method"]
    assert seg.num_foreground_components == cfg["perception_2d"]["num_foreground_components"]
    assert seg.spatial_refine == cfg["perception_2d"]["spatial_refine"]

