"""Real RTX 3060 GPU tests for ONNX export, TensorRT 10.9 engines, hooks, and benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import torch.nn.functional as F

from omnigda.core.config import load_config
from omnigda.core.schemas import SupportSample
from omnigda.deploy import (
    ONNXCLIPVisualHook,
    ONNXDINOv2Extractor,
    TensorRTCLIPVisualHook,
    TensorRTDINOv2Extractor,
    TensorRTRunner,
    build_trt_engine,
    clip_artifact_paths,
    compute_fidelity_metrics,
    create_clip_encoder,
    create_dinov2_extractor,
    dinov2_artifact_paths,
    export_clip_visual_onnx,
    export_dinov2_onnx,
    format_benchmark_markdown,
    run_full_benchmark,
)
from omnigda.perception_2d import DINOv2MultiScaleExtractor, SegmentGDAPlus
from omnigda.perception_3d.encoders import OpenCLIPMultimodalEncoder

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TensorRT 10.9 deployment tests require a CUDA GPU",
)


def _make_tabletop_bg(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Render a deterministic textured wood-grain tabletop background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grain = np.sin(yy * 0.12 + np.sin(xx * 0.025) * 2.5) * 12.0
    grad = (xx / w) * 18.0 - (yy / h) * 14.0
    noise = rng.normal(0.0, 3.0, size=(h, w, 3)).astype(np.float32)

    base = np.zeros((h, w, 3), dtype=np.float32)
    base[..., 0] = 175.0 + grain + grad
    base[..., 1] = 152.0 + grain * 0.8 + grad * 0.8
    base[..., 2] = 126.0 + grain * 0.6 + grad * 0.6
    return np.clip(base + noise, 0.0, 255.0).astype(np.uint8)


def _draw_object(
    img: np.ndarray,
    kind: str,
    cx: float,
    cy: float,
    scale: float = 1.0,
) -> np.ndarray:
    """Composite a shaded synthetic object onto `img` and return its boolean mask."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy

    if kind == "red_mug":
        rx, ry = 52.0 * scale, 60.0 * scale
        inside = (np.abs(dx) <= rx) & (np.abs(dy) <= ry)
        shade = np.clip(1.0 - 0.35 * (dx / max(rx, 1.0)), 0.55, 1.15)
        stripe = (np.abs(dy) < 10.0 * scale).astype(np.float32)
        col = np.zeros_like(img, dtype=np.float32)
        col[..., 0] = (210.0 * (1.0 - stripe) + 245.0 * stripe) * shade
        col[..., 1] = (35.0 * (1.0 - stripe) + 238.0 * stripe) * shade
        col[..., 2] = (40.0 * (1.0 - stripe) + 232.0 * stripe) * shade
        img[inside] = np.clip(col[inside], 0.0, 255.0).astype(np.uint8)
        mask[inside] = True
    elif kind == "blue_box":
        rx, ry = 62.0 * scale, 46.0 * scale
        inside = (np.abs(dx) <= rx) & (np.abs(dy) <= ry)
        checker = (((np.floor(dx / 14.0) + np.floor(dy / 14.0)) % 2) == 0).astype(np.float32)
        col = np.zeros_like(img, dtype=np.float32)
        col[..., 0] = 28.0 + 38.0 * checker
        col[..., 1] = 88.0 + 52.0 * checker
        col[..., 2] = 200.0 + 45.0 * checker
        img[inside] = np.clip(col[inside], 0.0, 255.0).astype(np.uint8)
        mask[inside] = True
    else:
        raise ValueError(f"Unknown synthetic object kind: {kind}")

    return mask


def _binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    inter = int(np.sum(pred & target))
    union = int(np.sum(pred | target))
    return float(inter / max(union, 1))


def test_dinov2_onnx_and_trt_export_cache() -> None:
    """Export DINOv2-ViT-S to ONNX and build a cached TensorRT 10.9 FP16 engine."""
    onnx_path, engine_path = dinov2_artifact_paths()
    exported = export_dinov2_onnx(output_path=onnx_path)
    assert exported.exists() and exported.stat().st_size > 1_000_000
    onnx.checker.check_model(str(exported))

    built = build_trt_engine(onnx_path=exported, engine_path=engine_path, fp16=True)
    assert built.exists() and built.stat().st_size > 1_000_000

    runner = TensorRTRunner(built, device="cuda:0")
    assert runner.input_names == ["input"]
    assert runner.output_names == ["features"]
    assert runner.input_shapes["input"] == (1, 3, 518, 518)
    assert runner.output_shapes["features"] == (1, 384, 37, 37)


def test_dinov2_extractors_numerical_parity() -> None:
    """Verify ONNX and TensorRT FP16 DINOv2 extractors match PyTorch FP32 (CosSim > 0.995)."""
    scene = _make_tabletop_bg(480, 640, seed=7)
    _draw_object(scene, "red_mug", cx=200.0, cy=250.0)
    _draw_object(scene, "blue_box", cx=440.0, cy=260.0)

    cfg = load_config()
    pt_ext = create_dinov2_extractor(cfg, backend="pytorch", fp16=False)
    onnx_ext = create_dinov2_extractor(cfg, backend="onnx")
    trt_ext = create_dinov2_extractor(cfg, backend="tensorrt", fp16=True)

    assert isinstance(pt_ext, DINOv2MultiScaleExtractor)
    assert isinstance(onnx_ext, ONNXDINOv2Extractor)
    assert isinstance(trt_ext, TensorRTDINOv2Extractor)
    assert trt_ext.feature_dim == pt_ext.feature_dim == 384
    assert onnx_ext.feature_dim == pt_ext.feature_dim == 384

    feat_pt = pt_ext.extract(scene)
    feat_onnx = onnx_ext.extract(scene)
    feat_trt = trt_ext.extract(scene)

    assert feat_pt.shape == (384, 37, 37)
    assert feat_onnx.shape == feat_pt.shape
    assert feat_trt.shape == feat_pt.shape

    # Verify per-token unit L2 norms across channel axis
    np.testing.assert_allclose(np.linalg.norm(feat_trt, axis=0), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(feat_onnx, axis=0), 1.0, atol=1e-4)

    cos_onnx, diff_onnx = compute_fidelity_metrics(feat_pt[None, ...], feat_onnx[None, ...])
    cos_trt, diff_trt = compute_fidelity_metrics(feat_pt[None, ...], feat_trt[None, ...])
    print(
        f"\n[DINOv2 Parity] ONNX CosSim={cos_onnx:.6f} MaxDiff={diff_onnx:.5f} | "
        f"TRT-FP16 CosSim={cos_trt:.6f} MaxDiff={diff_trt:.5f}"
    )

    assert cos_onnx > 0.999
    assert cos_trt > 0.995


def test_dinov2_trt_and_onnx_dropin_segment_gda_plus() -> None:
    """Verify TensorRTDINOv2Extractor and ONNXDINOv2Extractor drop directly into SegmentGDAPlus."""
    support_red = _make_tabletop_bg(480, 640, seed=11)
    mask_red_sup = _draw_object(support_red, "red_mug", cx=310.0, cy=245.0)

    support_blue = _make_tabletop_bg(480, 640, seed=12)
    mask_blue_sup = _draw_object(support_blue, "blue_box", cx=325.0, cy=255.0)

    supports = [
        SupportSample(class_name="red_mug", rgb=support_red, mask=mask_red_sup),
        SupportSample(class_name="blue_box", rgb=support_blue, mask=mask_blue_sup),
    ]

    query = _make_tabletop_bg(480, 640, seed=99)
    gt_red = _draw_object(query, "red_mug", cx=190.0, cy=250.0, scale=0.95)
    gt_blue = _draw_object(query, "blue_box", cx=455.0, cy=265.0, scale=1.05)

    pt_seg = SegmentGDAPlus(extractor=DINOv2MultiScaleExtractor(device="cuda", fp16=False))
    trt_seg = SegmentGDAPlus(extractor=TensorRTDINOv2Extractor(fp16=True))
    onnx_seg = SegmentGDAPlus(extractor=ONNXDINOv2Extractor())

    pt_seg.fit(supports)
    trt_seg.fit(supports)
    onnx_seg.fit(supports)

    res_pt = pt_seg.segment(query)
    res_trt = trt_seg.segment(query)
    res_onnx = onnx_seg.segment(query)

    iou_red_trt = _binary_iou(res_trt.masks["red_mug"], gt_red)
    iou_blue_trt = _binary_iou(res_trt.masks["blue_box"], gt_blue)
    iou_red_onnx = _binary_iou(res_onnx.masks["red_mug"], gt_red)
    iou_blue_onnx = _binary_iou(res_onnx.masks["blue_box"], gt_blue)
    agreement = float(np.mean(res_trt.class_map == res_pt.class_map))

    print(
        f"\n[SegmentGDA+ Drop-In] TRT IoU(red={iou_red_trt:.3f}, blue={iou_blue_trt:.3f}) | "
        f"ONNX IoU(red={iou_red_onnx:.3f}, blue={iou_blue_onnx:.3f}) | "
        f"TRT vs PyTorch pixel agreement={agreement * 100.0:.2f}%"
    )

    assert iou_red_trt >= 0.75
    assert iou_blue_trt >= 0.75
    assert iou_red_onnx >= 0.75
    assert iou_blue_onnx >= 0.75
    assert agreement >= 0.98


def test_clip_visual_onnx_and_trt_dynamic_batches_and_hook() -> None:
    """Verify OpenCLIP ViT-B-32 ONNX & TRT FP16 parity on batch=1, batch=4, and encoder hook."""
    onnx_path, engine_path = clip_artifact_paths(max_batch_size=16, fp16=True)
    exported = export_clip_visual_onnx(output_path=onnx_path)
    assert exported.exists()
    onnx.checker.check_model(str(exported))

    # Dynamic ONNX model without dynamic_shapes must raise a clear ValueError
    with pytest.raises(ValueError, match="dynamic_shapes"):
        build_trt_engine(
            onnx_path=exported,
            engine_path=engine_path.with_name("should_not_exist.engine"),
            fp16=True,
            dynamic_shapes=None,
            overwrite=True,
        )

    trt_hook = TensorRTCLIPVisualHook(fp16=True, max_batch_size=16)
    onnx_hook = ONNXCLIPVisualHook()
    assert trt_hook.engine_path.exists()
    assert trt_hook.embed_dim == 512
    assert onnx_hook.embed_dim == 512

    # Native PyTorch reference encoder (FP32)
    pt_encoder = OpenCLIPMultimodalEncoder(device="cuda", fp16=False)
    rng = np.random.default_rng(123)

    for batch_size in (1, 4):
        batch_np = rng.standard_normal((batch_size, 3, 224, 224)).astype(np.float32)
        with torch.inference_mode():
            pt_emb = pt_encoder.model.encode_image(torch.from_numpy(batch_np).cuda()).float()
            pt_emb = F.normalize(pt_emb, p=2, dim=-1, eps=1e-8).cpu().numpy()

        trt_emb = trt_hook(batch_np)
        onnx_emb = onnx_hook(batch_np)

        assert trt_emb.shape == (batch_size, 512)
        assert onnx_emb.shape == (batch_size, 512)
        np.testing.assert_allclose(np.linalg.norm(trt_emb, axis=-1), 1.0, atol=1e-4)

        cos_trt, diff_trt = compute_fidelity_metrics(pt_emb, trt_emb)
        cos_onnx, diff_onnx = compute_fidelity_metrics(pt_emb, onnx_emb)
        print(
            f"\n[OpenCLIP B={batch_size}] TRT-FP16 CosSim={cos_trt:.6f} MaxDiff={diff_trt:.5f} | "
            f"ONNX CosSim={cos_onnx:.6f} MaxDiff={diff_onnx:.5f}"
        )
        assert cos_trt > 0.995
        assert cos_onnx > 0.999

    # Verify automatic chunking when batch > max_batch_size (18 > 16)
    large_batch = rng.standard_normal((18, 3, 224, 224)).astype(np.float32)
    large_out = trt_hook(large_batch)
    assert large_out.shape == (18, 512)

    # Verify plug-in integration with OpenCLIPMultimodalEncoder.set_image_encoder_hook
    scene = _make_tabletop_bg(480, 640, seed=5)
    mug_mask = _draw_object(scene, "red_mug", cx=300.0, cy=240.0)

    pt_crop_emb = pt_encoder.encode_image_crop(scene, mask=mug_mask)

    pt_encoder.set_image_encoder_hook(trt_hook)
    trt_crop_emb = pt_encoder.encode_image_crop(scene, mask=mug_mask)

    pt_encoder.set_image_encoder_hook(onnx_hook)
    onnx_crop_emb = pt_encoder.encode_image_crop(scene, mask=mug_mask)

    cos_hook_trt, _ = compute_fidelity_metrics(pt_crop_emb, trt_crop_emb)
    cos_hook_onnx, _ = compute_fidelity_metrics(pt_crop_emb, onnx_crop_emb)
    print(
        f"[OpenCLIP Hook Crop] TRT CosSim={cos_hook_trt:.6f} | ONNX CosSim={cos_hook_onnx:.6f}"
    )
    assert cos_hook_trt > 0.995
    assert cos_hook_onnx > 0.999

    # Also test create_clip_encoder factory with backend='tensorrt'
    cfg = load_config()
    factory_trt_clip = create_clip_encoder(cfg, backend="tensorrt", fp16=True)
    factory_crop_emb = factory_trt_clip.encode_image_crop(scene, mask=mug_mask)
    cos_factory, _ = compute_fidelity_metrics(pt_crop_emb, factory_crop_emb)
    assert cos_factory > 0.995

    with pytest.raises(ValueError, match="Unsupported deployment backend"):
        create_dinov2_extractor(cfg, backend="unknown_backend")


def test_deploy_benchmark_suite_and_reports(tmp_path: Path) -> None:
    """Run standardized 50-iter RTX 3060 benchmark and verify speedup, fidelity, and reports."""
    json_out = tmp_path / "bench.json"
    md_out = tmp_path / "bench.md"

    results = run_full_benchmark(
        warmup_iters=10,
        benchmark_iters=50,
        clip_batch_sizes=(1, 4),
        json_path=json_out,
        markdown_path=md_out,
    )

    table = format_benchmark_markdown(results)
    print("\n" + table)

    assert len(results) == 12  # 4 backends x (dinov2 + clip B=1 + clip B=4)
    assert json_out.exists() and md_out.exists()

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 12
    assert "TensorRT" in md_out.read_text(encoding="utf-8")

    for r in results:
        assert r.mean_ms > 0.0
        assert r.p50_ms > 0.0
        assert r.p95_ms >= r.min_ms > 0.0
        assert r.fps > 0.0
        assert r.cosine_similarity > 0.995

    dino_trt = next(r for r in results if r.task == "dinov2" and r.backend == "tensorrt_fp16")
    clip_b1_trt = next(
        r
        for r in results
        if r.task == "clip_visual" and r.input_shape[0] == 1 and r.backend == "tensorrt_fp16"
    )
    clip_b4_trt = next(
        r
        for r in results
        if r.task == "clip_visual" and r.input_shape[0] == 4 and r.backend == "tensorrt_fp16"
    )

    assert dino_trt.speedup_vs_fp32 > 1.5
    assert clip_b1_trt.speedup_vs_fp32 > 1.5
    assert clip_b4_trt.speedup_vs_fp32 > 2.0

