"""RTX 3060 TensorRT 10.9 / ONNX / PyTorch Deployment & End-to-End Perception Benchmark.

Benchmarks:
1. Isolated backbone inference (`pytorch_fp32`, `pytorch_fp16`, `onnx_cuda`, `tensorrt_fp16`):
   - DINOv2-ViT-S multi-scale dense feature extractor `(1, 3, 518, 518)`
   - OpenCLIP ViT-B-32 visual tower `(1, 3, 224, 224)` and `(4, 3, 224, 224)`
2. End-to-end `OmniGDAAgent.perceive_frame(...)` on real ManiSkill 3 cluttered tabletop RGB-D
   frames (`seg2d_ms`, `gmda3d_ms`, `scene_graph_ms`, `total_ms`, `FPS`, `peak_vram_mb`)
   comparing `backend="pytorch"` (FP16) vs `backend="tensorrt"` (TensorRT 10.9 FP16).

Writes structured JSON results to `docs/results/deploy_benchmark.json`.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

from omnigda.agent.pipeline import OmniGDAAgent
from omnigda.core.config import load_config
from omnigda.core.schemas import RGBDFrame, SupportSample
from omnigda.deploy.benchmark import (
    DEFAULT_BENCHMARK_BACKENDS,
    format_benchmark_markdown,
    run_full_benchmark,
)
from omnigda.sim.clutter_env import (
    DEFAULT_BASE_OBJECTS,
    HELD_OUT_NOVEL_OBJECTS,
    OBJECT_CATALOG,
    ClutteredTabletopSim,
)
from omnigda.sim.templates import capture_few_shot_templates


def _cuda_used_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_b, total_b = torch.cuda.mem_get_info()
    return float(total_b - free_b) / float(1024 * 1024)


def _reset_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        _ = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _collect_sim_frames_and_templates(
    num_frames: int = 5,
    base_seed: int = 42,
    num_shots: int = 3,
) -> tuple[list[RGBDFrame], dict[str, list[SupportSample]], list[str], list[str]]:
    """Capture real ManiSkill 3 RGB-D frames and few-shot templates, then close SAPIEN."""
    base_names = list(DEFAULT_BASE_OBJECTS[:6])
    novel_names = list(HELD_OUT_NOVEL_OBJECTS)
    all_names = base_names + novel_names

    templates = capture_few_shot_templates(
        object_names=all_names,
        num_shots=num_shots,
        seed=base_seed,
    )

    frames: list[RGBDFrame] = []
    sim = ClutteredTabletopSim(
        num_objects=6,
        include_novel=True,
        seed=base_seed,
        sim_backend="cpu",
    )
    try:
        for i in range(num_frames):
            sim.reset(seed=base_seed + i)
            frame, _ = sim.capture_rgbd("base_camera")
            frames.append(frame)
    finally:
        sim.close()

    return frames, templates, base_names, novel_names


def benchmark_end_to_end_perception(
    frames: list[RGBDFrame],
    templates: dict[str, list[SupportSample]],
    base_names: list[str],
    novel_names: list[str],
    backends: tuple[str, ...] = ("pytorch", "tensorrt"),
    warmup_iters: int = 5,
    benchmark_iters: int = 25,
) -> dict[str, dict[str, Any]]:
    """Benchmark full OmniGDAAgent.perceive_frame latency & VRAM across backends."""
    base_cfg = load_config()
    e2e_results: dict[str, dict[str, Any]] = {}

    for backend in backends:
        _reset_gpu()
        base_vram_mb = _cuda_used_mb()

        cfg = copy.deepcopy(base_cfg)
        cfg["perception_2d"]["backend"] = backend
        cfg["perception_2d"]["fp16"] = True

        agent = OmniGDAAgent(config=cfg)

        # Register 6 base classes first, then time online registration of held-out novel objects
        base_reg_ms: list[float] = []
        for name in base_names:
            ms = agent.register_object_few_shot(
                name, templates[name], text_prompt=OBJECT_CATALOG[name].description
            )
            base_reg_ms.append(ms)

        novel_reg_ms: dict[str, float] = {}
        for name in novel_names:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = agent.register_object_few_shot(
                name, templates[name], text_prompt=OBJECT_CATALOG[name].description
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            novel_reg_ms[name] = float(ms)

        # Warmup iterations
        for i in range(warmup_iters):
            _ = agent.perceive_frame(frames[i % len(frames)])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        seg2d_list: list[float] = []
        gmda3d_list: list[float] = []
        sg_list: list[float] = []
        total_list: list[float] = []
        det_counts: list[int] = []

        for i in range(benchmark_iters):
            frame = frames[i % len(frames)]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = agent.perceive_frame(frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - t0) * 1000.0

            seg2d_list.append(float(out.seg2d_ms))
            gmda3d_list.append(float(out.gmda3d_ms))
            sg_list.append(float(out.scene_graph_ms))
            total_list.append(float(wall_ms))
            det_counts.append(len(out.detections))

        pt_peak_mb = (
            float(torch.cuda.max_memory_allocated()) / float(1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        driver_delta_mb = max(0.0, _cuda_used_mb() - base_vram_mb)
        peak_vram_mb = max(pt_peak_mb, driver_delta_mb)

        seg_arr = np.asarray(seg2d_list, dtype=np.float64)
        g3d_arr = np.asarray(gmda3d_list, dtype=np.float64)
        sg_arr = np.asarray(sg_list, dtype=np.float64)
        tot_arr = np.asarray(total_list, dtype=np.float64)

        e2e_results[backend] = {
            "backend": backend,
            "num_frames": len(frames),
            "warmup_iters": int(warmup_iters),
            "benchmark_iters": int(benchmark_iters),
            "num_registered_classes": len(base_names) + len(novel_names),
            "avg_detections_per_frame": float(np.mean(det_counts)),
            "seg2d_mean_ms": float(np.mean(seg_arr)),
            "seg2d_p50_ms": float(np.percentile(seg_arr, 50)),
            "seg2d_p95_ms": float(np.percentile(seg_arr, 95)),
            "gmda3d_mean_ms": float(np.mean(g3d_arr)),
            "gmda3d_p50_ms": float(np.percentile(g3d_arr, 50)),
            "gmda3d_p95_ms": float(np.percentile(g3d_arr, 95)),
            "scene_graph_mean_ms": float(np.mean(sg_arr)),
            "total_mean_ms": float(np.mean(tot_arr)),
            "total_p50_ms": float(np.percentile(tot_arr, 50)),
            "total_p95_ms": float(np.percentile(tot_arr, 95)),
            "fps": float(1000.0 / max(float(np.mean(tot_arr)), 1e-9)),
            "peak_vram_mb": float(peak_vram_mb),
            "novel_online_registration_ms": novel_reg_ms,
            "novel_online_registration_mean_ms": float(np.mean(list(novel_reg_ms.values()))),
            "base_registration_mean_ms": float(np.mean(base_reg_ms)),
        }

        del agent
        _reset_gpu()

    if "pytorch" in e2e_results and "tensorrt" in e2e_results:
        pt = e2e_results["pytorch"]
        tr = e2e_results["tensorrt"]
        tr["speedup_vs_pytorch_total"] = float(pt["total_mean_ms"] / max(tr["total_mean_ms"], 1e-9))
        tr["speedup_vs_pytorch_seg2d"] = float(pt["seg2d_mean_ms"] / max(tr["seg2d_mean_ms"], 1e-9))
        tr["speedup_vs_pytorch_gmda3d"] = float(
            pt["gmda3d_mean_ms"] / max(tr["gmda3d_mean_ms"], 1e-9)
        )
        pt["speedup_vs_pytorch_total"] = 1.0
        pt["speedup_vs_pytorch_seg2d"] = 1.0
        pt["speedup_vs_pytorch_gmda3d"] = 1.0

    return e2e_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RTX 3060 deployment + end-to-end OmniGDAAgent perception benchmark."
    )
    parser.add_argument("--warmup", type=int, default=15, help="Backbone warmup iterations")
    parser.add_argument("--iters", type=int, default=60, help="Backbone benchmark iterations")
    parser.add_argument(
        "--e2e-iters", type=int, default=25, help="End-to-end perceive_frame iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/results/deploy_benchmark.json",
        help="Path to write JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[1/2] Running isolated backbone benchmarks (DINOv2-ViT-S 518x518 & OpenCLIP ViT-B-32 B=1,4)...")
    backbone_results = run_full_benchmark(
        warmup_iters=args.warmup,
        benchmark_iters=args.iters,
        clip_batch_sizes=(1, 4),
        backends=DEFAULT_BENCHMARK_BACKENDS,
    )
    print(format_benchmark_markdown(backbone_results))

    print("[2/2] Running end-to-end OmniGDAAgent.perceive_frame() on real ManiSkill 3 RGB-D frames...")
    frames, templates, base_names, novel_names = _collect_sim_frames_and_templates(
        num_frames=5, base_seed=42, num_shots=3
    )
    e2e_results = benchmark_end_to_end_perception(
        frames=frames,
        templates=templates,
        base_names=base_names,
        novel_names=novel_names,
        backends=("pytorch", "tensorrt"),
        warmup_iters=5,
        benchmark_iters=args.e2e_iters,
    )

    payload: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "tensorrt_version": str(trt.__version__),
            "onnxruntime_version": str(ort.__version__),
        },
        "backbone_benchmarks": [r.to_dict() for r in backbone_results],
        "end_to_end_perception": e2e_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== End-to-End OmniGDAAgent.perceive_frame() Summary ===")
    print(
        "| Backend | Seg2D (ms) | GMDA3D (ms) | SceneGraph (ms) | Total (ms) | P95 (ms) | FPS | Speedup | Peak VRAM (MB) | Avg Dets |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, res in e2e_results.items():
        print(
            f"| `{name}` | {res['seg2d_mean_ms']:.2f} | {res['gmda3d_mean_ms']:.2f} "
            f"| {res['scene_graph_mean_ms']:.2f} | {res['total_mean_ms']:.2f} "
            f"| {res['total_p95_ms']:.2f} | {res['fps']:.2f} "
            f"| {res.get('speedup_vs_pytorch_total', 1.0):.2f}x "
            f"| {res['peak_vram_mb']:.1f} | {res['avg_detections_per_frame']:.2f} |"
        )
    print(f"\nSaved deployment benchmark JSON to: {out_path}")


if __name__ == "__main__":
    main()
