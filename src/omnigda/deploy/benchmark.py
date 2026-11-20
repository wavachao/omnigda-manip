"""Standardized latency, throughput, VRAM, and fidelity benchmarks for OmniGDA-Manip.

Compares `pytorch_fp32`, `pytorch_fp16`, `onnx_cuda`, and `tensorrt_fp16` across:
- DINOv2-ViT-S multi-scale dense feature extractor (`(1, 3, 518, 518)`)
- OpenCLIP ViT-B-32 visual tower (`(B, 3, 224, 224)`)
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import tensorrt as trt
import torch
import torch.nn.functional as F

from omnigda.deploy.exporter import (
    build_trt_engine,
    clip_artifact_paths,
    dinov2_artifact_paths,
    export_clip_visual_onnx,
    export_dinov2_onnx,
)
from omnigda.deploy.runtime import ONNXRuntimeRunner, TensorRTRunner
from omnigda.perception_2d.backbone import DINOv2MultiScaleExtractor

DEFAULT_BENCHMARK_BACKENDS: tuple[str, ...] = (
    "pytorch_fp32",
    "pytorch_fp16",
    "onnx_cuda",
    "tensorrt_fp16",
)


@dataclass(slots=True)
class BackendBenchmarkResult:
    """Measured latency, throughput, VRAM, and numerical fidelity for one backend."""

    task: str
    model_name: str
    backend: str
    active_provider: str
    input_shape: tuple[int, ...]
    warmup_iters: int
    benchmark_iters: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    fps: float
    peak_vram_mb: float
    cosine_similarity: float
    max_abs_diff: float
    speedup_vs_fp32: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_shape"] = list(self.input_shape)
        return data


def compute_fidelity_metrics(ref: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    """Compute `(mean_cosine_similarity, max_abs_diff)` between two output tensors.

    Supports:
    - 4D feature maps `(B, D, H, W)`: per-spatial-token cosine similarity across `D`.
    - 2D embeddings `(B, D)`: per-sample cosine similarity across `D`.
    - 1D vectors `(D,)`: scalar cosine similarity.
    """
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Output shape mismatch: ref={a.shape} vs test={b.shape}")

    max_abs_diff = float(np.max(np.abs(a - b)))

    if a.ndim == 4:
        # (B, D, H, W) -> (B * H * W, D)
        a_flat = np.transpose(a, (0, 2, 3, 1)).reshape(-1, a.shape[1])
        b_flat = np.transpose(b, (0, 2, 3, 1)).reshape(-1, b.shape[1])
    elif a.ndim == 2:
        a_flat, b_flat = a, b
    else:
        a_flat = a.reshape(1, -1)
        b_flat = b.reshape(1, -1)

    a_norm = np.linalg.norm(a_flat, axis=-1)
    b_norm = np.linalg.norm(b_flat, axis=-1)
    denom = np.maximum(a_norm * b_norm, 1e-12)
    cos = np.sum(a_flat * b_flat, axis=-1) / denom
    return float(np.mean(cos)), max_abs_diff


def _cuda_free_used_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_b, total_b = torch.cuda.mem_get_info()
    return float(total_b - free_b) / float(1024 * 1024)


def _reset_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        # Ensure CUDA context is initialized before baseline mem_get_info() deltas.
        _ = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _time_inference_loop(
    fn: Callable[[], Any],
    warmup_iters: int,
    benchmark_iters: int,
    sync_cuda: bool,
) -> tuple[float, float, float, float, float]:
    """Run warmup + timed loop and return `(mean_ms, p50_ms, p95_ms, min_ms, pytorch_peak_mb)`."""
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(int(warmup_iters)):
        fn()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(int(benchmark_iters)):
        t0 = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    pytorch_peak_mb = (
        float(torch.cuda.max_memory_allocated()) / float(1024 * 1024)
        if sync_cuda and torch.cuda.is_available()
        else 0.0
    )
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(np.min(arr)),
        pytorch_peak_mb,
    )


def _finalize_speedups(results: list[BackendBenchmarkResult]) -> list[BackendBenchmarkResult]:
    fp32_mean = next((r.mean_ms for r in results if r.backend == "pytorch_fp32"), None)
    if fp32_mean is not None and fp32_mean > 0.0:
        for r in results:
            r.speedup_vs_fp32 = float(fp32_mean / max(r.mean_ms, 1e-9))
    return results


def benchmark_dinov2(
    backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
    input_size: int = 518,
    feature_layers: Sequence[int] = (8, 10, 11),
    warmup_iters: int = 10,
    benchmark_iters: int = 50,
    backends: Sequence[str] = DEFAULT_BENCHMARK_BACKENDS,
    cache_dir: str | Path | None = None,
    seed: int = 42,
) -> list[BackendBenchmarkResult]:
    """Benchmark DINOv2 multi-scale dense feature extractor across requested backends."""
    rng = np.random.default_rng(seed)
    input_np = rng.standard_normal((1, 3, int(input_size), int(input_size))).astype(np.float32)
    input_shape = tuple(int(d) for d in input_np.shape)
    layers = tuple(int(i) for i in feature_layers)

    results: list[BackendBenchmarkResult] = []
    ref_out: np.ndarray | None = None

    need_ref = any(b != "pytorch_fp32" for b in backends)
    ordered_backends = list(backends)
    if need_ref and "pytorch_fp32" not in ordered_backends:
        ordered_backends = ["pytorch_fp32", *ordered_backends]

    for backend in ordered_backends:
        _reset_gpu_memory()
        base_used_mb = _cuda_free_used_mb()

        if backend in ("pytorch_fp32", "pytorch_fp16"):
            fp16 = backend == "pytorch_fp16"
            extractor = DINOv2MultiScaleExtractor(
                model_name=backbone_name,
                input_size=input_size,
                feature_layers=layers,
                device="cuda" if torch.cuda.is_available() else "cpu",
                fp16=fp16,
            )
            tensor = torch.from_numpy(input_np).to(extractor.device)

            def _run_pt(
                ext: DINOv2MultiScaleExtractor = extractor,
                x: torch.Tensor = tensor,
            ) -> torch.Tensor:
                with torch.inference_mode():
                    if ext.fp16:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            return ext.module(x)
                    return ext.module(x)

            out_np = _run_pt().detach().cpu().numpy().astype(np.float32)
            mean_ms, p50_ms, p95_ms, min_ms, pt_peak_mb = _time_inference_loop(
                _run_pt,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                sync_cuda=extractor.device.type == "cuda",
            )
            driver_delta_mb = max(0.0, _cuda_free_used_mb() - base_used_mb)
            peak_vram_mb = max(pt_peak_mb, driver_delta_mb)
            active_provider = extractor.backend_name
            del extractor, tensor

        elif backend == "onnx_cuda":
            onnx_path, _ = dinov2_artifact_paths(
                backbone_name=backbone_name,
                input_size=input_size,
                feature_layers=layers,
                cache_dir=cache_dir,
            )
            export_dinov2_onnx(
                backbone_name=backbone_name,
                input_size=input_size,
                feature_layers=layers,
                output_path=onnx_path,
            )
            runner = ONNXRuntimeRunner(onnx_path, prefer_cuda=True)

            def _run_ort(r: ONNXRuntimeRunner = runner) -> np.ndarray:
                res = r.infer(input_np)
                assert isinstance(res, np.ndarray)
                return res

            out_np = _run_ort().astype(np.float32, copy=False)
            mean_ms, p50_ms, p95_ms, min_ms, _ = _time_inference_loop(
                _run_ort,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                sync_cuda=runner.using_cuda,
            )
            peak_vram_mb = (
                max(0.0, _cuda_free_used_mb() - base_used_mb) if runner.using_cuda else 0.0
            )
            active_provider = runner.active_provider
            del runner

        elif backend == "tensorrt_fp16":
            onnx_path, engine_path = dinov2_artifact_paths(
                backbone_name=backbone_name,
                input_size=input_size,
                feature_layers=layers,
                fp16=True,
                cache_dir=cache_dir,
            )
            if not (engine_path.exists() and engine_path.stat().st_size > 0):
                export_dinov2_onnx(
                    backbone_name=backbone_name,
                    input_size=input_size,
                    feature_layers=layers,
                    output_path=onnx_path,
                )
                build_trt_engine(onnx_path=onnx_path, engine_path=engine_path, fp16=True)

            runner_trt = TensorRTRunner(engine_path, device="cuda:0")
            gpu_x = torch.from_numpy(input_np).to(runner_trt.device)

            def _run_trt(
                r: TensorRTRunner = runner_trt,
                x: torch.Tensor = gpu_x,
            ) -> dict[str, torch.Tensor]:
                return r.infer_torch(x)

            out_dict = _run_trt()
            out_np = out_dict[runner_trt.output_names[0]].detach().cpu().numpy().astype(np.float32)
            mean_ms, p50_ms, p95_ms, min_ms, pt_peak_mb = _time_inference_loop(
                _run_trt,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                sync_cuda=True,
            )
            driver_delta_mb = max(0.0, _cuda_free_used_mb() - base_used_mb)
            peak_vram_mb = max(pt_peak_mb + runner_trt.device_memory_mb, driver_delta_mb)
            active_provider = f"TensorRT-{trt.__version__}-FP16"
            del runner_trt, gpu_x

        else:
            raise ValueError(f"Unsupported benchmark backend {backend!r}")

        if backend == "pytorch_fp32":
            ref_out = out_np

        if backend not in backends:
            continue

        assert ref_out is not None
        cos_sim, max_diff = compute_fidelity_metrics(ref_out, out_np)
        results.append(
            BackendBenchmarkResult(
                task="dinov2",
                model_name=backbone_name,
                backend=backend,
                active_provider=active_provider,
                input_shape=input_shape,
                warmup_iters=int(warmup_iters),
                benchmark_iters=int(benchmark_iters),
                mean_ms=mean_ms,
                p50_ms=p50_ms,
                p95_ms=p95_ms,
                min_ms=min_ms,
                fps=1000.0 / max(mean_ms, 1e-9),
                peak_vram_mb=peak_vram_mb,
                cosine_similarity=cos_sim,
                max_abs_diff=max_diff,
            )
        )

    _reset_gpu_memory()
    return _finalize_speedups(results)


def benchmark_clip_visual(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    batch_size: int = 1,
    max_batch_size: int = 16,
    warmup_iters: int = 10,
    benchmark_iters: int = 50,
    backends: Sequence[str] = DEFAULT_BENCHMARK_BACKENDS,
    cache_dir: str | Path | None = None,
    seed: int = 42,
) -> list[BackendBenchmarkResult]:
    """Benchmark OpenCLIP visual tower across requested backends."""
    rng = np.random.default_rng(seed)
    input_np = rng.standard_normal((int(batch_size), 3, 224, 224)).astype(np.float32)
    input_shape = tuple(int(d) for d in input_np.shape)

    results: list[BackendBenchmarkResult] = []
    ref_out: np.ndarray | None = None

    need_ref = any(b != "pytorch_fp32" for b in backends)
    ordered_backends = list(backends)
    if need_ref and "pytorch_fp32" not in ordered_backends:
        ordered_backends = ["pytorch_fp32", *ordered_backends]

    for backend in ordered_backends:
        _reset_gpu_memory()
        base_used_mb = _cuda_free_used_mb()

        if backend in ("pytorch_fp32", "pytorch_fp16"):
            fp16 = backend == "pytorch_fp16" and torch.cuda.is_available()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            visual = model.visual.to(device).eval()
            tensor = torch.from_numpy(input_np).to(device)

            def _run_pt(
                v: torch.nn.Module = visual,
                x: torch.Tensor = tensor,
                use_fp16: bool = fp16,
            ) -> torch.Tensor:
                with torch.inference_mode():
                    if use_fp16:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            emb = v(x)
                    else:
                        emb = v(x)
                    return F.normalize(emb.float(), p=2, dim=-1, eps=1e-8)

            out_np = _run_pt().detach().cpu().numpy().astype(np.float32)
            mean_ms, p50_ms, p95_ms, min_ms, pt_peak_mb = _time_inference_loop(
                _run_pt,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                sync_cuda=device.type == "cuda",
            )
            driver_delta_mb = max(0.0, _cuda_free_used_mb() - base_used_mb)
            peak_vram_mb = max(pt_peak_mb, driver_delta_mb)
            active_provider = f"pytorch-{device.type}-{'fp16' if fp16 else 'fp32'}"
            del model, visual, tensor

        elif backend == "onnx_cuda":
            onnx_path, _ = clip_artifact_paths(
                model_name=model_name,
                pretrained=pretrained,
                max_batch_size=max_batch_size,
                cache_dir=cache_dir,
            )
            export_clip_visual_onnx(
                model_name=model_name,
                pretrained=pretrained,
                output_path=onnx_path,
            )
            runner = ONNXRuntimeRunner(onnx_path, prefer_cuda=True)

            def _run_ort(r: ONNXRuntimeRunner = runner) -> np.ndarray:
                res = r.infer(input_np)
                assert isinstance(res, np.ndarray)
                return res

            out_np = _run_ort().astype(np.float32, copy=False)
            mean_ms, p50_ms, p95_ms, min_ms, _ = _time_inference_loop(
                _run_ort,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                sync_cuda=runner.using_cuda,
            )
            peak_vram_mb = (
                max(0.0, _cuda_free_used_mb() - base_used_mb) if runner.using_cuda else 0.0
            )
            active_provider = runner.active_provider
            del runner

        elif backend == "tensorrt_fp16":
            onnx_path, engine_path = clip_artifact_paths(
                model_name=model_name,
                pretrained=pretrained,
                max_batch_size=max_batch_size,
                fp16=True,
                cache_dir=cache_dir,
            )
            if not (engine_path.exists() and engine_path.stat().st_size > 0):
                export_clip_visual_onnx(
                    model_name=model_name,
                    pretrained=pretrained,
                    output_path=onnx_path,
                )
                build_trt_engine(
                    onnx_path=onnx_path,
                    engine_path=engine_path,
                    fp16=True,
                    dynamic_shapes={
                        "input": (
                            (1, 3, 224, 224),
                            (min(4, max_batch_size), 3, 224, 224),
                            (max_batch_size, 3, 224, 224),
                        )
                    },
                )

            runner_trt = TensorRTRunner(engine_path, device="cuda:0")
            gpu_x = torch.from_numpy(input_np).to(runner_trt.device)

            def _run_trt(
                r: TensorRTRunner = runner_trt,
                x: torch.Tensor = gpu_x,
            ) -> dict[str, torch.Tensor]:
                return r.infer_torch(x)

            out_dict = _run_trt()
            out_np = out_dict[runner_trt.output_names[0]].detach().cpu().numpy().astype(np.float32)
            mean_ms, p50_ms, p95_ms, min_ms, pt_peak_mb = _time_inference_loop(
                _run_trt,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                sync_cuda=True,
            )
            driver_delta_mb = max(0.0, _cuda_free_used_mb() - base_used_mb)
            peak_vram_mb = max(pt_peak_mb + runner_trt.device_memory_mb, driver_delta_mb)
            active_provider = f"TensorRT-{trt.__version__}-FP16"
            del runner_trt, gpu_x

        else:
            raise ValueError(f"Unsupported benchmark backend {backend!r}")

        if backend == "pytorch_fp32":
            ref_out = out_np

        if backend not in backends:
            continue

        assert ref_out is not None
        cos_sim, max_diff = compute_fidelity_metrics(ref_out, out_np)
        results.append(
            BackendBenchmarkResult(
                task="clip_visual",
                model_name=f"{model_name} ({pretrained})",
                backend=backend,
                active_provider=active_provider,
                input_shape=input_shape,
                warmup_iters=int(warmup_iters),
                benchmark_iters=int(benchmark_iters),
                mean_ms=mean_ms,
                p50_ms=p50_ms,
                p95_ms=p95_ms,
                min_ms=min_ms,
                fps=1000.0 / max(mean_ms, 1e-9),
                peak_vram_mb=peak_vram_mb,
                cosine_similarity=cos_sim,
                max_abs_diff=max_diff,
            )
        )

    _reset_gpu_memory()
    return _finalize_speedups(results)


def format_benchmark_markdown(
    results: Sequence[BackendBenchmarkResult],
    title: str = "OmniGDA-Manip TensorRT / ONNX Deployment Benchmark",
) -> str:
    """Format benchmark results into a GitHub-flavored Markdown summary table."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    lines = [
        f"# {title}",
        "",
        f"- **GPU**: {gpu_name}",
        f"- **PyTorch**: `{torch.__version__}` | **TensorRT**: `{trt.__version__}`",
        "",
        "| Task | Model | Input Shape | Backend | Active Provider | Mean (ms) | P50 (ms) | P95 (ms) | FPS | Speedup | Peak VRAM (MB) | CosSim | MaxAbsDiff |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        shape_str = "x".join(str(d) for d in r.input_shape)
        lines.append(
            f"| {r.task} | {r.model_name} | `{shape_str}` | `{r.backend}` | `{r.active_provider}` "
            f"| {r.mean_ms:.2f} | {r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.fps:.1f} "
            f"| {r.speedup_vs_fp32:.2f}x | {r.peak_vram_mb:.1f} "
            f"| {r.cosine_similarity:.6f} | {r.max_abs_diff:.5f} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_benchmark_report(
    results: Sequence[BackendBenchmarkResult],
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    title: str = "OmniGDA-Manip TensorRT / ONNX Deployment Benchmark",
) -> None:
    """Persist benchmark results as JSON and/or Markdown files."""
    if json_path is not None:
        jp = Path(json_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "torch_version": torch.__version__,
            "tensorrt_version": str(trt.__version__),
            "results": [r.to_dict() for r in results],
        }
        jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if markdown_path is not None:
        mp = Path(markdown_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(format_benchmark_markdown(results, title=title), encoding="utf-8")


def run_full_benchmark(
    warmup_iters: int = 10,
    benchmark_iters: int = 50,
    clip_batch_sizes: Sequence[int] = (1,),
    backends: Sequence[str] = DEFAULT_BENCHMARK_BACKENDS,
    cache_dir: str | Path | None = None,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> list[BackendBenchmarkResult]:
    """Run standardized benchmarks on DINOv2-ViT-S (518x518) and OpenCLIP ViT-B-32 (224x224)."""
    all_results: list[BackendBenchmarkResult] = []
    all_results.extend(
        benchmark_dinov2(
            warmup_iters=warmup_iters,
            benchmark_iters=benchmark_iters,
            backends=backends,
            cache_dir=cache_dir,
        )
    )
    for bs in clip_batch_sizes:
        all_results.extend(
            benchmark_clip_visual(
                batch_size=int(bs),
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                backends=backends,
                cache_dir=cache_dir,
            )
        )

    save_benchmark_report(
        all_results,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    return all_results


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch FP32/FP16, ONNX Runtime, and TensorRT 10.9 FP16."
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Timed benchmark iterations")
    parser.add_argument(
        "--clip-batches",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark for OpenCLIP visual tower",
    )
    parser.add_argument("--json", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--md", type=str, default=None, help="Optional Markdown output path")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    results = run_full_benchmark(
        warmup_iters=args.warmup,
        benchmark_iters=args.iters,
        clip_batch_sizes=args.clip_batches,
        json_path=args.json,
        markdown_path=args.md,
    )
    print(format_benchmark_markdown(results))


if __name__ == "__main__":
    main()

