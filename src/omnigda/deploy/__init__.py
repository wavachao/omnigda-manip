"""ONNX / TensorRT 10.9 deployment, runtime runners, and benchmarking for OmniGDA-Manip."""

from omnigda.deploy.benchmark import (
    DEFAULT_BENCHMARK_BACKENDS,
    BackendBenchmarkResult,
    benchmark_clip_visual,
    benchmark_dinov2,
    compute_fidelity_metrics,
    format_benchmark_markdown,
    run_full_benchmark,
    save_benchmark_report,
)
from omnigda.deploy.exporter import (
    DEFAULT_TRT_CACHE_DIR,
    build_trt_engine,
    clip_artifact_paths,
    dinov2_artifact_paths,
    export_clip_visual_onnx,
    export_dinov2_onnx,
    get_gpu_cache_tag,
    resolve_cache_dir,
)
from omnigda.deploy.runtime import (
    ONNXCLIPVisualHook,
    ONNXDINOv2Extractor,
    ONNXRuntimeRunner,
    TensorRTCLIPVisualHook,
    TensorRTDINOv2Extractor,
    TensorRTRunner,
    create_clip_encoder,
    create_dinov2_extractor,
)

__all__ = [
    "DEFAULT_BENCHMARK_BACKENDS",
    "DEFAULT_TRT_CACHE_DIR",
    "BackendBenchmarkResult",
    "ONNXCLIPVisualHook",
    "ONNXDINOv2Extractor",
    "ONNXRuntimeRunner",
    "TensorRTCLIPVisualHook",
    "TensorRTDINOv2Extractor",
    "TensorRTRunner",
    "benchmark_clip_visual",
    "benchmark_dinov2",
    "build_trt_engine",
    "clip_artifact_paths",
    "compute_fidelity_metrics",
    "create_clip_encoder",
    "create_dinov2_extractor",
    "dinov2_artifact_paths",
    "export_clip_visual_onnx",
    "export_dinov2_onnx",
    "format_benchmark_markdown",
    "get_gpu_cache_tag",
    "resolve_cache_dir",
    "run_full_benchmark",
    "save_benchmark_report",
]

