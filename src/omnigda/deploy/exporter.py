"""ONNX export and TensorRT 10.9 engine builder utilities for OmniGDA-Manip.

Supports:
- DINOv2 multi-scale dense feature extractor (`_DINOv2MultiScaleModule`) export to ONNX
  and FP16/FP32 TensorRT 10.9 engine compilation.
- OpenCLIP visual tower export to ONNX with dynamic batch axis (`B in [1, 16]`)
  and FP16/FP32 TensorRT 10.9 engine compilation with optimization profiles.
- Deterministic caching of ONNX graphs and serialized `.engine` files under
  `/cache/trt_engines/` (tagged by GPU compute capability and TensorRT version).
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import onnx
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F

from omnigda.perception_2d.backbone import _DINOv2MultiScaleModule

DEFAULT_TRT_CACHE_DIR = Path("/cache/trt_engines")

ShapeSpec = (
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    | Mapping[str, Sequence[int]]
)


def _sanitize_token(text: str) -> str:
    """Convert model/tag strings into filesystem-safe tokens."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("_") or "model"


def resolve_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve and create a writable cache directory for ONNX & TensorRT artifacts."""
    if cache_dir is not None:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    env_dir = os.environ.get("OMNIGDA_TRT_CACHE_DIR")
    candidates = [
        Path(env_dir) if env_dir else DEFAULT_TRT_CACHE_DIR,
        Path.home() / ".cache" / "omnigda" / "trt_engines",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.touch(exist_ok=True)
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    raise PermissionError(
        f"Could not find a writable TensorRT cache directory among: {candidates}"
    )


def get_gpu_cache_tag() -> str:
    """Return a deterministic tag encoding GPU compute capability and TensorRT version."""
    trt_ver = _sanitize_token(str(getattr(trt, "__version__", "unknown")))
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm{major}{minor}_trt{trt_ver}"
    return f"cpu_trt{trt_ver}"


def dinov2_artifact_paths(
    backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
    input_size: int = 518,
    feature_layers: Sequence[int] = (8, 10, 11),
    fusion_mode: Literal["mean", "concat"] = "mean",
    fp16: bool = True,
    cache_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return deterministic `(onnx_path, engine_path)` cache locations for DINOv2."""
    root = resolve_cache_dir(cache_dir)
    layers_tag = "-".join(str(int(i)) for i in feature_layers)
    base = (
        f"dinov2_{_sanitize_token(backbone_name)}"
        f"_s{int(input_size)}_L{layers_tag}_{fusion_mode}"
    )
    prec = "fp16" if fp16 else "fp32"
    onnx_path = root / f"{base}.onnx"
    engine_path = root / f"{base}_{prec}_{get_gpu_cache_tag()}.engine"
    return onnx_path, engine_path


def clip_artifact_paths(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    max_batch_size: int = 16,
    fp16: bool = True,
    cache_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return deterministic `(onnx_path, engine_path)` cache locations for OpenCLIP visual."""
    root = resolve_cache_dir(cache_dir)
    base = f"openclip_visual_{_sanitize_token(model_name)}_{_sanitize_token(pretrained)}"
    prec = "fp16" if fp16 else "fp32"
    onnx_path = root / f"{base}.onnx"
    engine_path = root / f"{base}_bmax{int(max_batch_size)}_{prec}_{get_gpu_cache_tag()}.engine"
    return onnx_path, engine_path


class _CLIPVisualExportWrapper(nn.Module):
    """Export-friendly wrapper around an OpenCLIP visual tower returning L2-normalized embeddings."""

    def __init__(self, visual: nn.Module, normalize: bool = True) -> None:
        super().__init__()
        self.visual = visual
        self.normalize = bool(normalize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.visual(x).float()
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1, eps=1e-8)
        return emb


def _atomic_onnx_export(
    module: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    """Export a PyTorch module to ONNX atomically and verify with `onnx.checker`."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with torch.inference_mode():
            torch.onnx.export(
                module,
                (dummy_input,),
                str(tmp_path),
                export_params=True,
                opset_version=int(opset),
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
        onnx.checker.check_model(str(tmp_path))
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output_path


def export_dinov2_onnx(
    backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
    input_size: int = 518,
    feature_layers: Sequence[int] = (8, 10, 11),
    output_path: str | Path | None = None,
    opset: int = 17,
    fusion_mode: Literal["mean", "concat"] = "mean",
    pretrained: bool = True,
    dynamic_batch: bool = False,
    overwrite: bool = False,
) -> Path:
    """Export `_DINOv2MultiScaleModule` (`(1, 3, H, W) -> (1, D, Hf, Wf)`) to ONNX.

    Args:
        backbone_name: `timm` DINOv2 model identifier.
        input_size: Square input resolution (must be a multiple of ViT patch size).
        feature_layers: Transformer block indices to fuse.
        output_path: Destination `.onnx` path (defaults to `/cache/trt_engines/...`).
        opset: ONNX opset version (default 17).
        fusion_mode: Multi-layer fusion strategy (`"mean"` or `"concat"`).
        pretrained: Whether to load pretrained weights from timm/HF cache.
        dynamic_batch: If True, mark batch dimension 0 as dynamic.
        overwrite: If False and `output_path` exists and passes validation, reuse it.

    Returns:
        Path to the validated ONNX model file.
    """
    import timm

    layers = tuple(int(i) for i in feature_layers)
    if not layers:
        raise ValueError("feature_layers must contain at least one transformer block index")

    if output_path is None:
        dest, _ = dinov2_artifact_paths(
            backbone_name=backbone_name,
            input_size=input_size,
            feature_layers=layers,
            fusion_mode=fusion_mode,
        )
    else:
        dest = Path(output_path)

    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        onnx.checker.check_model(str(dest))
        return dest

    vit = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        img_size=int(input_size),
        dynamic_img_size=False,
    )
    patch_size = vit.patch_embed.patch_size
    patch_h = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)
    if int(input_size) <= 0 or int(input_size) % patch_h != 0:
        raise ValueError(
            f"input_size ({input_size}) must be a positive multiple of patch_size ({patch_h})"
        )

    num_blocks = len(vit.blocks)
    for idx in layers:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(
                f"feature_layer index {idx} out of range for {backbone_name} with {num_blocks} blocks"
            )

    vit.eval()
    module = _DINOv2MultiScaleModule(
        vit=vit,
        feature_layers=layers,
        fusion_mode=fusion_mode,
    ).cpu().eval()

    dummy = torch.randn(1, 3, int(input_size), int(input_size), dtype=torch.float32)
    dynamic_axes = (
        {"input": {0: "batch"}, "features": {0: "batch"}} if dynamic_batch else None
    )

    return _atomic_onnx_export(
        module=module,
        dummy_input=dummy,
        output_path=dest,
        input_names=["input"],
        output_names=["features"],
        opset=opset,
        dynamic_axes=dynamic_axes,
    )


def export_clip_visual_onnx(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    output_path: str | Path | None = None,
    opset: int = 17,
    normalize: bool = True,
    overwrite: bool = False,
) -> Path:
    """Export `open_clip` visual tower (`(B, 3, 224, 224) -> (B, D)`) with dynamic batch axis.

    Args:
        model_name: OpenCLIP model architecture (e.g. `"ViT-B-32"`).
        pretrained: OpenCLIP pretrained checkpoint tag (e.g. `"laion2b_s34b_b79k"`).
        output_path: Destination `.onnx` path (defaults to `/cache/trt_engines/...`).
        opset: ONNX opset version (default 17).
        normalize: If True, L2-normalize visual embeddings inside the exported graph.
        overwrite: If False and `output_path` exists and passes validation, reuse it.

    Returns:
        Path to the validated ONNX model file.
    """
    import open_clip

    if output_path is None:
        dest, _ = clip_artifact_paths(model_name=model_name, pretrained=pretrained)
    else:
        dest = Path(output_path)

    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        onnx.checker.check_model(str(dest))
        return dest

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    visual = model.visual.cpu().eval()

    img_size = getattr(visual, "image_size", 224)
    if isinstance(img_size, tuple | list):
        h, w = int(img_size[0]), int(img_size[1])
    else:
        h = w = int(img_size)

    wrapper = _CLIPVisualExportWrapper(visual=visual, normalize=normalize).cpu().eval()
    dummy = torch.randn(2, 3, h, w, dtype=torch.float32)

    return _atomic_onnx_export(
        module=wrapper,
        dummy_input=dummy,
        output_path=dest,
        input_names=["input"],
        output_names=["embedding"],
        opset=opset,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )


def _parse_shape_spec(spec: ShapeSpec) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Normalize a `(min, opt, max)` tuple or `{'min', 'opt', 'max'}` mapping."""
    if isinstance(spec, Mapping):
        try:
            min_s = tuple(int(d) for d in spec["min"])
            opt_s = tuple(int(d) for d in spec["opt"])
            max_s = tuple(int(d) for d in spec["max"])
        except KeyError as exc:
            raise ValueError(
                f"Dynamic shape mapping must contain 'min', 'opt', 'max' keys; got {list(spec.keys())}"
            ) from exc
        return min_s, opt_s, max_s

    if len(spec) != 3:
        raise ValueError(f"Expected (min_shape, opt_shape, max_shape) 3-tuple, got length {len(spec)}")
    return (
        tuple(int(d) for d in spec[0]),
        tuple(int(d) for d in spec[1]),
        tuple(int(d) for d in spec[2]),
    )


def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path | None = None,
    fp16: bool = True,
    dynamic_shapes: Mapping[str, ShapeSpec] | None = None,
    workspace_mb: int = 2048,
    overwrite: bool = False,
    verbose: bool = False,
) -> Path:
    """Build and serialize a TensorRT 10.9 engine from an ONNX model.

    Args:
        onnx_path: Path to the source `.onnx` model.
        engine_path: Destination `.engine` file path. If None, cached under `/cache/trt_engines/`.
        fp16: Enable `trt.BuilderFlag.FP16` when supported by the GPU.
        dynamic_shapes: Optional mapping `{input_name: (min_shape, opt_shape, max_shape)}`
            (or `{input_name: {"min": ..., "opt": ..., "max": ...}}`) for optimization profiles.
        workspace_mb: TensorRT workspace memory pool limit in MiB.
        overwrite: If False and `engine_path` already exists and is non-empty, reuse it.
        verbose: If True, enable INFO-level TensorRT builder logging.

    Returns:
        Path to the serialized `.engine` file.
    """
    src = Path(onnx_path)
    if not src.exists():
        raise FileNotFoundError(f"ONNX model not found: {src}")

    if engine_path is None:
        cache_root = resolve_cache_dir()
        prec = "fp16" if fp16 else "fp32"
        dyn_hash = ""
        if dynamic_shapes:
            norm = {k: _parse_shape_spec(v) for k, v in sorted(dynamic_shapes.items())}
            digest = hashlib.sha1(repr(norm).encode("utf-8")).hexdigest()[:8]
            dyn_hash = f"_dyn{digest}"
        dest = cache_root / f"{src.stem}{dyn_hash}_{prec}_{get_gpu_cache_tag()}.engine"
    else:
        dest = Path(engine_path)

    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    severity = trt.Logger.INFO if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(src)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(
            f"TensorRT OnnxParser failed on {src}:\n" + "\n".join(errors)
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * (1 << 20)
    )

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.log(
                trt.Logger.WARNING,
                "Requested FP16 TensorRT build, but GPU reports no fast FP16; falling back to FP32.",
            )

    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        network_inputs = {
            network.get_input(i).name for i in range(network.num_inputs)
        }
        for inp_name, spec in dynamic_shapes.items():
            if inp_name not in network_inputs:
                raise ValueError(
                    f"dynamic_shapes references unknown input {inp_name!r}; "
                    f"network inputs are {sorted(network_inputs)}"
                )
            min_s, opt_s, max_s = _parse_shape_spec(spec)
            if not (len(min_s) == len(opt_s) == len(max_s)):
                raise ValueError(
                    f"Rank mismatch in dynamic_shapes for {inp_name!r}: "
                    f"min={min_s}, opt={opt_s}, max={max_s}"
                )
            if any(not (0 < lo <= mid <= hi) for lo, mid, hi in zip(min_s, opt_s, max_s, strict=True)):
                raise ValueError(
                    f"Expected 0 < min <= opt <= max for {inp_name!r}: "
                    f"min={min_s}, opt={opt_s}, max={max_s}"
                )
            profile.set_shape(inp_name, min_s, opt_s, max_s)
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError("TensorRT failed to add optimization profile to builder config")
    else:
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            if any(int(d) < 0 for d in inp.shape):
                raise ValueError(
                    f"Network input {inp.name!r} has dynamic shape {tuple(inp.shape)}, "
                    "so `dynamic_shapes` must be provided to build_trt_engine()."
                )

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT builder returned None while compiling {src}")

    tmp_path = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(bytes(serialized))
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return dest
