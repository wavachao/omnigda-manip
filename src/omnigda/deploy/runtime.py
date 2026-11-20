"""ONNX Runtime and zero-pycuda TensorRT 10.9 inference runners, extractors, and hooks.

Provides:
- `TensorRTRunner`: TensorRT 10.9 execution engine backed purely by PyTorch CUDA tensors
  and `execute_async_v3` (zero `pycuda` dependency).
- `ONNXRuntimeRunner`: `onnxruntime-gpu` (`CUDAExecutionProvider` with automatic
  `CPUExecutionProvider` fallback) inference runner.
- `TensorRTDINOv2Extractor` & `ONNXDINOv2Extractor`: Drop-in `DenseFeatureExtractor`
  subclasses compatible with `SegmentGDAPlus(extractor=...)`.
- `TensorRTCLIPVisualHook` & `ONNXCLIPVisualHook`: Drop-in callables compatible with
  `OpenCLIPMultimodalEncoder.set_image_encoder_hook(fn)`.
- `create_dinov2_extractor(cfg)` & `create_clip_encoder(cfg)`: Config-driven factories
  supporting `backend in {"pytorch", "onnx", "tensorrt"}`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch

from omnigda.core.config import load_config
from omnigda.core.gda_math import l2_normalize
from omnigda.deploy.exporter import (
    build_trt_engine,
    clip_artifact_paths,
    dinov2_artifact_paths,
    export_clip_visual_onnx,
    export_dinov2_onnx,
)
from omnigda.perception_2d.backbone import (
    DenseFeatureExtractor,
    DINOv2MultiScaleExtractor,
    LightweightConvFeatureExtractor,
    preprocess_rgb_for_vit,
)
from omnigda.perception_3d.encoders import OpenCLIPMultimodalEncoder

_TRT_TO_TORCH_DTYPE: dict[trt.DataType, torch.dtype] = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT8: torch.int8,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.BOOL: torch.bool,
}

_ORT_TO_NP_DTYPE: dict[str, np.dtype[Any]] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(bool)": np.dtype(np.bool_),
}


def _trt_dtype_to_torch(dtype: trt.DataType) -> torch.dtype:
    if dtype not in _TRT_TO_TORCH_DTYPE:
        raise TypeError(f"Unsupported TensorRT DataType: {dtype}")
    return _TRT_TO_TORCH_DTYPE[dtype]


class TensorRTRunner:
    """Zero-pycuda TensorRT 10.9 GPU inference engine backed by PyTorch CUDA tensors."""

    def __init__(
        self,
        engine_path: str | Path,
        device: str | torch.device = "cuda:0",
        verbose: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRTRunner requires a CUDA-capable GPU.")

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"TensorRTRunner requires a CUDA device, got {self.device}")
        if self.device.index is None:
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")

        severity = trt.Logger.INFO if verbose else trt.Logger.WARNING
        self._logger = trt.Logger(severity)
        self._runtime = trt.Runtime(self._logger)

        with torch.cuda.device(self.device):
            engine_bytes = self.engine_path.read_bytes()
            engine = self._runtime.deserialize_cuda_engine(engine_bytes)
            if engine is None:
                raise RuntimeError(
                    f"Failed to deserialize TensorRT engine from {self.engine_path}"
                )
            self.engine: trt.ICudaEngine = engine
            context = self.engine.create_execution_context()
            if context is None:
                raise RuntimeError("Failed to create TensorRT execution context")
            self.context: trt.IExecutionContext = context
            # Dedicated non-default CUDA stream avoids TRT 10.9 default-stream sync penalties.
            self._stream = torch.cuda.Stream(device=self.device)

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.input_dtypes: dict[str, torch.dtype] = {}
        self.output_dtypes: dict[str, torch.dtype] = {}
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        self.output_shapes: dict[str, tuple[int, ...]] = {}

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            shape = tuple(int(d) for d in self.engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_dtypes[name] = dtype
                self.input_shapes[name] = shape
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
                self.output_dtypes[name] = dtype
                self.output_shapes[name] = shape

        self._output_buffers: dict[str, torch.Tensor] = {}

    @property
    def device_memory_mb(self) -> float:
        """Device memory required by the TensorRT engine activations in MiB."""
        size_bytes = getattr(self.engine, "device_memory_size_v2", None)
        if size_bytes is None:
            size_bytes = getattr(self.engine, "device_memory_size", 0)
        return float(size_bytes) / float(1024 * 1024)

    def _normalize_inputs(
        self,
        inputs: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
    ) -> tuple[dict[str, np.ndarray | torch.Tensor], bool]:
        if isinstance(inputs, Mapping):
            return dict(inputs), False
        if len(self.input_names) != 1:
            raise ValueError(
                f"Engine has {len(self.input_names)} inputs {self.input_names}; "
                "pass a dict mapping input names to tensors."
            )
        return {self.input_names[0]: inputs}, True

    def infer_torch(
        self,
        inputs: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
        clone_outputs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run asynchronous TensorRT inference and return CUDA `torch.Tensor` outputs."""
        feed, _ = self._normalize_inputs(inputs)
        missing = [n for n in self.input_names if n not in feed]
        if missing:
            raise KeyError(f"Missing required TensorRT input(s): {missing}")

        caller_stream = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(caller_stream)

        device_inputs: dict[str, torch.Tensor] = {}
        outputs: dict[str, torch.Tensor] = {}

        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            for name in self.input_names:
                val = feed[name]
                expected_dtype = self.input_dtypes[name]
                if isinstance(val, torch.Tensor):
                    tensor = val.to(device=self.device, dtype=expected_dtype, non_blocking=True)
                else:
                    arr = np.ascontiguousarray(val)
                    tensor = torch.from_numpy(arr).to(
                        device=self.device, dtype=expected_dtype, non_blocking=True
                    )
                tensor = tensor.contiguous()
                device_inputs[name] = tensor
                if not self.context.set_input_shape(name, tuple(tensor.shape)):
                    raise ValueError(
                        f"TensorRT rejected input shape {tuple(tensor.shape)} for {name!r}"
                    )
                self.context.set_tensor_address(name, int(tensor.data_ptr()))

            for name in self.output_names:
                out_shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
                if any(d < 0 for d in out_shape):
                    raise RuntimeError(
                        f"Unresolved dynamic output shape {out_shape} for {name!r}"
                    )
                out_dtype = self.output_dtypes[name]
                cached = self._output_buffers.get(name)
                if (
                    cached is None
                    or tuple(cached.shape) != out_shape
                    or cached.dtype != out_dtype
                ):
                    cached = torch.empty(out_shape, dtype=out_dtype, device=self.device)
                    self._output_buffers[name] = cached
                self.context.set_tensor_address(name, int(cached.data_ptr()))
                outputs[name] = cached

            stream_ptr = int(torch.cuda.current_stream(self.device).cuda_stream)
            ok = self.context.execute_async_v3(stream_ptr)
            if not ok:
                raise RuntimeError(
                    f"TensorRT execute_async_v3 failed for engine {self.engine_path}"
                )

            if clone_outputs:
                outputs = {k: v.clone() for k, v in outputs.items()}

        caller_stream.wait_stream(self._stream)
        return outputs

    def infer(
        self,
        inputs: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Run TensorRT inference and return CPU NumPy array(s)."""
        _, single_input = self._normalize_inputs(inputs)
        cuda_outputs = self.infer_torch(inputs, clone_outputs=False)
        np_outputs = {
            name: tensor.detach().cpu().numpy() for name, tensor in cuda_outputs.items()
        }
        if single_input and len(self.output_names) == 1:
            return np_outputs[self.output_names[0]]
        return np_outputs

    def __call__(
        self,
        inputs: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
    ) -> np.ndarray | dict[str, np.ndarray]:
        return self.infer(inputs)


class ONNXRuntimeRunner:
    """ONNX Runtime inference runner preferring `CUDAExecutionProvider` with CPU fallback."""

    def __init__(
        self,
        onnx_path: str | Path,
        prefer_cuda: bool = True,
        providers: Sequence[str] | None = None,
        intra_op_num_threads: int | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads is not None:
            sess_options.intra_op_num_threads = int(intra_op_num_threads)

        if providers is not None:
            requested = list(providers)
        elif prefer_cuda:
            requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            requested = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=requested,
        )
        self.providers: list[str] = list(self.session.get_providers())
        self.active_provider: str = (
            self.providers[0] if self.providers else "CPUExecutionProvider"
        )
        self.using_cuda: bool = self.active_provider == "CUDAExecutionProvider"

        self.input_names: list[str] = [inp.name for inp in self.session.get_inputs()]
        self.output_names: list[str] = [out.name for out in self.session.get_outputs()]
        self.input_shapes: dict[str, tuple[int | str, ...]] = {
            inp.name: tuple(inp.shape) for inp in self.session.get_inputs()
        }
        self.output_shapes: dict[str, tuple[int | str, ...]] = {
            out.name: tuple(out.shape) for out in self.session.get_outputs()
        }
        self.input_dtypes: dict[str, np.dtype[Any]] = {
            inp.name: _ORT_TO_NP_DTYPE.get(inp.type, np.dtype(np.float32))
            for inp in self.session.get_inputs()
        }

    def infer(
        self,
        inputs: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Run ONNX Runtime inference and return NumPy array(s)."""
        single_input = not isinstance(inputs, Mapping)
        if isinstance(inputs, Mapping):
            raw_feed = dict(inputs)
        else:
            if len(self.input_names) != 1:
                raise ValueError(
                    f"ONNX model has {len(self.input_names)} inputs {self.input_names}; "
                    "pass a dict mapping input names to arrays."
                )
            raw_feed = {self.input_names[0]: inputs}

        feed: dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name not in raw_feed:
                raise KeyError(f"Missing required ONNX input {name!r}")
            val = raw_feed[name]
            if isinstance(val, torch.Tensor):
                arr = val.detach().cpu().numpy()
            else:
                arr = np.asarray(val)
            expected_dtype = self.input_dtypes[name]
            if arr.dtype != expected_dtype:
                arr = arr.astype(expected_dtype, copy=False)
            feed[name] = np.ascontiguousarray(arr)

        raw_outputs = self.session.run(self.output_names, feed)
        np_outputs = {
            name: np.asarray(val)
            for name, val in zip(self.output_names, raw_outputs, strict=True)
        }
        if single_input and len(self.output_names) == 1:
            return np_outputs[self.output_names[0]]
        return np_outputs

    def __call__(
        self,
        inputs: np.ndarray | torch.Tensor | Mapping[str, np.ndarray | torch.Tensor],
    ) -> np.ndarray | dict[str, np.ndarray]:
        return self.infer(inputs)


class TensorRTDINOv2Extractor(DenseFeatureExtractor):
    """Drop-in `DenseFeatureExtractor` backed by a cached TensorRT 10.9 DINOv2 engine."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        input_size: int = 518,
        feature_layers: Sequence[int] = (8, 10, 11),
        fusion_mode: Literal["mean", "concat"] = "mean",
        fp16: bool = True,
        onnx_path: str | Path | None = None,
        engine_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        workspace_mb: int = 2048,
        device: str = "cuda:0",
        opset: int = 17,
        pretrained: bool = True,
    ) -> None:
        self.model_name = model_name
        self.input_size = int(input_size)
        self.feature_layers = tuple(int(i) for i in feature_layers)
        self.fusion_mode: Literal["mean", "concat"] = fusion_mode
        self.fp16 = bool(fp16)

        default_onnx, default_engine = dinov2_artifact_paths(
            backbone_name=model_name,
            input_size=self.input_size,
            feature_layers=self.feature_layers,
            fusion_mode=self.fusion_mode,
            fp16=self.fp16,
            cache_dir=cache_dir,
        )
        self.onnx_path = Path(onnx_path) if onnx_path is not None else default_onnx
        self.engine_path = Path(engine_path) if engine_path is not None else default_engine

        if not (self.engine_path.exists() and self.engine_path.stat().st_size > 0):
            export_dinov2_onnx(
                backbone_name=self.model_name,
                input_size=self.input_size,
                feature_layers=self.feature_layers,
                output_path=self.onnx_path,
                opset=opset,
                fusion_mode=self.fusion_mode,
                pretrained=pretrained,
            )
            build_trt_engine(
                onnx_path=self.onnx_path,
                engine_path=self.engine_path,
                fp16=self.fp16,
                workspace_mb=workspace_mb,
            )

        self.runner = TensorRTRunner(self.engine_path, device=device)
        out_name = self.runner.output_names[0]
        out_shape = self.runner.output_shapes[out_name]
        if len(out_shape) != 4 or out_shape[1] <= 0:
            raise RuntimeError(
                f"Unexpected DINOv2 TensorRT output shape {out_shape} for {out_name!r}"
            )
        self._feature_dim = int(out_shape[1])
        self._grid_size = int(out_shape[2])

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def backend_name(self) -> str:
        prec = "fp16" if self.fp16 else "fp32"
        return f"tensorrt-cuda-{prec}"

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        nchw = preprocess_rgb_for_vit(rgb, self.input_size)
        out = self.runner.infer(nchw)
        assert isinstance(out, np.ndarray)
        feat = out[0].astype(np.float32, copy=False)
        return l2_normalize(feat, axis=0, eps=1e-6).astype(np.float32, copy=False)


class ONNXDINOv2Extractor(DenseFeatureExtractor):
    """Drop-in `DenseFeatureExtractor` backed by an auto-exported ONNX Runtime DINOv2 model."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        input_size: int = 518,
        feature_layers: Sequence[int] = (8, 10, 11),
        fusion_mode: Literal["mean", "concat"] = "mean",
        onnx_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        prefer_cuda: bool = True,
        opset: int = 17,
        pretrained: bool = True,
    ) -> None:
        self.model_name = model_name
        self.input_size = int(input_size)
        self.feature_layers = tuple(int(i) for i in feature_layers)
        self.fusion_mode: Literal["mean", "concat"] = fusion_mode

        default_onnx, _ = dinov2_artifact_paths(
            backbone_name=model_name,
            input_size=self.input_size,
            feature_layers=self.feature_layers,
            fusion_mode=self.fusion_mode,
            cache_dir=cache_dir,
        )
        self.onnx_path = Path(onnx_path) if onnx_path is not None else default_onnx

        if not (self.onnx_path.exists() and self.onnx_path.stat().st_size > 0):
            export_dinov2_onnx(
                backbone_name=self.model_name,
                input_size=self.input_size,
                feature_layers=self.feature_layers,
                output_path=self.onnx_path,
                opset=opset,
                fusion_mode=self.fusion_mode,
                pretrained=pretrained,
            )

        self.runner = ONNXRuntimeRunner(self.onnx_path, prefer_cuda=prefer_cuda)
        out_name = self.runner.output_names[0]
        out_shape = self.runner.output_shapes[out_name]
        if len(out_shape) == 4 and isinstance(out_shape[1], int) and out_shape[1] > 0:
            self._feature_dim = int(out_shape[1])
            self._grid_size = int(out_shape[2]) if isinstance(out_shape[2], int) else 0
        else:
            probe = self.runner.infer(
                np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
            )
            assert isinstance(probe, np.ndarray)
            self._feature_dim = int(probe.shape[1])
            self._grid_size = int(probe.shape[2])

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def backend_name(self) -> str:
        dev = "cuda" if self.runner.using_cuda else "cpu"
        return f"onnx-{dev}-fp32"

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        nchw = preprocess_rgb_for_vit(rgb, self.input_size)
        out = self.runner.infer(nchw)
        assert isinstance(out, np.ndarray)
        feat = out[0].astype(np.float32, copy=False)
        return l2_normalize(feat, axis=0, eps=1e-6).astype(np.float32, copy=False)


class TensorRTCLIPVisualHook:
    """Callable `(batch_nchw: np.ndarray) -> np.ndarray` backed by TensorRT 10.9.

    Directly plugs into `OpenCLIPMultimodalEncoder.set_image_encoder_hook(hook)`.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        fp16: bool = True,
        min_batch_size: int = 1,
        opt_batch_size: int = 4,
        max_batch_size: int = 16,
        image_size: int = 224,
        onnx_path: str | Path | None = None,
        engine_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        workspace_mb: int = 2048,
        device: str = "cuda:0",
        opset: int = 17,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.fp16 = bool(fp16)
        self.min_batch_size = int(min_batch_size)
        self.opt_batch_size = int(opt_batch_size)
        self.max_batch_size = int(max_batch_size)
        self.image_size = int(image_size)

        default_onnx, default_engine = clip_artifact_paths(
            model_name=model_name,
            pretrained=pretrained,
            max_batch_size=self.max_batch_size,
            fp16=self.fp16,
            cache_dir=cache_dir,
        )
        self.onnx_path = Path(onnx_path) if onnx_path is not None else default_onnx
        self.engine_path = Path(engine_path) if engine_path is not None else default_engine

        if not (self.engine_path.exists() and self.engine_path.stat().st_size > 0):
            export_clip_visual_onnx(
                model_name=self.model_name,
                pretrained=self.pretrained,
                output_path=self.onnx_path,
                opset=opset,
            )
            build_trt_engine(
                onnx_path=self.onnx_path,
                engine_path=self.engine_path,
                fp16=self.fp16,
                dynamic_shapes={
                    "input": (
                        (self.min_batch_size, 3, self.image_size, self.image_size),
                        (self.opt_batch_size, 3, self.image_size, self.image_size),
                        (self.max_batch_size, 3, self.image_size, self.image_size),
                    )
                },
                workspace_mb=workspace_mb,
            )

        self.runner = TensorRTRunner(self.engine_path, device=device)
        out_name = self.runner.output_names[0]
        out_shape = self.runner.output_shapes[out_name]
        self._embed_dim = (
            int(out_shape[-1]) if len(out_shape) >= 2 and out_shape[-1] > 0 else 512
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def backend_name(self) -> str:
        prec = "fp16" if self.fp16 else "fp32"
        return f"tensorrt-cuda-{prec}"

    def __call__(self, batch_nchw: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(batch_nchw, torch.Tensor):
            arr = batch_nchw.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(batch_nchw, dtype=np.float32)

        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[1] != 3:
            raise ValueError(f"Expected input shape (B, 3, H, W), got {arr.shape}")

        batch = arr.shape[0]
        if batch == 0:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        if batch <= self.max_batch_size:
            out = self.runner.infer(arr)
            assert isinstance(out, np.ndarray)
            emb = out.astype(np.float32, copy=False)
        else:
            chunks: list[np.ndarray] = []
            for start in range(0, batch, self.max_batch_size):
                chunk = arr[start : start + self.max_batch_size]
                out = self.runner.infer(chunk)
                assert isinstance(out, np.ndarray)
                chunks.append(out.astype(np.float32, copy=False))
            emb = np.concatenate(chunks, axis=0)

        return l2_normalize(emb, axis=-1, eps=1e-8).astype(np.float32, copy=False)


class ONNXCLIPVisualHook:
    """Callable `(batch_nchw: np.ndarray) -> np.ndarray` backed by ONNX Runtime.

    Directly plugs into `OpenCLIPMultimodalEncoder.set_image_encoder_hook(hook)`.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        onnx_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        prefer_cuda: bool = True,
        opset: int = 17,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained

        default_onnx, _ = clip_artifact_paths(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )
        self.onnx_path = Path(onnx_path) if onnx_path is not None else default_onnx

        if not (self.onnx_path.exists() and self.onnx_path.stat().st_size > 0):
            export_clip_visual_onnx(
                model_name=self.model_name,
                pretrained=self.pretrained,
                output_path=self.onnx_path,
                opset=opset,
            )

        self.runner = ONNXRuntimeRunner(self.onnx_path, prefer_cuda=prefer_cuda)
        out_name = self.runner.output_names[0]
        out_shape = self.runner.output_shapes[out_name]
        self._embed_dim = (
            int(out_shape[-1])
            if len(out_shape) >= 2 and isinstance(out_shape[-1], int) and out_shape[-1] > 0
            else 512
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def backend_name(self) -> str:
        dev = "cuda" if self.runner.using_cuda else "cpu"
        return f"onnx-{dev}-fp32"

    def __call__(self, batch_nchw: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(batch_nchw, torch.Tensor):
            arr = batch_nchw.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(batch_nchw, dtype=np.float32)

        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[1] != 3:
            raise ValueError(f"Expected input shape (B, 3, H, W), got {arr.shape}")

        if arr.shape[0] == 0:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        out = self.runner.infer(arr)
        assert isinstance(out, np.ndarray)
        emb = out.astype(np.float32, copy=False)
        return l2_normalize(emb, axis=-1, eps=1e-8).astype(np.float32, copy=False)


def _normalize_backend_name(backend: str) -> str:
    key = backend.lower().strip()
    if key in ("pytorch", "torch", "dinov2"):
        return "pytorch"
    if key in ("onnx", "ort", "onnxruntime"):
        return "onnx"
    if key in ("tensorrt", "trt"):
        return "tensorrt"
    if key == "lightweight":
        return "lightweight"
    raise ValueError(
        f"Unsupported deployment backend {backend!r}. "
        "Expected one of {'pytorch', 'onnx', 'tensorrt', 'lightweight'}."
    )


def create_dinov2_extractor(
    cfg: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> DenseFeatureExtractor:
    """Create a DINOv2 `DenseFeatureExtractor` from OmniGDA config (`pytorch` | `onnx` | `tensorrt`)."""
    cfg_all = dict(cfg) if cfg is not None else load_config()
    p2d = cfg_all.get("perception_2d", cfg_all)
    proj = cfg_all.get("project", {})
    deploy = cfg_all.get("deploy", {})

    backend = _normalize_backend_name(
        str(overrides.get("backend", p2d.get("backend", deploy.get("backend", "pytorch"))))
    )
    model_name = str(
        overrides.get(
            "model_name",
            overrides.get("backbone", p2d.get("backbone", "vit_small_patch14_dinov2.lvd142m")),
        )
    )
    input_size = int(overrides.get("input_size", p2d.get("input_size", 518)))
    feature_layers = tuple(
        int(i) for i in overrides.get("feature_layers", p2d.get("feature_layers", (8, 10, 11)))
    )
    fusion_mode: Literal["mean", "concat"] = overrides.get(
        "fusion_mode", p2d.get("fusion_mode", "mean")
    )
    fp16 = bool(overrides.get("fp16", p2d.get("fp16", True)))
    device = str(overrides.get("device", proj.get("device", "cuda")))
    cache_dir = overrides.get("cache_dir", deploy.get("cache_dir"))

    if backend == "pytorch":
        return DINOv2MultiScaleExtractor(
            model_name=model_name,
            input_size=input_size,
            feature_layers=feature_layers,
            fusion_mode=fusion_mode,
            device=device,
            fp16=fp16,
            pretrained=bool(overrides.get("pretrained", True)),
        )
    if backend == "onnx":
        return ONNXDINOv2Extractor(
            model_name=model_name,
            input_size=input_size,
            feature_layers=feature_layers,
            fusion_mode=fusion_mode,
            onnx_path=overrides.get("onnx_path"),
            cache_dir=cache_dir,
            prefer_cuda=bool(overrides.get("prefer_cuda", device.startswith("cuda"))),
            opset=int(overrides.get("opset", deploy.get("opset", 17))),
            pretrained=bool(overrides.get("pretrained", True)),
        )
    if backend == "tensorrt":
        return TensorRTDINOv2Extractor(
            model_name=model_name,
            input_size=input_size,
            feature_layers=feature_layers,
            fusion_mode=fusion_mode,
            fp16=fp16,
            onnx_path=overrides.get("onnx_path"),
            engine_path=overrides.get("engine_path"),
            cache_dir=cache_dir,
            workspace_mb=int(overrides.get("workspace_mb", deploy.get("workspace_mb", 2048))),
            device=device if device.startswith("cuda") else "cuda:0",
            opset=int(overrides.get("opset", deploy.get("opset", 17))),
            pretrained=bool(overrides.get("pretrained", True)),
        )
    return LightweightConvFeatureExtractor()


def create_clip_encoder(
    cfg: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> OpenCLIPMultimodalEncoder:
    """Create an `OpenCLIPMultimodalEncoder` with optional ONNX/TensorRT visual encoder hook."""
    cfg_all = dict(cfg) if cfg is not None else load_config()
    p3d = cfg_all.get("perception_3d", cfg_all)
    p2d = cfg_all.get("perception_2d", {})
    proj = cfg_all.get("project", {})
    deploy = cfg_all.get("deploy", {})

    raw_backend = overrides.get(
        "backend",
        p3d.get(
            "backend",
            deploy.get("clip_backend", deploy.get("backend", p2d.get("backend", "pytorch"))),
        ),
    )
    backend = _normalize_backend_name(str(raw_backend))
    if backend == "lightweight":
        raise ValueError("CLIP encoder supports backend in {'pytorch', 'onnx', 'tensorrt'}.")

    model_name = str(overrides.get("model_name", p3d.get("clip_model", "ViT-B-32")))
    pretrained = str(
        overrides.get("pretrained", p3d.get("clip_pretrained", "laion2b_s34b_b79k"))
    )
    device = str(overrides.get("device", proj.get("device", "cuda")))
    fp16 = bool(overrides.get("fp16", p3d.get("fp16", p2d.get("fp16", True))))
    cache_dir = overrides.get("cache_dir", deploy.get("cache_dir"))

    encoder = OpenCLIPMultimodalEncoder(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        fp16=fp16,
    )

    if backend == "tensorrt":
        hook = TensorRTCLIPVisualHook(
            model_name=model_name,
            pretrained=pretrained,
            fp16=fp16,
            max_batch_size=int(overrides.get("max_batch_size", deploy.get("clip_max_batch", 16))),
            onnx_path=overrides.get("onnx_path"),
            engine_path=overrides.get("engine_path"),
            cache_dir=cache_dir,
            workspace_mb=int(overrides.get("workspace_mb", deploy.get("workspace_mb", 2048))),
            device=device if device.startswith("cuda") else "cuda:0",
            opset=int(overrides.get("opset", deploy.get("opset", 17))),
        )
        encoder.set_image_encoder_hook(hook)
    elif backend == "onnx":
        hook = ONNXCLIPVisualHook(
            model_name=model_name,
            pretrained=pretrained,
            onnx_path=overrides.get("onnx_path"),
            cache_dir=cache_dir,
            prefer_cuda=bool(overrides.get("prefer_cuda", device.startswith("cuda"))),
            opset=int(overrides.get("opset", deploy.get("opset", 17))),
        )
        encoder.set_image_encoder_hook(hook)

    return encoder

