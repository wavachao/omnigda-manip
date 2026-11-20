"""Dense visual feature extractors for SegmentGDAPlus.

Defines the backend-agnostic `DenseFeatureExtractor` interface so PyTorch
(`DINOv2MultiScaleExtractor`), ONNX Runtime, TensorRT, and lightweight CI
extractors are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omnigda.core.gda_math import l2_normalize

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    """Validate and convert RGB image to (H, W, 3) float32 in [0, 1]."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {arr.shape}")
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr_f = arr.astype(np.float32)
    if arr_f.size > 0 and float(np.max(arr_f)) > 1.5:
        arr_f = arr_f / 255.0
    return np.clip(arr_f, 0.0, 1.0)


def preprocess_rgb_for_vit(rgb: np.ndarray, input_size: int) -> np.ndarray:
    """Resize and ImageNet-normalize an RGB image into (1, 3, input_size, input_size) float32.

    Shared helper so PyTorch, ONNX Runtime, and TensorRT backends use identical preprocessing.
    """
    img = _validate_rgb(rgb)
    if img.shape[0] != input_size or img.shape[1] != input_size:
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    normed = (img - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    nchw = np.transpose(normed, (2, 0, 1))[None, ...].astype(np.float32, copy=False)
    return np.ascontiguousarray(nchw)


class DenseFeatureExtractor(ABC):
    """Abstract interface for dense patch-level visual feature extractors."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Channel dimension D of the returned dense feature map."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier (e.g. 'pytorch-cuda-fp16', 'onnx', 'tensorrt')."""

    @abstractmethod
    def extract(self, rgb: np.ndarray) -> np.ndarray:
        """Extract L2-normalized dense feature map of shape (D, Hf, Wf) float32."""

    def extract_batch(self, rgbs: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Extract dense feature maps for a sequence of RGB images."""
        return [self.extract(img) for img in rgbs]


class _DINOv2MultiScaleModule(nn.Module):
    """Export-friendly nn.Module wrapper producing fused L2-normalized dense features (B, D, Hf, Wf)."""

    def __init__(
        self,
        vit: nn.Module,
        feature_layers: Sequence[int],
        fusion_mode: Literal["mean", "concat"] = "mean",
    ) -> None:
        super().__init__()
        self.vit = vit
        self.feature_layers = tuple(int(i) for i in feature_layers)
        self.fusion_mode = fusion_mode
        patch_size = vit.patch_embed.patch_size
        self.patch_h = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)
        self.patch_w = int(patch_size[1] if isinstance(patch_size, tuple) else patch_size)
        self.embed_dim = int(vit.embed_dim)
        self.out_dim = (
            self.embed_dim * len(self.feature_layers)
            if fusion_mode == "concat"
            else self.embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_in, w_in = x.shape[-2], x.shape[-1]
        hf = h_in // self.patch_h
        wf = w_in // self.patch_w

        intermediates = self.vit.forward_intermediates(
            x,
            indices=list(self.feature_layers),
            return_prefix_tokens=False,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )

        normed_maps: list[torch.Tensor] = []
        for feat in intermediates:
            if feat.ndim == 3:
                # Defensive fallback if a timm backbone returns (B, N, C)
                feat = feat.transpose(1, 2).reshape(x.shape[0], self.embed_dim, hf, wf)
            normed_maps.append(F.normalize(feat.float(), p=2, dim=1, eps=1e-6))

        if self.fusion_mode == "concat":
            fused = torch.cat(normed_maps, dim=1)
        else:
            fused = torch.stack(normed_maps, dim=0).mean(dim=0)

        return F.normalize(fused, p=2, dim=1, eps=1e-6)


class DINOv2MultiScaleExtractor(DenseFeatureExtractor):
    """Multi-scale dense patch feature extractor backed by timm DINOv2 ViT."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        input_size: int = 518,
        feature_layers: Sequence[int] = (8, 10, 11),
        fusion_mode: Literal["mean", "concat"] = "mean",
        device: str = "cuda",
        fp16: bool = True,
        pretrained: bool = True,
    ) -> None:
        import timm

        if fusion_mode not in ("mean", "concat"):
            raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}; expected 'mean' or 'concat'")
        if not feature_layers:
            raise ValueError("feature_layers must contain at least one transformer block index")

        self.model_name = model_name
        self.input_size = int(input_size)
        self.feature_layers = tuple(int(i) for i in feature_layers)
        self.fusion_mode: Literal["mean", "concat"] = fusion_mode

        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(device if use_cuda else "cpu")
        self.fp16 = bool(fp16 and self.device.type == "cuda")

        vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=self.input_size,
            dynamic_img_size=True,
        )
        patch_size = vit.patch_embed.patch_size
        self.patch_size = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)
        if self.input_size <= 0 or self.input_size % self.patch_size != 0:
            raise ValueError(
                f"input_size ({self.input_size}) must be a positive multiple of patch_size ({self.patch_size})"
            )

        num_blocks = len(vit.blocks)
        for idx in self.feature_layers:
            if idx < 0 or idx >= num_blocks:
                raise ValueError(
                    f"feature_layer index {idx} out of range for {model_name} with {num_blocks} blocks"
                )

        vit.eval()
        self.module = _DINOv2MultiScaleModule(
            vit=vit,
            feature_layers=self.feature_layers,
            fusion_mode=self.fusion_mode,
        ).to(self.device)
        self.module.eval()

        self._feature_dim = self.module.out_dim
        self._grid_size = self.input_size // self.patch_size

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def backend_name(self) -> str:
        prec = "fp16" if self.fp16 else "fp32"
        return f"pytorch-{self.device.type}-{prec}"

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        nchw = preprocess_rgb_for_vit(rgb, self.input_size)
        tensor = torch.from_numpy(nchw).to(self.device, non_blocking=True)

        with torch.inference_mode():
            if self.fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    feats = self.module(tensor)
            else:
                feats = self.module(tensor)

        feat_np = feats[0].detach().cpu().numpy().astype(np.float32, copy=False)
        return l2_normalize(feat_np, axis=0, eps=1e-6).astype(np.float32, copy=False)

    def extract_batch(self, rgbs: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not rgbs:
            return []
        batch_np = np.concatenate(
            [preprocess_rgb_for_vit(img, self.input_size) for img in rgbs], axis=0
        )
        tensor = torch.from_numpy(batch_np).to(self.device, non_blocking=True)

        with torch.inference_mode():
            if self.fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    feats = self.module(tensor)
            else:
                feats = self.module(tensor)

        feats_np = feats.detach().cpu().numpy().astype(np.float32, copy=False)
        return [
            l2_normalize(feats_np[i], axis=0, eps=1e-6).astype(np.float32, copy=False)
            for i in range(feats_np.shape[0])
        ]


class LightweightConvFeatureExtractor(DenseFeatureExtractor):
    """Deterministic multi-scale color + Sobel/Laplacian dense patch descriptor.

    Intended strictly as an optional fast CPU/CI fallback (`backend='lightweight'`).
    """

    def __init__(
        self,
        input_size: int = 224,
        grid_size: int = 32,
        scales: Sequence[float] = (0.0, 1.5, 3.5),
    ) -> None:
        if input_size <= 0 or grid_size <= 0:
            raise ValueError("input_size and grid_size must be positive integers")
        self.input_size = int(input_size)
        self.grid_size = int(grid_size)
        self.scales = tuple(float(s) for s in scales)
        # Per scale: RGB mean (3) + RGB std (3) + Lab mean (3) + HSV mean (3)
        # + Sobel |gx|, |gy|, mag, Laplacian mean (4) = 16 channels per scale
        self._channels_per_scale = 16
        self._feature_dim = self._channels_per_scale * len(self.scales)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def backend_name(self) -> str:
        return "lightweight-cpu-fp32"

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        img_f = _validate_rgb(rgb)
        if img_f.shape[0] != self.input_size or img_f.shape[1] != self.input_size:
            img_f = cv2.resize(
                img_f, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
            )

        g = self.grid_size
        per_scale_maps: list[np.ndarray] = []

        for sigma in self.scales:
            if sigma > 0.0:
                blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
            else:
                blurred = img_f

            rgb_u8 = np.clip(blurred * 255.0, 0.0, 255.0).astype(np.uint8)
            lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] /= 180.0
            hsv[..., 1:] /= 255.0
            gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

            gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
            gmag = np.sqrt(gx * gx + gy * gy + 1e-8)
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

            rgb_mean = cv2.resize(blurred, (g, g), interpolation=cv2.INTER_AREA)
            rgb_sq_mean = cv2.resize(blurred * blurred, (g, g), interpolation=cv2.INTER_AREA)
            rgb_std = np.sqrt(np.maximum(rgb_sq_mean - rgb_mean * rgb_mean, 0.0) + 1e-6)
            lab_mean = cv2.resize(lab, (g, g), interpolation=cv2.INTER_AREA)
            hsv_mean = cv2.resize(hsv, (g, g), interpolation=cv2.INTER_AREA)

            grad_stack = np.stack([gx, gy, gmag, lap], axis=-1)
            grad_mean = cv2.resize(grad_stack, (g, g), interpolation=cv2.INTER_AREA)

            scale_feat = np.concatenate(
                [rgb_mean, rgb_std, lab_mean, hsv_mean, grad_mean], axis=-1
            )  # (g, g, 16)
            scale_chw = np.transpose(scale_feat, (2, 0, 1))
            per_scale_maps.append(l2_normalize(scale_chw, axis=0, eps=1e-6))

        fused = np.concatenate(per_scale_maps, axis=0).astype(np.float32, copy=False)
        return l2_normalize(fused, axis=0, eps=1e-6).astype(np.float32, copy=False)


def create_feature_extractor(
    backend: str = "pytorch",
    model_name: str = "vit_small_patch14_dinov2.lvd142m",
    input_size: int = 518,
    feature_layers: Sequence[int] = (8, 10, 11),
    fusion_mode: Literal["mean", "concat"] = "mean",
    device: str = "cuda",
    fp16: bool = True,
    pretrained: bool = True,
) -> DenseFeatureExtractor:
    """Factory for building a `DenseFeatureExtractor` from configuration parameters."""
    key = backend.lower().strip()
    if key in ("pytorch", "dinov2", "torch"):
        return DINOv2MultiScaleExtractor(
            model_name=model_name,
            input_size=input_size,
            feature_layers=feature_layers,
            fusion_mode=fusion_mode,
            device=device,
            fp16=fp16,
            pretrained=pretrained,
        )
    if key == "lightweight":
        return LightweightConvFeatureExtractor()
    raise ValueError(
        f"Unsupported perception_2d backend {backend!r}. Expected 'pytorch' or 'lightweight' "
        "(or pass a custom DenseFeatureExtractor instance such as ONNX/TensorRT)."
    )
