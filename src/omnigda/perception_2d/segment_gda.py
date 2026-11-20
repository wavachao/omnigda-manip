"""SegmentGDA+: Training-free few-shot 2D semantic/instance segmentation via Shrinkage GDA.

Upgrades SegmentGDA for robotic manipulation with:
- Multi-prototype GMM modeling for both foreground classes and heterogeneous backgrounds
- Adaptive covariance shrinkage (OAS / Ledoit-Wolf / Ridge) + Sherman-Morrison-Woodbury
  online incremental registration of unseen objects in sub-second latency
- Vectorized single-GEMM dense log-likelihood evaluation on ViT patch grids
- Fast Color Guided Filter (RGB-guided edge-preserving spatial refinement)
- Calibrated per-pixel confidence & normalized Shannon entropy maps
- Connected-component instance proposal extraction for downstream 3D multimodal GMDA
- Prototype memory state persistence (`save_state` / `load_state`)
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from omnigda.core.gda_math import (
    OnlineClassState,
    OnlineShrinkageGDA,
    l2_normalize,
    normalized_entropy,
    softmax,
)
from omnigda.core.schemas import SegmentationResult, SupportSample
from omnigda.perception_2d.backbone import (
    DenseFeatureExtractor,
    create_feature_extractor,
)

BACKGROUND_CLASS_NAME = "__background__"


@dataclass(slots=True)
class InstanceProposal:
    """Connected-component 2D object instance proposal feeding 3D multimodal GMDA."""

    instance_id: int
    class_name: str
    class_index: int  # 1..K matching SegmentationResult.class_map
    mask: np.ndarray  # (H, W) bool
    bbox_2d: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_px: int
    mean_confidence: float
    mean_entropy: float
    mean_log_likelihoods: np.ndarray  # (C,) float32 (index 0 = background)
    fg_log_likelihoods: np.ndarray  # (K,) float32 aligned with class_names


def fast_color_guided_filter(
    guide_rgb: np.ndarray,
    src_maps: np.ndarray,
    radius: int = 8,
    eps: float = 1e-3,
    subsample: int = 2,
) -> np.ndarray:
    """Fast Color Guided Filter (He & Sun) for RGB-guided edge-preserving refinement.

    Shares the precomputed 3x3 local RGB covariance inverse across all C log-likelihood channels.

    Args:
        guide_rgb: (H, W, 3) float32 image in [0, 1] (or uint8 in [0, 255]).
        src_maps: (C, H, W) float32 upsampled log-likelihood maps.
        radius: Filter window radius in full-resolution pixels.
        eps: Regularization epsilon for local 3x3 RGB covariance.
        subsample: Downsampling factor for fast guided coefficient estimation.

    Returns:
        refined: (C, H, W) float32 edge-aligned log-likelihood maps.
    """
    if src_maps.ndim != 3:
        raise ValueError(f"Expected src_maps shape (C, H, W), got {src_maps.shape}")
    c, h, w = src_maps.shape
    if c == 0 or h < 4 or w < 4:
        return src_maps.astype(np.float32, copy=True)

    guide = np.asarray(guide_rgb)
    if guide.dtype == np.uint8:
        guide = guide.astype(np.float32) / 255.0
    else:
        guide = guide.astype(np.float32, copy=False)
        if guide.size > 0 and float(np.max(guide)) > 1.5:
            guide = guide / 255.0

    s = max(1, int(subsample))
    if min(h, w) < 64:
        s = 1
    hs = max(4, h // s)
    ws = max(4, w // s)
    r_s = max(1, int(round(radius / float(s))))
    ksize = (2 * r_s + 1, 2 * r_s + 1)

    if s > 1:
        guide_s = cv2.resize(guide, (ws, hs), interpolation=cv2.INTER_AREA)
    else:
        guide_s = guide

    ir, ig, ib = guide_s[..., 0], guide_s[..., 1], guide_s[..., 2]

    def _box(arr: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(
            arr, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT_101
        )

    mean_r = _box(ir)
    mean_g = _box(ig)
    mean_b = _box(ib)

    var_rr = _box(ir * ir) - mean_r * mean_r + eps
    var_rg = _box(ir * ig) - mean_r * mean_g
    var_rb = _box(ir * ib) - mean_r * mean_b
    var_gg = _box(ig * ig) - mean_g * mean_g + eps
    var_gb = _box(ig * ib) - mean_g * mean_b
    var_bb = _box(ib * ib) - mean_b * mean_b + eps

    # Analytic 3x3 symmetric SPD inverse via cofactors
    c_rr = var_gg * var_bb - var_gb * var_gb
    c_rg = var_rb * var_gb - var_rg * var_bb
    c_rb = var_rg * var_gb - var_rb * var_gg
    c_gg = var_rr * var_bb - var_rb * var_rb
    c_gb = var_rg * var_rb - var_rr * var_gb
    c_bb = var_rr * var_gg - var_rg * var_rg

    det = np.maximum(var_rr * c_rr + var_rg * c_rg + var_rb * c_rb, 1e-12)
    inv_rr = c_rr / det
    inv_rg = c_rg / det
    inv_rb = c_rb / det
    inv_gg = c_gg / det
    inv_gb = c_gb / det
    inv_bb = c_bb / det

    full_r, full_g, full_b = guide[..., 0], guide[..., 1], guide[..., 2]
    out = np.empty((c, h, w), dtype=np.float32)

    for ch in range(c):
        p_full = src_maps[ch].astype(np.float32, copy=False)
        if s > 1:
            p_s = cv2.resize(p_full, (ws, hs), interpolation=cv2.INTER_AREA)
        else:
            p_s = p_full

        mean_p = _box(p_s)
        cov_rp = _box(ir * p_s) - mean_r * mean_p
        cov_gp = _box(ig * p_s) - mean_g * mean_p
        cov_bp = _box(ib * p_s) - mean_b * mean_p

        a_r = inv_rr * cov_rp + inv_rg * cov_gp + inv_rb * cov_bp
        a_g = inv_rg * cov_rp + inv_gg * cov_gp + inv_gb * cov_bp
        a_b = inv_rb * cov_rp + inv_gb * cov_gp + inv_bb * cov_bp
        b_s = mean_p - (a_r * mean_r + a_g * mean_g + a_b * mean_b)

        mean_ar = _box(a_r)
        mean_ag = _box(a_g)
        mean_ab = _box(a_b)
        mean_b_coef = _box(b_s)

        if s > 1:
            mean_ar = cv2.resize(mean_ar, (w, h), interpolation=cv2.INTER_LINEAR)
            mean_ag = cv2.resize(mean_ag, (w, h), interpolation=cv2.INTER_LINEAR)
            mean_ab = cv2.resize(mean_ab, (w, h), interpolation=cv2.INTER_LINEAR)
            mean_b_coef = cv2.resize(mean_b_coef, (w, h), interpolation=cv2.INTER_LINEAR)

        out[ch] = mean_ar * full_r + mean_ag * full_g + mean_ab * full_b + mean_b_coef

    return out


def extract_instance_masks(
    seg_result: SegmentationResult,
    min_area_px: int = 150,
) -> list[InstanceProposal]:
    """Split semantic segmentation masks into clean connected-component instance proposals.

    Args:
        seg_result: Output of `SegmentGDAPlus.segment`.
        min_area_px: Minimum pixel area to retain a connected component.

    Returns:
        List of `InstanceProposal` objects with bounding boxes and mean class log-likelihoods.
    """
    proposals: list[InstanceProposal] = []
    min_area = max(1, int(min_area_px))

    for class_idx, class_name in enumerate(seg_result.class_names, start=1):
        mask = seg_result.masks.get(class_name)
        if mask is None:
            mask = seg_result.class_map == class_idx
        mask_u8 = np.asarray(mask, dtype=np.uint8)
        if int(np.count_nonzero(mask_u8)) < min_area:
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            comp_mask = labels == lbl

            mean_conf = float(np.mean(seg_result.confidence_map[comp_mask]))
            mean_ent = float(np.mean(seg_result.entropy_map[comp_mask]))
            mean_ll = np.mean(seg_result.log_likelihoods[:, comp_mask], axis=1).astype(np.float32)
            fg_ll = mean_ll[1:].copy()

            proposals.append(
                InstanceProposal(
                    instance_id=-1,
                    class_name=class_name,
                    class_index=class_idx,
                    mask=comp_mask,
                    bbox_2d=(x, y, x + bw, y + bh),
                    area_px=area,
                    mean_confidence=mean_conf,
                    mean_entropy=mean_ent,
                    mean_log_likelihoods=mean_ll,
                    fg_log_likelihoods=fg_ll,
                )
            )

    proposals.sort(key=lambda p: (-p.mean_confidence, -p.area_px, p.class_name))
    for idx, prop in enumerate(proposals):
        prop.instance_id = idx
    return proposals


def _image_digest(rgb: np.ndarray) -> str:
    arr = np.ascontiguousarray(rgb)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


class SegmentGDAPlus:
    """Training-free few-shot 2D segmentation via Multi-Prototype Online Shrinkage GDA."""

    def __init__(
        self,
        extractor: DenseFeatureExtractor | None = None,
        backend: str = "pytorch",
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        input_size: int = 518,
        feature_layers: Sequence[int] = (8, 10, 11),
        device: str = "cuda",
        fp16: bool = True,
        shrinkage_method: str = "oas",
        ridge_alpha: float = 0.05,
        min_eig: float = 1e-4,
        num_foreground_components: int = 2,
        num_background_components: int = 4,
        confidence_threshold: float = 0.55,
        spatial_refine: bool = True,
        guided_radius: int = 8,
        guided_eps: float = 1e-3,
        guided_subsample: int = 2,
        temperature: float | None = None,
        background_bias: float = 0.0,
        max_bg_tokens: int = 1200,
        seed: int = 42,
    ) -> None:
        if extractor is not None:
            self.extractor: DenseFeatureExtractor = extractor
        else:
            self.extractor = create_feature_extractor(
                backend=backend,
                model_name=model_name,
                input_size=input_size,
                feature_layers=feature_layers,
                device=device,
                fp16=fp16,
            )

        self.shrinkage_method = shrinkage_method
        self.ridge_alpha = float(ridge_alpha)
        self.min_eig = float(min_eig)
        self.num_foreground_components = max(1, int(num_foreground_components))
        self.num_background_components = max(1, int(num_background_components))
        self.confidence_threshold = float(confidence_threshold)
        self.spatial_refine = bool(spatial_refine)
        self.guided_radius = int(guided_radius)
        self.guided_eps = float(guided_eps)
        self.guided_subsample = int(guided_subsample)
        self.temperature = None if temperature is None else float(temperature)
        self.background_bias = float(background_bias)
        self.max_bg_tokens = max(64, int(max_bg_tokens))
        self.seed = int(seed)

        self._gda = OnlineShrinkageGDA(
            shrinkage_method=self.shrinkage_method,
            ridge_alpha=self.ridge_alpha,
            min_eig=self.min_eig,
            normalize_features=True,
        )
        # Foreground class names in registration order (class index = i + 1; 0 = background)
        self._class_names: list[str] = []
        # Maps prototype key in self._gda -> class name (BACKGROUND_CLASS_NAME or foreground name)
        self._proto_to_class: dict[str, str] = {}
        # Normalized log mixture weight log(pi_{c,j}) per prototype key
        self._proto_log_weights: dict[str, float] = {}
        # Cached auto-calibrated temperature (updated on fit/register)
        self._calibrated_temperature: float = 1.0
        self.last_register_latency_ms: float = 0.0

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        extractor: DenseFeatureExtractor | None = None,
    ) -> SegmentGDAPlus:
        """Instantiate SegmentGDAPlus from a loaded OmniGDA YAML config dict."""
        p2d = cfg.get("perception_2d", {})
        proj = cfg.get("project", {})
        return cls(
            extractor=extractor,
            backend=p2d.get("backend", "pytorch"),
            model_name=p2d.get("backbone", "vit_small_patch14_dinov2.lvd142m"),
            input_size=int(p2d.get("input_size", 518)),
            feature_layers=tuple(p2d.get("feature_layers", (8, 10, 11))),
            device=proj.get("device", "cuda"),
            fp16=bool(p2d.get("fp16", True)),
            shrinkage_method=p2d.get("shrinkage_method", "oas"),
            ridge_alpha=float(p2d.get("ridge_alpha", 0.05)),
            num_foreground_components=int(p2d.get("num_foreground_components", 2)),
            num_background_components=int(p2d.get("num_background_components", 4)),
            confidence_threshold=float(p2d.get("confidence_threshold", 0.55)),
            spatial_refine=bool(p2d.get("spatial_refine", True)),
            seed=int(proj.get("seed", 42)),
        )

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def is_fitted(self) -> bool:
        return self._gda.precision is not None and len(self._class_names) > 0

    @property
    def effective_temperature(self) -> float:
        if self.temperature is not None:
            return max(self.temperature, 1e-6)
        return max(self._calibrated_temperature, 1.0)

    def _update_calibrated_temperature(self) -> None:
        if self._gda.precision is None or self._gda.dim <= 0:
            self._calibrated_temperature = 1.0
            return
        mean_diag_prec = float(np.trace(self._gda.precision)) / float(self._gda.dim)
        self._calibrated_temperature = max(1.0, float(np.sqrt(max(mean_diag_prec, 1.0))))

    def _extract_fg_tokens(self, feat_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Sample dense feature tokens inside a binary foreground mask."""
        d, hf, wf = feat_map.shape
        mask_f = np.asarray(mask, dtype=np.float32)
        if mask_f.ndim != 2:
            raise ValueError(f"Expected 2D mask (H, W), got shape {mask_f.shape}")
        cov = cv2.resize(mask_f, (wf, hf), interpolation=cv2.INTER_AREA)

        sel = cov >= 0.5
        if int(np.count_nonzero(sel)) < 4:
            sel = cov >= 0.25
        if int(np.count_nonzero(sel)) < 2:
            sel = cov > 0.0

        flat_feats = feat_map.reshape(d, -1).T  # (Hf*Wf, D)
        if int(np.count_nonzero(sel)) > 0:
            tokens = flat_feats[sel.reshape(-1)]
        else:
            # Ultimate fallback: pick top-4 patches with largest mask overlap
            top_idx = np.argsort(cov.reshape(-1))[-min(4, cov.size) :]
            tokens = flat_feats[top_idx]
        return l2_normalize(tokens.astype(np.float64), axis=-1)

    def _extract_bg_tokens(self, feat_map: np.ndarray, combined_fg_mask: np.ndarray) -> np.ndarray:
        """Sample clean background patch tokens strictly outside all dilated foreground masks."""
        d, hf, wf = feat_map.shape
        fg_u8 = (np.asarray(combined_fg_mask) > 0).astype(np.uint8)
        kernel = np.ones((5, 5), dtype=np.uint8)
        dilated = cv2.dilate(fg_u8, kernel, iterations=1).astype(np.float32)
        cov = cv2.resize(dilated, (wf, hf), interpolation=cv2.INTER_AREA)

        bg_sel = cov <= 0.05
        if int(np.count_nonzero(bg_sel)) < 8:
            bg_sel = cov < 0.5
        if int(np.count_nonzero(bg_sel)) == 0:
            bg_sel = cov <= float(np.min(cov)) + 1e-6

        flat_feats = feat_map.reshape(d, -1).T
        tokens = flat_feats[bg_sel.reshape(-1)]
        return l2_normalize(tokens.astype(np.float64), axis=-1)

    def _cluster_tokens(self, tokens: np.ndarray, max_components: int) -> list[np.ndarray]:
        """Partition token features into up to `max_components` sub-prototype clusters."""
        n = tokens.shape[0]
        min_per_cluster = 6
        k = min(max_components, max(1, n // min_per_cluster))
        if k <= 1:
            return [tokens]

        km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=self.seed)
        labels = km.fit_predict(tokens)

        clusters: list[np.ndarray] = []
        small_tokens: list[np.ndarray] = []
        for cid in range(k):
            pts = tokens[labels == cid]
            if pts.shape[0] >= 2:
                clusters.append(pts)
            elif pts.shape[0] > 0:
                small_tokens.append(pts)

        if not clusters:
            return [tokens]

        # Reassign any singleton cluster points to nearest valid cluster centroid
        if small_tokens:
            extra = np.concatenate(small_tokens, axis=0)
            centroids = np.stack([np.mean(c, axis=0) for c in clusters], axis=0)
            dists = np.linalg.norm(extra[:, None, :] - centroids[None, :, :], axis=-1)
            nearest = np.argmin(dists, axis=1)
            for idx, cid in enumerate(nearest):
                clusters[int(cid)] = np.vstack([clusters[int(cid)], extra[idx : idx + 1]])

        clusters.sort(key=lambda c: -c.shape[0])
        return clusters

    def _recompute_class_log_weights(self, target_class: str) -> None:
        """Compute normalized mixture log-weights log(pi_{c,j}) for all prototypes of `target_class`."""
        keys = [k for k in self._gda.class_order if self._proto_to_class.get(k) == target_class]
        if not keys:
            return
        counts = np.array([max(self._gda.classes[k].count, 1) for k in keys], dtype=np.float64)
        probs = counts / float(np.sum(counts))
        for k, p in zip(keys, probs, strict=True):
            self._proto_log_weights[k] = float(np.log(max(p, 1e-12)))

    def _validate_support_samples(self, support_samples: Sequence[SupportSample]) -> None:
        if not support_samples:
            raise ValueError("support_samples cannot be empty")
        for sample in support_samples:
            if sample.rgb is None or sample.mask is None:
                raise ValueError(
                    f"SupportSample for class {sample.class_name!r} requires both rgb and mask"
                )
            if sample.rgb.shape[:2] != sample.mask.shape[:2]:
                raise ValueError(
                    f"RGB shape {sample.rgb.shape[:2]} does not match mask shape {sample.mask.shape[:2]}"
                )

    def fit(self, support_samples: Sequence[SupportSample]) -> SegmentGDAPlus:
        """Fit multi-prototype shrinkage GDA from few-shot 2D support samples."""
        self._validate_support_samples(support_samples)

        # Deduplicate backbone extraction across samples sharing the same RGB image
        feat_cache: dict[str, np.ndarray] = {}
        union_masks: dict[str, np.ndarray] = {}
        fg_tokens_by_class: dict[str, list[np.ndarray]] = {}
        class_order: list[str] = []

        for sample in support_samples:
            assert sample.rgb is not None and sample.mask is not None
            cname = sample.class_name
            if cname == BACKGROUND_CLASS_NAME:
                raise ValueError(f"Class name {BACKGROUND_CLASS_NAME!r} is reserved for background")
            if cname not in fg_tokens_by_class:
                fg_tokens_by_class[cname] = []
                class_order.append(cname)

            digest = _image_digest(sample.rgb)
            if digest not in feat_cache:
                feat_cache[digest] = self.extractor.extract(sample.rgb)
                union_masks[digest] = np.zeros(sample.mask.shape[:2], dtype=bool)
            union_masks[digest] |= np.asarray(sample.mask, dtype=bool)

            fg_tok = self._extract_fg_tokens(feat_cache[digest], sample.mask)
            fg_tokens_by_class[cname].append(fg_tok)

        # Extract background tokens from outside the union foreground mask of each unique image
        bg_tokens_list = [
            self._extract_bg_tokens(feat_cache[digest], union_masks[digest])
            for digest in feat_cache
        ]
        bg_tokens = np.concatenate(bg_tokens_list, axis=0)
        if bg_tokens.shape[0] > self.max_bg_tokens:
            rng = np.random.default_rng(self.seed)
            sel_idx = rng.choice(bg_tokens.shape[0], size=self.max_bg_tokens, replace=False)
            bg_tokens = bg_tokens[np.sort(sel_idx)]

        features_by_proto: dict[str, np.ndarray] = {}
        self._proto_to_class = {}
        self._proto_log_weights = {}

        # Background sub-prototypes
        bg_clusters = self._cluster_tokens(bg_tokens, self.num_background_components)
        for j, cluster in enumerate(bg_clusters):
            pkey = f"{BACKGROUND_CLASS_NAME}__comp_{j}"
            features_by_proto[pkey] = cluster
            self._proto_to_class[pkey] = BACKGROUND_CLASS_NAME

        # Foreground sub-prototypes per class
        for cname in class_order:
            all_fg = np.concatenate(fg_tokens_by_class[cname], axis=0)
            fg_clusters = self._cluster_tokens(all_fg, self.num_foreground_components)
            for j, cluster in enumerate(fg_clusters):
                pkey = f"{cname}__comp_{j}"
                features_by_proto[pkey] = cluster
                self._proto_to_class[pkey] = cname

        self._gda = OnlineShrinkageGDA(
            shrinkage_method=self.shrinkage_method,
            ridge_alpha=self.ridge_alpha,
            min_eig=self.min_eig,
            normalize_features=True,
        )
        self._gda.fit(features_by_proto)
        self._class_names = list(class_order)

        self._recompute_class_log_weights(BACKGROUND_CLASS_NAME)
        for cname in self._class_names:
            self._recompute_class_log_weights(cname)

        self._update_calibrated_temperature()
        return self

    def register_object(
        self,
        class_name: str,
        support_samples: Sequence[SupportSample],
        update_background: bool = False,
    ) -> float:
        """Online-register a novel object class (or append shots to an existing class).

        Uses `OnlineShrinkageGDA.register_online` (Sherman-Morrison-Woodbury rank-k precision
        update) without refitting from raw previous support samples.

        Returns:
            Registration wall-clock latency in milliseconds.
        """
        t0 = time.perf_counter()
        if class_name == BACKGROUND_CLASS_NAME:
            raise ValueError(f"Class name {BACKGROUND_CLASS_NAME!r} is reserved for background")
        self._validate_support_samples(support_samples)

        if not self.is_fitted:
            normalized_samples = [
                SupportSample(
                    class_name=class_name,
                    rgb=s.rgb,
                    mask=s.mask,
                    depth=s.depth,
                    intrinsics=s.intrinsics,
                    camera_to_world=s.camera_to_world,
                    point_cloud_xyz=s.point_cloud_xyz,
                    text_prompt=s.text_prompt,
                    metadata=dict(s.metadata),
                )
                for s in support_samples
            ]
            self.fit(normalized_samples)
            self.last_register_latency_ms = (time.perf_counter() - t0) * 1000.0
            return self.last_register_latency_ms

        feat_cache: dict[str, np.ndarray] = {}
        union_masks: dict[str, np.ndarray] = {}
        fg_list: list[np.ndarray] = []

        for sample in support_samples:
            assert sample.rgb is not None and sample.mask is not None
            digest = _image_digest(sample.rgb)
            if digest not in feat_cache:
                feat_cache[digest] = self.extractor.extract(sample.rgb)
                union_masks[digest] = np.zeros(sample.mask.shape[:2], dtype=bool)
            union_masks[digest] |= np.asarray(sample.mask, dtype=bool)
            fg_list.append(self._extract_fg_tokens(feat_cache[digest], sample.mask))

        fg_tokens = np.concatenate(fg_list, axis=0)

        if class_name not in self._class_names:
            fg_clusters = self._cluster_tokens(fg_tokens, self.num_foreground_components)
            for j, cluster in enumerate(fg_clusters):
                pkey = f"{class_name}__comp_{j}"
                self._gda.register_online(pkey, cluster, use_woodbury=True)
                self._proto_to_class[pkey] = class_name
            self._class_names.append(class_name)
        else:
            existing_keys = [
                k for k in self._gda.class_order if self._proto_to_class.get(k) == class_name
            ]
            if len(existing_keys) == 1:
                self._gda.register_online(existing_keys[0], fg_tokens, use_woodbury=True)
            else:
                means = np.stack([self._gda.classes[k].mean for k in existing_keys], axis=0)
                dists = np.linalg.norm(fg_tokens[:, None, :] - means[None, :, :], axis=-1)
                nearest = np.argmin(dists, axis=1)
                for idx, pkey in enumerate(existing_keys):
                    pts = fg_tokens[nearest == idx]
                    if pts.shape[0] > 0:
                        self._gda.register_online(pkey, pts, use_woodbury=True)

        self._recompute_class_log_weights(class_name)

        if update_background:
            bg_list = [
                self._extract_bg_tokens(feat_cache[d], union_masks[d]) for d in feat_cache
            ]
            bg_tokens = np.concatenate(bg_list, axis=0)
            bg_keys = [
                k
                for k in self._gda.class_order
                if self._proto_to_class.get(k) == BACKGROUND_CLASS_NAME
            ]
            if bg_keys and bg_tokens.shape[0] > 0:
                max_new_bg = min(bg_tokens.shape[0], 128)
                if bg_tokens.shape[0] > max_new_bg:
                    rng = np.random.default_rng(self.seed)
                    idx_sel = rng.choice(bg_tokens.shape[0], size=max_new_bg, replace=False)
                    bg_tokens = bg_tokens[np.sort(idx_sel)]
                bg_means = np.stack([self._gda.classes[k].mean for k in bg_keys], axis=0)
                nearest = np.argmin(
                    np.linalg.norm(bg_tokens[:, None, :] - bg_means[None, :, :], axis=-1), axis=1
                )
                for idx, pkey in enumerate(bg_keys):
                    pts = bg_tokens[nearest == idx]
                    if pts.shape[0] > 0:
                        self._gda.register_online(pkey, pts, use_woodbury=True)
                self._recompute_class_log_weights(BACKGROUND_CLASS_NAME)

        self._update_calibrated_temperature()
        self.last_register_latency_ms = (time.perf_counter() - t0) * 1000.0
        return self.last_register_latency_ms

    def segment(self, rgb: np.ndarray) -> SegmentationResult:
        """Segment an RGB image into background + all registered object classes."""
        if not self.is_fitted:
            raise RuntimeError("SegmentGDAPlus must be fitted before calling segment().")

        t0 = time.perf_counter()
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {rgb.shape}")
        h, w = rgb.shape[:2]

        feat_map = self.extractor.extract(rgb)  # (D, Hf, Wf) float32
        d, hf, wf = feat_map.shape
        flat_feats = feat_map.reshape(d, -1).T  # (Hf*Wf, D)

        # Single vectorized GEMM for all prototype log-likelihoods
        proto_logits = self._gda.predict_logits(flat_feats)  # (N_tokens, P)
        temp = self.effective_temperature
        scaled_logits = proto_logits / temp

        # Aggregate sub-prototypes per class via log-sum-exp (class 0 = background, 1..K = fg)
        all_class_keys = [BACKGROUND_CLASS_NAME, *self._class_names]
        num_classes = len(all_class_keys)
        class_grid_ll = np.empty((num_classes, hf * wf), dtype=np.float32)

        for c_idx, cname in enumerate(all_class_keys):
            p_indices = [
                p_i
                for p_i, pkey in enumerate(self._gda.class_order)
                if self._proto_to_class.get(pkey) == cname
            ]
            log_w = np.array(
                [self._proto_log_weights.get(self._gda.class_order[p_i], 0.0) for p_i in p_indices],
                dtype=np.float64,
            )
            sub_logits = scaled_logits[:, p_indices] + log_w[None, :]
            ll = logsumexp(sub_logits, axis=1)
            if c_idx == 0 and self.background_bias != 0.0:
                ll = ll + self.background_bias
            class_grid_ll[c_idx] = ll.astype(np.float32)

        class_grid_ll = class_grid_ll.reshape(num_classes, hf, wf)

        # Bilinear upsampling to full image resolution (C, H, W)
        upsampled_ll = np.empty((num_classes, h, w), dtype=np.float32)
        for c_idx in range(num_classes):
            upsampled_ll[c_idx] = cv2.resize(
                class_grid_ll[c_idx], (w, h), interpolation=cv2.INTER_LINEAR
            )

        # RGB-guided edge-preserving spatial refinement
        if self.spatial_refine:
            refined_ll = fast_color_guided_filter(
                guide_rgb=rgb,
                src_maps=upsampled_ll,
                radius=self.guided_radius,
                eps=self.guided_eps,
                subsample=self.guided_subsample,
            )
        else:
            refined_ll = upsampled_ll

        probs = softmax(refined_ll.astype(np.float64), axis=0).astype(np.float32)  # (C, H, W)
        entropy_map = normalized_entropy(probs, axis=0).astype(np.float32)

        raw_class_map = np.argmax(probs, axis=0).astype(np.int32)
        raw_max_prob = np.max(probs, axis=0).astype(np.float32)

        # Reject low-confidence foreground predictions back to background
        class_map = raw_class_map.copy()
        low_conf_fg = (class_map > 0) & (raw_max_prob < self.confidence_threshold)
        class_map[low_conf_fg] = 0

        # Calibrated per-pixel confidence of the assigned class in [0, 1]
        confidence_map = np.clip(
            np.take_along_axis(probs, class_map[None, :, :], axis=0)[0],
            0.0,
            1.0,
        ).astype(np.float32)

        masks = {
            cname: (class_map == idx) for idx, cname in enumerate(self._class_names, start=1)
        }
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SegmentationResult(
            class_names=list(self._class_names),
            class_map=class_map,
            masks=masks,
            confidence_map=confidence_map,
            log_likelihoods=refined_ll.astype(np.float32),
            entropy_map=entropy_map,
            latency_ms=latency_ms,
        )

    def save_state(self, path: str | Path) -> Path:
        """Persist fitted GDA prototype memory and hyperparameters to an `.npz` file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save_state before SegmentGDAPlus is fitted.")
        assert (
            self._gda.scatter is not None
            and self._gda.covariance is not None
            and self._gda.precision is not None
        )

        out_path = Path(path)
        if out_path.suffix != ".npz":
            out_path = out_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        proto_order = list(self._gda.class_order)
        proto_means = np.stack([self._gda.classes[k].mean for k in proto_order], axis=0)
        proto_counts = np.array([self._gda.classes[k].count for k in proto_order], dtype=np.int64)

        meta = {
            "version": 1,
            "dim": int(self._gda.dim),
            "total_count": int(self._gda.total_count),
            "shrinkage_alpha": float(self._gda.shrinkage_alpha),
            "shrinkage_method": self.shrinkage_method,
            "ridge_alpha": self.ridge_alpha,
            "min_eig": self.min_eig,
            "normalize_features": bool(self._gda.normalize_features),
            "proto_order": proto_order,
            "class_names": list(self._class_names),
            "proto_to_class": dict(self._proto_to_class),
            "proto_log_weights": dict(self._proto_log_weights),
            "calibrated_temperature": float(self._calibrated_temperature),
            "temperature": self.temperature,
            "confidence_threshold": self.confidence_threshold,
            "spatial_refine": self.spatial_refine,
            "guided_radius": self.guided_radius,
            "guided_eps": self.guided_eps,
            "guided_subsample": self.guided_subsample,
            "background_bias": self.background_bias,
            "num_foreground_components": self.num_foreground_components,
            "num_background_components": self.num_background_components,
        }

        np.savez(
            out_path,
            meta_json=np.array(json.dumps(meta)),
            scatter=self._gda.scatter,
            covariance=self._gda.covariance,
            precision=self._gda.precision,
            proto_means=proto_means,
            proto_counts=proto_counts,
        )
        return out_path

    def load_state(self, path: str | Path) -> SegmentGDAPlus:
        """Restore fitted GDA prototype memory and hyperparameters from an `.npz` file."""
        in_path = Path(path)
        if not in_path.exists() and in_path.with_suffix(".npz").exists():
            in_path = in_path.with_suffix(".npz")
        if not in_path.exists():
            raise FileNotFoundError(f"SegmentGDAPlus state file not found: {in_path}")

        with np.load(in_path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"].item()))
            scatter = np.array(data["scatter"], dtype=np.float64)
            covariance = np.array(data["covariance"], dtype=np.float64)
            precision = np.array(data["precision"], dtype=np.float64)
            proto_means = np.array(data["proto_means"], dtype=np.float64)
            proto_counts = np.array(data["proto_counts"], dtype=np.int64)

        self.shrinkage_method = str(meta["shrinkage_method"])
        self.ridge_alpha = float(meta["ridge_alpha"])
        self.min_eig = float(meta["min_eig"])
        self.confidence_threshold = float(meta["confidence_threshold"])
        self.spatial_refine = bool(meta["spatial_refine"])
        self.guided_radius = int(meta["guided_radius"])
        self.guided_eps = float(meta["guided_eps"])
        self.guided_subsample = int(meta["guided_subsample"])
        self.temperature = None if meta["temperature"] is None else float(meta["temperature"])
        self.background_bias = float(meta["background_bias"])
        self.num_foreground_components = int(meta["num_foreground_components"])
        self.num_background_components = int(meta["num_background_components"])

        gda = OnlineShrinkageGDA(
            shrinkage_method=self.shrinkage_method,
            ridge_alpha=self.ridge_alpha,
            min_eig=self.min_eig,
            normalize_features=bool(meta["normalize_features"]),
        )
        gda.dim = int(meta["dim"])
        gda.total_count = int(meta["total_count"])
        gda.shrinkage_alpha = float(meta["shrinkage_alpha"])
        gda.scatter = scatter
        gda.covariance = covariance
        gda.precision = precision

        proto_order = list(meta["proto_order"])
        gda.class_order = proto_order
        gda.classes = {
            pkey: OnlineClassState(
                class_name=pkey,
                mean=proto_means[idx].copy(),
                count=int(proto_counts[idx]),
            )
            for idx, pkey in enumerate(proto_order)
        }
        gda._rebuild_discriminant_cache()

        self._gda = gda
        self._class_names = list(meta["class_names"])
        self._proto_to_class = {str(k): str(v) for k, v in meta["proto_to_class"].items()}
        self._proto_log_weights = {
            str(k): float(v) for k, v in meta["proto_log_weights"].items()
        }
        self._calibrated_temperature = float(meta["calibrated_temperature"])
        return self

