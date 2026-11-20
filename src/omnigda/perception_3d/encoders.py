"""4-Modality feature extractors for partial-view tabletop 3D objects.

Implements:
1. ``Geometric3DEncoder``: Rotation-invariant, partial-cloud-aware 3D geometric descriptor
   combining Open3D FPFH (mean + std pooling), multi-scale covariance eigenvalue shape
   descriptors, height-profile & radial-distance histograms, with an optional pluggable
   linear projection adapter hook (for T8 distilled 3D projection weights).
2. ``PhysicalSizeEncoder``: Absolute metric invariant descriptor (horizontal extents,
   vertical height, volume, aspect ratios, height above table) to separate visually
   identical objects of differing physical scales.
3. ``OpenCLIPMultimodalEncoder``: GPU FP16 OpenCLIP (ViT-B-32) image crop encoder (with
   pluggable TensorRT hook for T5) and multi-template prompt-ensembled text encoder for
   zero-shot open-vocabulary registration.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import open3d as o3d
import open_clip
import torch
from PIL import Image
from scipy.spatial import cKDTree

from omnigda.core.gda_math import l2_normalize
from omnigda.perception_3d.pointcloud import compute_oriented_bounding_box, subsample_pointcloud

DEFAULT_CLIP_PROMPT_TEMPLATES: tuple[str, ...] = (
    "a tabletop photo of a {name}",
    "a 3D {name} object",
    "a {name} for robotic grasping",
)


def _eigenvalue_shape_descriptors(pts: np.ndarray) -> np.ndarray:
    """Compute 6 standard 3D covariance eigenvalue shape descriptors from points (M, 3).

    Returns:
        Array of shape (6,): [linearity, planarity, sphericity, omnivariance,
        anisotropy, surface_variation].
    """
    if pts.shape[0] < 4:
        return np.zeros(6, dtype=np.float64)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = (centered.T @ centered) / float(pts.shape[0])
    cov = 0.5 * (cov + cov.T)
    evals = np.linalg.eigvalsh(cov)
    evals = np.maximum(evals[::-1], 1e-12)  # l1 >= l2 >= l3 > 0
    total = float(np.sum(evals))
    e = evals / total  # normalized eigenvalues summing to 1
    l1, l2, l3 = float(e[0]), float(e[1]), float(e[2])
    linearity = (l1 - l2) / max(l1, 1e-12)
    planarity = (l2 - l3) / max(l1, 1e-12)
    sphericity = l3 / max(l1, 1e-12)
    omnivariance = float((l1 * l2 * l3) ** (1.0 / 3.0))
    anisotropy = (l1 - l3) / max(l1, 1e-12)
    surface_variation = l3 / max(l1 + l2 + l3, 1e-12)
    return np.array(
        [linearity, planarity, sphericity, omnivariance, anisotropy, surface_variation],
        dtype=np.float64,
    )


class Geometric3DEncoder:
    """Rotation-invariant, partial-cloud-aware 3D geometric feature encoder.

    Raw descriptor composition (108 dims):
    - Open3D FPFH mean-pooled histogram (33 dims, L2-normalized block)
    - Open3D FPFH std-pooled histogram (33 dims, L2-normalized block)
    - Multi-scale covariance eigenvalue descriptors: global + 2 local radial scales (18 dims)
    - Normalized vertical height-profile histogram (12 dims)
    - Normalized horizontal radial-distance histogram (12 dims)

    Optionally projects the 108-dim descriptor through a pluggable linear adapter
    ``out = l2_normalize(raw @ W.T + b)`` to support distilled 3D weights from T8.
    """

    RAW_DIM: int = 108

    def __init__(
        self,
        max_points: int = 512,
        num_height_bins: int = 12,
        num_radial_bins: int = 12,
        local_query_points: int = 64,
        projection_matrix: np.ndarray | None = None,
        projection_bias: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        self.max_points = int(max_points)
        self.num_height_bins = int(num_height_bins)
        self.num_radial_bins = int(num_radial_bins)
        self.local_query_points = int(local_query_points)
        self.seed = int(seed)
        self._raw_dim = 66 + 18 + self.num_height_bins + self.num_radial_bins
        self._proj_weight: np.ndarray | None = None
        self._proj_bias: np.ndarray | None = None
        if projection_matrix is not None:
            self.set_projection_adapter(projection_matrix, projection_bias)

    @property
    def feature_dim(self) -> int:
        if self._proj_weight is not None:
            return int(self._proj_weight.shape[0])
        return self._raw_dim

    def set_projection_adapter(
        self,
        projection_matrix: np.ndarray | None,
        projection_bias: np.ndarray | None = None,
    ) -> None:
        """Attach or remove a linear projection adapter ``W`` of shape ``(D_out, D_raw)``."""
        if projection_matrix is None:
            self._proj_weight = None
            self._proj_bias = None
            return
        w = np.asarray(projection_matrix, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] != self._raw_dim:
            raise ValueError(
                f"projection_matrix must have shape (D_out, {self._raw_dim}), got {w.shape}"
            )
        b: np.ndarray | None = None
        if projection_bias is not None:
            b = np.asarray(projection_bias, dtype=np.float64)
            if b.shape != (w.shape[0],):
                raise ValueError(
                    f"projection_bias must have shape ({w.shape[0]},), got {b.shape}"
                )
        self._proj_weight = w
        self._proj_bias = b

    def load_projection_adapter(self, path: str | Path) -> None:
        """Load linear adapter weights from a ``.npz`` file containing ``weight`` (and optional ``bias``)."""
        data = np.load(Path(path), allow_pickle=False)
        weight = data["weight"]
        bias = data["bias"] if "bias" in data else None
        self.set_projection_adapter(weight, bias)

    def encode(self, xyz: np.ndarray) -> np.ndarray:
        """Encode a 3D point cloud ``(N, 3)`` into a unit-L2-normalized geometric descriptor."""
        pts_in = np.asarray(xyz, dtype=np.float64)
        if pts_in.ndim != 2 or pts_in.shape[1] != 3 or pts_in.shape[0] < 6:
            return np.zeros(self.feature_dim, dtype=np.float32)

        pts = subsample_pointcloud(pts_in.astype(np.float32), self.max_points, seed=self.seed).astype(
            np.float64
        )
        centroid = np.mean(pts, axis=0)
        rel = pts - centroid[None, :]
        bbox_span = np.max(pts, axis=0) - np.min(pts, axis=0)
        diag = float(np.linalg.norm(bbox_span))
        if diag < 1e-5:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # 1. Scale-adaptive Open3D FPFH (mean + std pooled)
        r_normal = max(0.14 * diag, 0.008)
        r_fpfh = max(0.28 * diag, 0.016)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=r_normal, max_nn=30)
        )
        # Consistently orient normals outward from centroid for rotation & viewpoint invariance
        normals = np.asarray(pcd.normals, dtype=np.float64)
        if normals.shape == pts.shape:
            dots = np.sum(normals * rel, axis=1)
            flip = dots < 0.0
            normals[flip] *= -1.0
            pcd.normals = o3d.utility.Vector3dVector(normals)

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=r_fpfh, max_nn=64),
        )
        fpfh_arr = np.asarray(fpfh.data, dtype=np.float64)  # (33, N)
        if fpfh_arr.ndim == 2 and fpfh_arr.shape == (33, pts.shape[0]):
            fpfh_mean = np.mean(fpfh_arr, axis=1)
            fpfh_std = np.std(fpfh_arr, axis=1)
        else:
            fpfh_mean = np.zeros(33, dtype=np.float64)
            fpfh_std = np.zeros(33, dtype=np.float64)

        fpfh_mean_norm = l2_normalize(fpfh_mean)
        fpfh_std_norm = l2_normalize(fpfh_std)

        # 2. Multi-scale covariance eigenvalue shape descriptors (global + 2 local scales)
        eig_global = _eigenvalue_shape_descriptors(pts)
        tree = cKDTree(pts)
        n_q = min(self.local_query_points, pts.shape[0])
        q_idx = np.linspace(0, pts.shape[0] - 1, num=n_q, dtype=np.int64)
        q_pts = pts[q_idx]

        eig_locals: list[np.ndarray] = []
        for frac in (0.22, 0.42):
            rad = max(frac * diag, 0.01)
            nbr_lists = tree.query_ball_point(q_pts, r=rad)
            accum = np.zeros(6, dtype=np.float64)
            valid_cnt = 0
            for nbrs in nbr_lists:
                if len(nbrs) >= 5:
                    accum += _eigenvalue_shape_descriptors(pts[nbrs])
                    valid_cnt += 1
            if valid_cnt > 0:
                eig_locals.append(accum / float(valid_cnt))
            else:
                eig_locals.append(eig_global.copy())

        eig_block = np.concatenate([eig_global, eig_locals[0], eig_locals[1]], axis=0)
        eig_block_norm = l2_normalize(eig_block)

        # 3. Normalized height-profile histogram (gravity-aligned Z distribution)
        z = pts[:, 2]
        z_min = float(np.min(z))
        z_span = max(float(np.max(z)) - z_min, 1e-4)
        z_norm = np.clip((z - z_min) / z_span, 0.0, 1.0)
        h_hist, _ = np.histogram(z_norm, bins=self.num_height_bins, range=(0.0, 1.0))
        h_hist = l2_normalize(h_hist.astype(np.float64))

        # 4. Normalized radial-distance histogram in horizontal XY plane
        r_xy = np.hypot(rel[:, 0], rel[:, 1])
        r_scale = max(float(np.percentile(r_xy, 95)), 1e-4)
        r_norm = np.clip(r_xy / r_scale, 0.0, 1.2)
        r_hist, _ = np.histogram(r_norm, bins=self.num_radial_bins, range=(0.0, 1.2))
        r_hist = l2_normalize(r_hist.astype(np.float64))

        # Balanced block weighting before final L2 normalization
        raw = np.concatenate(
            [
                1.0 * fpfh_mean_norm,
                0.7 * fpfh_std_norm,
                1.2 * eig_block_norm,
                0.8 * h_hist,
                0.8 * r_hist,
            ],
            axis=0,
        )
        raw = l2_normalize(raw)

        if self._proj_weight is not None:
            proj = raw @ self._proj_weight.T
            if self._proj_bias is not None:
                proj = proj + self._proj_bias
            return l2_normalize(proj).astype(np.float32)

        return raw.astype(np.float32)


class PhysicalSizeEncoder:
    """Encodes absolute physical metric invariants from a 3D object's OBB and position.

    Features are expressed in well-conditioned decimeter / log-volume units (NOT L2-normalized)
    so visually identical objects of different physical scales (e.g., short vs. tall cylinders,
    small vs. large cubes) remain cleanly separable in metric space.
    """

    FEATURE_DIM: int = 12

    @property
    def feature_dim(self) -> int:
        return self.FEATURE_DIM

    def encode(
        self,
        obb_extent: np.ndarray | None = None,
        obb_center: np.ndarray | None = None,
        table_z: float = 0.0,
        xyz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode metric size invariants from OBB parameters (or compute OBB from ``xyz``)."""
        if obb_extent is None:
            if xyz is None:
                raise ValueError("Either obb_extent or xyz must be provided to PhysicalSizeEncoder")
            obb_center_calc, obb_extent, _ = compute_oriented_bounding_box(xyz)
            if obb_center is None:
                obb_center = obb_center_calc

        ext = np.maximum(np.asarray(obb_extent, dtype=np.float64), 1e-4)
        dx, dy, dz = float(ext[0]), float(ext[1]), float(ext[2])
        h_max = max(dx, dy)
        h_min = min(dx, dy)

        vol = dx * dy * dz
        diag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        cbrt_vol = float(vol ** (1.0 / 3.0))
        log_vol = float(np.log(max(vol, 1e-8))) * 0.25

        xy_aspect = float(np.clip(h_max / max(h_min, 1e-4), 1.0, 10.0)) * 0.25
        mean_horiz = 0.5 * (h_max + h_min)
        tallness = float(np.clip(dz / max(mean_horiz, 1e-4), 0.05, 10.0)) * 0.5
        flatness = float(min(dx, dy, dz) / max(dx, dy, dz, 1e-4))
        footprint_scale = float(np.sqrt(max(h_max * h_min, 1e-8))) * 10.0

        if obb_center is not None:
            cz = float(obb_center[2])
            bottom_above_table = max(0.0, (cz - 0.5 * dz) - float(table_z)) * 10.0
            center_above_table = max(0.0, cz - float(table_z)) * 10.0
        else:
            bottom_above_table = 0.0
            center_above_table = 0.5 * dz * 10.0

        feat = np.array(
            [
                h_max * 10.0,
                h_min * 10.0,
                dz * 10.0,
                diag * 10.0,
                cbrt_vol * 10.0,
                log_vol,
                xy_aspect,
                tallness,
                flatness,
                bottom_above_table,
                center_above_table,
                footprint_scale,
            ],
            dtype=np.float32,
        )
        return feat


class OpenCLIPMultimodalEncoder:
    """OpenCLIP visual crop + multi-template text encoder on GPU (FP16).

    Supports:
    - Tightly cropped & background-masked object RGB encoding via ``encode_image_crop``.
    - Pluggable TensorRT/ONNX image encoder hook for T5 deployment acceleration.
    - Multi-template prompt-ensembled text encoding via ``encode_text`` for zero-shot
      open-vocabulary registration.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        fp16: bool = True,
        prompt_templates: tuple[str, ...] = DEFAULT_CLIP_PROMPT_TEMPLATES,
        image_encoder_hook: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.prompt_templates = tuple(prompt_templates)
        self._image_encoder_hook = image_encoder_hook
        self._text_cache: dict[tuple[str, bool], np.ndarray] = {}

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embed_dim = int(self.model.visual.output_dim)

    def set_image_encoder_hook(
        self,
        hook: Callable[[np.ndarray], np.ndarray] | None,
    ) -> None:
        """Register (or clear) a TensorRT/ONNX image encoder callable.

        The callable takes a preprocessed float32 tensor of shape ``(B, 3, 224, 224)`` and
        returns raw or normalized float32 embeddings of shape ``(B, embed_dim)``.
        """
        self._image_encoder_hook = hook

    @staticmethod
    def _extract_masked_square_crop(
        rgb: np.ndarray,
        mask: np.ndarray | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        pad_ratio: float = 0.12,
        bg_fill: int = 128,
    ) -> Image.Image:
        """Extract a tightly cropped, background-masked, aspect-preserving square PIL image."""
        img = np.asarray(rgb, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {img.shape}")
        h, w = img.shape[:2]

        mask_bool: np.ndarray | None = None
        if mask is not None:
            mask_bool = np.asarray(mask, dtype=bool)
            if mask_bool.shape != (h, w):
                raise ValueError(f"Mask shape {mask_bool.shape} != image spatial shape {(h, w)}")

        x1, y1, x2, y2 = 0, 0, w, h
        if bbox is not None and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
            x1, y1, x2, y2 = (
                int(np.clip(bbox[0], 0, w - 1)),
                int(np.clip(bbox[1], 0, h - 1)),
                int(np.clip(bbox[2], 1, w)),
                int(np.clip(bbox[3], 1, h)),
            )
        elif mask_bool is not None and np.any(mask_bool):
            ys, xs = np.nonzero(mask_bool)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1

        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        pad_x = int(round(bw * pad_ratio))
        pad_y = int(round(bh * pad_ratio))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = img[cy1:cy2, cx1:cx2].copy()
        if mask_bool is not None:
            crop_mask = mask_bool[cy1:cy2, cx1:cx2]
            if np.any(crop_mask):
                crop[~crop_mask] = np.uint8(bg_fill)

        ch, cw = crop.shape[:2]
        side = max(ch, cw, 1)
        square = np.full((side, side, 3), np.uint8(bg_fill), dtype=np.uint8)
        off_y = (side - ch) // 2
        off_x = (side - cw) // 2
        square[off_y : off_y + ch, off_x : off_x + cw] = crop
        return Image.fromarray(square, mode="RGB")

    def encode_image_crop(
        self,
        rgb: np.ndarray,
        mask: np.ndarray | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        pad_ratio: float = 0.12,
        bg_fill: int = 128,
    ) -> np.ndarray:
        """Encode a tightly cropped and background-masked object RGB image into ``(D,)`` float32."""
        pil_crop = self._extract_masked_square_crop(
            rgb=rgb,
            mask=mask,
            bbox=bbox,
            pad_ratio=pad_ratio,
            bg_fill=bg_fill,
        )
        tensor = self.preprocess(pil_crop).unsqueeze(0)  # (1, 3, 224, 224) float32

        if self._image_encoder_hook is not None:
            feats = np.asarray(
                self._image_encoder_hook(tensor.numpy().astype(np.float32)),
                dtype=np.float32,
            )
            if feats.ndim == 2:
                feats = feats[0]
            return l2_normalize(feats.astype(np.float64)).astype(np.float32)

        tensor = tensor.to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    emb = self.model.encode_image(tensor)
            else:
                emb = self.model.encode_image(tensor)
            emb = emb.float()
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return emb[0].detach().cpu().numpy().astype(np.float32)

    def _encode_raw_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            if self.fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    emb = self.model.encode_text(tokens)
            else:
                emb = self.model.encode_text(tokens)
            emb = emb.float()
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return emb.detach().cpu().numpy().astype(np.float32)

    def encode_text(
        self,
        prompts: list[str] | str,
        use_templates: bool = True,
    ) -> np.ndarray:
        """Compute multi-template prompt-ensembled L2-normalized CLIP text embeddings.

        Args:
            prompts: Either a single class/object name ``str`` (returns ``(D,)``) or a
                ``list[str]`` of class/object names (returns ``(M, D)``).
            use_templates: If True, ensembles each prompt over ``self.prompt_templates``.

        Returns:
            L2-normalized float32 embeddings of shape ``(D,)`` for a single string or
            ``(len(prompts), D)`` for a list of strings.
        """
        single = isinstance(prompts, str)
        prompt_list = [prompts] if single else list(prompts)
        if not prompt_list:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(prompt_list)
        uncached_indices: list[int] = []
        expanded_texts: list[str] = []
        spans: list[tuple[int, int]] = []

        for idx, raw_p in enumerate(prompt_list):
            clean_p = raw_p.strip()
            key = (clean_p, bool(use_templates))
            if key in self._text_cache:
                results[idx] = self._text_cache[key]
                continue
            uncached_indices.append(idx)
            start = len(expanded_texts)
            if use_templates and self.prompt_templates:
                for tmpl in self.prompt_templates:
                    expanded_texts.append(tmpl.format(name=clean_p))
            else:
                expanded_texts.append(clean_p)
            spans.append((start, len(expanded_texts)))

        if expanded_texts:
            all_embs = self._encode_raw_texts(expanded_texts)
            for idx, (s, e) in zip(uncached_indices, spans, strict=True):
                mean_emb = np.mean(all_embs[s:e].astype(np.float64), axis=0)
                norm_emb = l2_normalize(mean_emb).astype(np.float32)
                key = (prompt_list[idx].strip(), bool(use_templates))
                self._text_cache[key] = norm_emb
                results[idx] = norm_emb

        stacked = np.stack([r for r in results if r is not None], axis=0).astype(np.float32)
        return stacked[0] if single else stacked
