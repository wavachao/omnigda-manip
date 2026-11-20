"""Partial-View Aware Multimodal 3D GMDA Classifier with Open-Vocabulary Support.

Extends *Multimodal 3D Few-Shot Classification via Gaussian Mixture Discriminant Analysis*
to partial-view robotic tabletop manipulation:
- Modality-specific multi-component shrinkage GMDA models (``geom_3d``, ``size_3d``,
  ``visual_clip``) built on ``OnlineShrinkageGDA`` with Bayesian Normal-Inverse-Wishart
  covariance stabilization for 1-shot to many-shot regimes.
- Zero-shot open-vocabulary registration via prompt-ensembled OpenCLIP text prototypes
  (``text_clip``) and seamless fusion with external 2D segmentation logits (``seg2d``).
- Uncertainty-gated (entropy-weighted) Bayesian fusion modulated by partial point cloud
  completeness reliability.
- Calibrated Mahalanobis OOD distance scoring to reject unregistered distractors.
- Zero-pickle ``.npz`` memory persistence (``save_memory`` / ``load_memory``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from omnigda.core.config import load_config
from omnigda.core.gda_math import (
    OnlineShrinkageGDA,
    entropy_gated_bayesian_fusion,
    invert_spd,
    l2_normalize,
)
from omnigda.core.schemas import Object3DDetection, RGBDFrame, SupportSample
from omnigda.perception_3d.encoders import (
    Geometric3DEncoder,
    OpenCLIPMultimodalEncoder,
    PhysicalSizeEncoder,
)
from omnigda.perception_3d.pointcloud import (
    compute_oriented_bounding_box,
    confidence_weighted_backproject,
    estimate_pointcloud_completeness,
)


class OnlineShrinkageGMDA:
    """Multi-component tied-covariance Shrinkage GMDA with Bayesian covariance stabilization.

    Wraps ``OnlineShrinkageGDA`` to support up to ``num_components`` Gaussian sub-clusters per
    class (e.g., distinct partial-view poses), while using a dimension-calibrated Normal-Inverse-
    Wishart covariance floor so 1-shot registration and cross-modality Mahalanobis distances
    remain well-conditioned across 12-D, 108-D, and 512-D spaces.
    """

    def __init__(
        self,
        num_components: int = 2,
        shrinkage_method: str = "oas",
        ridge_alpha: float = 0.08,
        normalize_features: bool = True,
        prior_var: float = 0.025,
        prior_strength: float = 4.0,
        ref_dim: float = 12.0,
        seed: int = 42,
    ) -> None:
        self.num_components = max(int(num_components), 1)
        self.shrinkage_method = shrinkage_method
        self.ridge_alpha = float(ridge_alpha)
        self.normalize_features = bool(normalize_features)
        self.prior_var = float(prior_var)
        self.prior_strength = float(prior_strength)
        self.ref_dim = float(ref_dim)
        self.seed = int(seed)

        self.dim: int = 0
        self.class_order: list[str] = []
        self._raw_features: dict[str, np.ndarray] = {}  # class_name -> (N_k, D) float64
        self._gda: OnlineShrinkageGDA | None = None
        self._comp_keys_by_class: dict[str, list[str]] = {}
        self._comp_log_priors: dict[str, np.ndarray] = {}
        self._comp_means: np.ndarray | None = None  # (M_total, D)
        self._comp_index_map: dict[str, list[int]] = {}
        self._cal_precision: np.ndarray | None = None  # (D, D)
        self._cal_weights: np.ndarray | None = None  # (M_total, D)
        self._cal_bias: np.ndarray | None = None  # (M_total,)

    @property
    def is_fitted(self) -> bool:
        return self._cal_precision is not None and len(self.class_order) > 0

    def _prep(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if self.normalize_features:
            arr = l2_normalize(arr, axis=-1)
        return arr

    def add_samples(self, class_name: str, features: np.ndarray) -> None:
        """Add one or more support feature vectors for ``class_name`` and update GMDA state."""
        feats = self._prep(features)
        if feats.shape[0] == 0:
            return
        if self.dim == 0:
            self.dim = int(feats.shape[1])
        elif feats.shape[1] != self.dim:
            raise ValueError(f"Feature dim mismatch: expected {self.dim}, got {feats.shape[1]}")

        if class_name in self._raw_features:
            self._raw_features[class_name] = np.vstack([self._raw_features[class_name], feats])
        else:
            self._raw_features[class_name] = feats.copy()
            self.class_order.append(class_name)

        self._refit()

    def _cluster_class(self, feats: np.ndarray) -> list[np.ndarray]:
        n = feats.shape[0]
        if self.num_components <= 1 or n < 2 * self.num_components:
            return [feats]
        spread = float(np.mean(np.var(feats, axis=0)))
        if spread < 1e-8:
            return [feats]
        km = KMeans(
            n_clusters=self.num_components,
            n_init=5,
            random_state=self.seed,
        )
        labels = km.fit_predict(feats)
        clusters = [feats[labels == k] for k in range(self.num_components) if np.any(labels == k)]
        return clusters if clusters else [feats]

    def _refit(self) -> None:
        comp_features: dict[str, np.ndarray] = {}
        self._comp_keys_by_class.clear()
        self._comp_log_priors.clear()
        self._comp_index_map.clear()

        total_samples = 0
        total_components = 0
        comp_means_list: list[np.ndarray] = []

        for cname in self.class_order:
            feats = self._raw_features[cname]
            clusters = self._cluster_class(feats)
            n_class = float(feats.shape[0])
            keys: list[str] = []
            priors: list[float] = []
            indices: list[int] = []
            for m, sub in enumerate(clusters):
                key = f"{cname}__c{m}"
                comp_features[key] = sub
                keys.append(key)
                priors.append(float(sub.shape[0]) / n_class)
                indices.append(total_components)
                comp_means_list.append(np.mean(sub, axis=0))
                total_components += 1
            self._comp_keys_by_class[cname] = keys
            self._comp_log_priors[cname] = np.log(np.clip(priors, 1e-8, 1.0))
            self._comp_index_map[cname] = indices
            total_samples += feats.shape[0]

        gda = OnlineShrinkageGDA(
            shrinkage_method=self.shrinkage_method,
            ridge_alpha=self.ridge_alpha,
            min_eig=1e-5,
            normalize_features=False,
        )
        gda.fit(comp_features)
        self._gda = gda

        d = self.dim
        dof = max(total_samples - total_components, 0)
        prior_cov = self.prior_var * np.eye(d, dtype=np.float64)

        if dof == 0 or gda.covariance is None:
            cov_cal = prior_cov
        else:
            dim_scale = max(float(d) / self.ref_dim, 1.0)
            gda_cov_scaled = gda.covariance * dim_scale
            weight_emp = float(dof) / float(dof + self.prior_strength)
            cov_cal = weight_emp * gda_cov_scaled + (1.0 - weight_emp) * prior_cov
            cov_cal = 0.5 * (cov_cal + cov_cal.T)

        self._cal_precision = invert_spd(cov_cal)
        self._comp_means = np.stack(comp_means_list, axis=0)
        self._cal_weights = self._comp_means @ self._cal_precision
        self._cal_bias = -0.5 * np.sum(self._cal_weights * self._comp_means, axis=1)

    def _component_mahalanobis_sq(self, x: np.ndarray) -> np.ndarray:
        if (
            self._cal_precision is None
            or self._cal_weights is None
            or self._cal_bias is None
        ):
            raise RuntimeError("OnlineShrinkageGMDA must be fitted before prediction.")
        feats = self._prep(x)
        x_quad = np.sum((feats @ self._cal_precision) * feats, axis=1, keepdims=True)
        dist_sq = x_quad - 2.0 * (feats @ self._cal_weights.T) - 2.0 * self._cal_bias[None, :]
        return np.maximum(dist_sq, 0.0)

    def predict_class_scores(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-class GMDA log-likelihoods and min Mahalanobis squared distances."""
        d2_comp = self._component_mahalanobis_sq(x)[0]
        k = len(self.class_order)
        ll = np.empty(k, dtype=np.float64)
        min_d2 = np.empty(k, dtype=np.float64)
        for i, cname in enumerate(self.class_order):
            idxs = self._comp_index_map[cname]
            d2_sub = d2_comp[idxs]
            log_pi = self._comp_log_priors[cname]
            ll[i] = float(logsumexp(log_pi - 0.5 * d2_sub))
            min_d2[i] = float(np.min(d2_sub))
        return ll, min_d2

    def to_state_dict(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        meta = {
            "num_components": self.num_components,
            "shrinkage_method": self.shrinkage_method,
            "ridge_alpha": self.ridge_alpha,
            "normalize_features": self.normalize_features,
            "prior_var": self.prior_var,
            "prior_strength": self.prior_strength,
            "ref_dim": self.ref_dim,
            "seed": self.seed,
            "dim": self.dim,
            "class_order": list(self.class_order),
        }
        arrays = {cname: arr.astype(np.float64) for cname, arr in self._raw_features.items()}
        return meta, arrays

    @classmethod
    def from_state_dict(
        cls,
        meta: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> OnlineShrinkageGMDA:
        model = cls(
            num_components=int(meta["num_components"]),
            shrinkage_method=str(meta["shrinkage_method"]),
            ridge_alpha=float(meta["ridge_alpha"]),
            normalize_features=bool(meta["normalize_features"]),
            prior_var=float(meta["prior_var"]),
            prior_strength=float(meta["prior_strength"]),
            ref_dim=float(meta["ref_dim"]),
            seed=int(meta["seed"]),
        )
        for cname in meta["class_order"]:
            if cname in arrays and arrays[cname].shape[0] > 0:
                model._raw_features[cname] = np.asarray(arrays[cname], dtype=np.float64)
                model.class_order.append(cname)
                model.dim = int(arrays[cname].shape[1])
        if model.class_order:
            model._refit()
        return model


class PartialAwareMultimodalGMDA:
    """Partial-view aware multimodal 3D GMDA classifier with open-vocabulary registration."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        geom_encoder: Geometric3DEncoder | None = None,
        size_encoder: PhysicalSizeEncoder | None = None,
        clip_encoder: OpenCLIPMultimodalEncoder | None = None,
        enable_clip: bool = True,
        device: str | None = None,
        num_gmm_components: int | None = None,
        shrinkage_alpha: float | None = None,
        ood_mahalanobis_threshold: float | None = None,
        fusion_temperature: float | None = None,
        clip_logit_scale: float = 28.0,
    ) -> None:
        cfg_all = config if config is not None else load_config()
        cfg3d = cfg_all.get("perception_3d", {})
        cfg_proj = cfg_all.get("project", {})

        self.seed = int(cfg_proj.get("seed", 42))
        self.device = device if device is not None else str(cfg_proj.get("device", "cuda"))
        self.clip_model_name = str(cfg3d.get("clip_model", "ViT-B-32"))
        self.clip_pretrained = str(cfg3d.get("clip_pretrained", "laion2b_s34b_b79k"))
        self.point_num_samples = int(cfg3d.get("point_num_samples", 512))
        self.depth_min_m = float(cfg3d.get("depth_min_m", 0.15))
        self.depth_max_m = float(cfg3d.get("depth_max_m", 2.0))
        self.confidence_denoise_thresh = float(cfg3d.get("confidence_denoise_thresh", 0.45))
        self.sor_nb_neighbors = int(cfg3d.get("sor_nb_neighbors", 20))
        self.sor_std_ratio = float(cfg3d.get("sor_std_ratio", 1.5))
        self.num_gmm_components = (
            int(num_gmm_components)
            if num_gmm_components is not None
            else int(cfg3d.get("num_gmm_components", 2))
        )
        self.shrinkage_alpha = (
            float(shrinkage_alpha)
            if shrinkage_alpha is not None
            else float(cfg3d.get("shrinkage_alpha", 0.08))
        )
        self.ood_mahalanobis_threshold = (
            float(ood_mahalanobis_threshold)
            if ood_mahalanobis_threshold is not None
            else float(cfg3d.get("ood_mahalanobis_threshold", 45.0))
        )
        self.fusion_temperature = (
            float(fusion_temperature)
            if fusion_temperature is not None
            else float(cfg3d.get("fusion_temperature", 1.0))
        )
        self.clip_logit_scale = float(clip_logit_scale)
        self.enable_clip = bool(enable_clip)

        self.geom_encoder = (
            geom_encoder
            if geom_encoder is not None
            else Geometric3DEncoder(max_points=self.point_num_samples, seed=self.seed)
        )
        self.size_encoder = size_encoder if size_encoder is not None else PhysicalSizeEncoder()
        self._clip_encoder: OpenCLIPMultimodalEncoder | None = clip_encoder

        self.class_names: list[str] = []
        self._models: dict[str, OnlineShrinkageGMDA] = {
            "geom_3d": OnlineShrinkageGMDA(
                num_components=self.num_gmm_components,
                ridge_alpha=self.shrinkage_alpha,
                normalize_features=True,
                prior_var=0.015,
                prior_strength=4.0,
                seed=self.seed,
            ),
            "size_3d": OnlineShrinkageGMDA(
                num_components=self.num_gmm_components,
                ridge_alpha=self.shrinkage_alpha,
                normalize_features=False,
                prior_var=0.04,
                prior_strength=4.0,
                seed=self.seed,
            ),
            "visual_clip": OnlineShrinkageGMDA(
                num_components=self.num_gmm_components,
                ridge_alpha=self.shrinkage_alpha,
                normalize_features=True,
                prior_var=0.02,
                prior_strength=4.0,
                seed=self.seed,
            ),
        }
        self._text_prototypes: dict[str, np.ndarray] = {}
        self._text_prompts: dict[str, str] = {}

    @property
    def clip_encoder(self) -> OpenCLIPMultimodalEncoder | None:
        if self._clip_encoder is None and self.enable_clip:
            self._clip_encoder = OpenCLIPMultimodalEncoder(
                model_name=self.clip_model_name,
                pretrained=self.clip_pretrained,
                device=self.device,
            )
        return self._clip_encoder

    @property
    def registered_classes(self) -> list[str]:
        return list(self.class_names)

    def _ensure_class(self, class_name: str) -> None:
        if class_name not in self.class_names:
            self.class_names.append(class_name)

    def _extract_sample_pointcloud(self, sample: SupportSample) -> np.ndarray | None:
        if sample.point_cloud_xyz is not None and sample.point_cloud_xyz.shape[0] > 0:
            return np.asarray(sample.point_cloud_xyz, dtype=np.float32)
        if (
            sample.depth is not None
            and sample.intrinsics is not None
            and sample.mask is not None
        ):
            rgb = (
                sample.rgb
                if sample.rgb is not None
                else np.zeros((*sample.depth.shape, 3), dtype=np.uint8)
            )
            c2w = (
                sample.camera_to_world
                if sample.camera_to_world is not None
                else np.eye(4, dtype=np.float64)
            )
            frame = RGBDFrame(
                rgb=rgb,
                depth=sample.depth,
                intrinsics=sample.intrinsics,
                camera_to_world=c2w,
            )
            xyz, _ = confidence_weighted_backproject(
                frame=frame,
                mask=sample.mask,
                confidence_map=None,
                depth_min=self.depth_min_m,
                depth_max=self.depth_max_m,
                sor_nb_neighbors=self.sor_nb_neighbors,
                sor_std_ratio=self.sor_std_ratio,
                return_colors=False,
            )
            return xyz if xyz.shape[0] > 0 else None
        return None

    def register_few_shot(
        self,
        class_name: str,
        samples: list[SupportSample],
        register_text: bool = False,
        table_z: float = 0.0,
    ) -> None:
        """Online incremental few-shot registration across all available modalities."""
        if not samples:
            raise ValueError("samples list cannot be empty for register_few_shot")
        self._ensure_class(class_name)

        geom_feats: list[np.ndarray] = []
        size_feats: list[np.ndarray] = []
        vis_feats: list[np.ndarray] = []
        text_prompt: str | None = None

        for sample in samples:
            sample_table_z = float(sample.metadata.get("table_z", table_z))
            xyz = self._extract_sample_pointcloud(sample)
            if xyz is not None and xyz.shape[0] >= 6:
                obb_center, obb_extent, _ = compute_oriented_bounding_box(xyz)
                geom_feats.append(self.geom_encoder.encode(xyz))
                size_feats.append(
                    self.size_encoder.encode(
                        obb_extent=obb_extent,
                        obb_center=obb_center,
                        table_z=sample_table_z,
                    )
                )

            if sample.rgb is not None and self.clip_encoder is not None:
                bbox = sample.metadata.get("bbox_2d")
                vis_feats.append(
                    self.clip_encoder.encode_image_crop(
                        rgb=sample.rgb,
                        mask=sample.mask,
                        bbox=bbox,
                    )
                )

            if sample.text_prompt:
                text_prompt = sample.text_prompt

        if geom_feats:
            self._models["geom_3d"].add_samples(class_name, np.stack(geom_feats, axis=0))
        if size_feats:
            self._models["size_3d"].add_samples(class_name, np.stack(size_feats, axis=0))
        if vis_feats:
            self._models["visual_clip"].add_samples(class_name, np.stack(vis_feats, axis=0))

        if text_prompt is not None:
            self.register_zero_shot_text(class_name, text_prompt)
        elif register_text and self.clip_encoder is not None:
            self.register_zero_shot_text(class_name, class_name.replace("_", " "))

    def register_zero_shot_text(
        self,
        class_name: str,
        text_prompt: str | None = None,
    ) -> None:
        """Register a brand-new class purely from a natural-language description via OpenCLIP."""
        encoder = self.clip_encoder
        if encoder is None:
            raise RuntimeError("CLIP encoder is disabled; cannot register zero-shot text class.")
        self._ensure_class(class_name)
        prompt = (text_prompt or class_name.replace("_", " ")).strip()
        emb = encoder.encode_text(prompt, use_templates=True)
        self._text_prototypes[class_name] = l2_normalize(emb.astype(np.float64)).astype(np.float32)
        self._text_prompts[class_name] = prompt

    def _align_gmda_log_likelihood(
        self,
        model: OnlineShrinkageGMDA,
        feat: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Map modality GMDA log-likelihoods onto full ``self.class_names`` with open-set neutral fill."""
        ll_sub, d2_sub = model.predict_class_scores(feat)
        d2_by_class = {c: float(d2_sub[i]) for i, c in enumerate(model.class_order)}
        ll_by_class = {c: float(ll_sub[i]) for i, c in enumerate(model.class_order)}

        neutral_d2 = min(0.45 * self.ood_mahalanobis_threshold, 18.0)
        neutral_ll = -0.5 * neutral_d2

        aligned = np.full(len(self.class_names), neutral_ll, dtype=np.float64)
        for idx, cname in enumerate(self.class_names):
            if cname in ll_by_class:
                aligned[idx] = ll_by_class[cname]
        return aligned, d2_by_class

    def _align_external_logits(
        self,
        seg2d_logits: dict[str, float] | np.ndarray,
    ) -> np.ndarray | None:
        if isinstance(seg2d_logits, dict):
            vals = [float(v) for v in seg2d_logits.values()]
            if not vals:
                return None
            default_val = float(np.min(vals)) - 1.0
            arr = np.full(len(self.class_names), default_val, dtype=np.float64)
            matched = False
            for idx, cname in enumerate(self.class_names):
                if cname in seg2d_logits:
                    arr[idx] = float(seg2d_logits[cname])
                    matched = True
            return arr if matched else None

        arr = np.asarray(seg2d_logits, dtype=np.float64).ravel()
        if arr.shape[0] != len(self.class_names):
            raise ValueError(
                f"seg2d_logits length {arr.shape[0]} != number of registered classes {len(self.class_names)}"
            )
        return arr

    def _ensure_complete_text_prototypes(self) -> None:
        """Ensure every registered class has a text prototype whenever text_clip is active."""
        if not self._text_prototypes or self.clip_encoder is None:
            return
        for cname in self.class_names:
            if cname not in self._text_prototypes:
                prompt = cname.replace("_", " ")
                emb = self.clip_encoder.encode_text(prompt, use_templates=True)
                self._text_prototypes[cname] = l2_normalize(emb.astype(np.float64)).astype(
                    np.float32
                )
                self._text_prompts[cname] = prompt

    @staticmethod
    def _bbox_from_mask(mask_2d: np.ndarray | None) -> tuple[int, int, int, int]:
        if mask_2d is None or not np.any(mask_2d):
            return (0, 0, 0, 0)
        ys, xs = np.nonzero(mask_2d)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def classify_instance(
        self,
        instance_id: int = 0,
        xyz: np.ndarray | None = None,
        rgb_points: np.ndarray | None = None,
        rgb_image: np.ndarray | None = None,
        mask_2d: np.ndarray | None = None,
        bbox_2d: tuple[int, int, int, int] | None = None,
        frame: RGBDFrame | None = None,
        confidence_map: np.ndarray | None = None,
        seg2d_logits: dict[str, float] | np.ndarray | None = None,
        seg2d_reliability: float = 0.85,
        table_z: float = 0.0,
    ) -> Object3DDetection:
        """Classify a segmented 3D object instance via uncertainty-gated multimodal Bayesian fusion."""
        if not self.class_names:
            raise RuntimeError("No classes registered in PartialAwareMultimodalGMDA.")

        if xyz is None and frame is not None and mask_2d is not None:
            xyz, rgb_points = confidence_weighted_backproject(
                frame=frame,
                mask=mask_2d,
                confidence_map=confidence_map,
                conf_thresh=self.confidence_denoise_thresh,
                depth_min=self.depth_min_m,
                depth_max=self.depth_max_m,
                sor_nb_neighbors=self.sor_nb_neighbors,
                sor_std_ratio=self.sor_std_ratio,
                return_colors=True,
            )

        if rgb_image is None and frame is not None:
            rgb_image = frame.rgb

        pts = (
            np.asarray(xyz, dtype=np.float32)
            if xyz is not None
            else np.zeros((0, 3), dtype=np.float32)
        )
        obb_center, obb_extent, obb_rotation = compute_oriented_bounding_box(pts)
        centroid = (
            np.mean(pts.astype(np.float64), axis=0)
            if pts.shape[0] > 0
            else obb_center.copy()
        )
        resolved_bbox = bbox_2d if bbox_2d is not None else self._bbox_from_mask(mask_2d)

        completeness = estimate_pointcloud_completeness(pts, obb_extent)
        log_likelihoods: dict[str, np.ndarray] = {}
        reliabilities: dict[str, float] = {}
        modality_d2: dict[str, dict[str, float]] = {}

        # 1. External 2D segmentation logits (DINOv2-GDA)
        if seg2d_logits is not None:
            aligned_2d = self._align_external_logits(seg2d_logits)
            if aligned_2d is not None:
                log_likelihoods["seg2d"] = aligned_2d
                reliabilities["seg2d"] = float(np.clip(seg2d_reliability, 0.1, 1.0))

        # 2 & 3. 3D Geometry + Physical Size modalities
        if pts.shape[0] >= 6:
            if self._models["geom_3d"].is_fitted:
                g_feat = self.geom_encoder.encode(pts)
                ll_g, d2_g = self._align_gmda_log_likelihood(self._models["geom_3d"], g_feat)
                log_likelihoods["geom_3d"] = ll_g
                reliabilities["geom_3d"] = completeness
                modality_d2["geom_3d"] = d2_g

            if self._models["size_3d"].is_fitted:
                s_feat = self.size_encoder.encode(
                    obb_extent=obb_extent,
                    obb_center=obb_center,
                    table_z=table_z,
                )
                ll_s, d2_s = self._align_gmda_log_likelihood(self._models["size_3d"], s_feat)
                log_likelihoods["size_3d"] = ll_s
                reliabilities["size_3d"] = completeness
                modality_d2["size_3d"] = d2_s

        # 4 & 5. OpenCLIP Visual Few-Shot + Zero-Shot Text modalities
        img_emb: np.ndarray | None = None
        if rgb_image is not None and self.clip_encoder is not None:
            if self._models["visual_clip"].is_fitted or self._text_prototypes:
                img_emb = self.clip_encoder.encode_image_crop(
                    rgb=rgb_image,
                    mask=mask_2d,
                    bbox=resolved_bbox if resolved_bbox != (0, 0, 0, 0) else None,
                )
                vis_rel = 1.0
                if resolved_bbox != (0, 0, 0, 0):
                    bw = resolved_bbox[2] - resolved_bbox[0]
                    bh = resolved_bbox[3] - resolved_bbox[1]
                    if min(bw, bh) < 12:
                        vis_rel = 0.65

                if self._models["visual_clip"].is_fitted:
                    ll_v, d2_v = self._align_gmda_log_likelihood(
                        self._models["visual_clip"], img_emb
                    )
                    log_likelihoods["visual_clip"] = ll_v
                    reliabilities["visual_clip"] = vis_rel
                    modality_d2["visual_clip"] = d2_v

                if self._text_prototypes:
                    self._ensure_complete_text_prototypes()
                    proto_mat = np.stack(
                        [self._text_prototypes[c].astype(np.float64) for c in self.class_names],
                        axis=0,
                    )
                    sims = proto_mat @ img_emb.astype(np.float64)
                    log_likelihoods["text_clip"] = sims * self.clip_logit_scale
                    reliabilities["text_clip"] = 0.95 * vis_rel

        if not log_likelihoods:
            raise RuntimeError(
                "No active perception modalities could be evaluated for this instance. "
                "Provide 3D points, RGB image, or seg2d_logits matching registered modalities."
            )

        fused_probs, fused_entropy, modality_weights = entropy_gated_bayesian_fusion(
            log_likelihoods=log_likelihoods,
            reliability=reliabilities,
            temperature=self.fusion_temperature,
        )

        best_idx = int(np.argmax(fused_probs))
        pred_class = self.class_names[best_idx]
        confidence = float(fused_probs[best_idx])

        ood_score = self._compute_ood_score(
            pred_class=pred_class,
            modality_d2=modality_d2,
            reliabilities=reliabilities,
            img_emb=img_emb,
        )
        is_ood = bool(ood_score > self.ood_mahalanobis_threshold)

        posterior_probs = {
            cname: float(fused_probs[i]) for i, cname in enumerate(self.class_names)
        }

        return Object3DDetection(
            instance_id=int(instance_id),
            class_name=pred_class,
            confidence=confidence,
            entropy=float(fused_entropy),
            ood_score=float(ood_score),
            is_ood=is_ood,
            centroid=centroid.astype(np.float64),
            obb_center=obb_center.astype(np.float64),
            obb_extent=obb_extent.astype(np.float64),
            obb_rotation=obb_rotation.astype(np.float64),
            point_cloud_xyz=pts,
            point_cloud_rgb=(
                np.asarray(rgb_points, dtype=np.float32) if rgb_points is not None else None
            ),
            mask_2d=mask_2d.astype(bool) if mask_2d is not None else None,
            bbox_2d=resolved_bbox,
            modality_weights={k: float(v) for k, v in modality_weights.items()},
            posterior_probs=posterior_probs,
        )

    def _compute_ood_score(
        self,
        pred_class: str,
        modality_d2: dict[str, dict[str, float]],
        reliabilities: dict[str, float],
        img_emb: np.ndarray | None,
    ) -> float:
        """Compute reliability-weighted Mahalanobis OOD distance score."""
        weighted_sum = 0.0
        weight_total = 0.0
        reliable_d2: list[float] = []
        pred_covered_by_gmda = False

        for mod, d2_map in modality_d2.items():
            if not d2_map:
                continue
            if pred_class in d2_map:
                pred_covered_by_gmda = True
                d2_val = d2_map[pred_class]
            else:
                d2_val = min(d2_map.values())
            rel = float(reliabilities.get(mod, 0.5))
            weighted_sum += rel * d2_val
            weight_total += rel
            if rel >= 0.6:
                reliable_d2.append(d2_val)

        if not pred_covered_by_gmda and img_emb is not None and pred_class in self._text_prototypes:
            sim = float(
                np.dot(
                    self._text_prototypes[pred_class].astype(np.float64),
                    img_emb.astype(np.float64),
                )
            )
            gap = max(0.0, 0.30 - sim)
            return float((gap / 0.03) ** 2)

        if weight_total <= 0.0:
            return 0.0

        mean_d2 = weighted_sum / weight_total
        if reliable_d2:
            return float(0.6 * mean_d2 + 0.4 * max(reliable_d2))
        return float(mean_d2)

    def save_memory(self, path: str | Path) -> Path:
        """Persist registered class memory (GMDA statistics + text prototypes) to a ``.npz`` file."""
        out_path = Path(path)
        if out_path.suffix != ".npz":
            out_path = out_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        arrays_to_save: dict[str, np.ndarray] = {}
        models_meta: dict[str, Any] = {}

        for mod_name, model in self._models.items():
            meta, mod_arrays = model.to_state_dict()
            models_meta[mod_name] = meta
            for idx, cname in enumerate(meta["class_order"]):
                if cname in mod_arrays:
                    arrays_to_save[f"mod__{mod_name}__{idx}"] = mod_arrays[cname]

        text_class_order = list(self._text_prototypes.keys())
        for idx, cname in enumerate(text_class_order):
            arrays_to_save[f"text__{idx}"] = self._text_prototypes[cname].astype(np.float32)

        payload_meta = {
            "version": 1,
            "class_names": list(self.class_names),
            "models_meta": models_meta,
            "text_class_order": text_class_order,
            "text_prompts": dict(self._text_prompts),
            "hyperparams": {
                "num_gmm_components": self.num_gmm_components,
                "shrinkage_alpha": self.shrinkage_alpha,
                "ood_mahalanobis_threshold": self.ood_mahalanobis_threshold,
                "fusion_temperature": self.fusion_temperature,
                "clip_logit_scale": self.clip_logit_scale,
            },
        }
        arrays_to_save["__meta_json__"] = np.frombuffer(
            json.dumps(payload_meta).encode("utf-8"), dtype=np.uint8
        )
        np.savez_compressed(out_path, **arrays_to_save)
        return out_path

    def load_memory(self, path: str | Path) -> None:
        """Restore registered class memory from a ``.npz`` file created by ``save_memory``."""
        in_path = Path(path)
        if not in_path.exists() and in_path.with_suffix(".npz").exists():
            in_path = in_path.with_suffix(".npz")
        if not in_path.exists():
            raise FileNotFoundError(f"Memory file not found: {in_path}")

        with np.load(in_path, allow_pickle=False) as data:
            meta_bytes = data["__meta_json__"].tobytes()
            payload_meta = json.loads(meta_bytes.decode("utf-8"))

            self.class_names = list(payload_meta["class_names"])
            hp = payload_meta.get("hyperparams", {})
            self.num_gmm_components = int(hp.get("num_gmm_components", self.num_gmm_components))
            self.shrinkage_alpha = float(hp.get("shrinkage_alpha", self.shrinkage_alpha))
            self.ood_mahalanobis_threshold = float(
                hp.get("ood_mahalanobis_threshold", self.ood_mahalanobis_threshold)
            )
            self.fusion_temperature = float(hp.get("fusion_temperature", self.fusion_temperature))
            self.clip_logit_scale = float(hp.get("clip_logit_scale", self.clip_logit_scale))

            models_meta = payload_meta["models_meta"]
            for mod_name in ("geom_3d", "size_3d", "visual_clip"):
                meta = models_meta[mod_name]
                mod_arrays: dict[str, np.ndarray] = {}
                for idx, cname in enumerate(meta["class_order"]):
                    key = f"mod__{mod_name}__{idx}"
                    if key in data:
                        mod_arrays[cname] = data[key]
                self._models[mod_name] = OnlineShrinkageGMDA.from_state_dict(meta, mod_arrays)

            self._text_prototypes.clear()
            self._text_prompts = dict(payload_meta.get("text_prompts", {}))
            for idx, cname in enumerate(payload_meta.get("text_class_order", [])):
                key = f"text__{idx}"
                if key in data:
                    self._text_prototypes[cname] = data[key].astype(np.float32)

