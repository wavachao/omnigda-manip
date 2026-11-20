"""Core statistical & geometric mathematics for OmniGDA-Manip.

Implements:
- Adaptive covariance shrinkage (Ledoit-Wolf, OAS, Ridge) for N << D few-shot regimes
- Cholesky-stabilized precision matrix computation
- Sherman-Morrison-Woodbury online rank-k inverse covariance update for sub-second registration
- Vectorized tied-covariance GDA and multi-component GMDA log-likelihood & Mahalanobis OOD scoring
- Uncertainty-gated (entropy-weighted) Bayesian multi-modal fusion
- 3D rigid-body geometry helpers (SO(3) <-> quaternion, RGB-D backprojection)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.transform import Rotation


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    max_val = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_val
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True) + 1e-12)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax(logits, axis=axis))


def normalized_entropy(probs: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Normalized Shannon entropy in [0, 1]."""
    p = np.clip(probs, eps, 1.0)
    k = probs.shape[axis]
    if k <= 1:
        return np.zeros(probs.shape[:axis] + probs.shape[axis + 1 :], dtype=np.float64)
    ent = -np.sum(p * np.log(p), axis=axis)
    return np.clip(ent / np.log(float(k)), 0.0, 1.0)


def estimate_shrunk_covariance(
    centered: np.ndarray,
    method: str = "oas",
    ridge_alpha: float = 0.05,
    min_eig: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Estimate well-conditioned covariance matrix from centered residuals (N, D).

    Returns:
        cov: (D, D) symmetric positive-definite covariance matrix.
        shrinkage: scalar shrinkage intensity in [0, 1].
    """
    x = np.asarray(centered, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (N, D), got shape {x.shape}")
    n, d = x.shape
    if n <= 1:
        return np.eye(d, dtype=np.float64) * max(ridge_alpha, min_eig), 1.0

    emp_cov = (x.T @ x) / float(n)
    emp_cov = 0.5 * (emp_cov + emp_cov.T)
    mu = float(np.trace(emp_cov)) / float(d)
    target = mu * np.eye(d, dtype=np.float64)

    if method == "ridge":
        alpha = float(np.clip(ridge_alpha, 0.0, 1.0))
    elif method == "oas":
        # Oracle Approximating Shrinkage (Chen et al., IEEE TSP 2010)
        tr_s2 = float(np.sum(emp_cov * emp_cov))
        tr_s_sq = float(np.trace(emp_cov) ** 2)
        num = (1.0 - 2.0 / d) * tr_s2 + tr_s_sq
        den = (n + 1.0 - 2.0 / d) * (tr_s2 - tr_s_sq / d)
        alpha = 1.0 if den <= 1e-12 else float(np.clip(num / den, 0.0, 1.0))
    elif method == "ledoit_wolf":
        # Ledoit-Wolf shrinkage toward scaled identity
        x2 = x * x
        phi_mat = (x2.T @ x2) / float(n) - emp_cov * emp_cov
        phi = float(np.sum(phi_mat))
        gamma = float(np.sum((emp_cov - target) ** 2))
        alpha = 1.0 if gamma <= 1e-12 else float(np.clip(phi / (n * gamma), 0.0, 1.0))
    else:
        raise ValueError(f"Unknown shrinkage method: {method}")

    cov = (1.0 - alpha) * emp_cov + alpha * target
    cov.flat[:: d + 1] += min_eig
    return cov, alpha


def invert_spd(cov: np.ndarray) -> np.ndarray:
    """Invert a symmetric positive-definite matrix using Cholesky factorization."""
    d = cov.shape[0]
    c_fac = cho_factor(cov, lower=True, check_finite=False)
    inv = cho_solve(c_fac, np.eye(d, dtype=np.float64), check_finite=False)
    return 0.5 * (inv + inv.T)


def woodbury_precision_update(
    precision: np.ndarray,
    u_matrix: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Rank-k Woodbury update of Lambda = Sigma^{-1} for Sigma_new = Sigma + scale * U U^T.

    Given Lambda = Sigma^{-1} (D, D) and U (D, k):
        (Sigma + scale * U U^T)^{-1} = Lambda - Lambda U (scale^{-1} I_k + U^T Lambda U)^{-1} U^T Lambda
    """
    if u_matrix.size == 0 or abs(scale) < 1e-15:
        return precision
    lu = precision @ u_matrix  # (D, k)
    k = u_matrix.shape[1]
    inner = (1.0 / scale) * np.eye(k, dtype=np.float64) + u_matrix.T @ lu
    inner = 0.5 * (inner + inner.T)
    c_fac = cho_factor(inner, lower=True, check_finite=False)
    solved = cho_solve(c_fac, lu.T, check_finite=False)  # (k, D)
    updated = precision - lu @ solved
    return 0.5 * (updated + updated.T)


@dataclass
class OnlineClassState:
    class_name: str
    mean: np.ndarray  # (D,)
    count: int


class OnlineShrinkageGDA:
    """Tied-covariance Shrinkage GDA with Sherman-Morrison-Woodbury online class registration.

    For tied covariance Sigma across classes, the Bayes-optimal linear discriminant is:
        log p(x | y=k) = x^T (Lambda mu_k) - 0.5 * mu_k^T Lambda mu_k + const
    which evaluates in a single GEMM at inference time.
    """

    def __init__(
        self,
        shrinkage_method: str = "oas",
        ridge_alpha: float = 0.05,
        min_eig: float = 1e-4,
        normalize_features: bool = True,
    ) -> None:
        self.shrinkage_method = shrinkage_method
        self.ridge_alpha = ridge_alpha
        self.min_eig = min_eig
        self.normalize_features = normalize_features

        self.dim: int = 0
        self.total_count: int = 0
        self.shrinkage_alpha: float = ridge_alpha
        self.scatter: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.precision: np.ndarray | None = None
        self.classes: dict[str, OnlineClassState] = {}
        self.class_order: list[str] = []
        self._weights: np.ndarray | None = None  # (K, D) = Lambda @ mu_k
        self._bias: np.ndarray | None = None  # (K,) = -0.5 * mu_k^T Lambda mu_k

    def _prep(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if self.normalize_features:
            arr = l2_normalize(arr, axis=-1)
        return arr

    def _rebuild_discriminant_cache(self) -> None:
        if self.precision is None or not self.class_order:
            self._weights = None
            self._bias = None
            return
        means = np.stack([self.classes[c].mean for c in self.class_order], axis=0)  # (K, D)
        weights = means @ self.precision  # (K, D)
        bias = -0.5 * np.sum(weights * means, axis=1)  # (K,)
        self._weights = weights
        self._bias = bias

    def fit(self, features_by_class: dict[str, np.ndarray]) -> None:
        """Fit initial GDA model from dict mapping class_name -> (N_k, D) features."""
        if not features_by_class:
            raise ValueError("features_by_class cannot be empty")
        self.classes.clear()
        self.class_order = list(features_by_class.keys())

        residuals_list: list[np.ndarray] = []
        total = 0
        for cname in self.class_order:
            feats = self._prep(features_by_class[cname])
            self.dim = feats.shape[1]
            mean = np.mean(feats, axis=0)
            self.classes[cname] = OnlineClassState(class_name=cname, mean=mean, count=feats.shape[0])
            residuals_list.append(feats - mean[None, :])
            total += feats.shape[0]

        residuals = np.concatenate(residuals_list, axis=0)
        self.total_count = total
        self.scatter = residuals.T @ residuals
        self.covariance, self.shrinkage_alpha = estimate_shrunk_covariance(
            residuals,
            method=self.shrinkage_method,
            ridge_alpha=self.ridge_alpha,
            min_eig=self.min_eig,
        )
        self.precision = invert_spd(self.covariance)
        self._rebuild_discriminant_cache()

    def register_online(
        self,
        class_name: str,
        features: np.ndarray,
        use_woodbury: bool = True,
    ) -> None:
        """Register a brand-new class or append few-shot samples to an existing class online.

        Uses Sherman-Morrison-Woodbury rank-k precision matrix update when the model is already
        initialized, avoiding full O(D^3) inversion.
        """
        feats = self._prep(features)
        n_new, d = feats.shape
        if self.precision is None:
            self.fit({class_name: feats})
            return
        if d != self.dim:
            raise ValueError(f"Feature dim mismatch: expected {self.dim}, got {d}")

        n_old_total = self.total_count
        if class_name in self.classes:
            st = self.classes[class_name]
            n_old = st.count
            old_mean = st.mean
            new_batch_mean = np.mean(feats, axis=0)
            combined_count = n_old + n_new
            new_mean = (n_old * old_mean + n_new * new_batch_mean) / float(combined_count)
            # Parallel-axis (Chan's) residual vectors whose outer product equals delta scatter
            batch_res = feats - new_batch_mean[None, :]
            delta_mean = (new_batch_mean - old_mean) * np.sqrt((n_old * n_new) / float(combined_count))
            u_cols = np.vstack([batch_res, delta_mean[None, :]]).T  # (D, k)
            st.mean = new_mean
            st.count = combined_count
        else:
            batch_mean = np.mean(feats, axis=0)
            self.classes[class_name] = OnlineClassState(
                class_name=class_name, mean=batch_mean, count=n_new
            )
            self.class_order.append(class_name)
            u_cols = (feats - batch_mean[None, :]).T  # (D, n_new)

        assert self.scatter is not None and self.covariance is not None
        delta_scatter = u_cols @ u_cols.T
        self.scatter = self.scatter + delta_scatter
        self.total_count = n_old_total + n_new

        alpha = float(self.shrinkage_alpha)
        # Keep the same shrinkage target scale for fast exact Woodbury update:
        # Sigma_old = (1 - alpha) * S_old / N_old + alpha * mu_old * I + min_eig * I
        # Scale precision for denominator change N_old -> N_new, then apply rank-k Woodbury update.
        if use_woodbury and u_cols.shape[1] < d // 2:
            # Rescale Sigma_old_eff -> exact incremental covariance under fixed ridge target
            mu_old = float(np.trace(self.scatter - delta_scatter)) / float(max(n_old_total, 1) * d)
            mu_new = float(np.trace(self.scatter)) / float(self.total_count * d)
            # If trace shifts negligibly (< 5%), perform exact Woodbury on the scaled precision
            if abs(mu_new - mu_old) <= 0.05 * max(mu_old, 1e-6) and n_new <= max(16, int(0.15 * n_old_total)):
                # Approximate Sigma_new = Sigma_old + ((1-alpha)/N_new) * U U^T (conservative shrinkage)
                step_scale = (1.0 - alpha) / float(self.total_count)
                self.covariance = self.covariance + step_scale * delta_scatter
                self.precision = woodbury_precision_update(self.precision, u_cols, scale=step_scale)
            else:
                emp_cov = self.scatter / float(self.total_count)
                target = mu_new * np.eye(d, dtype=np.float64)
                cov = (1.0 - alpha) * emp_cov + alpha * target
                cov.flat[:: d + 1] += self.min_eig
                self.covariance = cov
                self.precision = invert_spd(cov)
        else:
            emp_cov = self.scatter / float(self.total_count)
            mu_new = float(np.trace(emp_cov)) / float(d)
            cov = (1.0 - alpha) * emp_cov + alpha * mu_new * np.eye(d, dtype=np.float64)
            cov.flat[:: d + 1] += self.min_eig
            self.covariance = cov
            self.precision = invert_spd(cov)

        self._rebuild_discriminant_cache()

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Compute tied-covariance GDA log-likelihood scores (up to class-independent constant)."""
        if self._weights is None or self._bias is None:
            raise RuntimeError("OnlineShrinkageGDA must be fitted before prediction.")
        feats = self._prep(x)
        return feats @ self._weights.T + self._bias[None, :]

    def predict_proba(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        logits = self.predict_logits(x) / max(float(temperature), 1e-6)
        return softmax(logits, axis=-1)

    def mahalanobis_sq(self, x: np.ndarray) -> np.ndarray:
        """Compute squared Mahalanobis distance to every class prototype: returns (N, K)."""
        if self.precision is None or not self.class_order:
            raise RuntimeError("OnlineShrinkageGDA must be fitted before computing distances.")
        feats = self._prep(x)  # (N, D)
        # d^2(x, mu_k) = x^T Lambda x - 2 x^T (Lambda mu_k) + mu_k^T Lambda mu_k
        x_quad = np.sum((feats @ self.precision) * feats, axis=1, keepdims=True)  # (N, 1)
        assert self._weights is not None and self._bias is not None
        dist_sq = x_quad - 2.0 * (feats @ self._weights.T) - 2.0 * self._bias[None, :]
        return np.maximum(dist_sq, 0.0)


def entropy_gated_bayesian_fusion(
    log_likelihoods: dict[str, np.ndarray],
    reliability: dict[str, float] | None = None,
    temperature: float = 1.0,
    min_weight: float = 0.05,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Fuse multi-modal class log-likelihood vectors via uncertainty-gated Bayesian weighting.

    Each modality m contributes weight:
        raw_w_m = max(reliability_m * (1.0 - H_norm(p_m)), min_weight)
        w_m = raw_w_m / sum_j raw_w_j
    Fused log-posterior:
        log p(y | x_1..M) = log_softmax(sum_m w_m * log_softmax(ll_m / T))

    Returns:
        fused_probs: (K,) fused posterior probability vector.
        fused_entropy: normalized Shannon entropy in [0, 1].
        weights: dict[modality_name -> normalized weight].
    """
    if not log_likelihoods:
        raise ValueError("log_likelihoods cannot be empty")
    rel = reliability or {}
    raw_weights: dict[str, float] = {}
    norm_log_probs: dict[str, np.ndarray] = {}

    t = max(float(temperature), 1e-6)
    for mod, ll in log_likelihoods.items():
        lp = log_softmax(np.asarray(ll, dtype=np.float64) / t, axis=-1)
        probs = np.exp(lp)
        ent = float(normalized_entropy(probs, axis=-1))
        r = float(np.clip(rel.get(mod, 1.0), 0.0, 1.0))
        raw_weights[mod] = max(r * (1.0 - ent), min_weight)
        norm_log_probs[mod] = lp

    total_w = sum(raw_weights.values())
    weights = {m: w / total_w for m, w in raw_weights.items()}

    fused_ll = sum(weights[m] * norm_log_probs[m] for m in weights)
    fused_probs = softmax(fused_ll, axis=-1)
    fused_ent = float(normalized_entropy(fused_probs, axis=-1))
    return fused_probs, fused_ent, weights


def rotation_matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot).as_quat().astype(np.float64)


def quat_xyzw_to_rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float64)


def backproject_depth(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_to_world: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    depth_min: float = 0.1,
    depth_max: float = 2.5,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Vectorized backprojection of depth map (H, W) into 3D points (N, 3) in world frame."""
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
    if mask is not None:
        valid &= mask.astype(bool)

    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        empty_rgb = np.zeros((0, 3), dtype=np.float32) if rgb is not None else None
        return np.zeros((0, 3), dtype=np.float32), empty_rgb

    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)

    if camera_to_world is not None:
        rot = camera_to_world[:3, :3]
        trans = camera_to_world[:3, 3]
        pts = pts_cam @ rot.T + trans[None, :]
    else:
        pts = pts_cam

    colors = None
    if rgb is not None:
        colors = rgb[ys, xs].astype(np.float32) / 255.0

    return pts.astype(np.float32), colors
