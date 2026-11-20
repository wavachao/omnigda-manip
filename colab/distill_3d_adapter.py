"""Fisher-SupCon 3D Discriminative Subspace Adapter Trainer.

Trains a compact linear projection adapter ``z = l2_norm(x @ W.T + b)`` mapping raw
108-D rotation-invariant ``Geometric3DEncoder`` descriptors into a lower-dimensional
discriminative subspace (default ``D_out = 64``) using:

    L = L_supcon + lambda_f * tr(S_W) / (tr(S_B) + eps) + lambda_o * ||W W^T - I||_F^2

Exported ``.npz`` weights are zero-pickle (``allow_pickle=False``) and directly
accepted by ``Geometric3DEncoder.load_projection_adapter(path)``.

CLI example:
    python -m colab.distill_3d_adapter --epochs 150 --out_dim 64 --output artifacts/geom3d_adapter_64d.npz
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omnigda.core.gda_math import (
    OnlineShrinkageGDA,
    backproject_depth,
    estimate_shrunk_covariance,
    invert_spd,
    l2_normalize,
)
from omnigda.perception_3d.encoders import Geometric3DEncoder


@dataclass
class Adapter3DTrainingResult:
    """Container for trained 3D projection adapter weights, curves, and evaluation metrics."""

    weight: np.ndarray  # (D_out, D_in) float64
    bias: np.ndarray  # (D_out,) float64
    history: dict[str, list[float]]
    metrics: dict[str, float]
    class_names: list[str]
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.15,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute numerically stable Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Args:
        embeddings: L2-normalized tensor of shape ``(N, D)``.
        labels: Integer class labels of shape ``(N,)``.
        temperature: Positive softmax temperature.
        eps: Numerical floor for log-probabilities.

    Returns:
        Scalar loss tensor.
    """
    n = embeddings.shape[0]
    if n <= 1:
        return embeddings.new_zeros(())

    tau = max(float(temperature), 1e-4)
    sim = torch.matmul(embeddings, embeddings.T) / tau  # (N, N)

    device = embeddings.device
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    not_self = ~self_mask

    # Shift logits per row using max over non-self entries for numerical stability
    sim_masked = sim.masked_fill(self_mask, float("-inf"))
    row_max, _ = sim_masked.max(dim=1, keepdim=True)
    logits = sim - row_max.detach()

    exp_logits = torch.exp(logits) * not_self.to(logits.dtype)
    log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=eps))
    log_prob = logits - log_denom  # (N, N)

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & not_self
    pos_counts = pos_mask.sum(dim=1)
    valid_anchors = pos_counts > 0
    if not torch.any(valid_anchors):
        return embeddings.new_zeros(())

    pos_log_prob_sum = (log_prob * pos_mask.to(log_prob.dtype)).sum(dim=1)
    mean_log_prob_pos = pos_log_prob_sum[valid_anchors] / pos_counts[valid_anchors].to(
        log_prob.dtype
    )
    return -mean_log_prob_pos.mean()


def fisher_discriminant_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute differentiable Fisher discriminant ratio ``tr(S_W) / (tr(S_B) + eps)``.

    Encourages tight intra-class clusters (small within-class scatter ``S_W``) and
    widely separated class centroids (large between-class scatter ``S_B``), directly
    aligning with downstream tied-covariance Gaussian Discriminant Analysis (GDA).
    """
    n = embeddings.shape[0]
    if n <= 1:
        return embeddings.new_zeros(())

    unique_classes = torch.unique(labels)
    if unique_classes.numel() <= 1:
        return embeddings.new_zeros(())

    global_mean = embeddings.mean(dim=0, keepdim=True)  # (1, D)
    trace_sw = embeddings.new_zeros(())
    trace_sb = embeddings.new_zeros(())

    for cls in unique_classes:
        mask = labels == cls
        cls_embs = embeddings[mask]
        n_c = cls_embs.shape[0]
        if n_c == 0:
            continue
        cls_mean = cls_embs.mean(dim=0, keepdim=True)
        diff_w = cls_embs - cls_mean
        trace_sw = trace_sw + (diff_w * diff_w).sum()
        diff_b = cls_mean - global_mean
        trace_sb = trace_sb + float(n_c) * (diff_b * diff_b).sum()

    trace_sw = trace_sw / float(n)
    trace_sb = trace_sb / float(n)
    return trace_sw / (trace_sb + eps)


def orthogonality_loss(weight: torch.Tensor) -> torch.Tensor:
    """Compute row-orthogonality penalty ``||W W^T - I||_F^2``."""
    out_dim = weight.shape[0]
    gram = torch.matmul(weight, weight.T)
    eye = torch.eye(out_dim, dtype=weight.dtype, device=weight.device)
    diff = gram - eye
    return torch.sum(diff * diff)


class LinearSubspaceAdapter(nn.Module):
    """Linear projection adapter matching ``Geometric3DEncoder`` inference semantics."""

    def __init__(self, in_dim: int, out_dim: int, init_weight: np.ndarray | None = None) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=True)
        with torch.no_grad():
            if init_weight is not None:
                self.linear.weight.copy_(torch.from_numpy(init_weight.astype(np.float32)))
            else:
                nn.init.orthogonal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), p=2, dim=-1, eps=1e-8)


def _compute_pca_orthogonal_init(
    features: np.ndarray,
    out_dim: int,
    seed: int = 42,
) -> np.ndarray:
    """Compute an exact row-orthonormal ``(out_dim, in_dim)`` init aligned with top PCA axes."""
    x = np.asarray(features, dtype=np.float64)
    _, in_dim = x.shape
    if out_dim > in_dim:
        raise ValueError(f"out_dim ({out_dim}) cannot exceed in_dim ({in_dim}) for orthogonal init")

    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(out_dim, vt.shape[0])
    basis = vt[:k].copy()  # (k, in_dim), orthonormal rows

    if k < out_dim:
        rng = np.random.default_rng(int(seed))
        extra = rng.standard_normal((out_dim - k, in_dim))
        rows = [basis[i] for i in range(k)]
        for row in extra:
            v = row.copy()
            for r in rows:
                v -= np.dot(v, r) * r
            norm = float(np.linalg.norm(v))
            if norm > 1e-8:
                rows.append(v / norm)
        mat = np.stack(rows[:out_dim], axis=0)
        q, _ = np.linalg.qr(mat.T)
        basis = q[:, :out_dim].T
    return basis.astype(np.float64)


def _stratified_train_val_split(
    features: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split ``(X, y)`` into stratified train and validation sets."""
    rng = np.random.default_rng(int(seed))
    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in np.unique(labels):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        n_cls = len(cls_idx)
        if n_cls <= 1:
            train_indices.extend(cls_idx.tolist())
            continue
        n_val = max(1, int(round(n_cls * float(val_ratio))))
        n_val = min(n_val, n_cls - 1)
        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())

    if not val_indices:
        return features, labels, features, labels

    tr = np.asarray(train_indices, dtype=np.int64)
    va = np.asarray(val_indices, dtype=np.int64)
    return features[tr], labels[tr], features[va], labels[va]


def _augment_partial_pointcloud(
    xyz: np.ndarray,
    rng: np.random.Generator,
    jitter_std: float = 0.0012,
    crop_min_frac: float = 0.65,
) -> np.ndarray:
    """Create realistic partial-view depth augmentation via directional cropping & sensor jitter."""
    pts = np.asarray(xyz, dtype=np.float32)
    if pts.shape[0] < 24:
        return pts.copy()

    direction = rng.standard_normal(3).astype(np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    proj = pts @ direction
    keep_frac = float(rng.uniform(crop_min_frac, 0.98))
    thresh = float(np.quantile(proj, 1.0 - keep_frac))
    cropped = pts[proj >= thresh]
    if cropped.shape[0] < 16:
        cropped = pts

    noise = rng.normal(0.0, jitter_std, size=cropped.shape).astype(np.float32)
    return cropped + noise


def collect_multiview_3d_features(
    object_names: Sequence[str] | None = None,
    num_template_shots: int = 8,
    num_clutter_episodes: int = 2,
    augmentations_per_view: int = 1,
    encoder: Geometric3DEncoder | None = None,
    image_width: int = 320,
    image_height: int = 240,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect multi-view partial 3D geometric features from ManiSkill 3 simulation.

    Combines clean multi-view renders from ``capture_few_shot_templates`` with
    cluttered multi-camera partial views from ``ClutteredTabletopSim`` (if
    ``num_clutter_episodes > 0``) and extracts raw 108-D ``Geometric3DEncoder``
    descriptors.

    Returns:
        Tuple ``(X, y, class_names)`` with ``X: (N, 108)`` float32, ``y: (N,)`` int64,
        and ordered ``class_names: list[str]``.
    """
    from omnigda.sim.clutter_env import (
        DEFAULT_BASE_OBJECTS,
        ClutteredTabletopSim,
        resolve_object_names,
    )
    from omnigda.sim.templates import capture_few_shot_templates

    resolved = resolve_object_names(
        object_names=object_names if object_names is not None else list(DEFAULT_BASE_OBJECTS)
    )
    class_names = list(resolved)
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    if encoder is None:
        enc = Geometric3DEncoder(seed=seed)
    else:
        # Ensure raw 108-D features are extracted even if caller passed an adapted encoder
        enc = Geometric3DEncoder(
            max_points=encoder.max_points,
            num_height_bins=encoder.num_height_bins,
            num_radial_bins=encoder.num_radial_bins,
            local_query_points=encoder.local_query_points,
            seed=encoder.seed,
        )

    rng = np.random.default_rng(int(seed))
    feats_list: list[np.ndarray] = []
    labels_list: list[int] = []

    def _add_cloud(xyz: np.ndarray, cls_idx: int) -> None:
        if xyz is None or xyz.ndim != 2 or xyz.shape[0] < 12:
            return
        desc = enc.encode(xyz)
        if np.linalg.norm(desc) > 1e-6:
            feats_list.append(desc.astype(np.float32))
            labels_list.append(cls_idx)
        for _ in range(int(augmentations_per_view)):
            aug_xyz = _augment_partial_pointcloud(xyz, rng=rng)
            aug_desc = enc.encode(aug_xyz)
            if np.linalg.norm(aug_desc) > 1e-6:
                feats_list.append(aug_desc.astype(np.float32))
                labels_list.append(cls_idx)

    # 1. Clean isolated multi-view studio templates
    if num_template_shots > 0:
        templates = capture_few_shot_templates(
            object_names=class_names,
            num_shots=int(num_template_shots),
            seed=int(seed),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        for name in class_names:
            cls_idx = name_to_idx[name]
            for sample in templates[name]:
                if sample.point_cloud_xyz is not None:
                    _add_cloud(sample.point_cloud_xyz, cls_idx)

    # 2. Multi-camera cluttered tabletop partial views
    if num_clutter_episodes > 0:
        with ClutteredTabletopSim(
            object_names=class_names,
            image_width=int(image_width),
            image_height=int(image_height),
            seed=int(seed),
            settle_steps=6,
        ) as sim:
            cam_names = sim.get_camera_names()
            for ep in range(int(num_clutter_episodes)):
                if ep > 0:
                    sim.reset(seed=int(seed) + ep * 17)
                for cam_name in cam_names:
                    frame, seg_map = sim.capture_rgbd(camera_name=cam_name)
                    if frame.gt_segmentation is None:
                        continue
                    for name in class_names:
                        obj_sids = [sid for sid, ename in seg_map.items() if ename == name]
                        if not obj_sids:
                            continue
                        mask = np.isin(frame.gt_segmentation, obj_sids)
                        if int(np.count_nonzero(mask)) < 20:
                            continue
                        pts_xyz, _ = backproject_depth(
                            depth=frame.depth,
                            fx=frame.intrinsics.fx,
                            fy=frame.intrinsics.fy,
                            cx=frame.intrinsics.cx,
                            cy=frame.intrinsics.cy,
                            camera_to_world=frame.camera_to_world,
                            mask=mask,
                            depth_min=0.05,
                            depth_max=2.5,
                        )
                        _add_cloud(pts_xyz, name_to_idx[name])

    if not feats_list:
        raise RuntimeError("No valid 3D point clouds were collected from simulation.")

    x_arr = np.stack(feats_list, axis=0).astype(np.float32)
    y_arr = np.asarray(labels_list, dtype=np.int64)
    return x_arr, y_arr, class_names


def generate_synthetic_3d_features(
    num_classes: int = 7,
    samples_per_class: int = 24,
    in_dim: int = 108,
    noise_std: float = 0.22,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate structured synthetic 108-D descriptors for fast CPU/unit verification."""
    rng = np.random.default_rng(int(seed))
    latent_dim = min(32, in_dim)
    basis, _ = np.linalg.qr(rng.standard_normal((in_dim, latent_dim)))
    centers_latent = rng.standard_normal((num_classes, latent_dim))
    centers = centers_latent @ basis.T

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for c in range(num_classes):
        samples = centers[c][None, :] + rng.normal(
            0.0, noise_std, size=(samples_per_class, in_dim)
        )
        xs.append(l2_normalize(samples, axis=-1))
        ys.extend([c] * samples_per_class)

    x_arr = np.concatenate(xs, axis=0).astype(np.float32)
    y_arr = np.asarray(ys, dtype=np.int64)
    class_names = [f"class_{i}" for i in range(num_classes)]
    return x_arr, y_arr, class_names


def _benchmark_cholesky_inversion_ms(dim: int, repeats: int = 200, seed: int = 42) -> float:
    """Benchmark average wall-clock latency (ms) of ``invert_spd`` on a ``(dim, dim)`` matrix."""
    rng = np.random.default_rng(int(seed))
    raw = rng.standard_normal((max(dim * 2, 64), dim))
    cov, _ = estimate_shrunk_covariance(raw, method="oas", min_eig=1e-4)

    for _ in range(10):
        _ = invert_spd(cov)

    t0 = time.perf_counter()
    for _ in range(int(repeats)):
        _ = invert_spd(cov)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / float(repeats)
    return float(elapsed_ms)


def _compute_fisher_ratio_np(features: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    """Compute ``tr(S_B) / (tr(S_W) + eps)`` (higher means better linear class separation)."""
    x = l2_normalize(np.asarray(features, dtype=np.float64), axis=-1)
    y = np.asarray(labels, dtype=np.int64)
    global_mean = np.mean(x, axis=0, keepdims=True)
    tr_sw = 0.0
    tr_sb = 0.0
    for cls in np.unique(y):
        sub = x[y == cls]
        if sub.shape[0] == 0:
            continue
        mu_c = np.mean(sub, axis=0, keepdims=True)
        tr_sw += float(np.sum((sub - mu_c) ** 2))
        tr_sb += float(sub.shape[0]) * float(np.sum((mu_c - global_mean) ** 2))
    return float(tr_sb / (tr_sw + eps))


def _evaluate_gda_top1(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Fit ``OnlineShrinkageGDA`` on train features and return Top-1 accuracy on validation set."""
    feats_by_class: dict[str, np.ndarray] = {}
    for cls in sorted(int(c) for c in np.unique(y_train)):
        feats_by_class[str(cls)] = x_train[y_train == cls]

    gda = OnlineShrinkageGDA(shrinkage_method="oas", min_eig=1e-4, normalize_features=True)
    gda.fit(feats_by_class)
    logits = gda.predict_logits(x_val)
    preds = np.asarray([int(gda.class_order[i]) for i in np.argmax(logits, axis=1)], dtype=np.int64)
    return float(np.mean(preds == y_val))


def evaluate_3d_adapter(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    cholesky_repeats: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate raw vs adapted 3D descriptors on held-out views and Cholesky inversion speed."""
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    in_dim = int(x_train.shape[1])
    out_dim = int(w.shape[0])

    x_tr_raw = l2_normalize(np.asarray(x_train, dtype=np.float64), axis=-1)
    x_va_raw = l2_normalize(np.asarray(x_val, dtype=np.float64), axis=-1)

    x_tr_proj = l2_normalize(x_tr_raw @ w.T + b[None, :], axis=-1)
    x_va_proj = l2_normalize(x_va_raw @ w.T + b[None, :], axis=-1)

    raw_top1 = _evaluate_gda_top1(x_tr_raw, y_train, x_va_raw, y_val)
    proj_top1 = _evaluate_gda_top1(x_tr_proj, y_train, x_va_proj, y_val)

    raw_fisher = _compute_fisher_ratio_np(x_va_raw, y_val)
    proj_fisher = _compute_fisher_ratio_np(x_va_proj, y_val)

    raw_chol_ms = _benchmark_cholesky_inversion_ms(in_dim, repeats=cholesky_repeats, seed=seed)
    proj_chol_ms = _benchmark_cholesky_inversion_ms(out_dim, repeats=cholesky_repeats, seed=seed)

    cov_reduction = float((in_dim * in_dim) / float(out_dim * out_dim))
    ortho_err = float(np.linalg.norm(w @ w.T - np.eye(out_dim, dtype=np.float64), ord="fro"))

    return {
        "raw_dim": float(in_dim),
        "out_dim": float(out_dim),
        "raw_top1_acc": raw_top1,
        "proj_top1_acc": proj_top1,
        "cov_reduction_factor": cov_reduction,
        "raw_cov_elements": float(in_dim * in_dim),
        "proj_cov_elements": float(out_dim * out_dim),
        "raw_cholesky_ms": raw_chol_ms,
        "proj_cholesky_ms": proj_chol_ms,
        "cholesky_speedup": float(raw_chol_ms / max(proj_chol_ms, 1e-9)),
        "raw_fisher_ratio": raw_fisher,
        "proj_fisher_ratio": proj_fisher,
        "final_ortho_fro_err": ortho_err,
    }


def save_3d_adapter(
    output_path: str | Path,
    weight: np.ndarray,
    bias: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export 3D linear adapter to zero-pickle ``.npz`` accepted by ``Geometric3DEncoder``.

    Writes ``weight`` and ``bias`` (keys required by
    ``Geometric3DEncoder.load_projection_adapter``) along with aliases ``W``, ``b``,
    and UTF-8 ``meta_json`` (safe under ``np.load(..., allow_pickle=False)``).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    meta_payload = dict(metadata) if metadata is not None else {}
    meta_payload.setdefault("in_dim", int(w.shape[1]))
    meta_payload.setdefault("out_dim", int(w.shape[0]))
    meta_payload.setdefault("adapter_type", "fisher_supcon_3d_linear")

    np.savez(
        path,
        weight=w,
        bias=b,
        W=w,
        b=b,
        meta_json=np.asarray(json.dumps(meta_payload, sort_keys=True), dtype=np.str_),
    )
    return path


def load_3d_adapter(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load 3D linear adapter weights and metadata from ``.npz`` without pickling."""
    data = np.load(Path(path), allow_pickle=False)
    weight = np.asarray(data["weight"] if "weight" in data else data["W"], dtype=np.float64)
    bias = np.asarray(
        data["bias"] if "bias" in data else data["b"],
        dtype=np.float64,
    )
    meta: dict[str, Any] = {}
    if "meta_json" in data:
        meta = json.loads(str(data["meta_json"].item()))
    return weight, bias, meta


def train_3d_metric_adapter(
    features: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    val_features: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    out_dim: int = 64,
    epochs: int = 150,
    lr: float = 2.5e-3,
    temperature: float = 0.15,
    lambda_fisher: float = 0.25,
    lambda_ortho: float = 0.02,
    val_ratio: float = 0.25,
    device: str = "cuda",
    seed: int = 42,
    output_path: str | Path | None = None,
    object_names: Sequence[str] | None = None,
    num_template_shots: int = 8,
    num_clutter_episodes: int = 2,
    augmentations_per_view: int = 1,
    cholesky_repeats: int = 200,
) -> Adapter3DTrainingResult:
    """Train a Fisher-SupCon 3D discriminative subspace adapter ``(D_out, D_in)``.

    Args:
        features: Optional pre-extracted ``(N, D_in)`` float features from ``Geometric3DEncoder``.
            If ``None``, multi-view partial point clouds are collected from ManiSkill 3.
        labels: Optional integer class labels ``(N,)`` matching ``features``.
        val_features: Optional held-out validation features ``(N_val, D_in)``.
        val_labels: Optional held-out validation labels ``(N_val,)``.
        class_names: Optional ordered class name list.
        out_dim: Target subspace dimension (default ``64``).
        epochs: Number of optimization epochs (default ``150``).
        lr: Initial Adam learning rate.
        temperature: SupCon temperature ``tau``.
        lambda_fisher: Weight ``lambda_f`` on ``tr(S_W) / (tr(S_B) + eps)``.
        lambda_ortho: Weight ``lambda_o`` on ``||W W^T - I||_F^2``.
        val_ratio: Stratified validation split ratio if ``val_features`` is ``None``.
        device: Torch device (``'cuda'`` falls back cleanly to ``'cpu'`` if unavailable).
        seed: Deterministic random seed.
        output_path: Optional ``.npz`` path to save adapter weights and metadata.
        object_names: Catalog objects used when collecting from simulation.
        num_template_shots: Isolated multi-view studio shots per object.
        num_clutter_episodes: Cluttered tabletop episodes for occluded partial views.
        augmentations_per_view: Partial-crop + jitter augmentations per captured view.
        cholesky_repeats: Repeat count for Cholesky inversion micro-benchmark.

    Returns:
        ``Adapter3DTrainingResult`` with learned ``weight``, ``bias``, loss history, and metrics.
    """
    if features is None or labels is None:
        features, labels, collected_names = collect_multiview_3d_features(
            object_names=object_names,
            num_template_shots=num_template_shots,
            num_clutter_episodes=num_clutter_episodes,
            augmentations_per_view=augmentations_per_view,
            seed=seed,
        )
        if class_names is None:
            class_names = collected_names

    x_all = l2_normalize(np.asarray(features, dtype=np.float32), axis=-1)
    y_all = np.asarray(labels, dtype=np.int64)
    if x_all.ndim != 2 or y_all.ndim != 1 or x_all.shape[0] != y_all.shape[0]:
        raise ValueError(f"Expected X (N, D) and y (N,), got X={x_all.shape}, y={y_all.shape}")

    in_dim = int(x_all.shape[1])
    out_dim = int(out_dim)
    if out_dim <= 0 or out_dim > in_dim:
        raise ValueError(f"out_dim must be in [1, {in_dim}], got {out_dim}")

    if val_features is not None and val_labels is not None:
        x_tr = x_all
        y_tr = y_all
        x_va = l2_normalize(np.asarray(val_features, dtype=np.float32), axis=-1)
        y_va = np.asarray(val_labels, dtype=np.int64)
    else:
        x_tr, y_tr, x_va, y_va = _stratified_train_val_split(
            x_all, y_all, val_ratio=val_ratio, seed=seed
        )

    if class_names is None:
        unique_ids = sorted(int(c) for c in np.unique(np.concatenate([y_tr, y_va])))
        resolved_names = [f"class_{c}" for c in unique_ids]
    else:
        resolved_names = [str(c) for c in class_names]

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    torch_device = torch.device(device if use_cuda else "cpu")

    init_w = _compute_pca_orthogonal_init(x_tr, out_dim=out_dim, seed=seed)
    model = LinearSubspaceAdapter(in_dim=in_dim, out_dim=out_dim, init_weight=init_w).to(
        torch_device
    )

    x_tensor = torch.from_numpy(x_tr.astype(np.float32)).to(torch_device)
    y_tensor = torch.from_numpy(y_tr.astype(np.int64)).to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=float(lr) * 0.05
    )

    history: dict[str, list[float]] = {
        "total_loss": [],
        "supcon_loss": [],
        "fisher_loss": [],
        "ortho_loss": [],
    }

    t_start = time.perf_counter()
    model.train()
    for _epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        z = model(x_tensor)
        loss_supcon = supervised_contrastive_loss(z, y_tensor, temperature=temperature)
        loss_fisher = fisher_discriminant_loss(z, y_tensor)
        loss_ortho = orthogonality_loss(model.linear.weight)
        loss_total = (
            loss_supcon
            + float(lambda_fisher) * loss_fisher
            + float(lambda_ortho) * loss_ortho
        )
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        history["total_loss"].append(float(loss_total.detach().cpu().item()))
        history["supcon_loss"].append(float(loss_supcon.detach().cpu().item()))
        history["fisher_loss"].append(float(loss_fisher.detach().cpu().item()))
        history["ortho_loss"].append(float(loss_ortho.detach().cpu().item()))

    train_time_sec = float(time.perf_counter() - t_start)

    weight_np = model.linear.weight.detach().cpu().numpy().astype(np.float64)
    bias_np = model.linear.bias.detach().cpu().numpy().astype(np.float64)

    metrics = evaluate_3d_adapter(
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_va,
        y_val=y_va,
        weight=weight_np,
        bias=bias_np,
        cholesky_repeats=cholesky_repeats,
        seed=seed,
    )
    metrics["train_time_sec"] = train_time_sec
    metrics["initial_loss"] = history["total_loss"][0] if history["total_loss"] else 0.0
    metrics["final_loss"] = history["total_loss"][-1] if history["total_loss"] else 0.0
    metrics["num_train_samples"] = float(x_tr.shape[0])
    metrics["num_val_samples"] = float(x_va.shape[0])

    metadata: dict[str, Any] = {
        "adapter_type": "fisher_supcon_3d_linear",
        "in_dim": in_dim,
        "out_dim": out_dim,
        "epochs": int(epochs),
        "lr": float(lr),
        "temperature": float(temperature),
        "lambda_fisher": float(lambda_fisher),
        "lambda_ortho": float(lambda_ortho),
        "seed": int(seed),
        "device": str(torch_device),
        "class_names": resolved_names,
        "metrics": metrics,
    }

    saved_path: Path | None = None
    if output_path is not None:
        saved_path = save_3d_adapter(output_path, weight_np, bias_np, metadata=metadata)

    return Adapter3DTrainingResult(
        weight=weight_np,
        bias=bias_np,
        history=history,
        metrics=metrics,
        class_names=resolved_names,
        output_path=saved_path,
        metadata=metadata,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Fisher-SupCon 3D Discriminative Subspace Adapter for Geometric3DEncoder."
    )
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--out_dim", type=int, default=64, help="Output subspace dimension.")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/geom3d_adapter_64d.npz",
        help="Destination .npz file path for exported adapter weights.",
    )
    parser.add_argument("--lr", type=float, default=2.5e-3, help="Initial Adam learning rate.")
    parser.add_argument("--temperature", type=float, default=0.15, help="SupCon temperature.")
    parser.add_argument(
        "--lambda_fisher",
        type=float,
        default=0.25,
        help="Weight for Fisher discriminant regularizer.",
    )
    parser.add_argument(
        "--lambda_ortho",
        type=float,
        default=0.02,
        help="Weight for row-orthogonality penalty ||W W^T - I||_F^2.",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=8,
        help="Multi-view studio template shots per object.",
    )
    parser.add_argument(
        "--num_clutter_episodes",
        type=int,
        default=2,
        help="Cluttered tabletop episodes for occluded partial point clouds.",
    )
    parser.add_argument(
        "--augmentations",
        type=int,
        default=1,
        help="Partial-view augmentations per captured cloud.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="Optional subset of catalog object names.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Validation split fraction.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run fast verification on synthetic 108-D features instead of ManiSkill 3.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Adapter3DTrainingResult:
    args = _build_arg_parser().parse_args(argv)
    features: np.ndarray | None = None
    labels: np.ndarray | None = None
    class_names: list[str] | None = None

    if args.synthetic:
        features, labels, class_names = generate_synthetic_3d_features(
            num_classes=7,
            samples_per_class=24,
            in_dim=Geometric3DEncoder.RAW_DIM,
            seed=args.seed,
        )

    result = train_3d_metric_adapter(
        features=features,
        labels=labels,
        class_names=class_names,
        out_dim=args.out_dim,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        lambda_fisher=args.lambda_fisher,
        lambda_ortho=args.lambda_ortho,
        val_ratio=args.val_ratio,
        device=args.device,
        seed=args.seed,
        output_path=args.output,
        object_names=args.objects,
        num_template_shots=args.num_shots,
        num_clutter_episodes=args.num_clutter_episodes,
        augmentations_per_view=args.augmentations,
    )

    m = result.metrics
    print(
        f"[3D Fisher-SupCon Adapter] {int(m['raw_dim'])}D -> {int(m['out_dim'])}D | "
        f"Loss: {m['initial_loss']:.4f} -> {m['final_loss']:.4f} ({m['train_time_sec']:.2f}s)"
    )
    print(
        f"  Held-out 3D GDA Top-1 Acc: {m['raw_top1_acc']*100:.1f}% -> {m['proj_top1_acc']*100:.1f}% | "
        f"Fisher Ratio: {m['raw_fisher_ratio']:.3f} -> {m['proj_fisher_ratio']:.3f}"
    )
    print(
        f"  Covariance Matrix: {int(m['raw_dim'])}x{int(m['raw_dim'])} -> "
        f"{int(m['out_dim'])}x{int(m['out_dim'])} ({m['cov_reduction_factor']:.2f}x smaller) | "
        f"Cholesky Inv: {m['raw_cholesky_ms']:.3f}ms -> {m['proj_cholesky_ms']:.3f}ms "
        f"({m['cholesky_speedup']:.2f}x speedup)"
    )
    if result.output_path is not None:
        print(f"  Saved adapter weights: {result.output_path}")
    return result


if __name__ == "__main__":
    main()
