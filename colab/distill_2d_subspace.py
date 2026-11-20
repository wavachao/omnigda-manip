"""Cross-Model 2D Discriminative Subspace Distillation Trainer.

Distills rich pairwise relational similarity geometry from a larger DINOv2 teacher
backbone (default ``vit_base_patch14_dinov2.lvd142m`` or ``vit_large_patch14_dinov2.lvd142m``
on Colab A100/L4) into a compact linear subspace projector (``384 -> 128``) on top of
student ``vit_small_patch14_dinov2.lvd142m`` dense patch features using:

    L = L_gram + alpha_cka * (1 - CKA(Z_S, X_T)) + lambda_supcon * L_supcon(Z_S, y)
        + lambda_ortho * ||W W^T - I||_F^2

Compressing 384-D DINOv2 patch descriptors into a distilled 128-D subspace cuts
``SegmentGDAPlus`` covariance matrix storage by exactly **9x** (``384x384 -> 128x128``)
and dramatically accelerates Cholesky precision inversion & online registration while
preserving discriminative geometry.

CLI example:
    python -m colab.distill_2d_subspace --epochs 150 --out_dim 128 --output artifacts/dinov2_subspace_128d.npz
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from colab.distill_3d_adapter import (
    LinearSubspaceAdapter,
    _benchmark_cholesky_inversion_ms,
    _compute_pca_orthogonal_init,
    _evaluate_gda_top1,
    _stratified_train_val_split,
    orthogonality_loss,
    supervised_contrastive_loss,
)
from omnigda.core.gda_math import l2_normalize
from omnigda.core.schemas import SupportSample
from omnigda.perception_2d.backbone import DenseFeatureExtractor, DINOv2MultiScaleExtractor

DEFAULT_STUDENT_MODEL = "vit_small_patch14_dinov2.lvd142m"
DEFAULT_TEACHER_MODEL = "vit_base_patch14_dinov2.lvd142m"


@dataclass
class Subspace2DTrainingResult:
    """Container for distilled 2D subspace projector weights, curves, and evaluation metrics."""

    weight: np.ndarray  # (D_out, D_student) float64
    bias: np.ndarray  # (D_out,) float64
    history: dict[str, list[float]]
    metrics: dict[str, float]
    class_names: list[str]
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _default_layers_for_model(model_name: str) -> tuple[int, ...]:
    """Select well-spaced late transformer block indices matching standard DINOv2 depths."""
    lower = model_name.lower()
    if "giant" in lower:
        return (30, 35, 39)
    if "large" in lower:
        return (18, 21, 23)
    return (8, 10, 11)


def _center_gram(gram: torch.Tensor) -> torch.Tensor:
    """Double-center an ``(N, N)`` Gram matrix: ``H G H`` in O(N^2) time."""
    row_mean = gram.mean(dim=1, keepdim=True)
    col_mean = gram.mean(dim=0, keepdim=True)
    total_mean = gram.mean()
    return gram - row_mean - col_mean + total_mean


def linear_cka_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute differentiable Linear CKA similarity in ``[0, 1]`` (Kornblith et al., ICML 2019).

    Args:
        x: Embeddings of shape ``(N, D_x)``.
        y: Target embeddings of shape ``(N, D_y)``.
        eps: Numerical stability floor.
    """
    if x.shape[0] <= 2:
        return x.new_ones(())
    gx = _center_gram(torch.matmul(x, x.T))
    gy = _center_gram(torch.matmul(y, y.T))
    hsic_xy = torch.sum(gx * gy)
    hsic_xx = torch.sum(gx * gx).clamp(min=eps)
    hsic_yy = torch.sum(gy * gy).clamp(min=eps)
    return hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)


def relational_cosine_distillation_loss(
    z_student: torch.Tensor,
    x_teacher: torch.Tensor,
    gram_weight: float = 4.0,
    cka_weight: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Relational Cosine Distillation Loss combining Gram MSE and Linear CKA alignment.

    Args:
        z_student: L2-normalized projected student embeddings ``(N, D_out)``.
        x_teacher: L2-normalized teacher embeddings ``(N, D_teacher)``.
        gram_weight: Relative weight on pairwise cosine Gram matrix MSE.
        cka_weight: Relative weight on ``1 - CKA(z_student, x_teacher)``.

    Returns:
        Tuple ``(total_relational_loss, gram_mse_loss, cka_loss)``.
    """
    gram_s = torch.matmul(z_student, z_student.T)
    gram_t = torch.matmul(x_teacher, x_teacher.T)
    gram_mse = F.mse_loss(gram_s, gram_t)
    cka_sim = linear_cka_similarity(z_student, x_teacher)
    cka_loss = 1.0 - cka_sim
    total = float(gram_weight) * gram_mse + float(cka_weight) * cka_loss
    return total, gram_mse, cka_loss


def _linear_cka_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """NumPy Linear CKA between ``(N, D_x)`` and ``(N, D_y)`` feature matrices."""
    if x.shape[0] <= 2:
        return 1.0
    gx = x @ x.T
    gy = y @ y.T
    gx = gx - gx.mean(axis=1, keepdims=True) - gx.mean(axis=0, keepdims=True) + gx.mean()
    gy = gy - gy.mean(axis=1, keepdims=True) - gy.mean(axis=0, keepdims=True) + gy.mean()
    num = float(np.sum(gx * gy))
    den = float(np.sqrt(max(float(np.sum(gx * gx)), eps) * max(float(np.sum(gy * gy)), eps)))
    return float(np.clip(num / max(den, eps), 0.0, 1.0))


def _gram_mse_np(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between pairwise cosine Gram matrices of unit-norm ``x`` and ``y``."""
    gx = x @ x.T
    gy = y @ y.T
    return float(np.mean((gx - gy) ** 2))


def save_2d_projector(
    output_path: str | Path,
    weight: np.ndarray,
    bias: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save 2D linear subspace projector weights to zero-pickle ``.npz``."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    meta_payload = dict(metadata) if metadata is not None else {}
    meta_payload.setdefault("in_dim", int(w.shape[1]))
    meta_payload.setdefault("out_dim", int(w.shape[0]))
    meta_payload.setdefault("adapter_type", "dinov2_relational_subspace_linear")

    np.savez(
        path,
        weight=w,
        bias=b,
        W=w,
        b=b,
        meta_json=np.asarray(json.dumps(meta_payload, sort_keys=True), dtype=np.str_),
    )
    return path


def load_2d_projector(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load 2D linear subspace projector weights and metadata from ``.npz`` without pickling."""
    data = np.load(Path(path), allow_pickle=False)
    weight = np.asarray(data["weight"] if "weight" in data else data["W"], dtype=np.float64)
    bias = np.asarray(data["bias"] if "bias" in data else data["b"], dtype=np.float64)
    meta: dict[str, Any] = {}
    if "meta_json" in data:
        meta = json.loads(str(data["meta_json"].item()))
    return weight, bias, meta


class ProjectedDenseFeatureExtractor(DenseFeatureExtractor):
    """Wraps any ``DenseFeatureExtractor`` with a distilled linear subspace projector.

    Drop-in compatible with ``SegmentGDAPlus``: projects ``(D_in, Hf, Wf)`` dense maps
    to L2-normalized ``(D_out, Hf, Wf)`` maps (e.g. ``384 -> 128``), cutting covariance
    matrix storage by ``(D_in / D_out)^2 = 9.0x``.
    """

    def __init__(
        self,
        base_extractor: DenseFeatureExtractor,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
    ) -> None:
        w = np.asarray(weight, dtype=np.float32)
        if w.ndim != 2 or w.shape[1] != base_extractor.feature_dim:
            raise ValueError(
                f"Expected weight shape (D_out, {base_extractor.feature_dim}), got {w.shape}"
            )
        b = (
            np.zeros(w.shape[0], dtype=np.float32)
            if bias is None
            else np.asarray(bias, dtype=np.float32)
        )
        if b.shape != (w.shape[0],):
            raise ValueError(f"Expected bias shape ({w.shape[0]},), got {b.shape}")

        self.base_extractor = base_extractor
        self.weight = w
        self.bias = b

    @classmethod
    def from_npz(
        cls,
        base_extractor: DenseFeatureExtractor,
        path: str | Path,
    ) -> ProjectedDenseFeatureExtractor:
        """Construct a projected dense extractor directly from an exported ``.npz`` file."""
        w, b, _ = load_2d_projector(path)
        return cls(base_extractor=base_extractor, weight=w, bias=b)

    @property
    def feature_dim(self) -> int:
        return int(self.weight.shape[0])

    @property
    def backend_name(self) -> str:
        return f"{self.base_extractor.backend_name}+subspace{self.feature_dim}d"

    def _project_map(self, feat_map: np.ndarray) -> np.ndarray:
        # feat_map: (D_in, Hf, Wf) -> (D_out, Hf, Wf)
        d_in, hf, wf = feat_map.shape
        flat = feat_map.reshape(d_in, -1)  # (D_in, Hf*Wf)
        proj = self.weight @ flat + self.bias[:, None]
        proj_map = proj.reshape(self.feature_dim, hf, wf)
        return l2_normalize(proj_map, axis=0, eps=1e-6).astype(np.float32, copy=False)

    def extract(self, rgb: np.ndarray) -> np.ndarray:
        return self._project_map(self.base_extractor.extract(rgb))

    def extract_batch(self, rgbs: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [self._project_map(f) for f in self.base_extractor.extract_batch(rgbs)]


def _sample_masked_patch_indices(
    mask: np.ndarray,
    hf: int,
    wf: int,
    max_tokens: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Select patch grid coordinates ``(ys, xs)`` covered by binary foreground mask."""
    mask_f = mask.astype(np.float32)
    mask_small = cv2.resize(mask_f, (wf, hf), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(mask_small >= 0.25)
    if ys.size == 0:
        # Fallback to highest-coverage patch cell(s) for very small objects
        flat_idx = int(np.argmax(mask_small))
        ys = np.asarray([flat_idx // wf], dtype=np.int64)
        xs = np.asarray([flat_idx % wf], dtype=np.int64)
    if ys.size > max_tokens:
        choice = rng.choice(ys.size, size=int(max_tokens), replace=False)
        ys = ys[choice]
        xs = xs[choice]
    return ys.astype(np.int64), xs.astype(np.int64)


def extract_dinov2_patch_dataset(
    support_samples: Sequence[SupportSample] | None = None,
    object_names: Sequence[str] | None = None,
    num_shots: int = 6,
    tokens_per_image: int = 24,
    student_model: str = DEFAULT_STUDENT_MODEL,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    input_size: int = 518,
    device: str = "cuda",
    fp16: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract spatially aligned student & teacher DINOv2 patch tokens for distillation.

    Extracts student features first, frees the student backbone, then extracts teacher
    features at identical patch coordinates so peak VRAM remains modest even with ViT-Large.

    Returns:
        Tuple ``(X_student, X_teacher, labels, class_names)`` with shapes
        ``(N, D_s)``, ``(N, D_t)``, ``(N,)``, and ordered class names.
    """
    if support_samples is None:
        from omnigda.sim.clutter_env import DEFAULT_BASE_OBJECTS, resolve_object_names
        from omnigda.sim.templates import capture_few_shot_templates

        resolved = resolve_object_names(
            object_names=object_names if object_names is not None else list(DEFAULT_BASE_OBJECTS)
        )
        templates = capture_few_shot_templates(
            object_names=resolved,
            num_shots=int(num_shots),
            seed=int(seed),
            image_width=320,
            image_height=240,
        )
        samples: list[SupportSample] = [
            s for name in resolved for s in templates[name] if s.rgb is not None and s.mask is not None
        ]
    else:
        samples = [s for s in support_samples if s.rgb is not None and s.mask is not None]

    if not samples:
        raise RuntimeError("No valid SupportSamples with rgb and mask available for 2D distillation.")

    class_names = list(dict.fromkeys(s.class_name for s in samples))
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    rgbs = [s.rgb for s in samples if s.rgb is not None]

    # Step 1: Student extraction
    student_ext = DINOv2MultiScaleExtractor(
        model_name=student_model,
        input_size=int(input_size),
        feature_layers=_default_layers_for_model(student_model),
        fusion_mode="mean",
        device=device,
        fp16=fp16,
        pretrained=True,
    )
    student_maps = [student_ext.extract(rgb) for rgb in rgbs]
    del student_ext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Teacher extraction
    teacher_ext = DINOv2MultiScaleExtractor(
        model_name=teacher_model,
        input_size=int(input_size),
        feature_layers=_default_layers_for_model(teacher_model),
        fusion_mode="mean",
        device=device,
        fp16=fp16,
        pretrained=True,
    )
    teacher_maps = [teacher_ext.extract(rgb) for rgb in rgbs]
    del teacher_ext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rng = np.random.default_rng(int(seed))
    s_tokens: list[np.ndarray] = []
    t_tokens: list[np.ndarray] = []
    y_tokens: list[int] = []

    for sample, s_map, t_map in zip(samples, student_maps, teacher_maps, strict=True):
        assert sample.mask is not None
        _, hf_s, wf_s = s_map.shape
        _, hf_t, wf_t = t_map.shape
        ys, xs = _sample_masked_patch_indices(
            sample.mask, hf=hf_s, wf=wf_s, max_tokens=int(tokens_per_image), rng=rng
        )
        # Map coordinates proportionally if teacher grid differs from student grid
        ys_t = np.clip(np.round(ys * (hf_t / float(hf_s))).astype(np.int64), 0, hf_t - 1)
        xs_t = np.clip(np.round(xs * (wf_t / float(wf_s))).astype(np.int64), 0, wf_t - 1)

        s_tok = s_map[:, ys, xs].T  # (M, D_s)
        t_tok = t_map[:, ys_t, xs_t].T  # (M, D_t)
        s_tokens.append(l2_normalize(s_tok, axis=-1))
        t_tokens.append(l2_normalize(t_tok, axis=-1))
        y_tokens.extend([name_to_idx[sample.class_name]] * int(ys.size))

    x_s = np.concatenate(s_tokens, axis=0).astype(np.float32)
    x_t = np.concatenate(t_tokens, axis=0).astype(np.float32)
    y = np.asarray(y_tokens, dtype=np.int64)
    return x_s, x_t, y, class_names


def generate_synthetic_2d_distill_data(
    num_classes: int = 8,
    tokens_per_class: int = 80,
    student_dim: int = 384,
    teacher_dim: int = 768,
    latent_dim: int = 48,
    student_noise_std: float = 0.08,
    teacher_noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate realistic correlated student (384-D) and teacher (768-D) token pairs.

    Models a stronger teacher backbone (cleaner semantic manifold, lower intra-class
    variance) paired with a compact noisy student backbone sharing underlying latent
    object geometry.
    """
    rng = np.random.default_rng(int(seed))
    k = min(int(latent_dim), int(student_dim), int(teacher_dim))
    basis_s, _ = np.linalg.qr(rng.standard_normal((student_dim, k)))
    basis_t, _ = np.linalg.qr(rng.standard_normal((teacher_dim, k)))

    # Semantic class prototypes + hierarchical inter-class correlations
    class_latents = rng.standard_normal((num_classes, k))
    class_latents = l2_normalize(class_latents, axis=-1)

    xs_list: list[np.ndarray] = []
    xt_list: list[np.ndarray] = []
    ys_list: list[int] = []

    for c in range(num_classes):
        # Shared instance-level semantic variation in latent space
        inst_var = rng.normal(0.0, 0.14, size=(tokens_per_class, k))
        latent = class_latents[c][None, :] + inst_var

        s_raw = latent @ basis_s.T + rng.normal(
            0.0, student_noise_std, size=(tokens_per_class, student_dim)
        )
        t_raw = latent @ basis_t.T + rng.normal(
            0.0, teacher_noise_std, size=(tokens_per_class, teacher_dim)
        )
        xs_list.append(l2_normalize(s_raw, axis=-1))
        xt_list.append(l2_normalize(t_raw, axis=-1))
        ys_list.extend([c] * tokens_per_class)

    x_s = np.concatenate(xs_list, axis=0).astype(np.float32)
    x_t = np.concatenate(xt_list, axis=0).astype(np.float32)
    y = np.asarray(ys_list, dtype=np.int64)
    class_names = [f"obj_{i}" for i in range(num_classes)]
    return x_s, x_t, y, class_names


def evaluate_2d_projector(
    x_s_train: np.ndarray,
    x_t_train: np.ndarray,
    y_train: np.ndarray,
    x_s_val: np.ndarray,
    x_t_val: np.ndarray,
    y_val: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    cholesky_repeats: int = 80,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate 2D subspace projector on relational alignment, GDA accuracy, and 9x cost reduction."""
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    in_dim = int(x_s_train.shape[1])
    out_dim = int(w.shape[0])
    teacher_dim = int(x_t_train.shape[1])

    xs_tr = l2_normalize(np.asarray(x_s_train, dtype=np.float64), axis=-1)
    xs_va = l2_normalize(np.asarray(x_s_val, dtype=np.float64), axis=-1)
    xt_va = l2_normalize(np.asarray(x_t_val, dtype=np.float64), axis=-1)

    zs_tr = l2_normalize(xs_tr @ w.T + b[None, :], axis=-1)
    zs_va = l2_normalize(xs_va @ w.T + b[None, :], axis=-1)

    raw_cka = _linear_cka_np(xs_va, xt_va)
    proj_cka = _linear_cka_np(zs_va, xt_va)
    raw_gram_mse = _gram_mse_np(xs_va, xt_va)
    proj_gram_mse = _gram_mse_np(zs_va, xt_va)

    raw_top1 = _evaluate_gda_top1(xs_tr, y_train, xs_va, y_val)
    proj_top1 = _evaluate_gda_top1(zs_tr, y_train, zs_va, y_val)

    raw_chol_ms = _benchmark_cholesky_inversion_ms(in_dim, repeats=cholesky_repeats, seed=seed)
    proj_chol_ms = _benchmark_cholesky_inversion_ms(out_dim, repeats=cholesky_repeats, seed=seed)

    cov_reduction = float((in_dim * in_dim) / float(out_dim * out_dim))
    ortho_err = float(np.linalg.norm(w @ w.T - np.eye(out_dim, dtype=np.float64), ord="fro"))

    return {
        "raw_dim": float(in_dim),
        "out_dim": float(out_dim),
        "teacher_dim": float(teacher_dim),
        "raw_cka": raw_cka,
        "proj_cka": proj_cka,
        "raw_gram_mse": raw_gram_mse,
        "proj_gram_mse": proj_gram_mse,
        "raw_top1_acc": raw_top1,
        "proj_top1_acc": proj_top1,
        "cov_reduction_factor": cov_reduction,
        "raw_cov_elements": float(in_dim * in_dim),
        "proj_cov_elements": float(out_dim * out_dim),
        "raw_cholesky_ms": raw_chol_ms,
        "proj_cholesky_ms": proj_chol_ms,
        "cholesky_speedup": float(raw_chol_ms / max(proj_chol_ms, 1e-9)),
        "final_ortho_fro_err": ortho_err,
    }


def train_2d_subspace_projector(
    student_features: np.ndarray | None = None,
    teacher_features: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    val_student_features: np.ndarray | None = None,
    val_teacher_features: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    out_dim: int = 128,
    epochs: int = 150,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 1024,
    gram_weight: float = 4.0,
    cka_weight: float = 2.0,
    lambda_supcon: float = 0.12,
    lambda_ortho: float = 0.01,
    temperature: float = 0.15,
    val_ratio: float = 0.25,
    device: str = "cuda",
    seed: int = 42,
    output_path: str | Path | None = None,
    student_model: str = DEFAULT_STUDENT_MODEL,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    object_names: Sequence[str] | None = None,
    num_shots: int = 6,
    tokens_per_image: int = 24,
    input_size: int = 518,
    cholesky_repeats: int = 80,
) -> Subspace2DTrainingResult:
    """Train a cross-model 2D DINOv2 linear subspace projector (e.g., ``384 -> 128``).

    Args:
        student_features: Optional ``(N, D_s)`` student patch features (default ``384``).
            If ``None``, extracts paired student/teacher tokens from ManiSkill 3 renders.
        teacher_features: Optional ``(N, D_t)`` teacher patch features (``768`` or ``1024``).
        labels: Optional integer class labels ``(N,)``.
        val_student_features: Optional held-out validation student features.
        val_teacher_features: Optional held-out validation teacher features.
        val_labels: Optional held-out validation labels.
        class_names: Optional ordered class name list.
        out_dim: Compact target subspace dimension (default ``128`` for 9x covariance cut).
        epochs: Training epochs.
        lr: Initial Adam learning rate.
        batch_size: Mini-batch size (runs exact full-batch when ``N <= batch_size``).
        cka_weight: Weight on Linear CKA loss ``1 - CKA(Z_S, X_T)``.
        lambda_supcon: Weight on class-separation SupCon loss.
        lambda_ortho: Weight on row-orthogonality penalty ``||W W^T - I||_F^2``.
        temperature: SupCon temperature.
        val_ratio: Stratified validation fraction when explicit val arrays are not given.
        device: Torch device (falls back cleanly to CPU when CUDA is unavailable).
        seed: Deterministic RNG seed.
        output_path: Optional ``.npz`` destination path.
        student_model: Student timm DINOv2 model name.
        teacher_model: Teacher timm DINOv2 model name.
        object_names: Catalog objects when extracting from simulation.
        num_shots: Isolated multi-view shots per object when extracting from simulation.
        tokens_per_image: Foreground tokens sampled per image.
        input_size: ViT input resolution (must be multiple of 14 for DINOv2).
        cholesky_repeats: Repeat count for Cholesky inversion benchmark.

    Returns:
        ``Subspace2DTrainingResult`` containing projector weights, curves, and metrics.
    """
    if student_features is None or teacher_features is None or labels is None:
        student_features, teacher_features, labels, extracted_names = extract_dinov2_patch_dataset(
            object_names=object_names,
            num_shots=num_shots,
            tokens_per_image=tokens_per_image,
            student_model=student_model,
            teacher_model=teacher_model,
            input_size=input_size,
            device=device,
            seed=seed,
        )
        if class_names is None:
            class_names = extracted_names

    xs_all = l2_normalize(np.asarray(student_features, dtype=np.float32), axis=-1)
    xt_all = l2_normalize(np.asarray(teacher_features, dtype=np.float32), axis=-1)
    y_all = np.asarray(labels, dtype=np.int64)

    if xs_all.ndim != 2 or xt_all.ndim != 2 or y_all.ndim != 1:
        raise ValueError(
            f"Expected 2D feature matrices and 1D labels, got {xs_all.shape}, {xt_all.shape}, {y_all.shape}"
        )
    if not (xs_all.shape[0] == xt_all.shape[0] == y_all.shape[0]):
        raise ValueError(
            f"Sample count mismatch: student={xs_all.shape[0]}, teacher={xt_all.shape[0]}, labels={y_all.shape[0]}"
        )

    in_dim = int(xs_all.shape[1])
    out_dim = int(out_dim)
    if out_dim <= 0 or out_dim > in_dim:
        raise ValueError(f"out_dim must be in [1, {in_dim}], got {out_dim}")

    if (
        val_student_features is not None
        and val_teacher_features is not None
        and val_labels is not None
    ):
        xs_tr, xt_tr, y_tr = xs_all, xt_all, y_all
        xs_va = l2_normalize(np.asarray(val_student_features, dtype=np.float32), axis=-1)
        xt_va = l2_normalize(np.asarray(val_teacher_features, dtype=np.float32), axis=-1)
        y_va = np.asarray(val_labels, dtype=np.int64)
    else:
        # Stratified split preserving alignment across (xs, xt, y)
        idx_all = np.arange(xs_all.shape[0], dtype=np.int64)
        idx_tr, y_tr, idx_va, y_va = _stratified_train_val_split(
            idx_all[:, None], y_all, val_ratio=val_ratio, seed=seed
        )
        tr_i = idx_tr[:, 0]
        va_i = idx_va[:, 0]
        xs_tr, xt_tr = xs_all[tr_i], xt_all[tr_i]
        xs_va, xt_va = xs_all[va_i], xt_all[va_i]

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

    init_w = _compute_pca_orthogonal_init(xs_tr, out_dim=out_dim, seed=seed)
    model = LinearSubspaceAdapter(in_dim=in_dim, out_dim=out_dim, init_weight=init_w).to(
        torch_device
    )

    xs_tensor = torch.from_numpy(xs_tr.astype(np.float32)).to(torch_device)
    xt_tensor = torch.from_numpy(xt_tr.astype(np.float32)).to(torch_device)
    y_tensor = torch.from_numpy(y_tr.astype(np.int64)).to(torch_device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(epochs), 1), eta_min=float(lr) * 0.05
    )

    history: dict[str, list[float]] = {
        "total_loss": [],
        "gram_loss": [],
        "cka_loss": [],
        "supcon_loss": [],
        "ortho_loss": [],
    }

    n_train = xs_tensor.shape[0]
    bs = max(int(batch_size), 16)
    gen = torch.Generator(device=torch_device)
    gen.manual_seed(int(seed))

    t_start = time.perf_counter()
    model.train()
    for _epoch in range(int(epochs)):
        if n_train <= bs:
            batches = [None]
        else:
            perm = torch.randperm(n_train, generator=gen, device=torch_device)
            batches = [perm[i : i + bs] for i in range(0, n_train, bs)]

        ep_total = 0.0
        ep_gram = 0.0
        ep_cka = 0.0
        ep_supcon = 0.0
        ep_ortho = 0.0
        n_steps = 0

        for idx in batches:
            if idx is None:
                bx_s, bx_t, by = xs_tensor, xt_tensor, y_tensor
            else:
                if idx.numel() < 4:
                    continue
                bx_s, bx_t, by = xs_tensor[idx], xt_tensor[idx], y_tensor[idx]

            optimizer.zero_grad(set_to_none=True)
            bz_s = model(bx_s)
            loss_rel, loss_gram, loss_cka = relational_cosine_distillation_loss(
                bz_s, bx_t, gram_weight=gram_weight, cka_weight=cka_weight
            )
            loss_supcon = supervised_contrastive_loss(bz_s, by, temperature=temperature)
            loss_ortho = orthogonality_loss(model.linear.weight)
            loss_total = (
                loss_rel
                + float(lambda_supcon) * loss_supcon
                + float(lambda_ortho) * loss_ortho
            )
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            ep_total += float(loss_total.detach().cpu().item())
            ep_gram += float(loss_gram.detach().cpu().item())
            ep_cka += float(loss_cka.detach().cpu().item())
            ep_supcon += float(loss_supcon.detach().cpu().item())
            ep_ortho += float(loss_ortho.detach().cpu().item())
            n_steps += 1

        scheduler.step()
        denom = float(max(n_steps, 1))
        history["total_loss"].append(ep_total / denom)
        history["gram_loss"].append(ep_gram / denom)
        history["cka_loss"].append(ep_cka / denom)
        history["supcon_loss"].append(ep_supcon / denom)
        history["ortho_loss"].append(ep_ortho / denom)

    train_time_sec = float(time.perf_counter() - t_start)

    weight_np = model.linear.weight.detach().cpu().numpy().astype(np.float64)
    bias_np = model.linear.bias.detach().cpu().numpy().astype(np.float64)

    metrics = evaluate_2d_projector(
        x_s_train=xs_tr,
        x_t_train=xt_tr,
        y_train=y_tr,
        x_s_val=xs_va,
        x_t_val=xt_va,
        y_val=y_va,
        weight=weight_np,
        bias=bias_np,
        cholesky_repeats=cholesky_repeats,
        seed=seed,
    )
    metrics["train_time_sec"] = train_time_sec
    metrics["initial_loss"] = history["total_loss"][0] if history["total_loss"] else 0.0
    metrics["final_loss"] = history["total_loss"][-1] if history["total_loss"] else 0.0
    metrics["num_train_tokens"] = float(xs_tr.shape[0])
    metrics["num_val_tokens"] = float(xs_va.shape[0])

    metadata: dict[str, Any] = {
        "adapter_type": "dinov2_relational_subspace_linear",
        "student_model": student_model,
        "teacher_model": teacher_model,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "teacher_dim": int(xt_tr.shape[1]),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "gram_weight": float(gram_weight),
        "cka_weight": float(cka_weight),
        "lambda_supcon": float(lambda_supcon),
        "lambda_ortho": float(lambda_ortho),
        "temperature": float(temperature),
        "seed": int(seed),
        "device": str(torch_device),
        "class_names": resolved_names,
        "metrics": metrics,
    }

    saved_path: Path | None = None
    if output_path is not None:
        saved_path = save_2d_projector(output_path, weight_np, bias_np, metadata=metadata)

    return Subspace2DTrainingResult(
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
        description="Cross-model 2D DINOv2 relational subspace distillation trainer (384D -> 128D)."
    )
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs.")
    parser.add_argument("--out_dim", type=int, default=128, help="Target subspace dimension.")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/dinov2_subspace_128d.npz",
        help="Destination .npz file path for distilled projector weights.",
    )
    parser.add_argument(
        "--student",
        type=str,
        default=DEFAULT_STUDENT_MODEL,
        help="Student DINOv2 timm model name (384-D).",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default=DEFAULT_TEACHER_MODEL,
        help="Teacher DINOv2 timm model name (e.g. vit_base or vit_large).",
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="Initial AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Mini-batch size.")
    parser.add_argument("--gram_weight", type=float, default=4.0, help="Gram MSE loss weight.")
    parser.add_argument("--cka_weight", type=float, default=2.0, help="Linear CKA loss weight.")
    parser.add_argument(
        "--lambda_supcon", type=float, default=0.12, help="Class-separation SupCon loss weight."
    )
    parser.add_argument(
        "--lambda_ortho",
        type=float,
        default=0.01,
        help="Row-orthogonality penalty ||W W^T - I||_F^2 weight.",
    )
    parser.add_argument("--temperature", type=float, default=0.15, help="SupCon temperature.")
    parser.add_argument("--num_shots", type=int, default=6, help="Sim shots per object.")
    parser.add_argument(
        "--tokens_per_image", type=int, default=24, help="Foreground tokens sampled per view."
    )
    parser.add_argument("--input_size", type=int, default=518, help="DINOv2 input image size.")
    parser.add_argument("--objects", nargs="+", default=None, help="Optional catalog object subset.")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Validation split fraction.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (cuda or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run fast synthetic 384D<-768D verification mode without loading DINOv2 teacher.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Subspace2DTrainingResult:
    args = _build_arg_parser().parse_args(argv)
    student_feats: np.ndarray | None = None
    teacher_feats: np.ndarray | None = None
    labels: np.ndarray | None = None
    class_names: list[str] | None = None

    if args.synthetic:
        student_feats, teacher_feats, labels, class_names = generate_synthetic_2d_distill_data(
            num_classes=8,
            tokens_per_class=80,
            student_dim=384,
            teacher_dim=768,
            seed=args.seed,
        )

    result = train_2d_subspace_projector(
        student_features=student_feats,
        teacher_features=teacher_feats,
        labels=labels,
        class_names=class_names,
        out_dim=args.out_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gram_weight=args.gram_weight,
        cka_weight=args.cka_weight,
        lambda_supcon=args.lambda_supcon,
        lambda_ortho=args.lambda_ortho,
        temperature=args.temperature,
        val_ratio=args.val_ratio,
        device=args.device,
        seed=args.seed,
        output_path=args.output,
        student_model=args.student,
        teacher_model=args.teacher,
        object_names=args.objects,
        num_shots=args.num_shots,
        tokens_per_image=args.tokens_per_image,
        input_size=args.input_size,
    )

    m = result.metrics
    print(
        f"[2D DINOv2 Subspace Distillation] {int(m['raw_dim'])}D -> {int(m['out_dim'])}D "
        f"(Teacher {int(m['teacher_dim'])}D) | Loss: {m['initial_loss']:.4f} -> "
        f"{m['final_loss']:.4f} ({m['train_time_sec']:.2f}s)"
    )
    print(
        f"  Relational CKA: {m['raw_cka']:.4f} -> {m['proj_cka']:.4f} | "
        f"Gram MSE: {m['raw_gram_mse']:.5f} -> {m['proj_gram_mse']:.5f} | "
        f"GDA Top-1: {m['raw_top1_acc']*100:.1f}% -> {m['proj_top1_acc']*100:.1f}%"
    )
    print(
        f"  Covariance Matrix: {int(m['raw_dim'])}x{int(m['raw_dim'])} -> "
        f"{int(m['out_dim'])}x{int(m['out_dim'])} ({m['cov_reduction_factor']:.2f}x smaller) | "
        f"Cholesky Inv: {m['raw_cholesky_ms']:.3f}ms -> {m['proj_cholesky_ms']:.3f}ms "
        f"({m['cholesky_speedup']:.2f}x speedup)"
    )
    if result.output_path is not None:
        print(f"  Saved projector weights: {result.output_path}")
    return result


if __name__ == "__main__":
    main()
