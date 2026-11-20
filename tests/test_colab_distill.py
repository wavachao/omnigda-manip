"""Tests for T8 Colab GPU subspace distillation trainers, encoder adapters, and notebook."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colab.distill_2d_subspace import (  # noqa: E402
    ProjectedDenseFeatureExtractor,
    generate_synthetic_2d_distill_data,
    linear_cka_similarity,
    load_2d_projector,
    relational_cosine_distillation_loss,
    train_2d_subspace_projector,
)
from colab.distill_3d_adapter import (  # noqa: E402
    collect_multiview_3d_features,
    fisher_discriminant_loss,
    load_3d_adapter,
    orthogonality_loss,
    supervised_contrastive_loss,
    train_3d_metric_adapter,
)
from omnigda.core.gda_math import l2_normalize  # noqa: E402
from omnigda.core.schemas import SupportSample  # noqa: E402
from omnigda.perception_2d.backbone import DINOv2MultiScaleExtractor  # noqa: E402
from omnigda.perception_2d.segment_gda import SegmentGDAPlus  # noqa: E402
from omnigda.perception_3d.encoders import Geometric3DEncoder  # noqa: E402
from omnigda.perception_3d.multimodal_gmda import PartialAwareMultimodalGMDA  # noqa: E402
from omnigda.sim.templates import capture_few_shot_templates  # noqa: E402


def test_distillation_loss_functions_mathematical_properties() -> None:
    """Verify SupCon, Fisher, Orthogonality, and CKA/Gram losses satisfy core invariants."""
    rng = np.random.default_rng(7)

    # 1. Well-separated unit clusters vs noisy overlapping clusters
    centers = np.eye(4, 16, dtype=np.float32)
    clean_z = np.concatenate(
        [centers[c][None, :] + rng.normal(0.0, 0.03, size=(8, 16)) for c in range(4)], axis=0
    )
    noisy_z = np.concatenate(
        [centers[c][None, :] + rng.normal(0.0, 0.45, size=(8, 16)) for c in range(4)], axis=0
    )
    clean_t = torch.from_numpy(l2_normalize(clean_z, axis=-1).astype(np.float32))
    noisy_t = torch.from_numpy(l2_normalize(noisy_z, axis=-1).astype(np.float32))
    labels = torch.repeat_interleave(torch.arange(4, dtype=torch.long), 8)

    assert float(supervised_contrastive_loss(clean_t, labels)) < float(
        supervised_contrastive_loss(noisy_t, labels)
    )
    assert float(fisher_discriminant_loss(clean_t, labels)) < float(
        fisher_discriminant_loss(noisy_t, labels)
    )

    # 2. Exact row-orthonormal matrix has zero orthogonality penalty
    q, _ = np.linalg.qr(rng.standard_normal((32, 16)))
    w_ortho = torch.from_numpy(q.T.astype(np.float32))  # (16, 32)
    assert float(orthogonality_loss(w_ortho)) == pytest.approx(0.0, abs=1e-5)

    # 3. Linear CKA and Relational Gram MSE invariants
    rel_self, gram_self, cka_self = relational_cosine_distillation_loss(clean_t, clean_t)
    assert float(linear_cka_similarity(clean_t, clean_t)) == pytest.approx(1.0, abs=1e-5)
    assert float(gram_self) == pytest.approx(0.0, abs=1e-6)
    assert float(cka_self) == pytest.approx(0.0, abs=1e-5)
    assert float(rel_self) == pytest.approx(0.0, abs=1e-5)


def test_3d_adapter_maniskill_training_export_and_gmda_integration(tmp_path: Path) -> None:
    """Train 3D Fisher-SupCon adapter on real ManiSkill 3 partial clouds & verify inside GMDA."""
    objects = ["red_mug", "blue_bottle", "green_box", "purple_bar"]
    x_raw, y_raw, class_names = collect_multiview_3d_features(
        object_names=objects,
        num_template_shots=6,
        num_clutter_episodes=1,
        augmentations_per_view=1,
        image_width=256,
        image_height=192,
        seed=42,
    )
    assert x_raw.ndim == 2 and x_raw.shape[1] == Geometric3DEncoder.RAW_DIM == 108
    assert x_raw.shape[0] == y_raw.shape[0]
    assert class_names == objects

    npz_path = tmp_path / "geom3d_adapter_64d.npz"
    result = train_3d_metric_adapter(
        features=x_raw,
        labels=y_raw,
        class_names=class_names,
        out_dim=64,
        epochs=120,
        lr=2.5e-3,
        temperature=0.15,
        lambda_fisher=0.25,
        lambda_ortho=0.02,
        device="cuda",
        seed=42,
        output_path=npz_path,
        cholesky_repeats=60,
    )

    # 1. Verify monotonic loss decrease
    losses = result.history["total_loss"]
    assert len(losses) == 120
    assert losses[-1] < losses[0] * 0.85, f"Expected substantial loss drop, got {losses[0]} -> {losses[-1]}"
    diffs = np.diff(np.asarray(losses, dtype=np.float64))
    assert float(np.max(diffs)) <= 1e-4, f"Loss increased unexpectedly by {float(np.max(diffs))}"

    # 2. Verify geometry & compression metrics
    m = result.metrics
    assert result.weight.shape == (64, 108)
    assert result.bias.shape == (64,)
    assert m["cov_reduction_factor"] == pytest.approx((108 * 108) / (64 * 64), rel=1e-6)
    assert m["proj_fisher_ratio"] > m["raw_fisher_ratio"]
    assert m["raw_cholesky_ms"] > 0.0 and m["proj_cholesky_ms"] > 0.0
    assert m["proj_top1_acc"] >= 0.70

    # 3. Verify zero-pickle .npz export and exact compatibility with Geometric3DEncoder
    assert npz_path.exists()
    raw_npz = np.load(npz_path, allow_pickle=False)
    for key in ("weight", "bias", "W", "b", "meta_json"):
        assert key in raw_npz
    w_loaded, b_loaded, meta_loaded = load_3d_adapter(npz_path)
    np.testing.assert_allclose(w_loaded, result.weight, atol=1e-12)
    np.testing.assert_allclose(b_loaded, result.bias, atol=1e-12)
    assert meta_loaded["out_dim"] == 64
    assert meta_loaded["in_dim"] == 108

    # 4. Load into Geometric3DEncoder and verify exact output shape, norm, and formula
    raw_enc = Geometric3DEncoder(seed=42)
    adapted_enc = Geometric3DEncoder(seed=42)
    adapted_enc.load_projection_adapter(npz_path)
    assert adapted_enc.feature_dim == 64

    templates = capture_few_shot_templates(
        object_names=objects,
        num_shots=3,
        seed=123,
        image_width=256,
        image_height=192,
    )
    sample_cloud = templates[objects[0]][0].point_cloud_xyz
    assert sample_cloud is not None and sample_cloud.shape[0] >= 16

    raw_vec = raw_enc.encode(sample_cloud)
    proj_vec = adapted_enc.encode(sample_cloud)
    assert raw_vec.shape == (108,)
    assert proj_vec.shape == (64,)
    assert proj_vec.dtype == np.float32
    assert float(np.linalg.norm(proj_vec)) == pytest.approx(1.0, abs=1e-5)

    expected_proj = l2_normalize(raw_vec.astype(np.float64) @ w_loaded.T + b_loaded).astype(
        np.float32
    )
    np.testing.assert_allclose(proj_vec, expected_proj, atol=1e-5)

    # 5. Seamless integration inside PartialAwareMultimodalGMDA (64-D geometry subspace)
    gmda = PartialAwareMultimodalGMDA(
        geom_encoder=adapted_enc,
        enable_clip=False,
    )
    for obj_name in objects:
        geom_only_samples = [
            SupportSample(
                class_name=s.class_name,
                point_cloud_xyz=s.point_cloud_xyz,
                text_prompt=None,
                metadata=dict(s.metadata),
            )
            for s in templates[obj_name][:2]
        ]
        gmda.register_few_shot(obj_name, geom_only_samples)

    assert gmda._models["geom_3d"].dim == 64
    assert gmda._models["geom_3d"]._cal_precision is not None
    assert gmda._models["geom_3d"]._cal_precision.shape == (64, 64)

    correct = 0
    for obj_name in objects:
        held_out = templates[obj_name][2]
        assert held_out.point_cloud_xyz is not None
        det = gmda.classify_instance(
            instance_id=0,
            xyz=held_out.point_cloud_xyz,
        )
        if det.class_name == obj_name:
            correct += 1
        assert det.confidence > 0.0
    assert correct >= 3, f"Expected >= 3/4 held-out GMDA classifications correct, got {correct}/4"


def test_2d_subspace_projector_convergence_roundtrip_and_segment_gda(tmp_path: Path) -> None:
    """Verify 2D DINOv2 subspace distillation (384 -> 128), .npz round-trip, and 9x cov cut."""
    xs, xt, y, class_names = generate_synthetic_2d_distill_data(
        num_classes=8,
        tokens_per_class=80,
        student_dim=384,
        teacher_dim=768,
        seed=42,
    )
    npz_path = tmp_path / "dinov2_subspace_128d.npz"

    result = train_2d_subspace_projector(
        student_features=xs,
        teacher_features=xt,
        labels=y,
        class_names=class_names,
        out_dim=128,
        epochs=120,
        lr=2e-3,
        weight_decay=1e-2,
        gram_weight=4.0,
        cka_weight=2.0,
        lambda_supcon=0.12,
        lambda_ortho=0.01,
        device="cuda",
        seed=42,
        output_path=npz_path,
        cholesky_repeats=40,
    )

    # 1. Verify smooth loss convergence & dimensionality
    losses = result.history["total_loss"]
    assert len(losses) == 120
    assert losses[-1] < losses[0] * 0.85
    # Verify monotonic decrease across 10-epoch windows and bounded per-step AdamW jitter
    window_means = [float(np.mean(losses[i : i + 10])) for i in range(0, 120, 10)]
    assert all(window_means[i + 1] <= window_means[i] for i in range(len(window_means) - 1))
    diffs = np.diff(np.asarray(losses, dtype=np.float64))
    assert float(np.max(diffs)) <= 5e-4

    assert result.weight.shape == (128, 384)
    assert result.bias.shape == (128,)

    # 2. Verify 9x covariance cut + improved relational alignment + preserved GDA accuracy
    m = result.metrics
    assert m["cov_reduction_factor"] == pytest.approx(9.0, abs=1e-6)
    assert m["raw_cov_elements"] == 384 * 384
    assert m["proj_cov_elements"] == 128 * 128
    assert m["proj_cka"] > m["raw_cka"]
    assert m["proj_gram_mse"] < m["raw_gram_mse"]
    assert m["proj_top1_acc"] >= 0.90
    assert m["cholesky_speedup"] > 1.5

    # 3. Verify zero-pickle .npz round-trip
    assert npz_path.exists()
    w_loaded, b_loaded, meta_loaded = load_2d_projector(npz_path)
    np.testing.assert_allclose(w_loaded, result.weight, atol=1e-12)
    np.testing.assert_allclose(b_loaded, result.bias, atol=1e-12)
    assert meta_loaded["in_dim"] == 384
    assert meta_loaded["out_dim"] == 128

    # 4. Verify ProjectedDenseFeatureExtractor wraps cached DINOv2 ViT-S and cuts SegmentGDAPlus cov 9x
    base_ext = DINOv2MultiScaleExtractor(
        model_name="vit_small_patch14_dinov2.lvd142m",
        input_size=224,
        feature_layers=(8, 10, 11),
        device="cuda",
        fp16=True,
        pretrained=True,
    )
    proj_ext = ProjectedDenseFeatureExtractor.from_npz(base_ext, npz_path)
    assert base_ext.feature_dim == 384
    assert proj_ext.feature_dim == 128

    rng = np.random.default_rng(99)
    rgb_a = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    rgb_a[20:70, 20:70] = np.array([220, 30, 30], dtype=np.uint8)
    mask_a = np.zeros((128, 128), dtype=bool)
    mask_a[20:70, 20:70] = True

    rgb_b = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    rgb_b[50:105, 50:105] = np.array([30, 60, 220], dtype=np.uint8)
    mask_b = np.zeros((128, 128), dtype=bool)
    mask_b[50:105, 50:105] = True

    feat_map = proj_ext.extract(rgb_a)
    assert feat_map.shape == (128, 16, 16)
    norms = np.linalg.norm(feat_map, axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    seg = SegmentGDAPlus(extractor=proj_ext, max_bg_tokens=64)
    seg.fit(
        [
            SupportSample(class_name="red_patch", rgb=rgb_a, mask=mask_a),
            SupportSample(class_name="blue_patch", rgb=rgb_b, mask=mask_b),
        ]
    )
    assert seg._gda.covariance is not None
    assert seg._gda.covariance.shape == (128, 128)
    seg_res = seg.segment(rgb_a)
    assert seg_res.class_map.shape == (128, 128)


def test_colab_notebook_valid_nbformat4() -> None:
    """Validate colab/train_adapters_colab.ipynb JSON, nbformat 4 schema, and Python syntax."""
    nb_path = REPO_ROOT / "colab" / "train_adapters_colab.ipynb"
    assert nb_path.exists(), f"Missing Colab notebook at {nb_path}"

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert isinstance(nb.get("nbformat_minor"), int) and nb["nbformat_minor"] >= 0
    assert "metadata" in nb and "kernelspec" in nb["metadata"]

    cells = nb.get("cells", [])
    assert isinstance(cells, list) and len(cells) >= 6

    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(md_cells) >= 3
    assert len(code_cells) >= 4

    full_code_text = []
    for idx, cell in enumerate(cells):
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else str(src)
        assert text.strip(), f"Cell {idx} ({cell.get('cell_type')}) has empty source"
        if cell["cell_type"] == "code":
            assert "outputs" in cell and isinstance(cell["outputs"], list)
            assert "execution_count" in cell
            full_code_text.append(text)

            # Strip shell magic lines (!...) and verify valid Python syntax
            py_lines = []
            in_shell_cont = False
            for line in text.splitlines():
                stripped = line.lstrip()
                if stripped.startswith("!"):
                    in_shell_cont = stripped.endswith("\\")
                    py_lines.append("")
                elif in_shell_cont:
                    in_shell_cont = stripped.endswith("\\")
                    py_lines.append("")
                else:
                    py_lines.append(line)
            compile("\n".join(py_lines), f"train_adapters_colab.ipynb:cell_{idx}", "exec")

    joined = "\n".join(full_code_text)
    for required_token in (
        "uv pip install",
        "distill_3d_adapter",
        "distill_2d_subspace",
        "Geometric3DEncoder",
        "load_projection_adapter",
        "TSNE",
        ".npz",
    ):
        assert required_token in joined, f"Expected {required_token!r} in Colab notebook code cells"
