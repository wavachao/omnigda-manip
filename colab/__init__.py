"""Offline GPU subspace distillation & adapter training utilities for OmniGDA-Manip.

Modules:
- ``colab.distill_3d_adapter``: Fisher-SupCon 3D discriminative subspace adapter trainer
  (108-D -> 64-D) exporting zero-pickle ``.npz`` weights for ``Geometric3DEncoder``.
- ``colab.distill_2d_subspace``: Cross-model 2D DINOv2 relational subspace distillation
  trainer (384-D -> 128-D) for 9x smaller ``SegmentGDAPlus`` covariance matrices.
"""
