"""2D dense visual perception and training-free few-shot segmentation (SegmentGDA+)."""

from omnigda.perception_2d.backbone import (
    DenseFeatureExtractor,
    DINOv2MultiScaleExtractor,
    LightweightConvFeatureExtractor,
    create_feature_extractor,
    preprocess_rgb_for_vit,
)

__all__ = [
    "DenseFeatureExtractor",
    "DINOv2MultiScaleExtractor",
    "LightweightConvFeatureExtractor",
    "create_feature_extractor",
    "preprocess_rgb_for_vit",
]
