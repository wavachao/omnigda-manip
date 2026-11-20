"""2D dense visual perception and training-free few-shot segmentation (SegmentGDA+)."""

from omnigda.perception_2d.backbone import (
    DenseFeatureExtractor,
    DINOv2MultiScaleExtractor,
    LightweightConvFeatureExtractor,
    create_feature_extractor,
    preprocess_rgb_for_vit,
)
from omnigda.perception_2d.segment_gda import (
    BACKGROUND_CLASS_NAME,
    InstanceProposal,
    SegmentGDAPlus,
    extract_instance_masks,
    fast_color_guided_filter,
)

__all__ = [
    "BACKGROUND_CLASS_NAME",
    "DenseFeatureExtractor",
    "DINOv2MultiScaleExtractor",
    "InstanceProposal",
    "LightweightConvFeatureExtractor",
    "SegmentGDAPlus",
    "create_feature_extractor",
    "extract_instance_masks",
    "fast_color_guided_filter",
    "preprocess_rgb_for_vit",
]
