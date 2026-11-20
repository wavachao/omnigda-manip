"""3D multimodal perception, partial-view GMDA classification, and 3D scene graph construction."""

from omnigda.perception_3d.encoders import (
    DEFAULT_CLIP_PROMPT_TEMPLATES,
    Geometric3DEncoder,
    OpenCLIPMultimodalEncoder,
    PhysicalSizeEncoder,
)
from omnigda.perception_3d.multimodal_gmda import (
    OnlineShrinkageGMDA,
    PartialAwareMultimodalGMDA,
)
from omnigda.perception_3d.pointcloud import (
    compute_oriented_bounding_box,
    confidence_weighted_backproject,
    estimate_pointcloud_completeness,
    fuse_multiview_pointclouds,
    subsample_pointcloud,
)
from omnigda.perception_3d.scene_graph import SceneGraphBuilder3D

__all__ = [
    "DEFAULT_CLIP_PROMPT_TEMPLATES",
    "Geometric3DEncoder",
    "OnlineShrinkageGMDA",
    "OpenCLIPMultimodalEncoder",
    "PartialAwareMultimodalGMDA",
    "PhysicalSizeEncoder",
    "SceneGraphBuilder3D",
    "compute_oriented_bounding_box",
    "confidence_weighted_backproject",
    "estimate_pointcloud_completeness",
    "fuse_multiview_pointclouds",
    "subsample_pointcloud",
]

