"""
Multi-dataset VLA training support.

This module provides unified dataset loading for various robot datasets
including RoboTwin, RoboCOIN, and HuggingFace LeRobot datasets.
"""

from .unified_sample import (
    ActiveComponents,
    RobotStateSpec,
    RobotActionSpec,
    UnifiedSample,
    compute_progress,
)
from .base_dataset import BaseVLADataset
from .robotwin_dataset import RoboTwinDataset
from .robocoin_dataset import RoboCOINDataset
from .lerobot_dataset import LeRobotVLADataset
from .multi_dataset import (
    MultiDatasetWrapper,
    WeightedMultiDatasetSampler,
    create_multi_dataset,
)

__all__ = [
    # Core types
    "ActiveComponents",
    "RobotStateSpec",
    "RobotActionSpec",
    "UnifiedSample",
    "compute_progress",
    # Base class
    "BaseVLADataset",
    # Dataset loaders
    "RoboTwinDataset",
    "RoboCOINDataset",
    "LeRobotVLADataset",
    # Multi-dataset support
    "MultiDatasetWrapper",
    "WeightedMultiDatasetSampler",
    "create_multi_dataset",
]
