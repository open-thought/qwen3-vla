"""
Abstract base class for VLA datasets.

All dataset loaders (RoboTwin, RoboCOIN, LeRobot) should inherit from this
class and implement the abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from .unified_sample import (
    UnifiedSample,
    RobotStateSpec,
    RobotActionSpec,
    ActiveComponents,
    compute_progress,
)


class BaseVLADataset(Dataset, ABC):
    """
    Abstract base class for all VLA datasets.

    Subclasses must implement:
        - _build_sample_index(): Build list of valid sample metadata
        - _load_raw_sample(): Load raw data for a single sample
        - DATASET_NAME: Class-level dataset identifier
        - SUPPORTED_CAMERAS: List of camera names this dataset provides

    The base class handles:
        - Common initialization (action horizon, image size, etc.)
        - Normalization loading and application
        - Idle action filtering
        - Progress computation
    """

    # Class-level dataset info (override in subclasses)
    DATASET_NAME: str = "base"
    SUPPORTED_CAMERAS: list[str] = []
    DEFAULT_STATE_SPEC: Optional[RobotStateSpec] = None
    DEFAULT_ACTION_SPEC: Optional[RobotActionSpec] = None

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
        action_horizon: int = 8,
        image_size: tuple[int, int] = (320, 240),
        action_type: str = "joint_delta",
        enable_augmentation: bool = False,
        idle_action_filter: bool = False,
        idle_threshold: float = 0.01,
        state_history_len: int = 0,
        symmetric_delta_norm: bool = True,
        **kwargs
    ):
        """
        Initialize base dataset.

        Args:
            dataset_root: Root directory for local datasets (None for HuggingFace)
            norm_stats_path: Path to normalization statistics JSON
            action_horizon: Number of future action steps to predict
            image_size: Target image size as (width, height)
            action_type: "joint_delta" or "eef_pose_delta"
            enable_augmentation: Whether to apply image augmentation
            idle_action_filter: Whether to filter out idle (zero-motion) samples
            idle_threshold: Max action magnitude to consider as idle
            state_history_len: Number of past states to include (0 = disabled)
            symmetric_delta_norm: Use symmetric normalization for delta actions
            **kwargs: Additional dataset-specific arguments
        """
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.action_type = action_type
        self.enable_augmentation = enable_augmentation
        self.idle_action_filter = idle_action_filter
        self.idle_threshold = idle_threshold
        self.state_history_len = state_history_len
        self.symmetric_delta_norm = symmetric_delta_norm

        # Store extra kwargs for subclasses
        self.extra_kwargs = kwargs

        # Load normalizer if stats path provided
        self.normalizer = None
        if norm_stats_path:
            self.normalizer = self._init_normalizer(norm_stats_path)

        # Build sample index (implemented by subclasses)
        self.samples = self._build_sample_index()

        # Apply idle action filtering if enabled
        if self.idle_action_filter:
            original_count = len(self.samples)
            self.samples = self._filter_idle_samples(self.samples)
            filtered_count = original_count - len(self.samples)
            if filtered_count > 0:
                print(f"  Filtered {filtered_count} idle samples ({filtered_count/original_count*100:.1f}%)")

    def _init_normalizer(self, stats_path: str):
        """Initialize normalizer from statistics file."""
        # Try relative import first, fall back to absolute
        try:
            from ..normalization import MultiRobotNormalizer
        except ImportError:
            from normalization import MultiRobotNormalizer
        return MultiRobotNormalizer(stats_path, symmetric_delta_norm=self.symmetric_delta_norm)

    @abstractmethod
    def _build_sample_index(self) -> list[dict]:
        """
        Build list of valid sample metadata.

        Each entry should be a dict with at least:
            - episode_id: str
            - frame_idx: int
            - robot_type: str

        Additional fields can include:
            - subtask_idx: int (if subtask info available)
            - max_action_magnitude: float (for idle filtering)

        Returns:
            List of sample metadata dicts
        """
        pass

    @abstractmethod
    def _load_raw_sample(self, sample_info: dict) -> dict[str, Any]:
        """
        Load raw data for a single sample.

        Args:
            sample_info: Sample metadata dict from _build_sample_index()

        Returns:
            Dict containing raw data with keys:
                - images: dict[str, np.ndarray] - Camera images
                - state: np.ndarray - Current robot state
                - actions: np.ndarray - Future action sequence (H, action_dim)
                - task_description: str - Episode-level instruction
                - subtask_description: Optional[str] - Subtask instruction
                - robot_type: str - Robot/embodiment name
                - episode_frame_idx: int
                - episode_total_frames: int
                - subtask_frame_idx: Optional[int]
                - subtask_total_frames: Optional[int]
                - state_history: Optional[np.ndarray] - (K, state_dim)
                - active_components: ActiveComponents
        """
        pass

    def _filter_idle_samples(self, samples: list[dict]) -> list[dict]:
        """
        Remove samples where all action deltas are below threshold.

        Subclasses can precompute 'max_action_magnitude' in sample_info
        for efficient filtering. If not present, samples are kept.
        """
        return [
            s for s in samples
            if s.get("max_action_magnitude", float('inf')) > self.idle_threshold
        ]

    def _process_images(self, images: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """
        Process raw images: resize, normalize, convert to tensor.

        Args:
            images: Dict of camera_name -> (H, W, C) uint8 numpy arrays

        Returns:
            Dict of camera_name -> (C, H, W) float32 tensors in [0, 1]
        """
        import torchvision.transforms.functional as TF
        from PIL import Image

        processed = {}
        target_w, target_h = self.image_size

        for cam_name, img in images.items():
            # Convert numpy to PIL
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)

            # Resize if needed
            if pil_img.size != (target_w, target_h):
                pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)

            # Convert to tensor (C, H, W) in [0, 1]
            tensor = TF.to_tensor(pil_img)
            processed[cam_name] = tensor

        return processed

    def _normalize_sample(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        robot_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize state and actions using loaded statistics.

        Args:
            state: Raw state vector
            actions: Raw action sequence (H, action_dim)
            robot_type: Robot type for stats lookup

        Returns:
            Tuple of (normalized_state, normalized_actions)
        """
        if self.normalizer is None:
            # No normalizer - return raw values
            return state.copy(), actions.copy()

        # Determine action category
        if self.action_type == "eef_pose_delta":
            action_category = "eef_delta_actions"
        else:
            action_category = "delta_actions"

        normalized_state = self.normalizer.normalize_state(state, robot_type)
        normalized_actions = self.normalizer.normalize_delta_actions(
            actions, robot_type, category=action_category
        )

        return normalized_state, normalized_actions

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UnifiedSample:
        """
        Load and process a single sample.

        This method:
            1. Loads raw data via _load_raw_sample()
            2. Processes images
            3. Normalizes state and actions
            4. Computes progress
            5. Returns UnifiedSample
        """
        sample_info = self.samples[idx]

        # Load raw data (implemented by subclass)
        raw = self._load_raw_sample(sample_info)

        # Process images
        images = self._process_images(raw["images"])

        # Normalize state and actions
        normalized_state, normalized_actions = self._normalize_sample(
            raw["state"],
            raw["actions"],
            raw["robot_type"],
        )

        # Compute progress
        progress = compute_progress(
            raw["episode_frame_idx"],
            raw["episode_total_frames"],
            raw.get("subtask_frame_idx"),
            raw.get("subtask_total_frames"),
        )

        # Determine if at subtask/episode end
        is_subtask_end = False
        is_episode_end = False

        if raw.get("subtask_frame_idx") is not None and raw.get("subtask_total_frames") is not None:
            is_subtask_end = raw["subtask_frame_idx"] >= raw["subtask_total_frames"] - 1

        is_episode_end = raw["episode_frame_idx"] >= raw["episode_total_frames"] - 1

        # Build UnifiedSample
        return UnifiedSample(
            # Images
            images=images,

            # Text
            task_description=raw["task_description"],
            subtask_description=raw.get("subtask_description"),
            robot_type=raw["robot_type"],

            # State
            state=raw["state"],
            state_normalized=normalized_state,
            state_spec=raw.get("state_spec", self.DEFAULT_STATE_SPEC),
            state_history=raw.get("state_history"),

            # Actions
            action_tokens=[],  # Filled by collator/tokenizer
            action_type=self.action_type,
            normalized_actions=normalized_actions,
            action_spec=raw.get("action_spec", self.DEFAULT_ACTION_SPEC),
            action_horizon=normalized_actions.shape[0],

            # Progress
            episode_frame_idx=raw["episode_frame_idx"],
            episode_total_frames=raw["episode_total_frames"],
            subtask_frame_idx=raw.get("subtask_frame_idx"),
            subtask_total_frames=raw.get("subtask_total_frames"),
            progress_percent=progress,
            is_subtask_end=is_subtask_end,
            is_episode_end=is_episode_end,

            # Metadata
            dataset_name=self.DATASET_NAME,
            episode_id=sample_info.get("episode_id", ""),
            active_components=raw.get("active_components", ActiveComponents.ALL),
        )

    def get_robot_types(self) -> list[str]:
        """Get list of unique robot types in this dataset."""
        return list(set(s.get("robot_type", "unknown") for s in self.samples))

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        robot_types = self.get_robot_types()
        return {
            "dataset_name": self.DATASET_NAME,
            "num_samples": len(self.samples),
            "robot_types": robot_types,
            "action_horizon": getattr(self, "action_horizon", None),
            "action_type": getattr(self, "action_type", None),
            "image_size": getattr(self, "image_size", None),
            "cameras": self.SUPPORTED_CAMERAS,
        }
