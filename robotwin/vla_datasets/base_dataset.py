"""
Abstract base class for VLA datasets.

All dataset loaders (RoboTwin, RoboCOIN, LeRobot) should inherit from this
class and implement the abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any
import json
import bisect

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


class ValidTimestepsIndex:
    """
    Index of valid (non-idle) timesteps per episode for efficient filtering.

    Provides:
    - Fast lookup of whether a timestep is valid
    - Progress calculation based on remaining valid frames
    """

    def __init__(self, valid_timesteps_path: str):
        """
        Load valid timesteps from JSON file.

        Args:
            valid_timesteps_path: Path to JSON file mapping episode_key -> list of valid timesteps
        """
        with open(valid_timesteps_path, "r") as f:
            self._valid_timesteps = json.load(f)

        # Convert lists to sets for O(1) lookup
        self._valid_sets: dict[str, set[int]] = {}
        for key, timesteps in self._valid_timesteps.items():
            self._valid_sets[key] = set(timesteps)

    def is_valid(self, episode_key: str, timestep: int) -> bool:
        """Check if a timestep is valid (non-idle)."""
        if episode_key not in self._valid_sets:
            # If episode not in index, assume all timesteps are valid
            return True
        return timestep in self._valid_sets[episode_key]

    def get_valid_timesteps(self, episode_key: str) -> list[int]:
        """Get sorted list of valid timesteps for an episode."""
        return self._valid_timesteps.get(episode_key, [])

    def compute_progress(
        self,
        episode_key: str,
        current_timestep: int,
        subtask_start: Optional[int] = None,
        subtask_end: Optional[int] = None,
    ) -> int:
        """
        Compute progress percentage based on valid frames remaining.

        Progress = 100 if no more valid frames remain in the range.

        Args:
            episode_key: Episode identifier
            current_timestep: Current timestep index
            subtask_start: Optional start of subtask range (inclusive)
            subtask_end: Optional end of subtask range (exclusive)

        Returns:
            Progress percentage [0, 100]
        """
        valid_list = self._valid_timesteps.get(episode_key, [])

        if not valid_list:
            # No valid timesteps info - fall back to frame-based progress
            return -1  # Sentinel value to use frame-based progress

        # Determine range to consider
        if subtask_start is not None and subtask_end is not None:
            # Filter to subtask range
            valid_in_range = [t for t in valid_list if subtask_start <= t < subtask_end]
        else:
            valid_in_range = valid_list

        if not valid_in_range:
            return 100  # No valid frames at all

        # Count valid frames from current position to end
        valid_remaining = sum(1 for t in valid_in_range if t > current_timestep)
        total_valid = len(valid_in_range)

        if total_valid == 0:
            return 100

        # Progress is (completed / total) * 100
        completed = total_valid - valid_remaining
        progress = int((completed / total_valid) * 100)

        # Ensure 100% only when truly at the last valid frame
        if valid_remaining == 0:
            progress = 100

        return min(progress, 100)

    def has_episode(self, episode_key: str) -> bool:
        """Check if episode exists in the index."""
        return episode_key in self._valid_timesteps

    def __len__(self) -> int:
        """Total number of episodes in index."""
        return len(self._valid_timesteps)


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
        valid_timesteps_path: Optional[str] = None,
        action_horizon: int = 8,
        image_size: tuple[int, int] = (320, 240),
        action_type: str = "joint_delta",
        enable_augmentation: bool = False,
        idle_action_filter: bool = False,
        idle_threshold: float = 0.01,
        state_history_len: int = 0,
        symmetric_delta_norm: bool = True,
        use_valid_timesteps_progress: bool = True,
        **kwargs
    ):
        """
        Initialize base dataset.

        Args:
            dataset_root: Root directory for local datasets (None for HuggingFace)
            norm_stats_path: Path to normalization statistics JSON
            valid_timesteps_path: Path to valid timesteps JSON (for filtering and progress)
            action_horizon: Number of future action steps to predict
            image_size: Target image size as (width, height)
            action_type: "joint_delta" or "eef_pose_delta"
            enable_augmentation: Whether to apply image augmentation
            idle_action_filter: Whether to filter out idle (zero-motion) samples
            idle_threshold: Max action magnitude to consider as idle
            state_history_len: Number of past states to include (0 = disabled)
            symmetric_delta_norm: Use symmetric normalization for delta actions
            use_valid_timesteps_progress: Use valid timesteps for progress calculation
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
        self.use_valid_timesteps_progress = use_valid_timesteps_progress

        # Store extra kwargs for subclasses
        self.extra_kwargs = kwargs

        # Load normalizer if stats path provided
        self.normalizer = None
        if norm_stats_path:
            self.normalizer = self._init_normalizer(norm_stats_path)

        # Load valid timesteps index if provided
        self.valid_timesteps_index: Optional[ValidTimestepsIndex] = None
        if valid_timesteps_path:
            self.valid_timesteps_index = ValidTimestepsIndex(valid_timesteps_path)
            print(f"  Loaded valid timesteps for {len(self.valid_timesteps_index)} episodes")

        # Build sample index (implemented by subclasses)
        self.samples = self._build_sample_index()

        # Apply filtering based on valid timesteps if index is loaded
        if self.valid_timesteps_index is not None:
            original_count = len(self.samples)
            self.samples = self._filter_with_valid_timesteps(self.samples)
            filtered_count = original_count - len(self.samples)
            if filtered_count > 0:
                print(f"  Filtered {filtered_count} invalid timesteps ({filtered_count/original_count*100:.1f}%)")
        # Fallback to simple idle filtering if no valid timesteps index
        elif self.idle_action_filter:
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

    def _filter_with_valid_timesteps(self, samples: list[dict]) -> list[dict]:
        """
        Filter samples to only include valid (non-idle) timesteps.

        Uses the pre-computed valid_timesteps_index for efficient filtering.
        Samples must have 'episode_key' and 'timestep' fields.
        """
        if self.valid_timesteps_index is None:
            return samples

        filtered = []
        for s in samples:
            # Get episode key - try common field names
            episode_key = s.get("episode_key")
            if episode_key is None:
                # Try to construct from other fields
                episode_id = s.get("episode_id", s.get("episode_idx", ""))
                task_name = s.get("task_name", "")
                if task_name and episode_id:
                    episode_key = f"{task_name}/episode{episode_id}"
                else:
                    episode_key = str(episode_id)

            timestep = s.get("timestep", s.get("frame_idx", 0))

            if self.valid_timesteps_index.is_valid(episode_key, timestep):
                # Store episode_key in sample for progress calculation
                s["_episode_key"] = episode_key
                filtered.append(s)

        return filtered

    def _get_episode_key(self, sample_info: dict) -> str:
        """Extract episode key from sample info for valid timesteps lookup."""
        # First check if we stored it during filtering
        if "_episode_key" in sample_info:
            return sample_info["_episode_key"]

        # Otherwise construct it
        episode_id = sample_info.get("episode_id", sample_info.get("episode_idx", ""))
        task_name = sample_info.get("task_name", "")
        if task_name and episode_id:
            return f"{task_name}/episode{episode_id}"
        return str(episode_id)

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
            4. Computes progress (using valid timesteps if available)
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

        # Compute progress - prefer valid timesteps-based if available
        progress = -1
        if self.valid_timesteps_index is not None and self.use_valid_timesteps_progress:
            episode_key = self._get_episode_key(sample_info)
            timestep = sample_info.get("timestep", raw["episode_frame_idx"])
            subtask_start = sample_info.get("subtask_start")
            subtask_end = sample_info.get("subtask_end")

            progress = self.valid_timesteps_index.compute_progress(
                episode_key, timestep, subtask_start, subtask_end
            )

        # Fall back to frame-based progress if valid timesteps progress not available
        if progress < 0:
            progress = compute_progress(
                raw["episode_frame_idx"],
                raw["episode_total_frames"],
                raw.get("subtask_frame_idx"),
                raw.get("subtask_total_frames"),
            )

        # Determine if at subtask/episode end based on valid frames
        is_subtask_end = False
        is_episode_end = False

        if self.valid_timesteps_index is not None and self.use_valid_timesteps_progress:
            # Use valid timesteps to determine end
            episode_key = self._get_episode_key(sample_info)
            timestep = sample_info.get("timestep", raw["episode_frame_idx"])
            subtask_start = sample_info.get("subtask_start")
            subtask_end = sample_info.get("subtask_end")

            valid_list = self.valid_timesteps_index.get_valid_timesteps(episode_key)

            if valid_list:
                # Check if this is the last valid timestep in subtask range
                if subtask_start is not None and subtask_end is not None:
                    valid_in_subtask = [t for t in valid_list if subtask_start <= t < subtask_end]
                    if valid_in_subtask:
                        is_subtask_end = timestep >= max(valid_in_subtask)

                # Check if this is the last valid timestep in episode
                is_episode_end = timestep >= max(valid_list)
        else:
            # Fall back to frame-based end detection
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

    def detect_active_components(
        self,
        actions: np.ndarray,
        action_spec: Optional[RobotActionSpec] = None,
        threshold: float = 0.01,
        num_arms: int = 2,
    ) -> ActiveComponents:
        """
        Detect which robot components are active based on action magnitudes.

        For bimanual robots, assumes joint/gripper dimensions are split evenly
        between left (first half) and right (second half) arms.

        Args:
            actions: Action array of shape (H, action_dim) for the horizon
            action_spec: Action specification describing the layout. If None,
                uses DEFAULT_ACTION_SPEC.
            threshold: Minimum absolute magnitude to consider active
            num_arms: Number of arms (1=single-arm, 2=bimanual)

        Returns:
            ActiveComponents flags indicating which parts are moving
        """
        spec = action_spec or self.DEFAULT_ACTION_SPEC
        if spec is None:
            return ActiveComponents.ALL  # Can't determine without spec

        # Get slice indices from spec
        indices = spec.get_slice_indices()

        # Track activity per arm
        left_arm_active = False
        right_arm_active = False
        left_gripper_active = False
        right_gripper_active = False

        # Check joint targets (delta positions)
        if "joint_targets" in indices:
            start, end = indices["joint_targets"]
            joints = actions[:, start:end]
            max_abs = np.abs(joints).max(axis=0)  # Max across horizon per DOF

            if num_arms == 2:
                # Bimanual: split in half
                mid = joints.shape[1] // 2
                left_arm_active = max_abs[:mid].max() > threshold
                right_arm_active = max_abs[mid:].max() > threshold
            else:
                # Single arm - treat as left
                left_arm_active = max_abs.max() > threshold

        # Check EEF position deltas
        if "eef_position_deltas" in indices:
            start, end = indices["eef_position_deltas"]
            eef_pos = actions[:, start:end]
            max_abs = np.abs(eef_pos).max(axis=0)

            if num_arms == 2:
                mid = eef_pos.shape[1] // 2
                left_arm_active = left_arm_active or max_abs[:mid].max() > threshold
                right_arm_active = right_arm_active or max_abs[mid:].max() > threshold
            else:
                left_arm_active = left_arm_active or max_abs.max() > threshold

        # Check EEF rotation deltas
        if "eef_rotation_deltas" in indices:
            start, end = indices["eef_rotation_deltas"]
            eef_rot = actions[:, start:end]
            max_abs = np.abs(eef_rot).max(axis=0)

            if num_arms == 2:
                mid = eef_rot.shape[1] // 2
                left_arm_active = left_arm_active or max_abs[:mid].max() > threshold
                right_arm_active = right_arm_active or max_abs[mid:].max() > threshold
            else:
                left_arm_active = left_arm_active or max_abs.max() > threshold

        # Check gripper commands - detect change from initial value
        if "gripper_commands" in indices:
            start, end = indices["gripper_commands"]
            grippers = actions[:, start:end]

            # Check for change across horizon (not just magnitude)
            gripper_change = np.abs(grippers - grippers[0:1, :]).max(axis=0)

            if num_arms == 2 and grippers.shape[1] >= 2:
                left_gripper_active = gripper_change[0] > threshold
                right_gripper_active = gripper_change[1] > threshold
            elif grippers.shape[1] >= 1:
                left_gripper_active = gripper_change[0] > threshold

        # Combine into ActiveComponents
        result = ActiveComponents.IDLE
        if left_arm_active:
            result |= ActiveComponents.LEFT_ARM
        if left_gripper_active:
            result |= ActiveComponents.LEFT_GRIPPER
        if right_arm_active:
            result |= ActiveComponents.RIGHT_ARM
        if right_gripper_active:
            result |= ActiveComponents.RIGHT_GRIPPER

        # Default to ALL if nothing detected (edge case)
        if result == ActiveComponents.IDLE:
            result = ActiveComponents.ALL

        return result

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
