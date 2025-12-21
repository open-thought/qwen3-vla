"""
Generic LeRobot dataset loader for HuggingFace VLA datasets.

Works with any LeRobot-compatible HuggingFace dataset including:
- HuggingFaceVLA/libero
- HuggingFaceVLA/bridge_v2
- HuggingFaceVLA/droid
- lerobot/pusht
- And many others...
"""

from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    print("Warning: lerobot not available. LeRobotVLADataset will not work.")

from .base_dataset import BaseVLADataset
from .unified_sample import (
    UnifiedSample,
    RobotStateSpec,
    RobotActionSpec,
    ActiveComponents,
    compute_progress,
)


class LeRobotVLADataset(BaseVLADataset):
    """
    Generic LeRobot dataset loader for HuggingFace VLA datasets.

    Automatically discovers:
    - Available camera views from dataset features
    - State/action dimensions from metadata
    - Task descriptions from episode info
    """

    DATASET_NAME = "lerobot"

    def __init__(
        self,
        repo_id: str,
        norm_stats_path: Optional[str] = None,
        action_horizon: int = 8,
        image_size: tuple[int, int] = (320, 240),
        action_type: str = "joint_delta",
        episodes: Optional[list[int]] = None,
        split: Optional[str] = None,
        robot_type: Optional[str] = None,
        use_percentile_normalization: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        **kwargs
    ):
        """
        Args:
            repo_id: HuggingFace repo ID (e.g., "HuggingFaceVLA/libero")
            norm_stats_path: Path to normalization statistics JSON (optional)
            action_horizon: Number of future timesteps to predict
            image_size: Target image size (width, height)
            action_type: "joint_delta" or "eef_pose_delta"
            episodes: Optional list of episode indices to include
            split: Optional dataset split (e.g., "train[:80%]")
            robot_type: Override robot type name (auto-detected if None)
            use_percentile_normalization: Use dataset-computed percentile normalization
            percentile_low: Low percentile for normalization
            percentile_high: High percentile for normalization
        """
        if not HAS_LEROBOT:
            raise RuntimeError("lerobot package is required for LeRobotVLADataset")

        self.repo_id = repo_id
        self.episodes_filter = episodes
        self.split = split
        self.robot_type_override = robot_type
        self.use_percentile_normalization = use_percentile_normalization
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

        # Load LeRobot dataset
        print(f"Loading LeRobot dataset: {repo_id}")
        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            episodes=episodes,
        )

        # Discover cameras
        self.available_cameras = self._discover_cameras()
        print(f"  Available cameras: {self.available_cameras}")

        # Get state/action dimensions
        self.state_dim = self._get_state_dim()
        self.action_dim = self._get_action_dim()
        print(f"  State dim: {self.state_dim}, Action dim: {self.action_dim}")

        # Compute normalization stats from dataset
        self._action_stats = None
        self._state_stats = None
        if use_percentile_normalization:
            self._compute_normalization_stats()

        # Call parent init
        super().__init__(
            dataset_root=None,  # No local files
            norm_stats_path=norm_stats_path,
            action_horizon=action_horizon,
            image_size=image_size,
            action_type=action_type,
            **kwargs
        )

        print(f"LeRobot dataset ready:")
        print(f"  Repo: {repo_id}")
        print(f"  Samples: {len(self.samples)}")

    @property
    def SUPPORTED_CAMERAS(self) -> list[str]:
        """Dynamically return discovered cameras."""
        return self.available_cameras

    def _discover_cameras(self) -> list[str]:
        """Auto-discover available camera views from dataset features or metadata."""
        cameras = []

        # First check HF dataset features (for datasets with images as tensors)
        for key in self.lerobot_dataset.hf_dataset.features.keys():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                cameras.append(cam_name)
            elif key.startswith("observation.image"):
                # Some datasets use observation.image, observation.image2, etc.
                cameras.append(key.replace("observation.", ""))

        # If no cameras found in features, check metadata camera_keys (for video-based datasets)
        if not cameras and hasattr(self.lerobot_dataset, "meta"):
            if hasattr(self.lerobot_dataset.meta, "camera_keys"):
                for key in self.lerobot_dataset.meta.camera_keys:
                    if key.startswith("observation.images."):
                        cam_name = key.replace("observation.images.", "")
                        cameras.append(cam_name)

        return cameras

    def _get_state_dim(self) -> int:
        """Get state dimension from dataset."""
        features = self.lerobot_dataset.hf_dataset.features
        if "observation.state" in features:
            feat = features["observation.state"]
            # Handle both Sequence (has .length) and Array (has .shape) types
            if hasattr(feat, "length"):
                return feat.length
            elif hasattr(feat, "shape"):
                return feat.shape[0] if feat.shape else 0
        return 0

    def _get_action_dim(self) -> int:
        """Get action dimension from dataset."""
        features = self.lerobot_dataset.hf_dataset.features
        if "action" in features:
            feat = features["action"]
            # Handle both Sequence (has .length) and Array (has .shape) types
            if hasattr(feat, "length"):
                return feat.length
            elif hasattr(feat, "shape"):
                return feat.shape[0] if feat.shape else 0
        return 0

    def _load_normalization_stats(self):
        """Load pre-computed normalization stats from dataset metadata."""
        print("  Loading normalization statistics from dataset metadata...")

        # LeRobot datasets include pre-computed stats in meta.stats
        if not hasattr(self.lerobot_dataset, "meta") or not hasattr(self.lerobot_dataset.meta, "stats"):
            print("  Warning: No pre-computed stats found, skipping normalization")
            return

        stats = self.lerobot_dataset.meta.stats

        # Load action stats
        if "action" in stats:
            action_stats = stats["action"]
            # Use q01/q99 percentiles if available, otherwise fall back to min/max
            q_low = action_stats.get("q01", action_stats.get("min"))
            q_high = action_stats.get("q99", action_stats.get("max"))
            if q_low is not None and q_high is not None:
                self._action_stats = {
                    "q_low": np.array(q_low).flatten(),
                    "q_high": np.array(q_high).flatten(),
                }

        # Load state stats
        if "observation.state" in stats:
            state_stats = stats["observation.state"]
            q_low = state_stats.get("q01", state_stats.get("min"))
            q_high = state_stats.get("q99", state_stats.get("max"))
            if q_low is not None and q_high is not None:
                self._state_stats = {
                    "q_low": np.array(q_low).flatten(),
                    "q_high": np.array(q_high).flatten(),
                }

    def _compute_normalization_stats(self):
        """Compute percentile-based normalization stats from dataset (fallback)."""
        # First try to load pre-computed stats
        self._load_normalization_stats()
        if self._action_stats is not None or self._state_stats is not None:
            return

        # Fall back to computing stats if not available in metadata
        print("  Computing normalization statistics (no pre-computed stats found)...")

        # Sample actions and states
        hf_dataset = self.lerobot_dataset.hf_dataset
        n_samples = min(10000, len(hf_dataset))
        indices = np.random.choice(len(hf_dataset), n_samples, replace=False)

        actions = []
        states = []

        for idx in indices:
            sample = hf_dataset[int(idx)]
            if "action" in sample:
                actions.append(np.array(sample["action"]))
            if "observation.state" in sample:
                states.append(np.array(sample["observation.state"]))

        if actions:
            actions = np.stack(actions)
            self._action_stats = {
                "q_low": np.percentile(actions, self.percentile_low, axis=0),
                "q_high": np.percentile(actions, self.percentile_high, axis=0),
            }

        if states:
            states = np.stack(states)
            self._state_stats = {
                "q_low": np.percentile(states, self.percentile_low, axis=0),
                "q_high": np.percentile(states, self.percentile_high, axis=0),
            }

    def _normalize_with_stats(
        self,
        values: np.ndarray,
        stats: dict,
    ) -> np.ndarray:
        """Normalize values using percentile stats."""
        if stats is None:
            return values

        q_low = stats["q_low"]
        q_high = stats["q_high"]
        range_val = q_high - q_low
        range_val = np.where(range_val < 1e-8, 1.0, range_val)  # Avoid division by zero

        normalized = 2.0 * (values - q_low) / range_val - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def _build_sample_index(self) -> list[dict]:
        """Build sample index from LeRobot dataset."""
        samples = []
        hf_dataset = self.lerobot_dataset.hf_dataset

        # Get episode boundaries - convert tensors to ints
        episode_indices_raw = hf_dataset["episode_index"]
        episode_indices = [int(x) if hasattr(x, 'item') else int(x) for x in episode_indices_raw]
        unique_episodes = sorted(set(episode_indices))

        # Build episode info
        episode_info = {}
        for i, ep_idx in enumerate(episode_indices):
            if ep_idx not in episode_info:
                episode_info[ep_idx] = {"start": i, "end": i + 1}
            else:
                episode_info[ep_idx]["end"] = i + 1

        # Get robot type
        robot_type = self.robot_type_override
        if robot_type is None:
            # Try to get from dataset metadata
            if hasattr(self.lerobot_dataset, "meta") and hasattr(self.lerobot_dataset.meta, "robot_type"):
                robot_type = self.lerobot_dataset.meta.robot_type
            else:
                robot_type = self.repo_id.replace("/", "_")

        # Create samples
        for ep_idx, info in episode_info.items():
            start = info["start"]
            end = info["end"]
            ep_len = end - start

            # Create sample for each valid timestep
            for t in range(start, end - 1):
                samples.append({
                    "global_idx": t,
                    "episode_idx": ep_idx,
                    "episode_start": start,
                    "episode_end": end,
                    "episode_frame_idx": t - start,
                    "episode_total_frames": ep_len,
                    "robot_type": robot_type,
                })

        return samples

    def _load_raw_sample(self, sample_info: dict) -> dict[str, Any]:
        """Load raw data for a single sample."""
        global_idx = sample_info["global_idx"]
        episode_start = sample_info["episode_start"]
        episode_end = sample_info["episode_end"]
        timestep = sample_info["episode_frame_idx"]
        total_frames = sample_info["episode_total_frames"]

        # Load from HuggingFace dataset
        hf_sample = self.lerobot_dataset.hf_dataset[global_idx]

        # Get state
        state = np.array(hf_sample.get("observation.state", np.zeros(self.state_dim)), dtype=np.float32)

        # Get future actions
        available_future = min(self.action_horizon, episode_end - global_idx - 1)
        future_end = global_idx + 1 + available_future

        future_actions = []
        for i in range(global_idx + 1, future_end):
            action = np.array(self.lerobot_dataset.hf_dataset[i]["action"], dtype=np.float32)
            future_actions.append(action)

        if future_actions:
            actions = np.stack(future_actions)
        else:
            actions = np.zeros((0, self.action_dim), dtype=np.float32)

        # Pad if needed
        if len(actions) < self.action_horizon:
            pad_count = self.action_horizon - len(actions)
            if len(actions) > 0:
                last_action = actions[-1:]
            else:
                last_action = np.zeros((1, self.action_dim), dtype=np.float32)
            actions = np.concatenate([actions, np.tile(last_action, (pad_count, 1))])

        # Load images - use LeRobotDataset directly for video-based datasets
        images = {}
        if self.available_cameras:
            # For video-based datasets, use lerobot_dataset[idx] which decodes videos
            lerobot_sample = self.lerobot_dataset[global_idx]
            for cam_name in self.available_cameras:
                key = f"observation.images.{cam_name}"
                if key in lerobot_sample:
                    img = lerobot_sample[key]
                    if isinstance(img, torch.Tensor):
                        img = img.numpy()
                    if img.ndim == 3 and img.shape[0] == 3:
                        # CHW -> HWC
                        img = np.transpose(img, (1, 2, 0))
                    # Normalize to 0-255 if needed
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    # Map camera names to standard format
                    std_name = self._map_camera_name(cam_name)
                    images[std_name] = img

        # Get task description
        task_desc = "Complete the manipulation task"
        if "task" in hf_sample:
            task_desc = hf_sample["task"]
        elif hasattr(self.lerobot_dataset, "meta") and hasattr(self.lerobot_dataset.meta, "tasks"):
            task_idx = hf_sample.get("task_index", torch.tensor(0))
            if isinstance(task_idx, torch.Tensor):
                task_idx = task_idx.item()
            # meta.tasks is a DataFrame with task descriptions as index
            tasks_df = self.lerobot_dataset.meta.tasks
            if task_idx < len(tasks_df):
                task_desc = tasks_df.index[task_idx]

        return {
            "images": images,
            "state": state,
            "actions": actions,
            "task_description": task_desc,
            "subtask_description": None,
            "robot_type": sample_info["robot_type"],
            "episode_frame_idx": timestep,
            "episode_total_frames": total_frames,
            "subtask_frame_idx": None,
            "subtask_total_frames": None,
            "state_history": None,
            "active_components": ActiveComponents.ALL,
        }

    def _map_camera_name(self, cam_name: str) -> str:
        """Map dataset camera names to standard names."""
        mapping = {
            "image": "head",
            "image2": "left_wrist",
            "cam_high_rgb": "head",
            "cam_left_wrist_rgb": "left_wrist",
            "cam_right_wrist_rgb": "right_wrist",
            "top": "head",
            "wrist": "left_wrist",
            "context": "head",  # Third-person/context view
        }
        return mapping.get(cam_name, cam_name)

    def __getitem__(self, idx: int) -> UnifiedSample:
        """Get a single sample with normalization."""
        sample_info = self.samples[idx]
        raw = self._load_raw_sample(sample_info)

        # Process images
        images = self._process_images(raw["images"])

        # Normalize state
        state = raw["state"]
        if self._state_stats is not None:
            state_normalized = self._normalize_with_stats(state, self._state_stats)
        elif self.normalizer is not None:
            state_normalized = self.normalizer.normalize_state(state, raw["robot_type"])
        else:
            state_normalized = state

        # Normalize actions
        actions = raw["actions"]
        if self._action_stats is not None:
            normalized_actions = self._normalize_with_stats(actions, self._action_stats)
        elif self.normalizer is not None:
            normalized_actions = self.normalizer.normalize_delta_actions(actions, raw["robot_type"])
        else:
            normalized_actions = actions

        # Compute progress
        progress = compute_progress(
            raw["episode_frame_idx"],
            raw["episode_total_frames"],
        )

        # Determine end flags
        is_episode_end = raw["episode_frame_idx"] >= raw["episode_total_frames"] - 1

        return UnifiedSample(
            images=images,
            task_description=raw["task_description"],
            subtask_description=raw.get("subtask_description"),
            robot_type=raw["robot_type"],
            state=state,
            state_normalized=state_normalized.astype(np.float32),
            state_spec=RobotStateSpec(joint_positions=self.state_dim),
            state_history=None,
            action_tokens=[],
            action_type=self.action_type,
            normalized_actions=normalized_actions.astype(np.float32),
            action_spec=RobotActionSpec(joint_targets=self.action_dim),
            action_horizon=normalized_actions.shape[0],
            episode_frame_idx=raw["episode_frame_idx"],
            episode_total_frames=raw["episode_total_frames"],
            subtask_frame_idx=None,
            subtask_total_frames=None,
            progress_percent=progress,
            is_subtask_end=False,
            is_episode_end=is_episode_end,
            dataset_name=self.DATASET_NAME,
            episode_id=str(sample_info["episode_idx"]),
            active_components=ActiveComponents.ALL,
        )
