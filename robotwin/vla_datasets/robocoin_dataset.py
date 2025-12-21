"""
RoboCOIN dataset loader extending BaseVLADataset.

Handles the RoboCOIN LeRobot-compatible format with:
- Parquet files for state/action data
- AV1 video decoding for camera views
- Subtask annotations (5 annotator levels)
- Multiple robot types with varying DoF
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import random

import numpy as np
import pandas as pd
import torch

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("Warning: PyAV not available. AV1 video decoding will not work.")

from .base_dataset import BaseVLADataset
from .unified_sample import (
    UnifiedSample,
    RobotStateSpec,
    RobotActionSpec,
    ActiveComponents,
    compute_progress,
)


@dataclass
class RoboCOINEpisodeInfo:
    """Information about a RoboCOIN episode."""
    task_dir: Path
    episode_idx: int
    parquet_path: Path
    video_paths: dict[str, Path]  # camera_name -> video_path
    num_frames: int
    robot_type: str
    task_description: str  # Full task description (from tasks.jsonl)
    state_dim: int
    action_dim: int


class RoboCOINDataset(BaseVLADataset):
    """
    RoboCOIN dataset loader using LeRobot format.

    Handles:
    - Parquet files for state/action data
    - AV1 video decoding for camera views
    - Subtask annotations (5 annotator levels, uses level 0)
    - Multiple robot types with varying DoF
    """

    DATASET_NAME = "robocoin"
    SUPPORTED_CAMERAS = ["cam_high_rgb", "cam_left_wrist_rgb", "cam_right_wrist_rgb"]

    def __init__(
        self,
        dataset_root: str,
        norm_stats_path: Optional[str] = None,
        action_horizon: int = 8,
        image_size: tuple[int, int] = (320, 240),
        action_type: str = "joint_delta",
        task_filter: Optional[list[str]] = None,
        robot_filter: Optional[list[str]] = None,
        subtask_annotator_level: int = 0,
        max_episodes_per_task: Optional[int] = None,
        cache_video_frames: bool = False,
        **kwargs
    ):
        """
        Args:
            dataset_root: Root directory of RoboCOIN dataset
            norm_stats_path: Path to normalization statistics JSON
            action_horizon: Number of future timesteps to predict
            image_size: Target image size (width, height)
            action_type: "joint_delta" (state-based) for RoboCOIN
            task_filter: Optional list of task names to include
            robot_filter: Optional list of robot types to include
            subtask_annotator_level: Which annotator level to use (0-4)
            max_episodes_per_task: Limit episodes per task (for debugging)
            cache_video_frames: Whether to cache decoded video frames
        """
        self.task_filter = task_filter
        self.robot_filter = robot_filter
        self.subtask_annotator_level = subtask_annotator_level
        self.max_episodes_per_task = max_episodes_per_task
        self.cache_video_frames = cache_video_frames

        # Episode metadata storage
        self.episodes: list[RoboCOINEpisodeInfo] = []
        self.task_info: dict[str, dict] = {}  # task_dir -> info.json
        self.subtask_annotations: dict[str, dict] = {}  # task_dir -> subtask_index -> text
        self.scene_annotations: dict[str, dict] = {}  # task_dir -> scene_index -> text

        # Parquet dataframe cache
        self._parquet_cache: dict[str, pd.DataFrame] = {}
        self._parquet_cache_size = 10

        # Video frame cache
        self._video_cache: dict[str, np.ndarray] = {}

        # Call parent init
        super().__init__(
            dataset_root=dataset_root,
            norm_stats_path=norm_stats_path,
            action_horizon=action_horizon,
            image_size=image_size,
            action_type=action_type,
            **kwargs
        )

        print(f"RoboCOIN dataset ready:")
        print(f"  Tasks: {len(self.task_info)}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  Samples: {len(self.samples)}")

    def _build_sample_index(self) -> list[dict]:
        """Build sample index from RoboCOIN dataset."""
        # Scan for task directories
        self.episodes = self._scan_episodes()

        if not self.episodes:
            print(f"Warning: No episodes found in {self.dataset_root}")
            return []

        # Build samples
        samples = []
        for ep in self.episodes:
            # Load parquet to get subtask annotations
            df = self._load_parquet(ep.parquet_path)
            subtask_ids = self._get_subtask_ids(df)

            # Find subtask boundaries
            subtask_boundaries = self._find_subtask_boundaries(subtask_ids)

            for start_frame, end_frame, subtask_id in subtask_boundaries:
                # Get subtask description to check if it's a special/invalid subtask
                task_key = str(ep.task_dir)
                subtask_text = self.subtask_annotations.get(task_key, {}).get(subtask_id, "")

                # Skip special subtasks (End, Static, Abnormal, null, or empty)
                if subtask_text.lower() in ["end", "static", "abnormal", "null", ""]:
                    continue

                subtask_len = end_frame - start_frame

                # Create samples for each valid timestep
                for t in range(start_frame, end_frame - 1):
                    samples.append({
                        "episode_info": ep,
                        "timestep": t,
                        "subtask_start": start_frame,
                        "subtask_end": end_frame,
                        "subtask_id": subtask_id,
                        "episode_total_frames": ep.num_frames,
                        "robot_type": ep.robot_type,
                    })

        return samples

    def _scan_episodes(self) -> list[RoboCOINEpisodeInfo]:
        """Scan dataset directory for episodes."""
        episodes = []

        # Find all task directories
        task_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]

        for task_dir in sorted(task_dirs):
            # Check for meta/info.json
            info_path = task_dir / "meta" / "info.json"
            if not info_path.exists():
                continue

            # Load task info
            with open(info_path) as f:
                info = json.load(f)

            robot_type = info.get("robot_type", "unknown")
            task_name = task_dir.name

            # Apply filters
            if self.robot_filter and robot_type not in self.robot_filter:
                continue
            if self.task_filter and task_name not in self.task_filter:
                continue

            # Store task info
            self.task_info[str(task_dir)] = info

            # Load annotations
            self._load_task_annotations(task_dir)

            # Load task descriptions
            tasks_path = task_dir / "meta" / "tasks.jsonl"
            task_descriptions = {}
            if tasks_path.exists():
                with open(tasks_path) as f:
                    for line in f:
                        d = json.loads(line)
                        task_descriptions[d["task_index"]] = d["task"]

            # Get state/action dimensions
            state_dim = info["features"]["observation.state"]["shape"][0]
            action_dim = info["features"]["action"]["shape"][0]

            # Scan parquet files
            data_dir = task_dir / "data"
            if not data_dir.exists():
                continue

            episode_count = 0
            for chunk_dir in sorted(data_dir.glob("chunk-*")):
                for parquet_path in sorted(chunk_dir.glob("episode_*.parquet")):
                    episode_idx = int(parquet_path.stem.split("_")[1])

                    # Get video paths
                    video_paths = self._get_video_paths(task_dir, chunk_dir.name, episode_idx)

                    # Get frame count from parquet
                    df = pd.read_parquet(parquet_path)
                    num_frames = len(df)

                    # Get task description
                    task_idx = df["task_index"].iloc[0] if "task_index" in df.columns else 0
                    task_desc = task_descriptions.get(task_idx, f"Complete the {task_name} task")

                    episodes.append(RoboCOINEpisodeInfo(
                        task_dir=task_dir,
                        episode_idx=episode_idx,
                        parquet_path=parquet_path,
                        video_paths=video_paths,
                        num_frames=num_frames,
                        robot_type=robot_type,
                        task_description=task_desc,
                        state_dim=state_dim,
                        action_dim=action_dim,
                    ))

                    episode_count += 1
                    if self.max_episodes_per_task and episode_count >= self.max_episodes_per_task:
                        break

                if self.max_episodes_per_task and episode_count >= self.max_episodes_per_task:
                    break

        return episodes

    def _get_video_paths(
        self,
        task_dir: Path,
        chunk_name: str,
        episode_idx: int,
    ) -> dict[str, Path]:
        """Get video paths for an episode."""
        video_paths = {}
        videos_dir = task_dir / "videos" / chunk_name

        for cam_name in self.SUPPORTED_CAMERAS:
            cam_dir = videos_dir / f"observation.images.{cam_name}"
            if cam_dir.exists():
                video_path = cam_dir / f"episode_{episode_idx:06d}.mp4"
                if video_path.exists():
                    video_paths[cam_name] = video_path

        return video_paths

    def _load_task_annotations(self, task_dir: Path):
        """Load subtask and scene annotations for a task."""
        task_key = str(task_dir)

        # Subtask annotations
        subtask_path = task_dir / "annotations" / "subtask_annotations.jsonl"
        if subtask_path.exists():
            subtasks = {}
            with open(subtask_path) as f:
                for line in f:
                    d = json.loads(line)
                    subtasks[d["subtask_index"]] = d["subtask"]
            self.subtask_annotations[task_key] = subtasks

        # Scene annotations
        scene_path = task_dir / "annotations" / "scene_annotations.jsonl"
        if scene_path.exists():
            scenes = {}
            with open(scene_path) as f:
                for line in f:
                    d = json.loads(line)
                    scenes[d["scene_index"]] = d["scene"]
            self.scene_annotations[task_key] = scenes

    def _load_parquet(self, parquet_path: Path) -> pd.DataFrame:
        """Load parquet file with caching."""
        path_str = str(parquet_path)
        if path_str in self._parquet_cache:
            return self._parquet_cache[path_str]

        # Evict oldest if cache is full
        if len(self._parquet_cache) >= self._parquet_cache_size:
            oldest = next(iter(self._parquet_cache))
            del self._parquet_cache[oldest]

        df = pd.read_parquet(parquet_path)
        self._parquet_cache[path_str] = df
        return df

    def _get_subtask_ids(self, df: pd.DataFrame) -> np.ndarray:
        """Get subtask IDs for each frame using specified annotator level."""
        if "subtask_annotation" not in df.columns:
            return np.zeros(len(df), dtype=np.int32)

        subtask_annotations = np.stack(df["subtask_annotation"].values)
        return subtask_annotations[:, self.subtask_annotator_level]

    def _find_subtask_boundaries(
        self,
        subtask_ids: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        """Find start/end frames for each subtask segment."""
        boundaries = []
        if len(subtask_ids) == 0:
            return boundaries

        current_id = subtask_ids[0]
        start_frame = 0

        for i in range(1, len(subtask_ids)):
            if subtask_ids[i] != current_id:
                boundaries.append((start_frame, i, current_id))
                current_id = subtask_ids[i]
                start_frame = i

        # Final segment
        boundaries.append((start_frame, len(subtask_ids), current_id))

        return boundaries

    def _load_video_frame(
        self,
        video_path: Path,
        frame_idx: int,
    ) -> np.ndarray:
        """Load a single frame from video using PyAV."""
        if not HAS_PYAV:
            raise RuntimeError("PyAV is required for video decoding")

        cache_key = f"{video_path}:{frame_idx}"

        if self.cache_video_frames and cache_key in self._video_cache:
            return self._video_cache[cache_key]

        container = av.open(str(video_path))
        stream = container.streams.video[0]

        # Seek to approximate position
        target_pts = int(frame_idx * stream.time_base.denominator / stream.average_rate)
        container.seek(target_pts, stream=stream)

        # Decode frames until we get the one we want
        frame_count = 0
        result = None
        for frame in container.decode(video=0):
            if frame_count >= frame_idx:
                result = frame.to_ndarray(format='rgb24')
                break
            frame_count += 1

        container.close()

        if result is None:
            # Fallback: return black frame
            result = np.zeros((480, 640, 3), dtype=np.uint8)

        if self.cache_video_frames:
            self._video_cache[cache_key] = result

        return result

    def _load_raw_sample(self, sample_info: dict) -> dict[str, Any]:
        """Load raw data for a single sample."""
        ep = sample_info["episode_info"]
        timestep = sample_info["timestep"]
        subtask_start = sample_info["subtask_start"]
        subtask_end = sample_info["subtask_end"]
        subtask_id = sample_info["subtask_id"]

        # Load parquet data
        df = self._load_parquet(ep.parquet_path)

        # Get state and action
        state = np.array(df["observation.state"].iloc[timestep], dtype=np.float32)

        # Get future actions
        available_future = min(self.action_horizon, subtask_end - timestep - 1)
        future_end = timestep + 1 + available_future

        future_states = np.stack([
            np.array(df["observation.state"].iloc[t])
            for t in range(timestep + 1, future_end)
        ]).astype(np.float32)

        # Compute deltas
        action_deltas = future_states - state[None, :]

        # Pad if needed
        if len(action_deltas) < self.action_horizon:
            pad_count = self.action_horizon - len(action_deltas)
            last_delta = action_deltas[-1:] if len(action_deltas) > 0 else np.zeros((1, ep.state_dim))
            action_deltas = np.concatenate([
                action_deltas,
                np.tile(last_delta, (pad_count, 1))
            ])

        # Load images
        images = {}
        for cam_name, video_path in ep.video_paths.items():
            try:
                img = self._load_video_frame(video_path, timestep)
                # Map camera names to standard format
                if cam_name == "cam_high_rgb":
                    images["head"] = img
                elif cam_name == "cam_left_wrist_rgb":
                    images["left_wrist"] = img
                elif cam_name == "cam_right_wrist_rgb":
                    images["right_wrist"] = img
                else:
                    images[cam_name] = img
            except Exception as e:
                print(f"Warning: Failed to load {cam_name} for episode {ep.episode_idx}: {e}")

        # Get subtask description
        task_key = str(ep.task_dir)
        subtask_desc = None
        if task_key in self.subtask_annotations:
            subtask_desc = self.subtask_annotations[task_key].get(subtask_id)

        # Determine active components (simplified - RoboCOIN doesn't have per-subtask arm annotations)
        active_components = ActiveComponents.ALL

        return {
            "images": images,
            "state": state,
            "actions": action_deltas,
            "task_description": ep.task_description,
            "subtask_description": subtask_desc,
            "robot_type": ep.robot_type,
            "episode_frame_idx": timestep,
            "episode_total_frames": ep.num_frames,
            "subtask_frame_idx": timestep - subtask_start,
            "subtask_total_frames": subtask_end - subtask_start,
            "state_history": None,  # TODO: implement state history
            "active_components": active_components,
        }

    def _create_state_spec(self, ep: RoboCOINEpisodeInfo) -> RobotStateSpec:
        """Create state spec from episode info."""
        # RoboCOIN has variable state dimensions
        # We treat everything as joint positions for simplicity
        return RobotStateSpec(joint_positions=ep.state_dim)

    def _create_action_spec(self, ep: RoboCOINEpisodeInfo) -> RobotActionSpec:
        """Create action spec from episode info."""
        return RobotActionSpec(joint_targets=ep.action_dim)

    def get_robot_types(self) -> list[str]:
        """Get list of unique robot types in dataset."""
        return list(set(ep.robot_type for ep in self.episodes))

    def get_task_names(self) -> list[str]:
        """Get list of unique task names in dataset."""
        return list(set(ep.task_dir.name for ep in self.episodes))
