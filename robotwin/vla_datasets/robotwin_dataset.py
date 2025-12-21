"""
RoboTwin dataset loader extending BaseVLADataset.

Supports two data formats:
1. Uncompressed format: {root}/data/episode*.hdf5 + subtask_annotations/
2. Zip-based format: {root}/{task}/*.zip containing HDF5 files

Features:
- Subtask annotations for boundary-aware sampling (uncompressed format)
- Episode-level sampling for zip-based format
- Support for both joint delta and EEF pose delta action types
"""

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import random

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from .base_dataset import BaseVLADataset
from .unified_sample import (
    UnifiedSample,
    RobotStateSpec,
    RobotActionSpec,
    ActiveComponents,
    compute_progress,
)


class ZipFileCache:
    """LRU cache for open zip file handles."""

    def __init__(self, max_open: int = 10):
        self.max_open = max_open
        self.cache: dict[str, tuple[zipfile.ZipFile, int]] = {}
        self.access_counter = 0

    def get(self, zip_path: Path) -> zipfile.ZipFile:
        """Get an open zip file, opening if needed."""
        path_str = str(zip_path)
        if path_str in self.cache:
            zf, _ = self.cache[path_str]
            self.access_counter += 1
            self.cache[path_str] = (zf, self.access_counter)
            return zf

        # Evict least recently used if cache is full
        if len(self.cache) >= self.max_open:
            lru_path = min(self.cache.items(), key=lambda x: x[1][1])[0]
            self.cache[lru_path][0].close()
            del self.cache[lru_path]

        # Open new zip file
        zf = zipfile.ZipFile(zip_path, "r")
        self.access_counter += 1
        self.cache[path_str] = (zf, self.access_counter)
        return zf

    def close_all(self):
        """Close all open zip files."""
        for zf, _ in self.cache.values():
            zf.close()
        self.cache.clear()


@dataclass
class SubtaskInfo:
    """Information about a single subtask."""
    subtask_id: int
    subtask_type: str
    start_frame: int
    end_frame: int
    instruction: str
    obj: Optional[str]
    base: Optional[str]
    arm: str
    metadata: dict


# Robot specifications for RoboTwin ALOHA-style robots
ROBOTWIN_STATE_SPEC = RobotStateSpec(
    joint_positions=12,  # 6 per arm
    gripper_states=2,    # 1 per gripper
)

ROBOTWIN_JOINT_ACTION_SPEC = RobotActionSpec(
    joint_targets=12,      # 6 per arm (deltas)
    gripper_commands=2,    # 1 per gripper
)

ROBOTWIN_EEF_ACTION_SPEC = RobotActionSpec(
    eef_position_deltas=6,   # 3 per arm
    eef_rotation_deltas=6,   # 3 per arm (axis-angle)
    gripper_commands=2,
)


class RoboTwinDataset(BaseVLADataset):
    """
    RoboTwin dataset loader supporting both recovery and zip-based formats.

    Supports:
    - Recovery format: Direct HDF5 with subtask annotations
    - Zip-based format: Original RoboTwin dataset in zip archives
    - Both joint delta and EEF pose delta action types
    - Multiple robot types with filtering
    """

    DATASET_NAME = "robotwin"
    SUPPORTED_CAMERAS = ["head", "left_wrist", "right_wrist"]
    DEFAULT_STATE_SPEC = ROBOTWIN_STATE_SPEC
    DEFAULT_ACTION_SPEC = ROBOTWIN_JOINT_ACTION_SPEC

    def __init__(
        self,
        dataset_root: str,
        norm_stats_path: Optional[str] = None,
        action_horizon: int = 8,
        image_size: tuple[int, int] = (320, 240),
        action_type: str = "joint_delta",
        robot_type: Optional[str] = None,  # None = auto-detect or use filter
        robot_types: Optional[list[str]] = None,  # Filter for zip format
        variants: Optional[list[str]] = None,  # Filter for zip format
        tasks: Optional[list[str]] = None,
        episode_filter: Optional[list[int]] = None,
        max_episodes_per_zip: Optional[int] = None,  # Limit for testing
        binarize_grippers: bool = False,
        gripper_open_threshold: float = 0.95,
        gripper_closed_threshold: float = 0.05,
        eef_action_ratio: float = 0.5,  # For mixed mode
        cache_size: int = 10,
        **kwargs
    ):
        """
        Args:
            dataset_root: Root directory (demo_recovery for recovery format,
                         or dataset root for zip format)
            norm_stats_path: Path to normalization statistics JSON
            action_horizon: Number of future timesteps to predict
            image_size: Target image size (width, height)
            action_type: "joint_delta", "eef_pose_delta", or "mixed"
            robot_type: Robot type for recovery format (default: "aloha-agilex")
            robot_types: Filter by robot types for zip format
            variants: Filter by variants (e.g., ["clean_50", "randomized_500"])
            tasks: Optional list of task names to include
            episode_filter: Optional list of episode indices to include
            max_episodes_per_zip: Max episodes per zip file (for testing)
            binarize_grippers: Whether to binarize gripper values
            gripper_open_threshold: Threshold for open gripper
            gripper_closed_threshold: Threshold for closed gripper
            eef_action_ratio: Ratio of EEF samples in mixed mode
            cache_size: Number of HDF5/zip files to keep open
        """
        self.robot_type_name = robot_type or "aloha-agilex"
        self.robot_types_filter = robot_types
        self.variants_filter = variants
        self.tasks_filter = tasks
        self.episode_filter = episode_filter
        self.max_episodes_per_zip = max_episodes_per_zip
        self.binarize_grippers = binarize_grippers
        self.gripper_open_threshold = gripper_open_threshold
        self.gripper_closed_threshold = gripper_closed_threshold
        self.eef_action_ratio = eef_action_ratio
        self.cache_size = cache_size

        # Caches
        self._hdf5_cache: dict[str, h5py.File] = {}
        self._zip_cache = ZipFileCache(max_open=cache_size)

        # Episode and subtask data (populated by _build_sample_index)
        self.episodes: list[dict] = []
        self.subtask_annotations: dict[int, dict] = {}

        # Format detection (set during _scan_episodes)
        self._format: str = "unknown"  # "uncompressed" or "zip"

        # Call parent init (which calls _build_sample_index)
        super().__init__(
            dataset_root=dataset_root,
            norm_stats_path=norm_stats_path,
            action_horizon=action_horizon,
            image_size=image_size,
            action_type=action_type,
            **kwargs
        )

        print(f"RoboTwin dataset ready:")
        print(f"  Format: {self._format}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Action type: {action_type}")

    def _build_sample_index(self) -> list[dict]:
        """Build sample index with subtask boundary information."""
        # First scan for episodes
        self.episodes = self._scan_episodes()

        if not self.episodes:
            print(f"Warning: No episodes found in {self.dataset_root}")
            return []

        # Route to format-specific sample building
        if self._format == "uncompressed":
            return self._build_samples_uncompressed()
        else:
            return self._build_samples_zip()

    def _build_samples_uncompressed(self) -> list[dict]:
        """Build samples for uncompressed format with subtask annotations."""
        # Load subtask annotations
        self.subtask_annotations = self._load_subtask_annotations()

        samples = []
        for ep in self.episodes:
            if ep["episode_idx"] not in self.subtask_annotations:
                continue

            annotations = self.subtask_annotations[ep["episode_idx"]]
            subtasks = annotations.get("subtasks", [])

            # Build episode key for valid timesteps lookup
            episode_key = f"{ep['task_name']}/episode{ep['episode_idx']}"

            for subtask in subtasks:
                start_frame = subtask["start_frame"]
                end_frame = subtask["end_frame"]

                # Create samples for each valid timestep
                for t in range(start_frame, end_frame - 1):
                    samples.append({
                        "format": "uncompressed",
                        "episode_idx": ep["episode_idx"],
                        "episode_key": episode_key,
                        "task_name": ep["task_name"],
                        "hdf5_path": ep["hdf5_path"],
                        "timestep": t,
                        "subtask_start": start_frame,
                        "subtask_end": end_frame,
                        "episode_total_frames": ep["num_timesteps"],
                        "robot_type": ep["robot_type"],
                        "subtask_info": SubtaskInfo(
                            subtask_id=subtask["subtask_id"],
                            subtask_type=subtask["type"],
                            start_frame=start_frame,
                            end_frame=end_frame,
                            instruction=subtask["instruction"],
                            obj=subtask.get("obj"),
                            base=subtask.get("base"),
                            arm=subtask["arm"],
                            metadata=subtask.get("metadata", {}),
                        ),
                    })

        return samples

    def _build_samples_zip(self) -> list[dict]:
        """Build samples for zip format (no subtask annotations)."""
        samples = []

        for ep in self.episodes:
            # Load episode to get timestep count
            try:
                hdf5_file = self._get_hdf5_file_zip(ep)
                num_timesteps = len(hdf5_file["joint_action"]["vector"])
                ep["num_timesteps"] = num_timesteps
            except Exception as e:
                print(f"  Warning: Could not load episode {ep['episode_idx']} from {ep['zip_path']}: {e}")
                continue

            # Build episode key for valid timesteps lookup
            episode_key = f"{ep['task_name']}/{ep['robot_type']}_{ep['variant']}/episode{ep['episode_idx']}"

            # Treat entire episode as one segment
            for t in range(num_timesteps - 1):
                samples.append({
                    "format": "zip",
                    "episode_idx": ep["episode_idx"],
                    "episode_key": episode_key,
                    "task_name": ep["task_name"],
                    "zip_path": ep["zip_path"],
                    "zip_stem": ep["zip_stem"],
                    "hdf5_path_in_zip": ep["hdf5_path_in_zip"],
                    "timestep": t,
                    "subtask_start": 0,
                    "subtask_end": num_timesteps,
                    "episode_total_frames": num_timesteps,
                    "robot_type": ep["robot_type"],
                    "variant": ep["variant"],
                    # No subtask info for zip format
                    "subtask_info": None,
                })

        return samples

    def _scan_episodes(self) -> list[dict]:
        """Scan dataset directory for episodes, auto-detecting format."""
        # Check for uncompressed format (direct HDF5 files)
        data_dir = self.dataset_root / "data"
        if data_dir.exists() and any(data_dir.glob("episode*.hdf5")):
            self._format = "uncompressed"
            return self._scan_episodes_uncompressed()

        # Check for zip-based format (task directories with zip files)
        task_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        if task_dirs and any(any(d.glob("*.zip")) for d in task_dirs):
            self._format = "zip"
            return self._scan_episodes_zip()

        return []

    def _scan_episodes_uncompressed(self) -> list[dict]:
        """Scan uncompressed format: direct HDF5 with subtask annotations."""
        episodes = []
        data_dir = self.dataset_root / "data"
        subtask_dir = self.dataset_root / "subtask_annotations"

        # Get task name from directory structure
        task_name = self.dataset_root.parent.name if self.dataset_root.name == "demo_recovery" else "unknown"

        # Apply task filter
        if self.tasks_filter and task_name not in self.tasks_filter:
            return episodes

        # Scan HDF5 files
        for hdf5_path in sorted(data_dir.glob("episode*.hdf5")):
            episode_idx = int(hdf5_path.stem.replace("episode", ""))

            # Apply episode filter
            if self.episode_filter and episode_idx not in self.episode_filter:
                continue

            subtask_path = subtask_dir / f"episode{episode_idx}.json"
            if not subtask_path.exists():
                continue

            # Get number of timesteps
            with h5py.File(hdf5_path, "r") as f:
                num_timesteps = len(f["joint_action"]["vector"])

            episodes.append({
                "format": "uncompressed",
                "task_name": task_name,
                "episode_idx": episode_idx,
                "hdf5_path": hdf5_path,
                "subtask_path": subtask_path,
                "num_timesteps": num_timesteps,
                "robot_type": self.robot_type_name,
            })

        return episodes

    def _scan_episodes_zip(self) -> list[dict]:
        """Scan zip-based format: task directories with zip archives."""
        episodes = []

        # Get all task directories
        task_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]

        if self.tasks_filter:
            task_dirs = [d for d in task_dirs if d.name in self.tasks_filter]

        for task_dir in sorted(task_dirs):
            task_name = task_dir.name

            # Find all zip files in this task directory
            for zip_path in sorted(task_dir.glob("*.zip")):
                # Parse zip filename: {robot_type}_{variant}.zip
                # e.g., "aloha-agilex_clean_50.zip" -> robot="aloha-agilex", variant="clean_50"
                stem = zip_path.stem
                parts = stem.rsplit("_", 2)

                if len(parts) >= 2:
                    robot_type = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
                    variant = f"{parts[-2]}_{parts[-1]}"
                else:
                    continue

                # Apply filters
                if self.robot_types_filter and robot_type not in self.robot_types_filter:
                    continue
                if self.variants_filter and variant not in self.variants_filter:
                    continue

                # Scan episodes in this zip
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        # Find episode HDF5 files
                        data_files = [
                            f for f in zf.namelist()
                            if f.startswith(f"{stem}/data/episode") and f.endswith(".hdf5")
                        ]

                        episode_count = 0
                        for data_file in sorted(data_files):
                            basename = Path(data_file).stem  # "episode0"
                            episode_idx = int(basename.replace("episode", ""))

                            # Apply episode filter
                            if self.episode_filter and episode_idx not in self.episode_filter:
                                continue

                            # Apply max episodes limit
                            if self.max_episodes_per_zip and episode_count >= self.max_episodes_per_zip:
                                break

                            episodes.append({
                                "format": "zip",
                                "task_name": task_name,
                                "episode_idx": episode_idx,
                                "zip_path": zip_path,
                                "zip_stem": stem,
                                "hdf5_path_in_zip": data_file,
                                "robot_type": robot_type,
                                "variant": variant,
                                # num_timesteps will be loaded lazily
                            })
                            episode_count += 1

                except Exception as e:
                    print(f"  Warning: Could not scan {zip_path}: {e}")

        return episodes

    def _load_subtask_annotations(self) -> dict[int, dict]:
        """Load subtask annotations for all episodes."""
        annotations = {}
        for ep in self.episodes:
            with open(ep["subtask_path"]) as f:
                annotations[ep["episode_idx"]] = json.load(f)
        return annotations

    def _get_hdf5_file(self, hdf5_path: Path) -> h5py.File:
        """Get HDF5 file from cache or open it (uncompressed format)."""
        path_str = str(hdf5_path)
        if path_str in self._hdf5_cache:
            return self._hdf5_cache[path_str]

        # Evict oldest if cache is full
        if len(self._hdf5_cache) >= self.cache_size:
            oldest_path = next(iter(self._hdf5_cache))
            self._hdf5_cache[oldest_path].close()
            del self._hdf5_cache[oldest_path]

        hdf5_file = h5py.File(hdf5_path, "r")
        self._hdf5_cache[path_str] = hdf5_file
        return hdf5_file

    def _get_hdf5_file_zip(self, ep: dict) -> h5py.File:
        """Get HDF5 file from zip archive, with caching."""
        cache_key = f"{ep['zip_path']}::{ep['hdf5_path_in_zip']}"

        if cache_key in self._hdf5_cache:
            return self._hdf5_cache[cache_key]

        # Evict oldest if cache is full
        if len(self._hdf5_cache) >= self.cache_size:
            oldest_key = next(iter(self._hdf5_cache))
            self._hdf5_cache[oldest_key].close()
            del self._hdf5_cache[oldest_key]

        # Read HDF5 from zip
        zf = self._zip_cache.get(ep["zip_path"])
        hdf5_bytes = zf.read(ep["hdf5_path_in_zip"])
        hdf5_file = h5py.File(io.BytesIO(hdf5_bytes), "r")

        self._hdf5_cache[cache_key] = hdf5_file
        return hdf5_file

    def _get_hdf5_for_sample(self, sample_info: dict) -> h5py.File:
        """Get HDF5 file for a sample, handling both formats."""
        if sample_info.get("format") == "zip":
            return self._get_hdf5_file_zip(sample_info)
        else:
            return self._get_hdf5_file(sample_info["hdf5_path"])

    def _load_raw_sample(self, sample_info: dict) -> dict[str, Any]:
        """Load raw data for a single sample."""
        hdf5_file = self._get_hdf5_for_sample(sample_info)
        timestep = sample_info["timestep"]
        subtask_info = sample_info["subtask_info"]
        subtask_start = sample_info["subtask_start"]
        subtask_end = sample_info["subtask_end"]

        # Load joint data
        left_arm = np.array(hdf5_file["joint_action"]["left_arm"])
        right_arm = np.array(hdf5_file["joint_action"]["right_arm"])
        left_gripper = np.array(hdf5_file["joint_action"]["left_gripper"])
        right_gripper = np.array(hdf5_file["joint_action"]["right_gripper"])

        full_joints = np.concatenate([left_arm, right_arm], axis=1)  # (T, 12)
        full_grippers = np.stack([left_gripper, right_gripper], axis=1)  # (T, 2)

        # Load EEF poses
        left_endpose = np.array(hdf5_file["endpose"]["left_endpose"])
        right_endpose = np.array(hdf5_file["endpose"]["right_endpose"])

        # Apply gripper binarization if enabled
        if self.binarize_grippers:
            full_grippers = self._binarize_grippers_array(
                full_grippers, subtask_start, subtask_end
            )

        # Current state
        current_joints = full_joints[timestep].astype(np.float32)
        current_grippers = full_grippers[timestep].astype(np.float32)

        # Future actions with subtask boundary padding
        actions, actual_action_type = self._extract_future_actions(
            timestep, subtask_end,
            full_joints, full_grippers,
            left_endpose, right_endpose,
            sample_info["robot_type"],
        )

        # Load and process images
        images = {
            "head": self._load_image(hdf5_file["observation"]["head_camera"]["rgb"][timestep]),
            "left_wrist": self._load_image(hdf5_file["observation"]["left_camera"]["rgb"][timestep]),
            "right_wrist": self._load_image(hdf5_file["observation"]["right_camera"]["rgb"][timestep]),
        }

        # Detect active components from action magnitudes
        active_components = self.detect_active_components(
            actions,
            action_spec=ROBOTWIN_JOINT_ACTION_SPEC if actual_action_type == "joint_delta" else ROBOTWIN_EEF_ACTION_SPEC,
        )

        # State history
        state_history = None
        if self.state_history_len > 0:
            state_history = self._extract_state_history(
                full_joints, full_grippers, timestep, subtask_start
            )

        # Task name - use from sample_info directly (works for both formats)
        task_name = sample_info["task_name"]

        # Get subtask description if available
        subtask_description = subtask_info.instruction if subtask_info is not None else None

        return {
            "images": images,
            "state": np.concatenate([current_joints, current_grippers]),
            "actions": actions,
            "task_description": f"Complete the {task_name.replace('_', ' ')} task",
            "subtask_description": subtask_description,
            "robot_type": sample_info["robot_type"],
            "episode_frame_idx": timestep,
            "episode_total_frames": sample_info["episode_total_frames"],
            "subtask_frame_idx": timestep - subtask_start if subtask_info is not None else None,
            "subtask_total_frames": subtask_end - subtask_start if subtask_info is not None else None,
            "state_history": state_history,
            "active_components": active_components,
            "action_type_override": actual_action_type,
            "state_spec": ROBOTWIN_STATE_SPEC,
            "action_spec": ROBOTWIN_JOINT_ACTION_SPEC if actual_action_type == "joint_delta" else ROBOTWIN_EEF_ACTION_SPEC,
        }

    def _extract_future_actions(
        self,
        timestep: int,
        subtask_end: int,
        full_joints: np.ndarray,
        full_grippers: np.ndarray,
        left_endpose: np.ndarray,
        right_endpose: np.ndarray,
        robot_type: str,
    ) -> tuple[np.ndarray, str]:
        """Extract future actions with subtask boundary padding."""
        # Determine action type for this sample
        if self.action_type == "mixed":
            actual_action_type = "eef_pose_delta" if random.random() < self.eef_action_ratio else "joint_delta"
        else:
            actual_action_type = self.action_type

        # Calculate available future frames
        available_future = min(self.action_horizon, subtask_end - timestep - 1)
        future_end = timestep + 1 + available_future

        if actual_action_type == "eef_pose_delta":
            # EEF pose deltas
            current_left_pose = left_endpose[timestep]
            current_right_pose = right_endpose[timestep]
            future_left_poses = left_endpose[timestep + 1:future_end]
            future_right_poses = right_endpose[timestep + 1:future_end]

            eef_deltas = self._compute_eef_pose_deltas(
                current_left_pose, current_right_pose,
                future_left_poses, future_right_poses,
            )

            # Pad if needed
            if len(eef_deltas) < self.action_horizon:
                pad_count = self.action_horizon - len(eef_deltas)
                last_delta = eef_deltas[-1:] if len(eef_deltas) > 0 else np.zeros((1, 12))
                eef_deltas = np.concatenate([eef_deltas, np.tile(last_delta, (pad_count, 1))])

            # Get future grippers
            future_grippers = full_grippers[timestep + 1:future_end]
            if len(future_grippers) < self.action_horizon:
                pad_count = self.action_horizon - len(future_grippers)
                last_grip = future_grippers[-1:] if len(future_grippers) > 0 else full_grippers[subtask_end - 1:subtask_end]
                future_grippers = np.concatenate([future_grippers, np.tile(last_grip, (pad_count, 1))])

            actions = np.concatenate([eef_deltas, future_grippers], axis=1)

        else:
            # Joint deltas
            current_joints = full_joints[timestep]
            future_joints = full_joints[timestep + 1:future_end]
            future_grippers = full_grippers[timestep + 1:future_end]

            # Pad if needed
            if len(future_joints) < self.action_horizon:
                pad_count = self.action_horizon - len(future_joints)
                boundary_joints = full_joints[subtask_end - 1:subtask_end]
                boundary_grippers = full_grippers[subtask_end - 1:subtask_end]
                future_joints = np.concatenate([future_joints, np.tile(boundary_joints, (pad_count, 1))])
                future_grippers = np.concatenate([future_grippers, np.tile(boundary_grippers, (pad_count, 1))])

            delta_joints = future_joints - current_joints[None, :]
            actions = np.concatenate([delta_joints, future_grippers], axis=1)

        return actions.astype(np.float32), actual_action_type

    def _compute_eef_pose_deltas(
        self,
        current_left_pose: np.ndarray,
        current_right_pose: np.ndarray,
        future_left_poses: np.ndarray,
        future_right_poses: np.ndarray,
    ) -> np.ndarray:
        """Compute EEF pose deltas for both arms."""
        H = len(future_left_poses)
        if H == 0:
            return np.zeros((0, 12), dtype=np.float32)

        deltas = np.zeros((H, 12), dtype=np.float32)

        for t in range(H):
            # Left arm
            deltas[t, :3] = future_left_poses[t, :3] - current_left_pose[:3]
            deltas[t, 3:6] = self._compute_rotation_delta(
                current_left_pose[3:7], future_left_poses[t, 3:7]
            )

            # Right arm
            deltas[t, 6:9] = future_right_poses[t, :3] - current_right_pose[:3]
            deltas[t, 9:12] = self._compute_rotation_delta(
                current_right_pose[3:7], future_right_poses[t, 3:7]
            )

        return deltas

    def _compute_rotation_delta(
        self,
        quat_current: np.ndarray,
        quat_future: np.ndarray,
    ) -> np.ndarray:
        """Compute rotation delta as axis-angle."""
        # RoboTwin uses [qw, qx, qy, qz], scipy uses [qx, qy, qz, qw]
        q_curr_scipy = np.array([quat_current[1], quat_current[2], quat_current[3], quat_current[0]])
        q_fut_scipy = np.array([quat_future[1], quat_future[2], quat_future[3], quat_future[0]])

        R_curr = Rotation.from_quat(q_curr_scipy)
        R_fut = Rotation.from_quat(q_fut_scipy)
        R_delta = R_fut * R_curr.inv()

        return R_delta.as_rotvec()

    def _binarize_grippers_array(
        self,
        full_grippers: np.ndarray,
        subtask_start: int,
        subtask_end: int,
    ) -> np.ndarray:
        """Binarize gripper values using forward-looking relabeling."""
        result = full_grippers.copy()

        for grip_idx in range(2):
            for t in range(subtask_start, subtask_end):
                value = full_grippers[t, grip_idx]

                if value > self.gripper_open_threshold:
                    result[t, grip_idx] = 1.0
                elif value < self.gripper_closed_threshold:
                    result[t, grip_idx] = 0.0
                else:
                    # Look forward
                    eventual = None
                    for future_t in range(t + 1, subtask_end):
                        future_val = full_grippers[future_t, grip_idx]
                        if future_val > self.gripper_open_threshold:
                            eventual = 1.0
                            break
                        elif future_val < self.gripper_closed_threshold:
                            eventual = 0.0
                            break
                    result[t, grip_idx] = eventual if eventual is not None else (1.0 if value > 0.5 else 0.0)

        return result

    def _extract_state_history(
        self,
        full_joints: np.ndarray,
        full_grippers: np.ndarray,
        timestep: int,
        subtask_start: int,
    ) -> np.ndarray:
        """Extract state history respecting subtask boundaries."""
        K = self.state_history_len
        state_dim = full_joints.shape[1] + full_grippers.shape[1]

        # Limit history to within current subtask
        start_idx = max(subtask_start, timestep - K + 1)
        collected_timesteps = list(range(start_idx, timestep + 1))

        # Collect states
        collected_states = []
        for t in collected_timesteps:
            state = np.concatenate([
                full_joints[t].astype(np.float32),
                full_grippers[t].astype(np.float32),
            ])
            collected_states.append(state)

        # Pad by replicating oldest state if needed
        if len(collected_states) < K:
            oldest = collected_states[0] if collected_states else np.zeros(state_dim, dtype=np.float32)
            padding = [oldest.copy() for _ in range(K - len(collected_states))]
            collected_states = padding + collected_states

        return np.stack(collected_states, axis=0)

    def _load_image(self, compressed_bytes: bytes) -> np.ndarray:
        """Load compressed image bytes to numpy array."""
        image = Image.open(io.BytesIO(compressed_bytes)).convert("RGB")
        return np.array(image)

    def __getitem__(self, idx: int) -> UnifiedSample:
        """Override to handle action type override from raw sample."""
        sample_info = self.samples[idx]
        raw = self._load_raw_sample(sample_info)

        # Handle action type override (for mixed mode)
        actual_action_type = raw.pop("action_type_override", self.action_type)

        # Process images
        images = self._process_images(raw["images"])

        # Normalize state and actions
        # Determine action category based on actual action type
        if self.normalizer is not None:
            normalized_state = self.normalizer.normalize_state(
                raw["state"][:12], raw["robot_type"]  # Just joints
            )
            normalized_state = np.concatenate([
                normalized_state,
                self.normalizer.normalize_grippers(raw["state"][12:], raw["robot_type"])
            ])

            action_category = "eef_delta_actions" if actual_action_type == "eef_pose_delta" else "delta_actions"
            normalized_actions = self.normalizer.normalize_delta_actions(
                raw["actions"][:, :-2], raw["robot_type"], category=action_category
            )
            normalized_grippers = self.normalizer.normalize_grippers(
                raw["actions"][:, -2:], raw["robot_type"]
            )
            normalized_actions = np.concatenate([normalized_actions, normalized_grippers], axis=1)
        else:
            normalized_state = raw["state"]
            normalized_actions = raw["actions"]

        # Normalize state history if present
        state_history = None
        if raw.get("state_history") is not None and self.normalizer is not None:
            sh = raw["state_history"]
            sh_joints = self.normalizer.normalize_state(sh[:, :12], raw["robot_type"])
            sh_grippers = self.normalizer.normalize_grippers(sh[:, 12:], raw["robot_type"])
            state_history = np.concatenate([sh_joints, sh_grippers], axis=1)
        elif raw.get("state_history") is not None:
            state_history = raw["state_history"]

        # Compute progress
        progress = compute_progress(
            raw["episode_frame_idx"],
            raw["episode_total_frames"],
            raw.get("subtask_frame_idx"),
            raw.get("subtask_total_frames"),
        )

        # Determine end flags
        is_subtask_end = False
        is_episode_end = False
        if raw.get("subtask_frame_idx") is not None and raw.get("subtask_total_frames") is not None:
            is_subtask_end = raw["subtask_frame_idx"] >= raw["subtask_total_frames"] - 1
        is_episode_end = raw["episode_frame_idx"] >= raw["episode_total_frames"] - 1

        return UnifiedSample(
            images=images,
            task_description=raw["task_description"],
            subtask_description=raw.get("subtask_description"),
            robot_type=raw["robot_type"],
            state=raw["state"],
            state_normalized=normalized_state,
            state_spec=raw.get("state_spec", self.DEFAULT_STATE_SPEC),
            state_history=state_history,
            action_tokens=[],
            action_type=actual_action_type,
            normalized_actions=normalized_actions,
            action_spec=raw.get("action_spec", self.DEFAULT_ACTION_SPEC),
            action_horizon=normalized_actions.shape[0],
            episode_frame_idx=raw["episode_frame_idx"],
            episode_total_frames=raw["episode_total_frames"],
            subtask_frame_idx=raw.get("subtask_frame_idx"),
            subtask_total_frames=raw.get("subtask_total_frames"),
            progress_percent=progress,
            is_subtask_end=is_subtask_end,
            is_episode_end=is_episode_end,
            dataset_name=self.DATASET_NAME,
            episode_id=f"{sample_info['episode_idx']}",
            active_components=raw.get("active_components", ActiveComponents.ALL),
        )

    def close(self):
        """Close all open HDF5 files and zip handles."""
        for f in self._hdf5_cache.values():
            try:
                f.close()
            except (TypeError, ValueError):
                # h5py files from BytesIO may not close cleanly
                pass
        self._hdf5_cache.clear()
        self._zip_cache.close_all()

    def __del__(self):
        self.close()
