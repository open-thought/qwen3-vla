"""
Efficient batch loader for RoboTwin dataset.

Key features:
- Streams data directly from zip files without full extraction
- Builds lightweight metadata index across all tasks
- Samples episodes from diverse tasks for each batch
- Lazy loading with LRU caching for performance
"""

import json
import os
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import io

import h5py
import numpy as np
from PIL import Image


@dataclass
class EpisodeMetadata:
    """Metadata for a single episode."""

    task_name: str
    robot_type: str  # e.g., "arx-x5", "franka", "ur5"
    variant: str  # "clean_50" or "randomized_500"
    zip_path: Path
    episode_idx: int
    num_timesteps: int = 53  # typical value


class RoboTwinDatasetIndex:
    """Build and maintain index of all episodes across compressed archives."""

    def __init__(self, dataset_root: str = "/mnt/robotwin/dataset"):
        self.dataset_root = Path(dataset_root)
        self.episodes: List[EpisodeMetadata] = []
        self.episodes_by_task: dict = defaultdict(list)

    def build_index(
        self,
        robot_types: Optional[List[str]] = None,
        variants: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
    ) -> None:
        """
        Build index by scanning zip files.

        Args:
            robot_types: Filter by robot types (e.g., ["arx-x5", "franka"])
            variants: Filter by variants (e.g., ["clean_50", "randomized_500"])
            tasks: Filter by task names (e.g., ["adjust_bottle", "click_bell"])
        """
        print("Building dataset index...")

        # Get all task directories
        task_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]

        if tasks:
            task_dirs = [d for d in task_dirs if d.name in tasks]

        for task_dir in sorted(task_dirs):
            task_name = task_dir.name

            # Find all zip files in this task directory
            for zip_path in sorted(task_dir.glob("*.zip")):
                # Parse zip filename: {robot_type}_{variant}.zip
                stem = zip_path.stem  # e.g., "arx-x5_clean_50"
                parts = stem.rsplit(
                    "_", 2
                )  # Split from right to handle robot names with hyphens

                if len(parts) >= 2:
                    robot_type = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
                    variant = f"{parts[-2]}_{parts[-1]}"
                else:
                    continue

                # Apply filters
                if robot_types and robot_type not in robot_types:
                    continue
                if variants and variant not in variants:
                    continue

                # Peek into zip to count episodes
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        # Count episode files in the data/ subdirectory
                        data_files = [
                            f
                            for f in zf.namelist()
                            if f.startswith(f"{stem}/data/episode")
                            and f.endswith(".hdf5")
                        ]

                        for data_file in data_files:
                            # Extract episode number
                            basename = Path(data_file).stem  # "episode0"
                            episode_idx = int(basename.replace("episode", ""))

                            metadata = EpisodeMetadata(
                                task_name=task_name,
                                robot_type=robot_type,
                                variant=variant,
                                zip_path=zip_path,
                                episode_idx=episode_idx,
                            )
                            self.episodes.append(metadata)
                            self.episodes_by_task[task_name].append(metadata)

                    print(
                        f"  Indexed {len(data_files)} episodes from {task_name}/{zip_path.name}"
                    )
                except Exception as e:
                    print(f"  Warning: Could not index {zip_path}: {e}")

        print(f"\nIndexing complete!")
        print(f"  Total episodes: {len(self.episodes)}")
        print(f"  Tasks: {len(self.episodes_by_task)}")
        print(
            f"  Episodes per task: {[len(eps) for eps in self.episodes_by_task.values()][:10]}..."
        )

    def sample_episodes(
        self, num_episodes: int, strategy: str = "task_balanced"
    ) -> List[EpisodeMetadata]:
        """
        Sample episodes for a batch.

        Args:
            num_episodes: Number of episodes to sample
            strategy: Sampling strategy
                - "random": Uniform random sampling
                - "task_balanced": Sample equally from different tasks
                - "task_diverse": Maximize task diversity in each batch

        Returns:
            List of episode metadata
        """
        if strategy == "random":
            return random.sample(self.episodes, min(num_episodes, len(self.episodes)))

        elif strategy == "task_balanced":
            # Sample approximately equal number from each task
            tasks = list(self.episodes_by_task.keys())
            episodes_per_task = max(1, num_episodes // len(tasks))

            sampled = []
            for task in random.sample(tasks, min(len(tasks), num_episodes)):
                task_episodes = self.episodes_by_task[task]
                n_sample = min(episodes_per_task, len(task_episodes))
                sampled.extend(random.sample(task_episodes, n_sample))

                if len(sampled) >= num_episodes:
                    break

            return sampled[:num_episodes]

        elif strategy == "task_diverse":
            # Maximize task diversity by cycling through tasks
            tasks = list(self.episodes_by_task.keys())
            random.shuffle(tasks)

            sampled = []
            task_idx = 0

            while len(sampled) < num_episodes:
                task = tasks[task_idx % len(tasks)]
                task_episodes = self.episodes_by_task[task]

                # Sample one episode from this task
                if task_episodes:
                    sampled.append(random.choice(task_episodes))

                task_idx += 1

            return sampled[:num_episodes]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        stats = {
            "total_episodes": len(self.episodes),
            "num_tasks": len(self.episodes_by_task),
            "episodes_per_task": {
                task: len(eps) for task, eps in self.episodes_by_task.items()
            },
            "robot_types": set(ep.robot_type for ep in self.episodes),
            "variants": set(ep.variant for ep in self.episodes),
        }
        return stats


class ZipFileCache:
    """LRU cache for open zip file handles."""

    def __init__(self, max_open: int = 10):
        self.max_open = max_open
        self.cache: dict = {}  # Path -> (zipfile, access_count)
        self.access_counter = 0

    def get(self, zip_path: Path) -> zipfile.ZipFile:
        """Get an open zip file, opening if needed."""
        if zip_path in self.cache:
            zf, _ = self.cache[zip_path]
            self.access_counter += 1
            self.cache[zip_path] = (zf, self.access_counter)
            return zf

        # Need to open new file
        if len(self.cache) >= self.max_open:
            # Evict least recently used
            lru_path = min(self.cache.items(), key=lambda x: x[1][1])[0]
            self.cache[lru_path][0].close()
            del self.cache[lru_path]

        # Open new zip file
        zf = zipfile.ZipFile(zip_path, "r")
        self.access_counter += 1
        self.cache[zip_path] = (zf, self.access_counter)
        return zf

    def close_all(self):
        """Close all open zip files."""
        for zf, _ in self.cache.values():
            zf.close()
        self.cache.clear()


class RoboTwinBatchLoader:
    """Efficient batch loader that streams from zip files."""

    def __init__(
        self,
        index: RoboTwinDatasetIndex,
        batch_size: int = 8,
        sequence_length: int = 10,
        sampling_strategy: str = "task_diverse",
        cache_size: int = 10,
    ):
        """
        Args:
            index: Dataset index
            batch_size: Number of episodes per batch
            sequence_length: Number of timesteps to sample from each episode
            sampling_strategy: Episode sampling strategy
            cache_size: Number of zip files to keep open
        """
        self.index = index
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.sampling_strategy = sampling_strategy
        self.zip_cache = ZipFileCache(max_open=cache_size)

    def load_episode_data(self, metadata: EpisodeMetadata) -> dict:
        """
        Load episode data from zip file.

        Returns:
            Dictionary with episode data (observations, actions, instruction)
        """
        # Get zip file from cache
        zf = self.zip_cache.get(metadata.zip_path)

        # Construct paths within zip
        stem = f"{metadata.robot_type}_{metadata.variant}"
        episode_name = f"episode{metadata.episode_idx}"

        hdf5_path = f"{stem}/data/{episode_name}.hdf5"
        instruction_path = f"{stem}/instructions/{episode_name}.json"

        # Load HDF5 data
        hdf5_bytes = zf.read(hdf5_path)
        hdf5_file = h5py.File(io.BytesIO(hdf5_bytes), "r")

        # Load instruction
        instruction_bytes = zf.read(instruction_path)
        instruction_data = json.loads(instruction_bytes.decode("utf-8"))

        # Sample random instruction variation
        instruction = random.choice(
            instruction_data["seen"] + instruction_data["unseen"]
        )

        # Extract data from HDF5
        data = {
            "task": metadata.task_name,
            "instruction": instruction,
            "robot_type": metadata.robot_type,
            # Actions (14-dim: 6 joints × 2 arms + 2 grippers)
            "actions": np.array(hdf5_file["joint_action"]["vector"]),
            # End effector poses
            "left_endpose": np.array(hdf5_file["endpose"]["left_endpose"]),
            "right_endpose": np.array(hdf5_file["endpose"]["right_endpose"]),
            "left_gripper": np.array(hdf5_file["endpose"]["left_gripper"]),
            "right_gripper": np.array(hdf5_file["endpose"]["right_gripper"]),
            # Camera info (we'll decode images on demand)
            "head_camera_rgb": hdf5_file["observation"]["head_camera"]["rgb"],
            "left_camera_rgb": hdf5_file["observation"]["left_camera"]["rgb"],
            "right_camera_rgb": hdf5_file["observation"]["right_camera"]["rgb"],
            # Keep h5py file open for lazy image loading
            "_hdf5_file": hdf5_file,
        }

        return data

    def decode_image(self, compressed_bytes: bytes) -> np.ndarray:
        """Decode compressed image bytes to numpy array."""
        image = Image.open(io.BytesIO(compressed_bytes))
        return np.array(image)

    def sample_subsequence(self, episode_data: dict) -> dict:
        """Sample a subsequence from the episode."""
        num_timesteps = len(episode_data["actions"])

        # Random start point
        if num_timesteps > self.sequence_length:
            start_idx = random.randint(0, num_timesteps - self.sequence_length)
            end_idx = start_idx + self.sequence_length
        else:
            start_idx = 0
            end_idx = num_timesteps

        # Extract subsequence
        subseq = {
            "task": episode_data["task"],
            "instruction": episode_data["instruction"],
            "robot_type": episode_data["robot_type"],
            "actions": episode_data["actions"][start_idx:end_idx],
            "left_endpose": episode_data["left_endpose"][start_idx:end_idx],
            "right_endpose": episode_data["right_endpose"][start_idx:end_idx],
            "left_gripper": episode_data["left_gripper"][start_idx:end_idx],
            "right_gripper": episode_data["right_gripper"][start_idx:end_idx],
        }

        # Decode images for this subsequence (lazy loading)
        subseq["head_camera_rgb"] = np.stack(
            [
                self.decode_image(episode_data["head_camera_rgb"][i])
                for i in range(start_idx, end_idx)
            ]
        )
        subseq["left_camera_rgb"] = np.stack(
            [
                self.decode_image(episode_data["left_camera_rgb"][i])
                for i in range(start_idx, end_idx)
            ]
        )
        subseq["right_camera_rgb"] = np.stack(
            [
                self.decode_image(episode_data["right_camera_rgb"][i])
                for i in range(start_idx, end_idx)
            ]
        )

        # Close HDF5 file
        episode_data["_hdf5_file"].close()

        return subseq

    def get_batch(self) -> List[dict]:
        """
        Generate one batch of data.

        Returns:
            List of episode subsequences
        """
        # Sample episodes
        episode_metadatas = self.index.sample_episodes(
            self.batch_size, strategy=self.sampling_strategy
        )

        batch = []
        for metadata in episode_metadatas:
            try:
                # Load full episode
                episode_data = self.load_episode_data(metadata)

                # Sample subsequence
                subseq = self.sample_subsequence(episode_data)
                batch.append(subseq)

            except Exception as e:
                print(
                    f"Warning: Failed to load {metadata.task_name}/episode{metadata.episode_idx}: {e}"
                )
                continue

        return batch

    def __iter__(self) -> Iterator[List[dict]]:
        """Iterate indefinitely over batches."""
        while True:
            yield self.get_batch()

    def close(self):
        """Clean up resources."""
        self.zip_cache.close_all()


# Example usage
if __name__ == "__main__":
    # Build index (fast - just scans zip metadata)
    print("Step 1: Building index...")
    index = RoboTwinDatasetIndex("/mnt/robotwin/dataset")

    # Optional: filter to specific robots/variants/tasks
    index.build_index(
        robot_types=["arx-x5"],  # Only arx-x5 robot
        variants=["clean_50"],  # Only clean_50 variant
        # tasks=["adjust_bottle", "click_bell"]  # Optional: specific tasks
    )

    # Print statistics
    print("\nStep 2: Dataset statistics")
    stats = index.get_statistics()
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Tasks: {stats['num_tasks']}")
    print(f"  Robot types: {stats['robot_types']}")

    # Create batch loader
    print("\nStep 3: Creating batch loader...")
    loader = RoboTwinBatchLoader(
        index=index,
        batch_size=8,
        sequence_length=10,
        sampling_strategy="task_diverse",  # Maximize task diversity
        cache_size=10,  # Keep 10 zip files open
    )

    # Generate a few batches
    print("\nStep 4: Generating sample batches...")
    for i, batch in enumerate(loader):
        if i >= 3:  # Just show 3 batches
            break

        print(f"\nBatch {i+1}:")
        print(f"  Batch size: {len(batch)}")

        for j, episode in enumerate(batch):
            print(f"  Episode {j+1}:")
            print(f"    Task: {episode['task']}")
            print(f"    Instruction: {episode['instruction'][:80]}...")
            print(f"    Robot: {episode['robot_type']}")
            print(f"    Actions shape: {episode['actions'].shape}")
            print(f"    Head camera shape: {episode['head_camera_rgb'].shape}")

    loader.close()
    print("\nDone!")
