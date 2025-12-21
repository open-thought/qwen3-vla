"""
Multi-dataset wrapper for combined VLA training.

Supports weighted sampling across multiple datasets with different
robot types, camera configurations, and action spaces.
"""

from typing import Optional, Union
import random

import numpy as np
from torch.utils.data import Dataset, Sampler

from .base_dataset import BaseVLADataset
from .unified_sample import UnifiedSample


class MultiDatasetWrapper(Dataset):
    """
    Combines multiple VLA datasets with configurable sampling weights.

    Supports:
    - Weighted sampling across datasets
    - Per-dataset normalization
    - Unified output format (UnifiedSample)
    - Dataset-aware batch sampling
    """

    def __init__(
        self,
        datasets: dict[str, BaseVLADataset],
        sampling_weights: Optional[dict[str, float]] = None,
        sampling_strategy: str = "weighted",
    ):
        """
        Args:
            datasets: Dict mapping dataset name to dataset instance
            sampling_weights: Optional dict mapping dataset name to sampling weight
                             (will be normalized to sum to 1.0)
            sampling_strategy: Sampling strategy:
                - "weighted": Sample according to weights
                - "uniform": Equal probability for each dataset
                - "proportional": Proportional to dataset size
        """
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())

        if not self.datasets:
            raise ValueError("At least one dataset must be provided")

        # Compute sampling probabilities
        self.weights = self._compute_weights(sampling_weights, sampling_strategy)
        print(f"MultiDatasetWrapper initialized:")
        print(f"  Datasets: {self.dataset_names}")
        print(f"  Weights: {self.weights}")

        # Build unified index mapping global idx -> (dataset_name, local_idx)
        self._build_index()

    def _compute_weights(
        self,
        sampling_weights: Optional[dict[str, float]],
        strategy: str,
    ) -> dict[str, float]:
        """Compute normalized sampling weights."""
        if strategy == "proportional":
            total_samples = sum(len(d) for d in self.datasets.values())
            weights = {
                name: len(d) / total_samples
                for name, d in self.datasets.items()
            }
        elif strategy == "uniform":
            n = len(self.datasets)
            weights = {name: 1.0 / n for name in self.datasets}
        else:  # weighted
            if sampling_weights is None:
                # Default to uniform
                n = len(self.datasets)
                weights = {name: 1.0 / n for name in self.datasets}
            else:
                # Normalize provided weights
                total = sum(sampling_weights.values())
                weights = {
                    name: sampling_weights.get(name, 0.0) / total
                    for name in self.datasets
                }

        return weights

    def _build_index(self):
        """Build mapping from global index to (dataset_name, local_index)."""
        self._index_map = []
        self._dataset_offsets = {}

        offset = 0
        for name, dataset in self.datasets.items():
            self._dataset_offsets[name] = offset
            for local_idx in range(len(dataset)):
                self._index_map.append((name, local_idx))
            offset += len(dataset)

        self._total_samples = len(self._index_map)

    def __len__(self) -> int:
        return self._total_samples

    def __getitem__(self, idx: int) -> UnifiedSample:
        """Get sample by global index."""
        dataset_name, local_idx = self._index_map[idx]
        return self.datasets[dataset_name][local_idx]

    def sample_by_weight(self) -> UnifiedSample:
        """Sample a random sample according to dataset weights."""
        # Choose dataset
        dataset_name = random.choices(
            self.dataset_names,
            weights=[self.weights[n] for n in self.dataset_names],
            k=1,
        )[0]

        # Choose random sample from dataset
        dataset = self.datasets[dataset_name]
        local_idx = random.randint(0, len(dataset) - 1)

        return dataset[local_idx]

    def get_dataset_stats(self) -> dict:
        """Get statistics about all datasets."""
        stats = {
            "total_samples": self._total_samples,
            "num_datasets": len(self.datasets),
            "datasets": {},
        }

        for name, dataset in self.datasets.items():
            ds_stats = dataset.get_stats() if hasattr(dataset, "get_stats") else {}
            ds_stats["num_samples"] = len(dataset)
            ds_stats["weight"] = self.weights[name]
            stats["datasets"][name] = ds_stats

        return stats


class WeightedMultiDatasetSampler(Sampler):
    """
    Sampler that samples from multiple datasets according to weights.

    Unlike the default sampler that goes through indices sequentially,
    this sampler respects the dataset weights for each sample.
    """

    def __init__(
        self,
        multi_dataset: MultiDatasetWrapper,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        """
        Args:
            multi_dataset: MultiDatasetWrapper instance
            num_samples: Number of samples per epoch (default: total samples)
            replacement: Whether to sample with replacement
        """
        self.multi_dataset = multi_dataset
        self.num_samples = num_samples or len(multi_dataset)
        self.replacement = replacement

        # Compute per-sample weights
        self._sample_weights = self._compute_sample_weights()

    def _compute_sample_weights(self) -> np.ndarray:
        """Compute weight for each sample based on dataset weights."""
        weights = np.zeros(len(self.multi_dataset))

        for name, dataset in self.multi_dataset.datasets.items():
            dataset_weight = self.multi_dataset.weights[name]
            offset = self.multi_dataset._dataset_offsets[name]
            n_samples = len(dataset)

            # Each sample in this dataset gets weight = dataset_weight / n_samples
            # This ensures total weight for dataset = dataset_weight
            if n_samples > 0:
                per_sample_weight = dataset_weight / n_samples
                weights[offset:offset + n_samples] = per_sample_weight

        # Normalize
        weights = weights / weights.sum()
        return weights

    def __iter__(self):
        indices = np.random.choice(
            len(self.multi_dataset),
            size=self.num_samples,
            replace=self.replacement,
            p=self._sample_weights,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def create_multi_dataset(
    config: dict,
    norm_stats_dir: Optional[str] = None,
) -> MultiDatasetWrapper:
    """
    Factory function to create MultiDatasetWrapper from config.

    Config format:
    ```yaml
    datasets:
      robotwin:
        type: "robotwin"
        root: "/path/to/robotwin/data"
        norm_stats: "data/robotwin_norm_stats.json"
        weight: 0.4
        action_type: "mixed"

      robocoin:
        type: "robocoin"
        root: "/path/to/robocoin"
        weight: 0.4

      libero:
        type: "lerobot"
        repo_id: "HuggingFaceVLA/libero"
        weight: 0.2
    ```

    Args:
        config: Configuration dict with datasets section
        norm_stats_dir: Base directory for normalization stats files

    Returns:
        MultiDatasetWrapper instance
    """
    from .robotwin_dataset import RoboTwinDataset
    from .robocoin_dataset import RoboCOINDataset
    from .lerobot_dataset import LeRobotVLADataset

    datasets = {}
    weights = {}

    datasets_config = config.get("datasets", {})

    for name, ds_config in datasets_config.items():
        ds_type = ds_config.get("type", name)
        weight = ds_config.get("weight", 1.0)

        # Get common params
        common_params = {
            "action_horizon": config.get("action_horizon", 8),
            "image_size": tuple(config.get("image_size", [320, 240])),
            "action_type": ds_config.get("action_type", config.get("action_type", "joint_delta")),
            "idle_action_filter": config.get("idle_action_filter", False),
            "idle_threshold": config.get("idle_threshold", 0.01),
            "state_history_len": config.get("state_history_len", 0),
        }

        # Get norm stats path
        norm_stats = ds_config.get("norm_stats")
        if norm_stats and norm_stats_dir:
            from pathlib import Path
            norm_stats = str(Path(norm_stats_dir) / norm_stats)

        if ds_type == "robotwin":
            dataset = RoboTwinDataset(
                dataset_root=ds_config["root"],
                norm_stats_path=norm_stats,
                robot_type=ds_config.get("robot_type", "aloha-agilex"),
                tasks=ds_config.get("tasks"),
                episode_filter=ds_config.get("episodes"),
                **common_params,
            )
        elif ds_type == "robocoin":
            dataset = RoboCOINDataset(
                dataset_root=ds_config["root"],
                norm_stats_path=norm_stats,
                task_filter=ds_config.get("tasks"),
                robot_filter=ds_config.get("robots"),
                max_episodes_per_task=ds_config.get("max_episodes_per_task"),
                **common_params,
            )
        elif ds_type == "lerobot":
            dataset = LeRobotVLADataset(
                repo_id=ds_config["repo_id"],
                norm_stats_path=norm_stats,
                episodes=ds_config.get("episodes"),
                split=ds_config.get("split"),
                robot_type=ds_config.get("robot_type"),
                **common_params,
            )
        else:
            raise ValueError(f"Unknown dataset type: {ds_type}")

        datasets[name] = dataset
        weights[name] = weight

    # Determine sampling strategy
    strategy = config.get("sampling_strategy", "weighted")

    return MultiDatasetWrapper(
        datasets=datasets,
        sampling_weights=weights,
        sampling_strategy=strategy,
    )
