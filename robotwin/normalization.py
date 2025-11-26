"""
Normalization utilities for robot states and delta actions.

Maps values to [-1, 1] range using pre-computed 1% and 99% percentiles.
Values outside [q01, q99] are clamped to ensure they stay within [-1, 1].
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


class MultiRobotNormalizer:
    """
    Normalizer that handles multiple robot types with different DoF.

    Each robot type has its own normalization statistics (q01, q99 percentiles).
    Normalizes values to [-1, 1] range and handles denormalization.
    """

    def __init__(self, stats_path: str):
        """
        Initialize normalizer with statistics from JSON file.

        Args:
            stats_path: Path to JSON file containing normalization statistics
        """
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        self.robot_types = list(self.stats.keys())

        # Convert lists to numpy arrays for faster operations
        for robot_type in self.robot_types:
            for category in ["state", "delta_actions", "grippers"]:
                self.stats[robot_type][category]["q01"] = np.array(
                    self.stats[robot_type][category]["q01"]
                )
                self.stats[robot_type][category]["q99"] = np.array(
                    self.stats[robot_type][category]["q99"]
                )

    def normalize_state(
        self,
        state: np.ndarray,
        robot_type: str,
        return_torch: bool = False
    ) -> np.ndarray | torch.Tensor:
        """
        Normalize robot state to [-1, 1] range.

        Args:
            state: Robot state array, shape (2*dof,) or (batch, 2*dof)
            robot_type: Type of robot (e.g., "franka", "aloha-agilex")
            return_torch: If True, return torch.Tensor instead of np.ndarray

        Returns:
            Normalized state in [-1, 1] range
        """
        q01 = self.stats[robot_type]["state"]["q01"]
        q99 = self.stats[robot_type]["state"]["q99"]

        # Normalize to [-1, 1]
        # (value - q01) / (q99 - q01) maps [q01, q99] to [0, 1]
        # Then * 2 - 1 maps [0, 1] to [-1, 1]
        normalized = 2.0 * (state - q01) / (q99 - q01 + 1e-8) - 1.0

        if return_torch:
            return torch.from_numpy(normalized).float()
        return normalized

    def denormalize_state(
        self,
        normalized_state: np.ndarray | torch.Tensor,
        robot_type: str
    ) -> np.ndarray:
        """
        Denormalize robot state from [-1, 1] range back to original scale.

        Args:
            normalized_state: Normalized state in [-1, 1] range
            robot_type: Type of robot

        Returns:
            Denormalized state in original scale
        """
        if isinstance(normalized_state, torch.Tensor):
            normalized_state = normalized_state.cpu().numpy()

        q01 = self.stats[robot_type]["state"]["q01"]
        q99 = self.stats[robot_type]["state"]["q99"]

        # Reverse: [-1, 1] -> [0, 1] -> [q01, q99]
        denormalized = (normalized_state + 1.0) / 2.0 * (q99 - q01) + q01
        return denormalized

    def normalize_delta_actions(
        self,
        delta_actions: np.ndarray,
        robot_type: str,
        return_torch: bool = False
    ) -> np.ndarray | torch.Tensor:
        """
        Normalize delta actions to [-1, 1] range.

        Args:
            delta_actions: Delta action array, shape (action_horizon, 2*dof) or (batch, action_horizon, 2*dof)
            robot_type: Type of robot
            return_torch: If True, return torch.Tensor instead of np.ndarray

        Returns:
            Normalized delta actions in [-1, 1] range
        """
        q01 = self.stats[robot_type]["delta_actions"]["q01"]
        q99 = self.stats[robot_type]["delta_actions"]["q99"]

        # Normalize to [-1, 1]
        normalized = 2.0 * (delta_actions - q01) / (q99 - q01 + 1e-8) - 1.0

        if return_torch:
            return torch.from_numpy(normalized).float()
        return normalized

    def denormalize_delta_actions(
        self,
        normalized_deltas: np.ndarray | torch.Tensor,
        robot_type: str
    ) -> np.ndarray:
        """
        Denormalize delta actions from [-1, 1] range back to original scale.

        Args:
            normalized_deltas: Normalized delta actions in [-1, 1] range
            robot_type: Type of robot

        Returns:
            Denormalized delta actions in original scale
        """
        if isinstance(normalized_deltas, torch.Tensor):
            normalized_deltas = normalized_deltas.cpu().numpy()

        q01 = self.stats[robot_type]["delta_actions"]["q01"]
        q99 = self.stats[robot_type]["delta_actions"]["q99"]

        # Reverse normalization
        denormalized = (normalized_deltas + 1.0) / 2.0 * (q99 - q01) + q01
        return denormalized

    def normalize_grippers(
        self,
        grippers: np.ndarray,
        robot_type: str,
        return_torch: bool = False
    ) -> np.ndarray | torch.Tensor:
        """
        Normalize gripper states to [-1, 1] range.

        Args:
            grippers: Gripper state array, shape (2,) or (batch, 2)
            robot_type: Type of robot
            return_torch: If True, return torch.Tensor instead of np.ndarray

        Returns:
            Normalized gripper states in [-1, 1] range
        """
        q01 = self.stats[robot_type]["grippers"]["q01"]
        q99 = self.stats[robot_type]["grippers"]["q99"]

        # Normalize to [-1, 1]
        normalized = 2.0 * (grippers - q01) / (q99 - q01 + 1e-8) - 1.0

        if return_torch:
            return torch.from_numpy(normalized).float()
        return normalized

    def denormalize_grippers(
        self,
        normalized_grippers: np.ndarray | torch.Tensor,
        robot_type: str
    ) -> np.ndarray:
        """
        Denormalize gripper states from [-1, 1] range back to original scale.

        Args:
            normalized_grippers: Normalized gripper states in [-1, 1] range
            robot_type: Type of robot

        Returns:
            Denormalized gripper states in original scale
        """
        if isinstance(normalized_grippers, torch.Tensor):
            normalized_grippers = normalized_grippers.cpu().numpy()

        q01 = self.stats[robot_type]["grippers"]["q01"]
        q99 = self.stats[robot_type]["grippers"]["q99"]

        # Reverse normalization
        denormalized = (normalized_grippers + 1.0) / 2.0 * (q99 - q01) + q01
        return denormalized

    def get_robot_metadata(self, robot_type: str) -> Dict[str, Any]:
        """
        Get metadata for a specific robot type.

        Args:
            robot_type: Type of robot

        Returns:
            Dictionary with dof, action_horizon, num_samples
        """
        return self.stats[robot_type]["metadata"]


def discretize_normalized_values(
    normalized_values: np.ndarray | torch.Tensor,
    num_bins: int = 256
) -> np.ndarray:
    """
    Discretize normalized values in [-1, 1] to integer bins [0, num_bins-1].

    Values outside [-1, 1] are clamped to the extreme bins.
    This is used for converting robot states to discrete text tokens.

    Args:
        normalized_values: Normalized values in [-1, 1] range
        num_bins: Number of discrete bins (default: 256)

    Returns:
        Integer array with values in [0, num_bins-1]
    """
    if isinstance(normalized_values, torch.Tensor):
        normalized_values = normalized_values.cpu().numpy()

    # Clamp to [-1, 1] first
    clamped = np.clip(normalized_values, -1.0, 1.0)

    # Map [-1, 1] to [0, num_bins-1]
    # [-1, 1] -> [0, 1] -> [0, num_bins-1]
    discretized = ((clamped + 1.0) / 2.0 * (num_bins - 1)).astype(np.int32)

    # Ensure values are in valid range (handle edge cases)
    discretized = np.clip(discretized, 0, num_bins - 1)

    return discretized


def undiscretize_to_normalized(
    discretized_values: np.ndarray,
    num_bins: int = 256
) -> np.ndarray:
    """
    Convert discrete bins [0, num_bins-1] back to normalized values in [-1, 1].

    Args:
        discretized_values: Integer array with values in [0, num_bins-1]
        num_bins: Number of discrete bins (default: 256)

    Returns:
        Normalized values in [-1, 1] range
    """
    # Map [0, num_bins-1] -> [0, 1] -> [-1, 1]
    normalized = (discretized_values.astype(np.float32) / (num_bins - 1)) * 2.0 - 1.0
    return normalized
