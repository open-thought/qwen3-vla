"""
Unified sample format and specifications for multi-dataset VLA training.

This module defines the common data structures used across all dataset loaders
to ensure compatibility in the training pipeline.
"""

from dataclasses import dataclass, field
from enum import IntFlag, auto
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch


class ActiveComponents(IntFlag):
    """
    Bit flags indicating which robot components have significant motion.

    Used for selective prediction to only encode/decode moving parts.
    """
    IDLE = 0
    LEFT_ARM = auto()       # 1
    LEFT_GRIPPER = auto()   # 2
    RIGHT_ARM = auto()      # 4
    RIGHT_GRIPPER = auto()  # 8

    # Common combinations for convenience
    LEFT = LEFT_ARM | LEFT_GRIPPER            # 3
    RIGHT = RIGHT_ARM | RIGHT_GRIPPER         # 12
    BOTH_ARMS = LEFT_ARM | RIGHT_ARM          # 5
    BOTH_GRIPPERS = LEFT_GRIPPER | RIGHT_GRIPPER  # 10
    ALL = LEFT | RIGHT                        # 15

    def to_component_name(self) -> str:
        """Convert active components to selective tokenizer component name."""
        # Map to component token names used in selective_tokenizer.py
        mapping = {
            ActiveComponents.IDLE: "idle",
            ActiveComponents.LEFT_ARM: "left_arm",
            ActiveComponents.LEFT_GRIPPER: "left_gripper",
            ActiveComponents.RIGHT_ARM: "right_arm",
            ActiveComponents.RIGHT_GRIPPER: "right_gripper",
            ActiveComponents.LEFT: "left_arm_gripper",
            ActiveComponents.RIGHT: "right_arm_gripper",
            ActiveComponents.BOTH_ARMS: "both_arms",
            ActiveComponents.ALL: "both_arms_grippers",
        }
        return mapping.get(self, "both_arms_grippers")


@dataclass
class RobotStateSpec:
    """
    Specification for robot state observation (may include read-only values).

    Each field specifies the number of dimensions for that component,
    or None if not present for this robot.

    IMPORTANT: When constructing state vectors, components MUST appear in the
    order defined here (top to bottom). Use get_slice_indices() to get the
    correct index ranges for each component.

    Component Order:
        1. joint_positions
        2. joint_velocities
        3. joint_efforts
        4. eef_positions
        5. eef_rotations
        6. gripper_states
        7. additional_sensors
    """
    joint_positions: Optional[int] = None       # Number of joint position dims
    joint_velocities: Optional[int] = None      # Number of joint velocity dims (read-only)
    joint_efforts: Optional[int] = None         # Number of effort/torque dims (read-only)
    eef_positions: Optional[int] = None         # EEF position dims (3D per arm)
    eef_rotations: Optional[int] = None         # EEF rotation dims (quaternion per arm)
    gripper_states: Optional[int] = None        # Gripper openness dims
    additional_sensors: Optional[int] = None    # IMU, force sensors, etc.

    # Define the canonical order of components
    COMPONENT_ORDER: tuple[str, ...] = (
        'joint_positions', 'joint_velocities', 'joint_efforts',
        'eef_positions', 'eef_rotations', 'gripper_states',
        'additional_sensors'
    )

    @property
    def total_dim(self) -> int:
        """Total dimensionality of the state vector."""
        return sum(v for v in [getattr(self, name) for name in self.COMPONENT_ORDER] if v is not None)

    def get_slice_indices(self) -> OrderedDict[str, tuple[int, int]]:
        """
        Get (start, end) indices for each component in the flattened state vector.

        Returns:
            OrderedDict mapping component name to (start_idx, end_idx).
            Only includes components that are present (not None).
        """
        indices = OrderedDict()
        current = 0
        for name in self.COMPONENT_ORDER:
            dim = getattr(self, name)
            if dim is not None:
                indices[name] = (current, current + dim)
                current += dim
        return indices

    def get_component_dims(self) -> OrderedDict[str, int]:
        """Get dimensions of each present component in order."""
        dims = OrderedDict()
        for name in self.COMPONENT_ORDER:
            dim = getattr(self, name)
            if dim is not None:
                dims[name] = dim
        return dims

    def extract_component(self, state: np.ndarray, component: str) -> Optional[np.ndarray]:
        """
        Extract a specific component from a state vector.

        Args:
            state: State vector of shape (..., total_dim)
            component: Component name (e.g., 'joint_positions')

        Returns:
            Component values of shape (..., component_dim) or None if not present
        """
        indices = self.get_slice_indices()
        if component not in indices:
            return None
        start, end = indices[component]
        return state[..., start:end]

    def validate_state(self, state: np.ndarray) -> bool:
        """Check if state vector has correct total dimension."""
        return state.shape[-1] == self.total_dim


@dataclass
class RobotActionSpec:
    """
    Specification for robot action commands (controllable values only).

    Unlike state, actions only include values that can be directly commanded.

    IMPORTANT: When constructing action vectors, components MUST appear in the
    order defined here (top to bottom). Use get_slice_indices() to get the
    correct index ranges for each component.

    Component Order:
        1. joint_targets
        2. eef_position_deltas
        3. eef_rotation_deltas
        4. gripper_commands
    """
    joint_targets: Optional[int] = None         # Joint position targets
    eef_position_deltas: Optional[int] = None   # EEF position deltas (3D per arm)
    eef_rotation_deltas: Optional[int] = None   # EEF rotation deltas (quaternion per arm)
    gripper_commands: Optional[int] = None      # Gripper commands

    # Define the canonical order of components
    COMPONENT_ORDER: tuple[str, ...] = (
        'joint_targets', 'eef_position_deltas',
        'eef_rotation_deltas', 'gripper_commands'
    )

    @property
    def total_dim(self) -> int:
        """Total dimensionality of the action vector."""
        return sum(v for v in [getattr(self, name) for name in self.COMPONENT_ORDER] if v is not None)

    def get_slice_indices(self) -> OrderedDict[str, tuple[int, int]]:
        """Get (start, end) indices for each component in the flattened action vector."""
        indices = OrderedDict()
        current = 0
        for name in self.COMPONENT_ORDER:
            dim = getattr(self, name)
            if dim is not None:
                indices[name] = (current, current + dim)
                current += dim
        return indices

    def get_component_dims(self) -> OrderedDict[str, int]:
        """Get dimensions of each present component in order."""
        dims = OrderedDict()
        for name in self.COMPONENT_ORDER:
            dim = getattr(self, name)
            if dim is not None:
                dims[name] = dim
        return dims

    def extract_component(self, actions: np.ndarray, component: str) -> Optional[np.ndarray]:
        """
        Extract a specific component from an action vector.

        Args:
            actions: Action vector of shape (..., total_dim)
            component: Component name (e.g., 'joint_targets')

        Returns:
            Component values of shape (..., component_dim) or None if not present
        """
        indices = self.get_slice_indices()
        if component not in indices:
            return None
        start, end = indices[component]
        return actions[..., start:end]

    def validate_actions(self, actions: np.ndarray) -> bool:
        """Check if action vector has correct total dimension."""
        return actions.shape[-1] == self.total_dim


@dataclass
class UnifiedSample:
    """
    Unified sample format for all VLA datasets.

    This dataclass provides a common interface that all dataset loaders
    must produce, ensuring compatibility with the training pipeline.
    """

    # === Images (variable number of cameras) ===
    images: dict[str, torch.Tensor]  # {"cam_name": (C, H, W) tensor}
    # Standard camera names: "head", "left_wrist", "right_wrist", "overview"

    # === Text ===
    task_description: str               # Episode-level task description
    subtask_description: Optional[str]  # Subtask instruction (if available)
    robot_type: str                     # Robot platform/embodiment identifier

    # === State (observation) ===
    state: np.ndarray                   # Full robot state observation (raw)
    state_normalized: np.ndarray        # Normalized to [-1, 1]
    state_spec: Optional[RobotStateSpec] = None  # Spec describing state layout
    state_history: Optional[np.ndarray] = None   # (K, state_dim) past states if available

    # === Actions ===
    action_tokens: list[int] = field(default_factory=list)  # Tokenized action sequence
    action_type: str = "joint_delta"    # "joint_delta" or "eef_pose_delta"
    normalized_actions: Optional[np.ndarray] = None  # (H, action_dim) normalized
    action_spec: Optional[RobotActionSpec] = None    # Spec describing action layout
    action_horizon: int = 1             # Number of future timesteps in this sample

    # === Progress/Boundary Information ===
    episode_frame_idx: int = 0          # Current frame index in episode
    episode_total_frames: int = 1       # Total frames in episode
    subtask_frame_idx: Optional[int] = None     # Frame index within subtask
    subtask_total_frames: Optional[int] = None  # Total frames in subtask
    progress_percent: int = 0           # 0-100 progress within (sub)task, rounded to 10s
    is_subtask_end: bool = False        # True if this is the last frame of subtask
    is_episode_end: bool = False        # True if this is the last frame of episode

    # === Metadata ===
    dataset_name: str = ""              # "robotwin", "robocoin", "lerobot"
    episode_id: str = ""                # Unique episode identifier
    active_components: ActiveComponents = ActiveComponents.ALL  # Bit flags for selective prediction

    def get_active_instruction(self) -> str:
        """Get the most specific instruction available (subtask if present, else task)."""
        return self.subtask_description or self.task_description

    def has_subtask_info(self) -> bool:
        """Check if subtask-level information is available."""
        return self.subtask_frame_idx is not None and self.subtask_total_frames is not None

    def get_state_component(self, component: str) -> Optional[np.ndarray]:
        """Extract a specific component from the state vector."""
        if self.state_spec is None:
            raise ValueError("state_spec not set, cannot extract component")
        return self.state_spec.extract_component(self.state, component)

    def get_action_component(self, component: str) -> Optional[np.ndarray]:
        """Extract a specific component from the action vector."""
        if self.action_spec is None:
            raise ValueError("action_spec not set, cannot extract component")
        if self.normalized_actions is None:
            return None
        return self.action_spec.extract_component(self.normalized_actions, component)


def compute_progress(
    frame_idx: int,
    total_frames: int,
    subtask_frame_idx: Optional[int] = None,
    subtask_total_frames: Optional[int] = None,
) -> int:
    """
    Compute progress percentage (0-100), rounded to nearest 10%.

    If subtask info available, use subtask progress.
    Otherwise, use episode progress.

    Args:
        frame_idx: Current frame index in episode
        total_frames: Total frames in episode
        subtask_frame_idx: Optional frame index within subtask
        subtask_total_frames: Optional total frames in subtask

    Returns:
        Progress percentage (0, 10, 20, ..., 100)
    """
    if subtask_frame_idx is not None and subtask_total_frames is not None:
        # Subtask progress
        progress = (subtask_frame_idx / max(subtask_total_frames - 1, 1)) * 100
    else:
        # Episode progress
        progress = (frame_idx / max(total_frames - 1, 1)) * 100

    # Round to nearest 10%
    return min(100, int(round(progress / 10) * 10))
