"""
Action tokenizers for VLA training.

Supports two tokenization methods:
1. BSpline - B-spline based tokenizer for smooth trajectory encoding
2. Bin (OpenVLA-style) - simple uniform discretization, one token per action dimension

Both tokenizers map to extended vocabulary tokens (starting at VOCAB_OFFSET).
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch

from bspline_tokenizer import BSplineTokenizer


class BaseActionTokenizer(ABC):
    """Abstract base class for action tokenizers."""

    # Token range for action tokens in the extended vocabulary
    # Original Qwen3-VL vocab: 151936 tokens (0-151935)
    VOCAB_OFFSET = 151936

    @abstractmethod
    def encode(
        self,
        normalized_actions: np.ndarray | torch.Tensor,
        return_torch: bool = False
    ) -> list[list[int]] | torch.Tensor:
        """Encode normalized actions into discrete tokens."""
        pass

    @abstractmethod
    def decode(
        self,
        tokens: list[list[int]] | torch.Tensor,
        action_horizon: int,
        action_dim: int
    ) -> np.ndarray:
        """Decode tokens back to normalized actions."""
        pass

    @abstractmethod
    def get_token_range(self) -> tuple[int, int]:
        """Get the range of token IDs used by this tokenizer."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get the vocabulary size of this tokenizer."""
        pass


class BinTokenizer(BaseActionTokenizer):
    """
    Simple bin-based action tokenizer (OpenVLA-style).

    Discretizes each action dimension independently into N bins.
    One token per action dimension, so action_horizon * action_dim tokens total.

    For exact zero reconstruction, use n_bins=255 or 257 (odd number) so that
    one bin center falls exactly at 0.
    """

    def __init__(
        self,
        n_bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        """
        Initialize bin tokenizer.

        Args:
            n_bins: Number of bins for discretization (default: 256).
                    Use 257 for exact zero reconstruction (bin center at 0).
            min_action: Minimum action value (default: -1.0)
            max_action: Maximum action value (default: 1.0)
        """
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action

        # Create uniform bin centers directly
        # With n_bins centers from min to max, we get exact 0 when n_bins is odd
        # and min_action = -max_action
        self.bin_centers = np.linspace(min_action, max_action, n_bins)

        # Compute bin edges (boundaries between centers)
        # Each bin spans from midpoint to previous center to midpoint to next center
        half_width = (self.bin_centers[1] - self.bin_centers[0]) / 2.0 if n_bins > 1 else 0.5
        self.bin_edges = np.concatenate([
            [min_action - half_width],  # Left edge of first bin
            (self.bin_centers[:-1] + self.bin_centers[1:]) / 2.0,  # Edges between bins
            [max_action + half_width],  # Right edge of last bin
        ])

        # Vocabulary size is the number of bins
        self._vocab_size = n_bins

        # Check if 0 is a bin center (for informational purposes)
        zero_is_center = any(np.isclose(self.bin_centers, 0.0, atol=1e-10))

        print(f"Initialized BinTokenizer")
        print(f"  Bins: {n_bins}")
        print(f"  Action range: [{min_action}, {max_action}]")
        print(f"  Bin width: {2.0 * half_width:.6f}")
        print(f"  Zero is bin center: {zero_is_center}")
        print(f"  Token range: [{self.VOCAB_OFFSET}, {self.VOCAB_OFFSET + n_bins - 1}]")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(
        self,
        normalized_actions: np.ndarray | torch.Tensor,
        return_torch: bool = False
    ) -> list[list[int]] | torch.Tensor:
        """
        Encode normalized actions into discrete tokens.

        Each action value becomes one token, so output length = action_horizon * action_dim.

        Args:
            normalized_actions: Normalized actions in [-1, 1] range
                Shape: (batch, action_horizon, action_dim) or (action_horizon, action_dim)
            return_torch: If True, return torch.Tensor instead of list

        Returns:
            Token sequences with vocab offset applied
        """
        # Convert to numpy if needed
        if isinstance(normalized_actions, torch.Tensor):
            normalized_actions = normalized_actions.cpu().numpy()

        # Ensure 3D: (batch, action_horizon, action_dim)
        if normalized_actions.ndim == 2:
            normalized_actions = normalized_actions[None, :]

        batch_size, action_horizon, action_dim = normalized_actions.shape

        # Clip to valid range
        clipped = np.clip(normalized_actions, self.min_action, self.max_action)

        # Digitize using bin edges: find which bin each value falls into
        # np.digitize returns indices in [0, n_bins] for bin_edges with n_bins+1 edges
        # We want indices in [0, n_bins-1]
        discretized = np.digitize(clipped, self.bin_edges[1:])  # Skip first edge

        # Clip to valid bin range [0, n_bins-1]
        discretized = np.clip(discretized, 0, self.n_bins - 1)

        # Flatten to (batch, action_horizon * action_dim) and add vocab offset
        tokens_flat = discretized.reshape(batch_size, -1) + self.VOCAB_OFFSET

        # Convert to list of lists
        tokens_list = [list(seq) for seq in tokens_flat]

        if return_torch:
            return torch.tensor(tokens_flat, dtype=torch.long)

        return tokens_list

    def decode(
        self,
        tokens: list[list[int]] | torch.Tensor,
        action_horizon: int,
        action_dim: int
    ) -> np.ndarray:
        """
        Decode tokens back to normalized actions.

        Args:
            tokens: Token sequences with vocab offset
            action_horizon: Number of timesteps in action chunk
            action_dim: Action dimension

        Returns:
            Normalized actions in [-1, 1] range
            Shape: (batch, action_horizon, action_dim)
        """
        # Convert to numpy array
        if isinstance(tokens, torch.Tensor):
            tokens_array = tokens.cpu().numpy()
        else:
            # Pad to same length if needed
            max_len = max(len(seq) for seq in tokens)
            tokens_array = np.zeros((len(tokens), max_len), dtype=np.int64)
            for i, seq in enumerate(tokens):
                tokens_array[i, :len(seq)] = seq

        batch_size = tokens_array.shape[0]
        expected_len = action_horizon * action_dim

        # Remove vocab offset to get bin indices
        bin_indices = tokens_array - self.VOCAB_OFFSET

        # Clip to valid bin range
        bin_indices = np.clip(bin_indices, 0, len(self.bin_centers) - 1)

        # Take only the expected number of tokens
        if bin_indices.shape[1] >= expected_len:
            bin_indices = bin_indices[:, :expected_len]
        else:
            # Pad with zeros (center bin) if not enough tokens
            padded = np.zeros((batch_size, expected_len), dtype=np.int64)
            padded[:, :bin_indices.shape[1]] = bin_indices
            bin_indices = padded

        # Map bin indices to bin centers
        actions_flat = self.bin_centers[bin_indices]

        # Reshape to (batch, action_horizon, action_dim)
        actions = actions_flat.reshape(batch_size, action_horizon, action_dim)

        return actions.astype(np.float32)

    def get_token_range(self) -> tuple[int, int]:
        """Get the range of token IDs used."""
        return (self.VOCAB_OFFSET, self.VOCAB_OFFSET + self.n_bins - 1)


class BSplineActionTokenizer(BaseActionTokenizer):
    """
    Wrapper for B-spline action tokenizer.

    Encodes normalized delta actions (in [-1, 1] range) into discrete tokens
    using B-spline trajectory representation.

    The B-spline tokenizer fits smooth trajectories to the action sequence
    and encodes the control points as tokens. This provides a compact
    representation with smooth interpolation.
    """

    def __init__(
        self,
        n_control_points: int = 8,
        degree: int = 4,
        bounds: tuple[float, float] = (-1.0, 1.0),
        n_bins: int = 255,
        token_order: Literal['basis_first', 'joint_first'] = 'basis_first',
    ):
        """
        Initialize B-spline action tokenizer.

        Args:
            n_control_points: Number of B-spline control points per DoF
            degree: B-spline polynomial degree
            bounds: (lower, upper) bounds for control point values
            n_bins: Number of quantization bins (255 recommended for exact zero with symmetric bounds)
            token_order: Order of tokens in output:
                - 'basis_first': [cp0_j0, cp0_j1, ..., cp0_jN, cp1_j0, ...]
                - 'joint_first': [cp0_j0, cp1_j0, ..., cpM_j0, cp0_j1, ...]
        """
        self.n_control_points = n_control_points
        self.degree = degree
        self.bounds = bounds
        self.n_bins = n_bins
        self.token_order = token_order

        # Vocab size is the number of bins
        self._vocab_size = n_bins

        # We'll create the actual tokenizer lazily since n_dof depends on input
        self._tokenizers: dict[int, BSplineTokenizer] = {}

        print(f"Initialized BSplineActionTokenizer")
        print(f"  Control points: {n_control_points}, Degree: {degree}")
        print(f"  Bounds: {bounds}, Bins: {n_bins}")
        print(f"  Token order: {token_order}")
        print(f"  Token range: [{self.VOCAB_OFFSET}, {self.VOCAB_OFFSET + n_bins - 1}]")

    def _get_tokenizer(self, n_dof: int) -> BSplineTokenizer:
        """Get or create a BSplineTokenizer for the given number of DoFs."""
        if n_dof not in self._tokenizers:
            self._tokenizers[n_dof] = BSplineTokenizer(
                n_dof=n_dof,
                n_control_points=self.n_control_points,
                degree=self.degree,
                bounds=self.bounds,
                n_bins=self.n_bins,
                token_order=self.token_order,
            )
        return self._tokenizers[n_dof]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(
        self,
        normalized_actions: np.ndarray | torch.Tensor,
        return_torch: bool = False
    ) -> list[list[int]] | torch.Tensor:
        """
        Encode normalized actions into discrete tokens.

        Args:
            normalized_actions: Normalized actions in [-1, 1] range
                Shape: (batch, action_horizon, action_dim) or (action_horizon, action_dim)
            return_torch: If True, return torch.Tensor instead of list

        Returns:
            Token sequences with vocab offset applied
        """
        # Convert to numpy if needed
        if isinstance(normalized_actions, torch.Tensor):
            normalized_actions = normalized_actions.cpu().numpy()

        # Ensure 3D: (batch, action_horizon, action_dim)
        if normalized_actions.ndim == 2:
            normalized_actions = normalized_actions[None, :]

        batch_size, action_horizon, action_dim = normalized_actions.shape

        # Get tokenizer for this action_dim
        tokenizer = self._get_tokenizer(action_dim)

        # Create normalized time values for B-spline fitting
        t = np.linspace(0, 1, action_horizon)

        # Encode each sample in the batch
        tokens_with_offset = []
        for i in range(batch_size):
            # Encode trajectory to tokens
            tokens = tokenizer.encode(t, normalized_actions[i])
            # Apply vocabulary offset
            tokens_with_offset.append([int(tok) + self.VOCAB_OFFSET for tok in tokens])

        if return_torch:
            # All sequences should have the same length for BSpline tokenizer
            # (n_control_points * action_dim tokens)
            max_len = max(len(seq) for seq in tokens_with_offset)
            padded_tokens = torch.full(
                (batch_size, max_len),
                self.VOCAB_OFFSET,
                dtype=torch.long
            )
            for i, seq in enumerate(tokens_with_offset):
                padded_tokens[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            return padded_tokens

        return tokens_with_offset

    def decode(
        self,
        tokens: list[list[int]] | torch.Tensor,
        action_horizon: int,
        action_dim: int
    ) -> np.ndarray:
        """
        Decode tokens back to normalized actions.

        Args:
            tokens: Token sequences with vocab offset
            action_horizon: Number of timesteps in action chunk
            action_dim: Action dimension

        Returns:
            Normalized actions in [-1, 1] range
            Shape: (batch, action_horizon, action_dim)
        """
        # Convert tensor to list if needed
        if isinstance(tokens, torch.Tensor):
            tokens_list = []
            for seq in tokens:
                valid_tokens = seq[seq >= self.VOCAB_OFFSET].tolist()
                tokens_list.append(valid_tokens)
            tokens = tokens_list

        # Get tokenizer for this action_dim
        tokenizer = self._get_tokenizer(action_dim)

        # Expected number of tokens
        expected_tokens = self.n_control_points * action_dim

        # Decode each sample
        batch_size = len(tokens)
        decoded_actions = np.zeros((batch_size, action_horizon, action_dim), dtype=np.float32)

        # Create evaluation time points
        t_eval = np.linspace(0, 1, action_horizon)

        for i, token_seq in enumerate(tokens):
            # Remove vocabulary offset
            tokens_no_offset = np.array([t - self.VOCAB_OFFSET for t in token_seq], dtype=np.int32)

            # Handle token length mismatch
            if len(tokens_no_offset) < expected_tokens:
                # Pad with center bin value (should map to 0)
                center_bin = self.n_bins // 2
                tokens_no_offset = np.concatenate([
                    tokens_no_offset,
                    np.full(expected_tokens - len(tokens_no_offset), center_bin, dtype=np.int32)
                ])
            elif len(tokens_no_offset) > expected_tokens:
                tokens_no_offset = tokens_no_offset[:expected_tokens]

            # Decode to BSplineTrajectory and evaluate
            trajectory = tokenizer.decode(tokens_no_offset)
            decoded_actions[i] = trajectory.evaluate(t_eval)

        return decoded_actions

    def get_token_range(self) -> tuple[int, int]:
        """Get the range of token IDs used."""
        return (self.VOCAB_OFFSET, self.VOCAB_OFFSET + self._vocab_size - 1)

    def get_num_tokens(self, action_dim: int) -> int:
        """Get the number of tokens for a given action dimension."""
        return self.n_control_points * action_dim


def create_action_tokenizer(
    tokenizer_type: Literal["bspline", "bin"] = "bspline",
    n_bins: int = 255,
    n_control_points: int = 8,
    degree: int = 4,
    bounds: tuple[float, float] = (-1.0, 1.0),
    token_order: Literal['basis_first', 'joint_first'] = 'basis_first',
    **kwargs
) -> BaseActionTokenizer:
    """
    Factory function to create an action tokenizer.

    Args:
        tokenizer_type: Type of tokenizer to create
            - "bspline": B-spline tokenizer (smooth trajectory encoding)
            - "bin": Simple bin tokenizer (OpenVLA-style, fixed-length output)
        n_bins: Number of bins for quantization (default: 255).
                Use 255 for exact zero reconstruction with symmetric bounds.
        n_control_points: Number of B-spline control points (for bspline tokenizer)
        degree: B-spline polynomial degree (for bspline tokenizer)
        bounds: (lower, upper) bounds for values (for bspline tokenizer)
        token_order: Token ordering mode (for bspline tokenizer)
        **kwargs: Additional arguments passed to the tokenizer constructor

    Returns:
        Action tokenizer instance
    """
    if tokenizer_type == "bspline":
        return BSplineActionTokenizer(
            n_control_points=n_control_points,
            degree=degree,
            bounds=bounds,
            n_bins=n_bins,
            token_order=token_order,
            **kwargs
        )
    elif tokenizer_type == "bin":
        return BinTokenizer(n_bins=n_bins, **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Must be 'bspline' or 'bin'.")


def _test_tokenizer(tokenizer: BaseActionTokenizer, name: str, action_horizon: int = 16, action_dim: int = 14):
    """Test a single tokenizer."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")

    # Create synthetic normalized actions (smooth trajectories for better B-spline fit)
    batch_size = 4
    np.random.seed(42)

    # Generate smooth trajectories using sine waves
    t = np.linspace(0, 1, action_horizon)
    actions = np.zeros((batch_size, action_horizon, action_dim), dtype=np.float32)
    for b in range(batch_size):
        for d in range(action_dim):
            freq = np.random.uniform(0.5, 2)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.3, 0.8)
            offset = np.random.uniform(-0.2, 0.2)
            actions[b, :, d] = amplitude * np.sin(2 * np.pi * freq * t + phase) + offset

    print(f"\nInput actions shape: {actions.shape}")
    print(f"Input actions range: [{actions.min():.4f}, {actions.max():.4f}]")

    # Test encoding
    tokens = tokenizer.encode(actions, return_torch=False)
    print(f"\nEncoded tokens:")
    print(f"  Num sequences: {len(tokens)}")
    print(f"  Sequence lengths: {[len(seq) for seq in tokens]}")
    print(f"  Token range: [{min(min(seq) for seq in tokens)}, {max(max(seq) for seq in tokens)}]")

    # Test decoding
    decoded_actions = tokenizer.decode(tokens, action_horizon=action_horizon, action_dim=action_dim)
    print(f"\nDecoded actions shape: {decoded_actions.shape}")
    print(f"Decoded actions range: [{decoded_actions.min():.4f}, {decoded_actions.max():.4f}]")

    # Check reconstruction error
    reconstruction_error = np.abs(actions - decoded_actions).mean()
    print(f"\nReconstruction error (MAE): {reconstruction_error:.6f}")

    # Expected error thresholds
    # BinTokenizer: ~0.004 (256 bins over [-1,1] = bin width 0.0078)
    # BSplineActionTokenizer: varies based on smoothness, typically < 0.05 for smooth trajectories
    threshold = 0.01 if name == "BinTokenizer" else 0.05

    if reconstruction_error < threshold:
        print(f"✓ Reconstruction error within acceptable range (< {threshold})")
    else:
        print(f"✗ WARNING: Reconstruction error is large: {reconstruction_error}")

    return reconstruction_error


def test_action_tokenizers():
    """Test both action tokenizers."""
    print("Testing Action Tokenizers")
    print("=" * 60)

    action_horizon = 16
    action_dim = 14  # 12 joints + 2 grippers

    # Test BinTokenizer
    bin_tokenizer = BinTokenizer(n_bins=256)
    bin_error = _test_tokenizer(bin_tokenizer, "BinTokenizer", action_horizon, action_dim)

    # Test BSplineActionTokenizer
    bspline_tokenizer = BSplineActionTokenizer(
        n_control_points=8,
        degree=4,
        bounds=(-1.0, 1.0),
        n_bins=255,
    )
    bspline_error = _test_tokenizer(bspline_tokenizer, "BSplineActionTokenizer", action_horizon, action_dim)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"BinTokenizer:          MAE = {bin_error:.6f}, tokens = {action_horizon * action_dim} (fixed)")
    print(f"BSplineActionTokenizer: MAE = {bspline_error:.6f}, tokens = {bspline_tokenizer.get_num_tokens(action_dim)} (fixed)")
    print(f"\n✓ All tokenizer tests completed!")


if __name__ == "__main__":
    test_action_tokenizers()
