"""
B-Spline Tokenizer for Robotics Trajectories

This module provides a BSplineTokenizer class for encoding multi-DoF trajectories
as discrete tokens using clamped B-spline representation with uniform quantization.

Key features:
- Multi-DoF support (e.g., 7-DoF robot arm)
- Configurable B-spline degree and number of control points
- Bounded fitting for predictable quantization range
- Two token ordering modes: 'basis_first' and 'joint_first'
- Uniform quantization (optimal for bounded B-spline control points)

Example usage:
    tokenizer = BSplineTokenizer(
        n_dof=7,
        n_control_points=8,
        degree=4,
        bounds=(-1.5, 1.5),
        n_bins=256,
        token_order='basis_first'
    )

    # Encode trajectory to tokens
    tokens = tokenizer.encode(t_data, trajectory_data)

    # Decode tokens to BSplineTrajectory object
    trajectory = tokenizer.decode(tokens)

    # Evaluate at any normalized time point(s) in [0, 1]
    values = trajectory.evaluate(np.array([0.0, 0.5, 1.0]))

    # Or evaluate at a single point
    value = trajectory(0.5)
"""

import numpy as np
from typing import Tuple, Optional, Literal

from .bspline_trajectory import BSplineTrajectory


class BSplineTokenizer:
    """
    B-Spline based tokenizer for multi-DoF robot trajectories.

    Encodes trajectories as discrete tokens by:
    1. Fitting clamped B-splines to each DoF
    2. Extracting control points
    3. Quantizing control points to integer tokens

    Attributes:
        n_dof: Number of degrees of freedom
        n_control_points: Number of B-spline control points per DoF
        degree: B-spline degree
        bounds: (lower, upper) bounds for control point values
        n_bins: Number of quantization bins
        token_order: 'basis_first' or 'joint_first'
        vocab_size: Total vocabulary size (equals n_bins)
        n_tokens: Total number of tokens per trajectory (n_dof * n_control_points)
    """

    def __init__(
        self,
        n_dof: int = 7,
        n_control_points: int = 8,
        degree: int = 4,
        bounds: Tuple[float, float] = (-1.5, 1.5),
        n_bins: int = 255,
        token_order: Literal['basis_first', 'joint_first'] = 'basis_first'
    ):
        """
        Initialize the B-Spline tokenizer.

        Args:
            n_dof: Number of degrees of freedom (e.g., 7 for a 7-DoF robot arm)
            n_control_points: Number of B-spline control points per DoF
            degree: B-spline polynomial degree (must satisfy n_control_points >= degree + 1)
            bounds: (lower, upper) bounds for control point values during fitting and quantization
            n_bins: Number of quantization bins. Default is 255 (odd) to ensure zero maps
                    exactly to the center bin with symmetric bounds, and can be padded to 256
                    for vocabulary size.
            token_order: Order of tokens in output:
                - 'basis_first': [cp0_j0, cp0_j1, ..., cp0_jN, cp1_j0, cp1_j1, ...]
                  (all joints for basis 0, then all joints for basis 1, etc.)
                - 'joint_first': [cp0_j0, cp1_j0, ..., cpM_j0, cp0_j1, cp1_j1, ...]
                  (all control points for joint 0, then all for joint 1, etc.)
        """
        if n_control_points < degree + 1:
            raise ValueError(f"n_control_points ({n_control_points}) must be >= degree + 1 ({degree + 1})")

        if token_order not in ('basis_first', 'joint_first'):
            raise ValueError(f"token_order must be 'basis_first' or 'joint_first', got '{token_order}'")

        self.n_dof = n_dof
        self.n_control_points = n_control_points
        self.degree = degree
        self.bounds = bounds
        self.n_bins = n_bins
        self.token_order = token_order

        # Derived properties
        self.vocab_size = n_bins
        self.n_tokens = n_dof * n_control_points

    def _quantize(self, values: np.ndarray) -> np.ndarray:
        """
        Quantize continuous values to integer tokens using uniform quantization.

        Args:
            values: Continuous values (should be within self.bounds)

        Returns:
            Integer tokens in range [0, n_bins - 1]
        """
        lower, upper = self.bounds
        normalized = (values - lower) / (upper - lower)
        tokens = np.round(normalized * (self.n_bins - 1)).astype(np.int32)
        return np.clip(tokens, 0, self.n_bins - 1)

    def _dequantize(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert integer tokens back to continuous values.

        Args:
            tokens: Integer tokens in range [0, n_bins - 1]

        Returns:
            Continuous values within self.bounds
        """
        lower, upper = self.bounds
        normalized = tokens.astype(np.float64) / (self.n_bins - 1)
        return normalized * (upper - lower) + lower

    def _control_points_to_tokens(self, control_points: np.ndarray) -> np.ndarray:
        """
        Convert control points matrix to flat token array with specified ordering.

        Args:
            control_points: Array of shape (n_dof, n_control_points)

        Returns:
            Flat array of tokens with length n_dof * n_control_points
        """
        if self.token_order == 'basis_first':
            # Transpose so we iterate over basis functions first
            # Result: [cp0_j0, cp0_j1, ..., cp0_jN, cp1_j0, cp1_j1, ...]
            flat = control_points.T.flatten()
        else:  # joint_first
            # Keep as-is, iterate over joints first
            # Result: [cp0_j0, cp1_j0, ..., cpM_j0, cp0_j1, cp1_j1, ...]
            flat = control_points.flatten()

        return self._quantize(flat)

    def _tokens_to_control_points(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert flat token array back to control points matrix.

        Args:
            tokens: Flat array of tokens with length n_dof * n_control_points

        Returns:
            Array of shape (n_dof, n_control_points)
        """
        values = self._dequantize(tokens)

        if self.token_order == 'basis_first':
            # Reshape to (n_control_points, n_dof) then transpose
            return values.reshape(self.n_control_points, self.n_dof).T
        else:  # joint_first
            return values.reshape(self.n_dof, self.n_control_points)

    def encode(self, t: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
        """
        Encode a trajectory as discrete tokens.

        Args:
            t: Time/parameter values of shape (n_timesteps,), should be in [0, 1]
            trajectory: Trajectory data of shape (n_timesteps, n_dof)

        Returns:
            Integer token array of shape (n_dof * n_control_points,)
        """
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)

        if trajectory.shape[1] != self.n_dof:
            raise ValueError(f"Expected trajectory with {self.n_dof} DoFs, got {trajectory.shape[1]}")

        # Use BSplineTrajectory.fit() with bounds for quantization
        bspline_traj = BSplineTrajectory.fit(
            t, trajectory,
            n_control_points=self.n_control_points,
            degree=self.degree,
            bounds=self.bounds
        )
        return self._control_points_to_tokens(bspline_traj.control_points)

    def decode(self, tokens: np.ndarray) -> BSplineTrajectory:
        """
        Decode tokens to a BSplineTrajectory object.

        Args:
            tokens: Integer token array of shape (n_dof * n_control_points,)

        Returns:
            BSplineTrajectory object that can be evaluated at any time in [0, 1]
        """
        if len(tokens) != self.n_tokens:
            raise ValueError(f"Expected {self.n_tokens} tokens, got {len(tokens)}")

        control_points = self._tokens_to_control_points(tokens)
        return BSplineTrajectory(control_points, degree=self.degree)

    def get_control_points_from_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert tokens to control points without evaluating the trajectory.

        Args:
            tokens: Integer token array of shape (n_dof * n_control_points,)

        Returns:
            Control points array of shape (n_dof, n_control_points)
        """
        return self._tokens_to_control_points(tokens)

    def get_tokens_from_control_points(self, control_points: np.ndarray) -> np.ndarray:
        """
        Convert control points to tokens without fitting.

        Args:
            control_points: Array of shape (n_dof, n_control_points)

        Returns:
            Integer token array of shape (n_dof * n_control_points,)
        """
        return self._control_points_to_tokens(control_points)

    def compute_reconstruction_error(self, t: np.ndarray, trajectory: np.ndarray,
                                      t_eval: Optional[np.ndarray] = None) -> dict:
        """
        Compute reconstruction error metrics.

        Args:
            t: Original time values
            trajectory: Original trajectory
            t_eval: Optional different time values for evaluation (default: use t)

        Returns:
            Dictionary with error metrics
        """
        if t_eval is None:
            t_eval = t

        tokens = self.encode(t, trajectory)
        bspline_traj = self.decode(tokens)
        reconstructed = bspline_traj.evaluate(t_eval)

        if t_eval is t or (len(t_eval) == len(t) and np.allclose(t_eval, t)):
            original = trajectory
        else:
            # Interpolate original to t_eval for comparison
            original = np.zeros_like(reconstructed)
            for dof in range(self.n_dof):
                original[:, dof] = np.interp(t_eval, t, trajectory[:, dof])

        error = reconstructed - original

        return {
            'mae': np.mean(np.abs(error)),
            'max_error': np.max(np.abs(error)),
            'rmse': np.sqrt(np.mean(error ** 2)),
            'mae_per_dof': np.mean(np.abs(error), axis=0),
            'max_error_per_dof': np.max(np.abs(error), axis=0),
        }

    def __repr__(self) -> str:
        return (
            f"BSplineTokenizer(\n"
            f"  n_dof={self.n_dof},\n"
            f"  n_control_points={self.n_control_points},\n"
            f"  degree={self.degree},\n"
            f"  bounds={self.bounds},\n"
            f"  n_bins={self.n_bins},\n"
            f"  token_order='{self.token_order}',\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  n_tokens={self.n_tokens}\n"
            f")"
        )


# Convenience function for quick tokenization
def tokenize_trajectory(
    t: np.ndarray,
    trajectory: np.ndarray,
    n_control_points: int = 8,
    degree: int = 4,
    bounds: Tuple[float, float] = (-1.5, 1.5),
    n_bins: int = 255,
    token_order: Literal['basis_first', 'joint_first'] = 'basis_first'
) -> Tuple[np.ndarray, BSplineTokenizer]:
    """
    Convenience function to tokenize a trajectory.

    Args:
        t: Time values (should be in [0, 1])
        trajectory: Trajectory of shape (n_timesteps, n_dof) or (n_timesteps,)
        n_control_points: Number of B-spline control points
        degree: B-spline degree
        bounds: Control point bounds
        n_bins: Quantization bins
        token_order: Token ordering mode

    Returns:
        Tuple of (tokens, tokenizer)
    """
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)

    n_dof = trajectory.shape[1]

    tokenizer = BSplineTokenizer(
        n_dof=n_dof,
        n_control_points=n_control_points,
        degree=degree,
        bounds=bounds,
        n_bins=n_bins,
        token_order=token_order
    )

    tokens = tokenizer.encode(t, trajectory)

    return tokens, tokenizer


if __name__ == "__main__":
    # Demo / test
    np.random.seed(42)

    # Create a sample 7-DoF trajectory (e.g., robot arm)
    n_timesteps = 50
    n_dof = 7
    t = np.linspace(0, 1, n_timesteps)

    # Generate smooth random trajectories for each joint
    trajectory = np.zeros((n_timesteps, n_dof))
    for dof in range(n_dof):
        # Random smooth trajectory using sine waves
        freq = np.random.uniform(0.5, 2)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.3, 0.8)
        offset = np.random.uniform(-0.3, 0.3)
        trajectory[:, dof] = amplitude * np.sin(2 * np.pi * freq * t + phase) + offset

    print("="*70)
    print("BSplineTokenizer Demo")
    print("="*70)

    # Test both token orders
    for order in ['basis_first', 'joint_first']:
        print(f"\n--- Token order: {order} ---")

        tokenizer = BSplineTokenizer(
            n_dof=n_dof,
            n_control_points=8,
            degree=4,
            bounds=(-1.5, 1.5),
            n_bins=256,
            token_order=order  # type: ignore
        )

        print(tokenizer)

        # Encode
        tokens = tokenizer.encode(t, trajectory)
        print(f"\nTokens shape: {tokens.shape}")
        print(f"Token range: [{tokens.min()}, {tokens.max()}]")
        print(f"First 16 tokens: {tokens[:16]}")

        # Decode to BSplineTrajectory
        bspline_traj = tokenizer.decode(tokens)
        print(f"\nDecoded trajectory: {bspline_traj}")

        # Evaluate at multiple points
        t_eval = np.linspace(0, 1, 100)
        reconstructed = bspline_traj.evaluate(t_eval)
        print(f"Evaluated shape: {reconstructed.shape}")

        # Evaluate at a single point using callable interface
        single_value = bspline_traj(0.5)
        print(f"Value at t=0.5: {single_value[:3]}... (first 3 DoFs)")

        # Compute error
        errors = tokenizer.compute_reconstruction_error(t, trajectory, t_eval)
        print(f"\nReconstruction errors:")
        print(f"  MAE: {errors['mae']:.6f}")
        print(f"  Max error: {errors['max_error']:.6f}")
        print(f"  RMSE: {errors['rmse']:.6f}")

    print("\n" + "="*70)
    print("Demo complete!")
