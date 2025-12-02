"""
B-Spline Trajectory for Robotics

This module provides the BSplineTrajectory class and B-spline helper functions
for representing and evaluating multi-DoF trajectories using clamped B-splines.

Key features:
- Clamped B-splines that pass through first and last control points
- Multi-DoF support (e.g., 7-DoF robot arm)
- Bounded or unconstrained least-squares fitting
- Efficient evaluation at arbitrary time points

Example usage:
    # Fit a B-spline trajectory to data
    traj = BSplineTrajectory.fit(t, trajectory, n_control_points=8)

    # Evaluate at any normalized time point(s) in [0, 1]
    values = traj.evaluate(np.array([0.0, 0.5, 1.0]))

    # Or evaluate at a single point
    value = traj(0.5)

    # Create from control points directly
    traj = BSplineTrajectory(control_points, degree=4)
"""

import numpy as np
from functools import lru_cache
from typing import Tuple, Optional
from scipy.optimize import lsq_linear


@lru_cache(maxsize=32)
def create_clamped_knot_vector(n_control_points: int, degree: int) -> np.ndarray:
    """
    Create a clamped (open) knot vector for B-splines (memoized).

    For a clamped B-spline:
    - First (degree+1) knots are 0
    - Last (degree+1) knots are 1
    - Interior knots are uniformly spaced

    This ensures the curve passes through the first and last control points.

    Args:
        n_control_points: Number of control points
        degree: Degree of the B-spline (order = degree + 1)

    Returns:
        Knot vector of length (n_control_points + degree + 1)
    """
    n_knots = n_control_points + degree + 1
    n_interior = n_knots - 2 * (degree + 1)

    knots = np.zeros(n_knots)

    if n_interior > 0:
        interior = np.linspace(0, 1, n_interior + 2)[1:-1]
        knots[degree + 1:degree + 1 + n_interior] = interior

    knots[-(degree + 1):] = 1.0

    return knots


def bspline_basis(i: int, degree: int, t: float, knots: np.ndarray) -> float:
    """
    Compute B-spline basis function B_{i,degree}(t) using Cox-de Boor recursion.

    Args:
        i: Index of the basis function
        degree: Degree of the B-spline
        t: Parameter value
        knots: Knot vector

    Returns:
        Value of basis function B_{i,degree}(t)
    """
    if degree == 0:
        if t == knots[-1] and knots[i] <= t <= knots[i + 1]:
            return 1.0
        if knots[i] <= t < knots[i + 1]:
            return 1.0
        return 0.0

    left = 0.0
    right = 0.0

    denom_left = knots[i + degree] - knots[i]
    if denom_left != 0:
        left = (t - knots[i]) / denom_left * bspline_basis(i, degree - 1, t, knots)

    denom_right = knots[i + degree + 1] - knots[i + 1]
    if denom_right != 0:
        right = (knots[i + degree + 1] - t) / denom_right * bspline_basis(i + 1, degree - 1, t, knots)

    return left + right


def bspline_basis_matrix(t_values: np.ndarray, n_control_points: int, degree: int,
                         knots: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the B-spline basis matrix for given parameter values.

    Args:
        t_values: Array of parameter values
        n_control_points: Number of control points
        degree: Degree of the B-spline
        knots: Optional knot vector (created if not provided)

    Returns:
        Matrix of shape (len(t_values), n_control_points) where
        M[i, j] = B_{j,degree}(t_values[i])
    """
    if knots is None:
        knots = create_clamped_knot_vector(n_control_points, degree)

    n_points = len(t_values)
    basis_matrix = np.zeros((n_points, n_control_points))

    for i, t in enumerate(t_values):
        for j in range(n_control_points):
            basis_matrix[i, j] = bspline_basis(j, degree, t, knots)

    return basis_matrix


class BSplineTrajectory:
    """
    A B-spline trajectory that can be evaluated at any time point in [0, 1].

    This class holds the control points and knot vector for a multi-DoF B-spline
    trajectory and provides efficient evaluation at arbitrary time points.

    Attributes:
        control_points: Array of shape (n_dof, n_control_points)
        n_dof: Number of degrees of freedom
        n_control_points: Number of control points per DoF
        degree: B-spline polynomial degree
        knots: knot vector

    Example:
        # Fit directly from trajectory data
        traj = BSplineTrajectory.fit(t, trajectory, n_control_points=8)
        values = traj.evaluate(np.linspace(0, 1, 100))

        # Or create from control points
        traj = BSplineTrajectory(control_points, degree=4)
    """

    def __init__(
        self,
        control_points: np.ndarray,
        degree: int = 4,
        knots: Optional[np.ndarray] = None
    ):
        """
        Initialize a B-spline trajectory.

        Args:
            control_points: Array of shape (n_dof, n_control_points)
            degree: B-spline polynomial degree
            knots: Optional knot vector. If None, creates a clamped knot vector.
        """
        if control_points.ndim == 1:
            control_points = control_points.reshape(1, -1)

        self.control_points = control_points
        self.n_dof = control_points.shape[0]
        self.n_control_points = control_points.shape[1]
        self.degree = degree

        if knots is None:
            self.knots = create_clamped_knot_vector(self.n_control_points, degree)
        else:
            self.knots = knots

    @classmethod
    def fit(
        cls,
        t: np.ndarray,
        trajectory: np.ndarray,
        n_control_points: int = 8,
        degree: int = 4,
        bounds: Optional[Tuple[float, float]] = None
    ) -> 'BSplineTrajectory':
        """
        Fit a B-spline trajectory to data.

        This is a factory method that creates a BSplineTrajectory by fitting
        B-splines to the provided trajectory data.

        Args:
            t: Time/parameter values of shape (n_timesteps,), should be in [0, 1]
            trajectory: Trajectory data of shape (n_timesteps, n_dof) or (n_timesteps,)
            n_control_points: Number of B-spline control points per DoF
            degree: B-spline polynomial degree (must satisfy n_control_points >= degree + 1)
            bounds: Optional (lower, upper) bounds for control point values.
                    If None, uses unconstrained least squares fitting.

        Returns:
            BSplineTrajectory fitted to the data

        Example:
            t = np.linspace(0, 1, 100)
            trajectory = np.sin(2 * np.pi * t).reshape(-1, 1)
            traj = BSplineTrajectory.fit(t, trajectory, n_control_points=8)
        """
        if n_control_points < degree + 1:
            raise ValueError(f"n_control_points ({n_control_points}) must be >= degree + 1 ({degree + 1})")

        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)

        n_dof = trajectory.shape[1]
        knots = create_clamped_knot_vector(n_control_points, degree)
        basis_matrix = bspline_basis_matrix(t, n_control_points, degree, knots)

        control_points = np.zeros((n_dof, n_control_points))

        for dof in range(n_dof):
            if bounds is not None:
                result = lsq_linear(basis_matrix, trajectory[:, dof], bounds=bounds)
                control_points[dof] = result.x
            else:
                # Unconstrained least squares
                control_points[dof], _, _, _ = np.linalg.lstsq(
                    basis_matrix, trajectory[:, dof], rcond=None
                )

        return cls(control_points, degree=degree, knots=knots)

    def evaluate(self, t: float | np.ndarray) -> np.ndarray:
        """
        Evaluate the trajectory at given time point(s).

        Args:
            t: Time/parameter value(s) in [0, 1]. Can be a scalar float, 1D array,
               or any array-like that will be converted to array.

        Returns:
            Trajectory values of shape (len(t), n_dof) if t is array-like,
            or shape (n_dof,) if t is a scalar.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        scalar_input = (t.shape == (1,))

        # Validate time range
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Time values must be in [0, 1], got range [{t.min()}, {t.max()}]")

        basis_matrix = bspline_basis_matrix(t, self.n_control_points, self.degree, self.knots)

        # Evaluate all DoFs: (n_points, n_cp) @ (n_cp, n_dof).T -> need different approach
        # control_points is (n_dof, n_control_points)
        # basis_matrix is (n_points, n_control_points)
        # Result should be (n_points, n_dof)
        trajectory = basis_matrix @ self.control_points.T

        if scalar_input:
            return trajectory[0]
        return trajectory

    def __call__(self, t: float) -> np.ndarray:
        """
        Evaluate the trajectory at a single time point.

        Args:
            t: Time/parameter value in [0, 1]

        Returns:
            Action values of shape (n_dof,)
        """
        return self.evaluate(t)

    def __repr__(self) -> str:
        return (
            f"BSplineTrajectory(n_dof={self.n_dof}, "
            f"n_control_points={self.n_control_points}, "
            f"degree={self.degree})"
        )
