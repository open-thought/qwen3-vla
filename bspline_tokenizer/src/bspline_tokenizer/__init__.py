"""
B-Spline Tokenizer for Robotics Trajectories

A module for encoding multi-DoF robot trajectories as discrete tokens
using clamped B-spline representation with uniform quantization.
"""

from .bspline_tokenizer import (
    BSplineTokenizer,
    BSplineTrajectory,
    tokenize_trajectory,
    create_clamped_knot_vector,
    bspline_basis_matrix,
)

from .temporal_ensemble import (
    BSplineTemporalEnsemble,
)

__all__ = [
    'BSplineTokenizer',
    'BSplineTrajectory',
    'tokenize_trajectory',
    'create_clamped_knot_vector',
    'bspline_basis_matrix',
    'BSplineTemporalEnsemble',
]

__version__ = '0.1.0'
