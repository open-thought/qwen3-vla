"""
B-Spline Tokenizer for Robotics Trajectories

A module for encoding multi-DoF robot trajectories as discrete tokens
using clamped B-spline representation with uniform quantization.
"""

from .bspline_trajectory import (
    BSplineTrajectory,
    create_clamped_knot_vector,
    bspline_basis,
    bspline_basis_matrix,
)

from .bspline_tokenizer import (
    BSplineTokenizer,
    tokenize_trajectory,
)

from .temporal_ensemble import (
    BSplineTemporalEnsemble,
)

__all__ = [
    'BSplineTrajectory',
    'BSplineTokenizer',
    'BSplineTemporalEnsemble',
    'tokenize_trajectory',
    'create_clamped_knot_vector',
    'bspline_basis',
    'bspline_basis_matrix',
]

__version__ = '0.1.0'
