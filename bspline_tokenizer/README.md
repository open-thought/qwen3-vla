# B-Spline Tokenizer

A library for encoding robot trajectories as discrete tokens using clamped B-spline representation with uniform quantization.

## Installation

```bash
pip install -e .
```

## Usage

### Basic Tokenization

```python
import numpy as np
from bspline_tokenizer import BSplineTokenizer

# Create tokenizer for 7-DoF robot
tokenizer = BSplineTokenizer(
    n_dof=7,
    n_control_points=8,
    degree=4,
    bounds=(-1.5, 1.5),
    n_bins=256,
)

# Sample trajectory: 50 timesteps, 7 joints
t = np.linspace(0, 1, 50)
trajectory = np.random.randn(50, 7) * 0.5

# Encode to tokens
tokens = tokenizer.encode(t, trajectory)
print(f"Tokens: {tokens.shape}")  # (56,) = 7 * 8

# Decode back to trajectory
decoded = tokenizer.decode(tokens)
reconstructed = decoded.evaluate(t)
```

### Quick Tokenization

```python
from bspline_tokenizer import tokenize_trajectory

t = np.linspace(0, 1, 100)
trajectory = np.sin(2 * np.pi * t).reshape(-1, 1)

tokens, tokenizer = tokenize_trajectory(t, trajectory, n_control_points=8)
```

### Working with BSplineTrajectory

```python
from bspline_tokenizer import BSplineTrajectory

# Fit trajectory to data
traj = BSplineTrajectory.fit(t, trajectory, n_control_points=8, degree=4)

# Evaluate at any time in [0, 1]
values = traj.evaluate(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

# Or use callable interface
value_at_half = traj(0.5)
```

### Temporal Ensembling

For real-time control with overlapping action chunks:

```python
from bspline_tokenizer import BSplineTemporalEnsemble

ensemble = BSplineTemporalEnsemble(n_dof=7, weights="exponential", exp_decay=0.5)

# Add predictions as they arrive
ensemble.add_trajectory(traj1, start_timestep=0)
ensemble.add_trajectory(traj2, start_timestep=8)

# Get blended action at any timestep
action = ensemble.get_action(timestep=10)
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_dof` | Degrees of freedom | 7 |
| `n_control_points` | Control points per DoF | 8 |
| `degree` | B-spline degree | 4 |
| `bounds` | Control point bounds | (-1.5, 1.5) |
| `n_bins` | Quantization bins (odd for zero-centered) | 255 |
| `token_order` | `'basis_first'` or `'joint_first'` | `'basis_first'` |

## Testing

```bash
pytest tests/ -v
```
