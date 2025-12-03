"""
Temporal Ensembling for B-Spline Tokenized Trajectories

Implements the temporal ensembling technique from the ACT paper:
"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"

The key idea is to query the policy more frequently than the action chunk duration,
creating overlapping predictions that are averaged with exponential weighting
to produce smoother actions.

Example:
    - Action chunk duration: 0.8s (8 control points at some internal resolution)
    - Query frequency: 10 Hz (every 0.1s)
    - At any time t, we have multiple overlapping predictions from queries at
      t, t-0.1s, t-0.2s, etc., each predicting what the action should be at time t
    - We combine these with exponential weights: w_i = exp(-m * i)
      where i=0 is the newest prediction

Usage:
    ensemble = BSplineTemporalEnsemble(
        tokenizer=tokenizer,
        chunk_duration=0.8,
        query_frequency=10.0,
        exp_weight_decay=0.01
    )

    # At each timestep, add new prediction and get ensembled action
    ensemble.add_prediction(tokens)
    action = ensemble.get_action()
"""

import numpy as np
from typing import Optional
from collections import deque
from .bspline_tokenizer import BSplineTokenizer
from .bspline_trajectory import BSplineTrajectory


class BSplineTemporalEnsemble:
    """
    Temporal ensembling for B-spline tokenized action chunks.

    Maintains a buffer of overlapping action chunk predictions and combines
    them using exponential weighting to produce smooth actions.

    Attributes:
        tokenizer: BSplineTokenizer instance for decoding tokens
        chunk_duration: Duration of each action chunk in seconds
        query_frequency: How often the policy is queried (Hz)
        exp_weight_decay: Decay factor m for exponential weights w_i = exp(-m * i)
        max_buffer_size: Maximum number of predictions to keep in buffer
    """

    def __init__(
        self,
        tokenizer: BSplineTokenizer,
        chunk_duration: float = 0.8,
        query_frequency: float = 10.0,
        exp_weight_decay: float = 0.01,
        max_buffer_size: Optional[int] = None
    ):
        """
        Initialize the temporal ensemble.

        Args:
            tokenizer: BSplineTokenizer instance for decoding tokens to trajectories
            chunk_duration: Duration of each action chunk in seconds (e.g., 0.8s)
            query_frequency: How often the policy is queried in Hz (e.g., 10 Hz)
            exp_weight_decay: Decay factor m for weights w_i = exp(-m * i).
                             Larger values = faster decay = more weight on recent predictions.
                             Typical values: 0.01 (slow decay) to 0.1 (fast decay)
            max_buffer_size: Maximum predictions to keep. If None, computed from
                            chunk_duration and query_frequency.
        """
        self.tokenizer = tokenizer
        self.chunk_duration = chunk_duration
        self.query_frequency = query_frequency
        self.query_period = 1.0 / query_frequency  # Time between queries
        self.exp_weight_decay = exp_weight_decay

        # Calculate how many predictions can overlap at any point in time
        # A prediction made at time t covers [t, t + chunk_duration]
        # So predictions from [t - chunk_duration, t] all cover time t
        self.max_overlap = int(np.ceil(chunk_duration * query_frequency))

        if max_buffer_size is None:
            max_buffer_size = self.max_overlap + 1
        self.max_buffer_size = max_buffer_size

        # Buffer stores (query_time, tokens, trajectory) tuples
        # query_time is relative to when the ensemble started (or was reset)
        self._buffer: deque = deque(maxlen=max_buffer_size)

        # Current time (incremented with each query)
        self._current_time = 0.0
        self._query_count = 0

    def reset(self):
        """Clear the buffer and reset time."""
        self._buffer.clear()
        self._current_time = 0.0
        self._query_count = 0

    def add_prediction(self, tokens: np.ndarray) -> None:
        """
        Add a new action chunk prediction to the buffer.

        This should be called each time the policy produces a new prediction.
        The prediction is assumed to start at the current time.

        Args:
            tokens: Token array from the policy, shape (n_tokens,)
        """
        # Decode tokens to BSplineTrajectory immediately
        trajectory = self.tokenizer.decode(tokens)

        self._buffer.append({
            'query_time': self._current_time,
            'trajectory': trajectory,
        })

        # Advance time
        self._current_time += self.query_period
        self._query_count += 1

    def _get_action_from_prediction(self, pred: dict, target_time: float) -> Optional[np.ndarray]:
        """
        Get the action from a single prediction at a specific time.

        Args:
            pred: Prediction dict with query_time and BSplineTrajectory
            target_time: The absolute time we want the action for

        Returns:
            Action array of shape (n_dof,) or None if target_time is outside
            this prediction's valid range.
        """
        query_time = pred['query_time']

        # Check if target_time falls within this prediction's range
        # Prediction covers [query_time, query_time + chunk_duration]
        time_into_chunk = target_time - query_time

        if time_into_chunk < 0 or time_into_chunk > self.chunk_duration:
            return None

        # Normalize to [0, 1] for B-spline evaluation
        t_normalized = time_into_chunk / self.chunk_duration

        # Evaluate B-spline directly at the exact time point
        trajectory: BSplineTrajectory = pred['trajectory']
        return trajectory(t_normalized)

    def get_action(self, time_offset: float = 0.0) -> np.ndarray:
        """
        Get the ensembled action for the current time (or with an offset).

        Combines overlapping predictions using exponential weighting where
        newer predictions have higher weight.

        Args:
            time_offset: Offset from current time in seconds.
                        0.0 = current time (most common)
                        Positive = future (within current predictions)
                        Negative = past

        Returns:
            Ensembled action array of shape (n_dof,)
        """
        if len(self._buffer) == 0:
            raise ValueError("No predictions in buffer. Call add_prediction() first.")

        target_time = self._current_time + time_offset

        # Collect all predictions that cover target_time
        actions = []
        weights = []

        for pred in self._buffer:
            action = self._get_action_from_prediction(pred, target_time)
            if action is not None:
                # Age of prediction: how many query steps ago it was made
                age = (self._current_time - pred['query_time']) / self.query_period

                # Exponential weight: newer = higher weight
                weight = np.exp(-self.exp_weight_decay * age)

                actions.append(action)
                weights.append(weight)

        if len(actions) == 0:
            # No predictions cover target_time, return most recent valid action
            # This can happen at the very beginning or if time_offset is too large
            latest_pred = self._buffer[-1]
            # Return action at start of latest chunk
            return self._get_action_from_prediction(latest_pred, latest_pred['query_time'])

        # Weighted average
        actions = np.array(actions)  # (n_predictions, n_dof)
        weights = np.array(weights)  # (n_predictions,)
        weights = weights / weights.sum()  # Normalize

        ensembled_action = np.sum(actions * weights[:, np.newaxis], axis=0)

        return ensembled_action

    def get_action_trajectory(self, n_steps: int, step_duration: float) -> np.ndarray:
        """
        Get a trajectory of ensembled actions for the next n_steps.

        Useful for visualization or for robots that want a short trajectory
        rather than a single action.

        Args:
            n_steps: Number of steps in the trajectory
            step_duration: Duration of each step in seconds

        Returns:
            Trajectory array of shape (n_steps, n_dof)
        """
        trajectory = np.zeros((n_steps, self.tokenizer.n_dof))

        for i in range(n_steps):
            time_offset = i * step_duration
            trajectory[i] = self.get_action(time_offset)

        return trajectory

    def get_ensemble_info(self) -> dict:
        """
        Get information about the current ensemble state.

        Returns:
            Dictionary with ensemble statistics
        """
        target_time = self._current_time

        # Count overlapping predictions
        n_overlapping = 0
        ages = []
        weights = []

        for pred in self._buffer:
            time_into_chunk = target_time - pred['query_time']
            if 0 <= time_into_chunk <= self.chunk_duration:
                n_overlapping += 1
                age = (self._current_time - pred['query_time']) / self.query_period
                ages.append(age)
                weights.append(np.exp(-self.exp_weight_decay * age))

        weights = np.array(weights)
        if len(weights) > 0:
            weights = weights / weights.sum()

        return {
            'current_time': self._current_time,
            'query_count': self._query_count,
            'buffer_size': len(self._buffer),
            'n_overlapping': n_overlapping,
            'max_overlap': self.max_overlap,
            'prediction_ages': ages,
            'normalized_weights': weights.tolist() if len(weights) > 0 else [],
        }

    @property
    def current_time(self) -> float:
        """Current time in seconds since start/reset."""
        return self._current_time

    @property
    def query_count(self) -> int:
        """Number of predictions added."""
        return self._query_count

    def __repr__(self) -> str:
        return (
            f"BSplineTemporalEnsemble(\n"
            f"  chunk_duration={self.chunk_duration}s,\n"
            f"  query_frequency={self.query_frequency}Hz,\n"
            f"  query_period={self.query_period:.3f}s,\n"
            f"  exp_weight_decay={self.exp_weight_decay},\n"
            f"  max_overlap={self.max_overlap},\n"
            f"  buffer_size={len(self._buffer)}/{self.max_buffer_size},\n"
            f"  current_time={self._current_time:.3f}s,\n"
            f"  n_dof={self.tokenizer.n_dof}\n"
            f")"
        )


if __name__ == "__main__":
    # Demo
    from .bspline_tokenizer import BSplineTokenizer
    import matplotlib.pyplot as plt

    print("="*70)
    print("Temporal Ensemble Demo")
    print("="*70)

    # Create tokenizer
    tokenizer = BSplineTokenizer(
        n_dof=2,
        n_control_points=8,
        degree=3,
        bounds=(-1.5, 1.5),
        n_bins=256
    )

    # Create ensemble
    ensemble = BSplineTemporalEnsemble(
        tokenizer=tokenizer,
        chunk_duration=0.8,
        query_frequency=10.0,
        exp_weight_decay=0.05
    )

    print(ensemble)

    # Simulate: generate slightly different predictions over time
    np.random.seed(42)
    n_queries = 20

    # Base trajectory
    t_base = np.linspace(0, 1, 50)
    base_traj = np.column_stack([
        np.sin(2 * np.pi * t_base),
        np.cos(2 * np.pi * t_base)
    ])

    # Store results
    all_actions = []
    all_times = []

    for i in range(n_queries):
        # Add noise to simulate varying predictions
        noise = np.random.randn(*base_traj.shape) * 0.1
        noisy_traj = base_traj + noise

        # Encode and add to ensemble
        tokens = tokenizer.encode(t_base, noisy_traj)
        ensemble.add_prediction(tokens)

        # Get ensembled action
        action = ensemble.get_action()
        all_actions.append(action)
        all_times.append(ensemble.current_time - ensemble.query_period)

        if i < 3 or i == n_queries - 1:
            info = ensemble.get_ensemble_info()
            print(f"\nQuery {i+1}:")
            print(f"  Overlapping predictions: {info['n_overlapping']}")
            print(f"  Weights: {[f'{w:.3f}' for w in info['normalized_weights']]}")

    all_actions = np.array(all_actions)
    all_times = np.array(all_times)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(all_times, all_actions[:, 0], 'b-o', label='DoF 0 (ensembled)')
    ax.plot(all_times, all_actions[:, 1], 'r-o', label='DoF 1 (ensembled)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Action')
    ax.set_title('Temporal Ensembled Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Show weight decay
    ages = np.arange(0, ensemble.max_overlap + 1)
    weights = np.exp(-ensemble.exp_weight_decay * ages)
    weights_normalized = weights / weights.sum()
    ax.bar(ages, weights_normalized, color='steelblue', edgecolor='black')
    ax.set_xlabel('Prediction Age (query steps)')
    ax.set_ylabel('Normalized Weight')
    ax.set_title(f'Exponential Weights (decay={ensemble.exp_weight_decay})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('temporal_ensemble_demo.png', dpi=150)
    print("\nSaved plot to temporal_ensemble_demo.png")
    plt.show()
