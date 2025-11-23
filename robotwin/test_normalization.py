"""Test script for normalization utilities."""

import numpy as np
import h5py

from normalization import MultiRobotNormalizer, discretize_normalized_values, undiscretize_to_normalized


def test_normalization():
    """Test normalization and denormalization round-trip."""
    print("Testing normalization utilities...")
    print("=" * 60)

    # Load normalizer
    normalizer = MultiRobotNormalizer("data/test_norm_stats.json")
    print(f"\nLoaded normalizer for robot types: {normalizer.robot_types}")

    # Load test data
    with h5py.File("data/test_delta_actions.hdf5", "r") as f:
        for robot_type in normalizer.robot_types:
            print(f"\n{'='*60}")
            print(f"Testing {robot_type}")
            print(f"{'='*60}")

            # Get metadata
            metadata = normalizer.get_robot_metadata(robot_type)
            print(f"\nMetadata:")
            print(f"  DoF per arm: {metadata['dof']}")
            print(f"  Action horizon: {metadata['action_horizon']}")
            print(f"  Num samples: {metadata['num_samples']}")

            # Load sample data
            states = f[f"{robot_type}/states"][:100]  # First 100 samples
            deltas = f[f"{robot_type}/delta_actions"][:100]
            grippers = f[f"{robot_type}/grippers"][:100]

            print(f"\nOriginal data ranges:")
            print(f"  States: [{states.min():.4f}, {states.max():.4f}]")
            print(f"  Deltas: [{deltas.min():.4f}, {deltas.max():.4f}]")
            print(f"  Grippers: [{grippers.min():.4f}, {grippers.max():.4f}]")

            # Test state normalization
            normalized_states = normalizer.normalize_state(states, robot_type)
            print(f"\nNormalized states range: [{normalized_states.min():.4f}, {normalized_states.max():.4f}]")
            assert normalized_states.min() >= -1.0 and normalized_states.max() <= 1.0, \
                "Normalized states outside [-1, 1] range!"

            # Test state denormalization (round-trip)
            denormalized_states = normalizer.denormalize_state(normalized_states, robot_type)
            state_error = np.abs(states - denormalized_states).mean()
            print(f"State round-trip error: {state_error:.6f}")
            assert state_error < 2e-3, f"State round-trip error too large: {state_error}"

            # Test delta action normalization
            normalized_deltas = normalizer.normalize_delta_actions(deltas, robot_type)
            print(f"\nNormalized deltas range: [{normalized_deltas.min():.4f}, {normalized_deltas.max():.4f}]")
            assert normalized_deltas.min() >= -1.0 and normalized_deltas.max() <= 1.0, \
                "Normalized deltas outside [-1, 1] range!"

            # Test delta action denormalization (round-trip)
            denormalized_deltas = normalizer.denormalize_delta_actions(normalized_deltas, robot_type)
            delta_error = np.abs(deltas - denormalized_deltas).mean()
            print(f"Delta round-trip error: {delta_error:.6f}")
            assert delta_error < 2e-3, f"Delta round-trip error too large: {delta_error}"

            # Test gripper normalization
            normalized_grippers = normalizer.normalize_grippers(grippers, robot_type)
            print(f"\nNormalized grippers range: [{normalized_grippers.min():.4f}, {normalized_grippers.max():.4f}]")
            assert normalized_grippers.min() >= -1.0 and normalized_grippers.max() <= 1.0, \
                "Normalized grippers outside [-1, 1] range!"

            # Test gripper denormalization (round-trip)
            denormalized_grippers = normalizer.denormalize_grippers(normalized_grippers, robot_type)
            gripper_error = np.abs(grippers - denormalized_grippers).mean()
            print(f"Gripper round-trip error: {gripper_error:.6f}")
            assert gripper_error < 2e-3, f"Gripper round-trip error too large: {gripper_error}"

            # Test discretization
            discretized = discretize_normalized_values(normalized_states)
            print(f"\nDiscretized state range: [{discretized.min()}, {discretized.max()}]")
            assert discretized.min() >= 0 and discretized.max() <= 255, \
                "Discretized values outside [0, 255] range!"

            # Test undiscretization
            undiscretized = undiscretize_to_normalized(discretized)
            print(f"Undiscretized state range: [{undiscretized.min():.4f}, {undiscretized.max():.4f}]")
            discretization_error = np.abs(normalized_states - undiscretized).mean()
            print(f"Discretization round-trip error: {discretization_error:.6f}")
            # Note: Some error is expected due to quantization
            assert discretization_error < 0.01, f"Discretization error too large: {discretization_error}"

            print(f"\n✓ All tests passed for {robot_type}")

    print("\n" + "=" * 60)
    print("✓ All normalization tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_normalization()
