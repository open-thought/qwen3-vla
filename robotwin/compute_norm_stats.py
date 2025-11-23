"""
Compute normalization statistics from extracted delta actions.

For each robot type, computes 1% and 99% percentiles for:
- Robot states (current joint positions)
- Delta actions (future movements)
- Gripper states

These statistics are used during training to normalize inputs to [-1, 1] range.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def compute_statistics(delta_actions_path: str, output_json: str):
    """
    Compute normalization statistics from extracted delta actions.

    Args:
        delta_actions_path: Path to HDF5 file with extracted delta actions
        output_json: Path to output JSON file for statistics
    """
    print(f"Loading data from {delta_actions_path}...")

    with h5py.File(delta_actions_path, "r") as f:
        stats = {}

        print("\nComputing statistics per robot type:")
        print("-" * 60)

        for robot_type in f.keys():
            print(f"\nProcessing {robot_type}...")

            # Load data
            states = f[f"{robot_type}/states"][:]  # (N, 2*dof)
            deltas = f[f"{robot_type}/delta_actions"][:]  # (N, action_horizon, 2*dof)
            grippers = f[f"{robot_type}/grippers"][:]  # (N, 2)

            # Get metadata
            action_horizon = f[robot_type].attrs["action_horizon"]
            dof = f[robot_type].attrs["dof"]
            num_samples = f[robot_type].attrs["num_samples"]

            print(f"  Samples: {num_samples}")
            print(f"  DoF per arm: {dof}")
            print(f"  Action horizon: {action_horizon}")
            print(f"  States shape: {states.shape}")
            print(f"  Delta actions shape: {deltas.shape}")
            print(f"  Grippers shape: {grippers.shape}")

            # Compute percentiles for states (current joint positions)
            # Shape: (2*dof,) - percentiles computed across all samples
            state_q01 = np.percentile(states, 1, axis=0)
            state_q99 = np.percentile(states, 99, axis=0)

            print(f"  State range (q01 to q99):")
            print(f"    Min: {state_q01.min():.4f}, Max: {state_q99.max():.4f}")

            # Compute percentiles for delta actions
            # Flatten across samples and time steps, then compute per joint
            # Shape: (N * action_horizon, 2*dof) -> percentiles: (2*dof,)
            deltas_flat = deltas.reshape(-1, deltas.shape[-1])
            delta_q01 = np.percentile(deltas_flat, 1, axis=0)
            delta_q99 = np.percentile(deltas_flat, 99, axis=0)

            print(f"  Delta action range (q01 to q99):")
            print(f"    Min: {delta_q01.min():.4f}, Max: {delta_q99.max():.4f}")

            # Compute percentiles for grippers
            # Shape: (2,) - one for left, one for right gripper
            gripper_q01 = np.percentile(grippers, 1, axis=0)
            gripper_q99 = np.percentile(grippers, 99, axis=0)

            print(f"  Gripper range (q01 to q99):")
            print(f"    Left: [{gripper_q01[0]:.4f}, {gripper_q99[0]:.4f}]")
            print(f"    Right: [{gripper_q01[1]:.4f}, {gripper_q99[1]:.4f}]")

            # Store statistics
            stats[robot_type] = {
                "metadata": {
                    "dof": int(dof),
                    "action_horizon": int(action_horizon),
                    "num_samples": int(num_samples),
                },
                "state": {
                    "q01": state_q01.tolist(),
                    "q99": state_q99.tolist(),
                },
                "delta_actions": {
                    "q01": delta_q01.tolist(),
                    "q99": delta_q99.tolist(),
                },
                "grippers": {
                    "q01": gripper_q01.tolist(),
                    "q99": gripper_q99.tolist(),
                },
            }

    # Save to JSON
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving statistics to {output_json}...")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone!")
    print(f"\nStatistics saved for {len(stats)} robot types:")
    for robot_type in stats.keys():
        print(f"  - {robot_type}")

    print("\nStatistics structure per robot type:")
    print("  metadata/")
    print("    - dof: degrees of freedom per arm")
    print("    - action_horizon: prediction horizon")
    print("    - num_samples: number of training samples")
    print("  state/")
    print("    - q01: 1% percentile of current joint positions (2*dof,)")
    print("    - q99: 99% percentile of current joint positions (2*dof,)")
    print("  delta_actions/")
    print("    - q01: 1% percentile of delta movements (2*dof,)")
    print("    - q99: 99% percentile of delta movements (2*dof,)")
    print("  grippers/")
    print("    - q01: 1% percentile of gripper states (2,)")
    print("    - q99: 99% percentile of gripper states (2,)")


def main():
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics from extracted delta actions"
    )
    parser.add_argument(
        "--delta-actions",
        type=str,
        default="data/delta_actions.hdf5",
        help="Path to HDF5 file with extracted delta actions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/norm_stats.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    compute_statistics(
        delta_actions_path=args.delta_actions,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
