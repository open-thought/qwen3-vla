"""
Compute idle frame masks for RoboTwin dataset.

For each episode, identifies timesteps where the robot is idle (not moving).
An idle frame is defined as one where max(abs(delta_actions)) < threshold.

The output is a JSON file mapping episode keys to lists of valid (non-idle) timesteps.
This can be used by the dataloader to skip idle frames during training.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
from tqdm import tqdm

from efficient_batch_loader import RoboTwinDatasetIndex


def compute_idle_masks(
    dataset_root: str,
    output_path: str,
    action_horizon: int = 16,
    idle_threshold: float = 0.001,
    robot_types: Optional[List[str]] = None,
    variants: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
):
    """
    Compute idle frame masks for all episodes.

    Args:
        dataset_root: Root directory of RoboTwin dataset
        output_path: Path to output JSON file
        action_horizon: Number of future timesteps to predict
        idle_threshold: Maximum delta action magnitude to consider as "idle"
        robot_types: Optional filter for robot types
        variants: Optional filter for variants
        tasks: Optional filter for tasks
    """
    # Build index
    print("Building dataset index...")
    index = RoboTwinDatasetIndex(dataset_root)
    index.build_index(robot_types=robot_types, variants=variants, tasks=tasks)

    stats = index.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Tasks: {stats['num_tasks']}")
    print(f"  Robot types: {stats['robot_types']}")
    print(f"  Action horizon: {action_horizon}")
    print(f"  Idle threshold: {idle_threshold}")

    # Results: episode_key -> list of valid (non-idle) timesteps
    valid_timesteps = {}

    # Statistics
    total_frames = 0
    idle_frames = 0
    active_frames = 0

    # Track current open zip file
    current_zip_path = None
    current_zf = None

    print(f"\nComputing idle masks for {len(index.episodes)} episodes...")

    try:
        for ep_meta in tqdm(index.episodes, desc="Processing episodes"):
            # Check if we need to switch to a new zip file
            if ep_meta.zip_path != current_zip_path:
                if current_zf is not None:
                    current_zf.close()
                current_zf = zipfile.ZipFile(ep_meta.zip_path, "r")
                current_zip_path = ep_meta.zip_path

            # Load episode joint data
            try:
                stem = f"{ep_meta.robot_type}_{ep_meta.variant}"
                episode_name = f"episode{ep_meta.episode_idx}"
                hdf5_path = f"{stem}/data/{episode_name}.hdf5"

                hdf5_bytes = current_zf.read(hdf5_path)
                hdf5_file = h5py.File(io.BytesIO(hdf5_bytes), "r")

                left_arm = np.array(hdf5_file["joint_action"]["left_arm"])
                right_arm = np.array(hdf5_file["joint_action"]["right_arm"])
                hdf5_file.close()

                full_joints = np.concatenate([left_arm, right_arm], axis=1)
                num_timesteps = len(full_joints)

            except Exception as e:
                print(f"Warning: Failed to load {ep_meta.task_name}/episode{ep_meta.episode_idx}: {e}")
                continue

            # Create episode key
            episode_key = f"{ep_meta.task_name}/{ep_meta.robot_type}_{ep_meta.variant}/episode{ep_meta.episode_idx}"

            # For each timestep, compute max delta over the action horizon
            ep_valid_timesteps = []

            for t in range(num_timesteps - 1):  # Last timestep has no future
                # Get available future timesteps
                available = min(action_horizon, num_timesteps - t - 1)
                future_states = full_joints[t + 1:t + 1 + available]
                current_state = full_joints[t]

                # Compute delta actions
                delta_actions = future_states - current_state

                # Check if idle (max absolute delta below threshold)
                max_delta = np.abs(delta_actions).max()

                total_frames += 1
                if max_delta < idle_threshold:
                    idle_frames += 1
                else:
                    active_frames += 1
                    ep_valid_timesteps.append(t)

            valid_timesteps[episode_key] = ep_valid_timesteps

    finally:
        if current_zf is not None:
            current_zf.close()

    # Save results
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving idle masks to {output_path}...")
    with open(output_path_obj, "w") as f:
        json.dump(valid_timesteps, f)

    # Print statistics
    print("\n" + "=" * 60)
    print("IDLE FRAME ANALYSIS")
    print("=" * 60)
    print(f"Total frames analyzed: {total_frames}")
    print(f"Idle frames (max_delta < {idle_threshold}): {idle_frames} ({idle_frames / total_frames * 100:.1f}%)")
    print(f"Active frames: {active_frames} ({active_frames / total_frames * 100:.1f}%)")
    print()
    print(f"Output saved to: {output_path}")
    print(f"  Episodes: {len(valid_timesteps)}")
    print(f"  Format: {{episode_key: [list of valid timestep indices]}}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute idle frame masks for RoboTwin dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/robotwin/dataset",
        help="Root directory of RoboTwin dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/robotwin_valid_timesteps.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="Number of future timesteps to predict",
    )
    parser.add_argument(
        "--idle-threshold",
        type=float,
        default=0.001,
        help="Maximum delta action magnitude to consider as idle",
    )
    parser.add_argument(
        "--robot-types",
        type=str,
        nargs="+",
        default=None,
        help="Filter by robot types",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help="Filter by variants",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Filter by tasks",
    )

    args = parser.parse_args()

    compute_idle_masks(
        dataset_root=args.dataset_root,
        output_path=args.output,
        action_horizon=args.action_horizon,
        idle_threshold=args.idle_threshold,
        robot_types=args.robot_types,
        variants=args.variants,
        tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
