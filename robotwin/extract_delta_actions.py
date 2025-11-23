"""
Extract delta action chunks from RoboTwin dataset.

For each timestep t in each episode:
- Current state: joint_action[t] (concatenated left + right arms)
- Future states: joint_action[t+1:t+1+action_horizon]
- Delta actions: future_states - current_state

Handles episode boundaries by repeating last action to fill horizon.
Organized by robot type to handle different DoF (Franka: 7, others: 6).
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import List, Optional
from collections import defaultdict
import io

import h5py
import numpy as np
from tqdm import tqdm

from efficient_batch_loader import RoboTwinDatasetIndex, EpisodeMetadata


def extract_episode_delta_actions(zf: zipfile.ZipFile, metadata: EpisodeMetadata, action_horizon: int) -> dict:
    """
    Extract delta action chunks from an episode.

    Args:
        zf: Open zipfile handle
        metadata: Episode metadata
        action_horizon: Number of future timesteps to predict

    Returns:
        Dictionary with states and delta_actions arrays, or None if failed
    """
    try:
        # Construct path within zip
        stem = f"{metadata.robot_type}_{metadata.variant}"
        episode_name = f"episode{metadata.episode_idx}"
        hdf5_path = f"{stem}/data/{episode_name}.hdf5"

        # Load HDF5 data
        hdf5_bytes = zf.read(hdf5_path)
        hdf5_file = h5py.File(io.BytesIO(hdf5_bytes), "r")

        # Extract joint data
        left_arm = np.array(hdf5_file["joint_action"]["left_arm"])  # (T, 6 or 7)
        right_arm = np.array(hdf5_file["joint_action"]["right_arm"])  # (T, 6 or 7)
        left_gripper = np.array(hdf5_file["joint_action"]["left_gripper"])  # (T,)
        right_gripper = np.array(hdf5_file["joint_action"]["right_gripper"])  # (T,)

        hdf5_file.close()

        # Concatenate left and right arms to form full state
        # Shape: (T, 2*dof) where dof is 6 or 7
        full_joints = np.concatenate([left_arm, right_arm], axis=1)
        full_grippers = np.stack([left_gripper, right_gripper], axis=1)  # (T, 2)

        num_timesteps = len(full_joints)

        # We can only use timesteps where we have action_horizon future steps
        # Skip last action_horizon timesteps
        if num_timesteps <= action_horizon:
            return None

        states_list = []
        delta_actions_list = []
        grippers_list = []

        # For each valid timestep
        for t in range(num_timesteps - action_horizon):
            current_state = full_joints[t]  # (2*dof,)
            current_gripper = full_grippers[t]  # (2,)

            # Get future states
            future_states = full_joints[t+1:t+1+action_horizon]  # (action_horizon, 2*dof)

            # Compute delta actions (relative to current state)
            delta_actions = future_states - current_state[None, :]  # (action_horizon, 2*dof)

            states_list.append(current_state)
            delta_actions_list.append(delta_actions)
            grippers_list.append(current_gripper)

        if len(states_list) == 0:
            return None

        return {
            "states": np.array(states_list),  # (num_valid_timesteps, 2*dof)
            "delta_actions": np.array(delta_actions_list),  # (num_valid_timesteps, action_horizon, 2*dof)
            "grippers": np.array(grippers_list),  # (num_valid_timesteps, 2)
        }

    except Exception as e:
        print(f"Warning: Failed to load {metadata.task_name}/episode{metadata.episode_idx}: {e}")
        return None


def extract_delta_actions(
    dataset_root: str,
    output_path: str,
    action_horizon: int = 50,
    robot_types: Optional[List[str]] = None,
    variants: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
):
    """
    Extract delta action chunks from all episodes and save to HDF5.

    Args:
        dataset_root: Root directory of RoboTwin dataset
        output_path: Path to output HDF5 file
        action_horizon: Number of future timesteps to predict
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
    print(f"  Variants: {stats['variants']}")
    print(f"  Action horizon: {action_horizon}")

    # Organize data by robot type
    robot_data = defaultdict(lambda: {
        "states": [],
        "delta_actions": [],
        "grippers": [],
    })

    # Track open zip files
    open_zips = {}

    print(f"\nExtracting delta actions from {len(index.episodes)} episodes...")

    try:
        for ep_meta in tqdm(index.episodes, desc="Processing episodes"):
            # Get or open zip file
            if ep_meta.zip_path not in open_zips:
                open_zips[ep_meta.zip_path] = zipfile.ZipFile(ep_meta.zip_path, "r")
            zf = open_zips[ep_meta.zip_path]

            # Extract delta actions for this episode
            episode_data = extract_episode_delta_actions(zf, ep_meta, action_horizon)

            if episode_data is None:
                continue

            # Append to robot-specific data
            robot_type = ep_meta.robot_type
            rd = robot_data[robot_type]

            rd["states"].append(episode_data["states"])
            rd["delta_actions"].append(episode_data["delta_actions"])
            rd["grippers"].append(episode_data["grippers"])

    finally:
        # Close all zip files
        for zf in open_zips.values():
            zf.close()

    # Concatenate data per robot type
    print("\nConcatenating data per robot type...")
    for robot_type, rd in robot_data.items():
        if len(rd["states"]) == 0:
            continue

        rd["states"] = np.concatenate(rd["states"], axis=0)
        rd["delta_actions"] = np.concatenate(rd["delta_actions"], axis=0)
        rd["grippers"] = np.concatenate(rd["grippers"], axis=0)

        print(f"\n{robot_type}:")
        print(f"  Samples: {len(rd['states'])}")
        print(f"  States shape: {rd['states'].shape}")
        print(f"  Delta actions shape: {rd['delta_actions'].shape}")
        print(f"  Grippers shape: {rd['grippers'].shape}")

    # Save to HDF5
    print(f"\nSaving to {output_path}...")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path_obj, "w") as f:
        for robot_type, rd in robot_data.items():
            if len(rd["states"]) == 0:
                continue

            robot_group = f.create_group(robot_type)
            robot_group.create_dataset("states", data=rd["states"], compression="gzip")
            robot_group.create_dataset("delta_actions", data=rd["delta_actions"], compression="gzip")
            robot_group.create_dataset("grippers", data=rd["grippers"], compression="gzip")

            # Store metadata
            robot_group.attrs["num_samples"] = len(rd["states"])
            robot_group.attrs["action_horizon"] = action_horizon
            robot_group.attrs["dof"] = rd["states"].shape[1] // 2  # Total joints / 2 arms

        # Store overall metadata
        f.attrs["action_horizon"] = action_horizon
        f.attrs["robot_types"] = json.dumps(list(robot_data.keys()))

    print("Done!")
    print(f"\nData saved to {output_path}")
    print("Structure:")
    print("  <robot_type>/")
    print("    - states: (num_samples, 2*dof)")
    print("    - delta_actions: (num_samples, action_horizon, 2*dof)")
    print("    - grippers: (num_samples, 2)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract delta action chunks from RoboTwin dataset"
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
        default="data/delta_actions.hdf5",
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=50,
        help="Number of future timesteps to predict",
    )
    parser.add_argument(
        "--robot-types",
        type=str,
        nargs="+",
        default=None,
        help="Filter by robot types (e.g., arx-x5 franka)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help="Filter by variants (e.g., clean_50 randomized_500)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Filter by tasks (e.g., adjust_bottle click_bell)",
    )

    args = parser.parse_args()

    extract_delta_actions(
        dataset_root=args.dataset_root,
        output_path=args.output,
        action_horizon=args.action_horizon,
        robot_types=args.robot_types,
        variants=args.variants,
        tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
