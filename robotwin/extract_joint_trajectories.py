"""
Extract all joint state trajectories from RoboTwin dataset.

Reads all episodes and creates a single HDF5 file with data organized by robot type:
- Joint states (left/right arm, stored separately from grippers) per robot type
- Gripper states (left/right, stored separately) per robot type
- Metadata (task, robot_type, variant, episode_idx) per robot type

Different robots have different DoF (e.g., Franka: 7 DoF, arx-x5: 6 DoF),
so data is stored separately per robot type.

This enables efficient computation of statistics over all joint values.
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import List, Optional, Dict
from collections import defaultdict
import io

import h5py
import numpy as np
from tqdm import tqdm

from efficient_batch_loader import RoboTwinDatasetIndex, EpisodeMetadata


def load_joint_data(zf: zipfile.ZipFile, metadata: EpisodeMetadata) -> dict:
    """
    Load joint and gripper data from an episode.

    Args:
        zf: Open zipfile handle
        metadata: Episode metadata

    Returns:
        Dictionary with joint and gripper arrays, or None if failed
    """
    try:
        # Construct path within zip
        stem = f"{metadata.robot_type}_{metadata.variant}"
        episode_name = f"episode{metadata.episode_idx}"
        hdf5_path = f"{stem}/data/{episode_name}.hdf5"

        # Load HDF5 data
        hdf5_bytes = zf.read(hdf5_path)
        hdf5_file = h5py.File(io.BytesIO(hdf5_bytes), "r")

        # Extract joint and gripper data separately (not using 'vector')
        data = {
            "left_arm_joints": np.array(hdf5_file["joint_action"]["left_arm"]),
            "right_arm_joints": np.array(hdf5_file["joint_action"]["right_arm"]),
            "left_gripper": np.array(hdf5_file["joint_action"]["left_gripper"]),
            "right_gripper": np.array(hdf5_file["joint_action"]["right_gripper"]),
        }

        hdf5_file.close()
        return data

    except Exception as e:
        print(f"Warning: Failed to load {metadata.task_name}/episode{metadata.episode_idx}: {e}")
        return None


def extract_trajectories(
    dataset_root: str,
    output_path: str,
    robot_types: Optional[List[str]] = None,
    variants: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
):
    """
    Extract all trajectories and save to HDF5 file.

    Args:
        dataset_root: Root directory of RoboTwin dataset
        output_path: Path to output HDF5 file
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

    # Prepare data structure organized by robot type
    # Different robots have different DoF (e.g., Franka: 7, arx-x5: 6)
    robot_data = defaultdict(lambda: {
        "left_arm_joints": [],
        "right_arm_joints": [],
        "left_gripper": [],
        "right_gripper": [],
        "trajectory_metadata": [],
        "trajectory_boundaries": [0],
    })

    # Track open zip files to reuse handles
    open_zips = {}  # Path -> zipfile handle

    print(f"\nExtracting trajectories from {len(index.episodes)} episodes...")

    try:
        for ep_meta in tqdm(index.episodes, desc="Processing episodes"):
            # Get or open zip file
            if ep_meta.zip_path not in open_zips:
                open_zips[ep_meta.zip_path] = zipfile.ZipFile(ep_meta.zip_path, "r")
            zf = open_zips[ep_meta.zip_path]

            # Load joint data
            joint_data = load_joint_data(zf, ep_meta)

            if joint_data is None:
                continue

            # Get data structure for this robot type
            robot_type = ep_meta.robot_type
            rd = robot_data[robot_type]

            # Append joint and gripper data
            rd["left_arm_joints"].append(joint_data["left_arm_joints"])
            rd["right_arm_joints"].append(joint_data["right_arm_joints"])
            rd["left_gripper"].append(joint_data["left_gripper"])
            rd["right_gripper"].append(joint_data["right_gripper"])

            # Record trajectory boundary
            num_timesteps = len(joint_data["left_arm_joints"])
            rd["trajectory_boundaries"].append(rd["trajectory_boundaries"][-1] + num_timesteps)

            # Store metadata
            rd["trajectory_metadata"].append({
                "task": ep_meta.task_name,
                "robot_type": ep_meta.robot_type,
                "variant": ep_meta.variant,
                "episode_idx": ep_meta.episode_idx,
                "num_timesteps": num_timesteps,
                "zip_path": str(ep_meta.zip_path),
            })

    finally:
        # Close all zip files
        for zf in open_zips.values():
            zf.close()

    total_trajectories = sum(len(rd["trajectory_metadata"]) for rd in robot_data.values())
    print(f"\nSuccessfully loaded {total_trajectories} trajectories")

    # Concatenate trajectories per robot type
    print("\nConcatenating trajectory data per robot type...")
    for robot_type, rd in robot_data.items():
        print(f"\n{robot_type}:")
        rd["left_arm_joints"] = np.concatenate(rd["left_arm_joints"], axis=0)
        rd["right_arm_joints"] = np.concatenate(rd["right_arm_joints"], axis=0)
        rd["left_gripper"] = np.concatenate(rd["left_gripper"], axis=0)
        rd["right_gripper"] = np.concatenate(rd["right_gripper"], axis=0)
        rd["trajectory_boundaries"] = np.array(rd["trajectory_boundaries"], dtype=np.int64)

        print(f"  Trajectories: {len(rd['trajectory_metadata'])}")
        print(f"  Total timesteps: {len(rd['left_arm_joints'])}")
        print(f"  Left arm shape: {rd['left_arm_joints'].shape} (DoF: {rd['left_arm_joints'].shape[1]})")
        print(f"  Right arm shape: {rd['right_arm_joints'].shape} (DoF: {rd['right_arm_joints'].shape[1]})")
        print(f"  Gripper shapes: {rd['left_gripper'].shape}, {rd['right_gripper'].shape}")

    # Save to HDF5 (organized by robot type)
    print(f"\nSaving to {output_path}...")
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path_obj, "w") as f:
        # Store data per robot type in separate groups
        for robot_type, rd in robot_data.items():
            robot_group = f.create_group(robot_type)

            # Create joint state datasets
            robot_group.create_dataset("left_arm_joints", data=rd["left_arm_joints"], compression="gzip")
            robot_group.create_dataset("right_arm_joints", data=rd["right_arm_joints"], compression="gzip")
            robot_group.create_dataset("left_gripper", data=rd["left_gripper"], compression="gzip")
            robot_group.create_dataset("right_gripper", data=rd["right_gripper"], compression="gzip")

            # Create trajectory boundary dataset
            robot_group.create_dataset("trajectory_boundaries", data=rd["trajectory_boundaries"], compression="gzip")

            # Create metadata datasets
            tasks = np.array([m["task"] for m in rd["trajectory_metadata"]], dtype=h5py.string_dtype())
            variants = np.array([m["variant"] for m in rd["trajectory_metadata"]], dtype=h5py.string_dtype())
            episode_indices = np.array([m["episode_idx"] for m in rd["trajectory_metadata"]], dtype=np.int32)
            num_timesteps_arr = np.array([m["num_timesteps"] for m in rd["trajectory_metadata"]], dtype=np.int32)
            zip_paths = np.array([m["zip_path"] for m in rd["trajectory_metadata"]], dtype=h5py.string_dtype())

            metadata_group = robot_group.create_group("metadata")
            metadata_group.create_dataset("tasks", data=tasks, compression="gzip")
            metadata_group.create_dataset("variants", data=variants, compression="gzip")
            metadata_group.create_dataset("episode_indices", data=episode_indices, compression="gzip")
            metadata_group.create_dataset("num_timesteps", data=num_timesteps_arr, compression="gzip")
            metadata_group.create_dataset("zip_paths", data=zip_paths, compression="gzip")

            # Store per-robot statistics as attributes
            robot_group.attrs["num_trajectories"] = len(rd["trajectory_metadata"])
            robot_group.attrs["total_timesteps"] = len(rd["left_arm_joints"])
            robot_group.attrs["dof"] = rd["left_arm_joints"].shape[1]
            robot_group.attrs["num_tasks"] = len(set(m["task"] for m in rd["trajectory_metadata"]))

        # Store overall dataset statistics as attributes
        all_robot_types = list(robot_data.keys())
        total_traj = sum(len(rd["trajectory_metadata"]) for rd in robot_data.values())
        total_ts = sum(len(rd["left_arm_joints"]) for rd in robot_data.values())
        all_tasks = set()
        all_variants = set()
        for rd in robot_data.values():
            all_tasks.update(m["task"] for m in rd["trajectory_metadata"])
            all_variants.update(m["variant"] for m in rd["trajectory_metadata"])

        f.attrs["total_trajectories"] = total_traj
        f.attrs["total_timesteps"] = total_ts
        f.attrs["num_tasks"] = len(all_tasks)
        f.attrs["robot_types"] = json.dumps(all_robot_types)
        f.attrs["variants"] = json.dumps(list(all_variants))

    print("Done!")
    print(f"\nDataset structure (organized by robot type):")
    print("  <robot_type>/")
    print("    - left_arm_joints: (timesteps, DoF)")
    print("    - right_arm_joints: (timesteps, DoF)")
    print("    - left_gripper: (timesteps,)")
    print("    - right_gripper: (timesteps,)")
    print("    - trajectory_boundaries: (num_trajectories + 1,)")
    print("    - metadata/tasks: (num_trajectories,)")
    print("    - metadata/variants: (num_trajectories,)")
    print("    - metadata/episode_indices: (num_trajectories,)")
    print("    - metadata/num_timesteps: (num_trajectories,)")

    # Quick statistics per robot type
    print(f"\nQuick statistics per robot:")
    for robot_type, rd in robot_data.items():
        print(f"\n{robot_type}:")
        dof = rd["left_arm_joints"].shape[1]
        print(f"  DoF: {dof}")
        print(f"  Left arm joint ranges:")
        for i in range(dof):
            print(f"    Joint {i}: [{rd['left_arm_joints'][:, i].min():.3f}, {rd['left_arm_joints'][:, i].max():.3f}]")
        print(f"  Right arm joint ranges:")
        for i in range(dof):
            print(f"    Joint {i}: [{rd['right_arm_joints'][:, i].min():.3f}, {rd['right_arm_joints'][:, i].max():.3f}]")
        print(f"  Left gripper range: [{rd['left_gripper'].min():.3f}, {rd['left_gripper'].max():.3f}]")
        print(f"  Right gripper range: [{rd['right_gripper'].min():.3f}, {rd['right_gripper'].max():.3f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Extract joint trajectories from RoboTwin dataset"
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
        default="data/joint_trajectories.hdf5",
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--robot-types",
        type=str,
        nargs="+",
        default=None,
        help="Filter by robot types (e.g., arx-x5 franka ur5)",
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

    extract_trajectories(
        dataset_root=args.dataset_root,
        output_path=args.output,
        robot_types=args.robot_types,
        variants=args.variants,
        tasks=args.tasks,
    )


if __name__ == "__main__":
    main()
