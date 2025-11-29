#!/usr/bin/env python3
"""
Extract all episodes for a robot type into a single HDF5 file.

Uses per-episode groups for natural access patterns during training.

Output HDF5 structure:
    /episodes/{task}_{variant}_ep{idx}/
        cameras/
            left/
                rgb           - (T,) variable-length JPEG bytes
                intrinsic     - (T, 3, 3) float32 - camera intrinsics (OpenCV)
                extrinsic     - (T, 3, 4) float32 - camera extrinsics (OpenCV)
            right/
                rgb           - (T,) variable-length JPEG bytes
                intrinsic     - (T, 3, 3) float32
                extrinsic     - (T, 3, 4) float32
            head/
                rgb           - (T,) variable-length JPEG bytes
                intrinsic     - (T, 3, 3) float32
                extrinsic     - (T, 3, 4) float32
            front/
                rgb           - (T,) variable-length JPEG bytes
                intrinsic     - (T, 3, 3) float32
                extrinsic     - (T, 3, 4) float32
        state                 - (T, DOF*2 + 2) float32 - joints + grippers
        endpose/
            left              - (T, 7) float32 - xyz + quat
            right             - (T, 7) float32 - xyz + quat
        instructions/
            seen              - (N,) strings
            unseen            - (N,) strings
        [attrs: task, variant, episode_idx, num_frames]

    # Root attributes
    robot_type, num_episodes, total_frames, dof
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def extract_episode(
    zf: zipfile.ZipFile,
    stem: str,
    episode_idx: int,
) -> dict:
    """Extract a single episode from a zip file."""
    episode_name = f"episode{episode_idx}"
    hdf5_path = f"{stem}/data/{episode_name}.hdf5"
    instruction_path = f"{stem}/instructions/{episode_name}.json"

    # Load HDF5 data
    hdf5_bytes = zf.read(hdf5_path)
    with h5py.File(io.BytesIO(hdf5_bytes), "r") as hdf5_file:
        obs = hdf5_file["observation"]
        num_frames = len(obs["left_camera"]["rgb"])

        # Extract camera data (rgb as compressed JPEG bytes, plus intrinsics/extrinsics)
        cameras = {}
        for cam_name in ["left", "right", "head", "front"]:
            src_name = f"{cam_name}_camera"
            cam = obs[src_name]
            cameras[cam_name] = {
                "rgb": [bytes(cam["rgb"][t]) for t in range(num_frames)],
                "intrinsic": np.array(cam["intrinsic_cv"]),
                "extrinsic": np.array(cam["extrinsic_cv"]),
            }

        # Extract joint actions
        joint_action = hdf5_file["joint_action"]
        left_arm = np.array(joint_action["left_arm"])
        right_arm = np.array(joint_action["right_arm"])
        left_gripper = np.array(joint_action["left_gripper"])
        right_gripper = np.array(joint_action["right_gripper"])

        # Combine into state: [left_arm, right_arm, left_gripper, right_gripper]
        state = np.concatenate([
            left_arm,
            right_arm,
            left_gripper[:, np.newaxis],
            right_gripper[:, np.newaxis],
        ], axis=1)

        # Extract end-effector poses (skip redundant grippers, already in state)
        endpose_grp = hdf5_file["endpose"]
        endpose = {
            "left": np.array(endpose_grp["left_endpose"]),
            "right": np.array(endpose_grp["right_endpose"]),
        }

    # Load instructions
    instruction_bytes = zf.read(instruction_path)
    instruction_data = json.loads(instruction_bytes.decode("utf-8"))

    return {
        "cameras": cameras,
        "state": state,
        "endpose": endpose,
        "instructions_seen": instruction_data["seen"],
        "instructions_unseen": instruction_data["unseen"],
        "num_frames": num_frames,
    }


def find_all_episodes(
    dataset_root: Path,
    robot_type: str,
    variants: list[str] | None = None,
    tasks: list[str] | None = None,
) -> list[dict]:
    """Find all episodes for a robot type."""
    episodes = []

    # Get all task directories
    task_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    if tasks:
        task_dirs = [d for d in task_dirs if d.name in tasks]

    for task_dir in task_dirs:
        task_name = task_dir.name

        # Find zip files for this robot type
        zip_files = list(task_dir.glob(f"{robot_type}_*.zip"))

        for zip_path in zip_files:
            # Parse variant from filename: {robot_type}_{variant}.zip
            variant = zip_path.stem.replace(f"{robot_type}_", "")

            if variants and variant not in variants:
                continue

            # Count episodes in zip
            with zipfile.ZipFile(zip_path, "r") as zf:
                stem = f"{robot_type}_{variant}"
                data_files = [
                    f for f in zf.namelist()
                    if f.startswith(f"{stem}/data/episode") and f.endswith(".hdf5")
                ]
                num_episodes = len(data_files)

            for ep_idx in range(num_episodes):
                episodes.append({
                    "task": task_name,
                    "variant": variant,
                    "robot_type": robot_type,
                    "episode_idx": ep_idx,
                    "zip_path": str(zip_path),
                })

    return episodes


def main():
    parser = argparse.ArgumentParser(
        description="Extract episodes for a robot type into a single HDF5 file"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/robotwin/dataset",
        help="Path to RoboTwin dataset",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="aloha-agilex",
        help="Robot type to extract",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help="Variants to include (default: all)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to include (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HDF5 file path (default: data/{robot_type}_episodes.hdf5)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_path = args.output or f"data/{args.robot_type}_episodes.hdf5"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all episodes
    print(f"Scanning dataset at {dataset_root}...")
    episodes = find_all_episodes(
        dataset_root,
        robot_type=args.robot_type,
        variants=args.variants,
        tasks=args.tasks,
    )
    print(f"Found {len(episodes)} episodes")

    if not episodes:
        print("No episodes found. Check robot_type and dataset_root.")
        return

    # Group episodes by zip file for efficient reading
    episodes_by_zip: dict[str, list[dict]] = {}
    for ep in episodes:
        zip_path = ep["zip_path"]
        if zip_path not in episodes_by_zip:
            episodes_by_zip[zip_path] = []
        episodes_by_zip[zip_path].append(ep)

    # Create output HDF5 file
    print(f"\nWriting to {output_path}...")

    # Create variable-length types
    str_dtype = h5py.special_dtype(vlen=str)
    bytes_dtype = h5py.special_dtype(vlen=np.uint8)

    with h5py.File(output_path, "w") as out_f:
        # Create episodes group
        episodes_grp = out_f.create_group("episodes")

        total_frames = 0
        dof: int | None = None

        # Process each zip file
        for zip_path, zip_episodes in tqdm(
            episodes_by_zip.items(), desc="Processing zip files"
        ):
            with zipfile.ZipFile(zip_path, "r") as zf:
                stem = f"{args.robot_type}_{zip_episodes[0]['variant']}"

                for ep_info in tqdm(
                    zip_episodes, desc=f"  {Path(zip_path).stem}", leave=False
                ):
                    # Extract episode data
                    ep_data = extract_episode(zf, stem, ep_info["episode_idx"])
                    num_frames = ep_data["num_frames"]

                    # Episode name
                    ep_name = f"{ep_info['task']}_{ep_info['variant']}_ep{ep_info['episode_idx']}"

                    # Get DOF from first episode
                    if dof is None:
                        # state shape is (T, DOF*2 + 2)
                        dof = (ep_data["state"].shape[1] - 2) // 2

                    # Create episode group
                    ep_grp = episodes_grp.create_group(ep_name)

                    # Store cameras (grouped by camera with rgb, intrinsic, extrinsic)
                    cam_grp = ep_grp.create_group("cameras")
                    for cam_name, cam_data in ep_data["cameras"].items():
                        cam_subgrp = cam_grp.create_group(cam_name)

                        # RGB images as variable-length bytes
                        rgb_ds = cam_subgrp.create_dataset(
                            "rgb", shape=(num_frames,), dtype=bytes_dtype
                        )
                        for t, img_bytes in enumerate(cam_data["rgb"]):
                            rgb_ds[t] = np.frombuffer(img_bytes, dtype=np.uint8)

                        # Intrinsics and extrinsics
                        cam_subgrp.create_dataset(
                            "intrinsic",
                            data=cam_data["intrinsic"],
                            compression="gzip",
                        )
                        cam_subgrp.create_dataset(
                            "extrinsic",
                            data=cam_data["extrinsic"],
                            compression="gzip",
                        )

                    # Store state (joints + grippers)
                    ep_grp.create_dataset(
                        "state",
                        data=ep_data["state"],
                        compression="gzip",
                    )

                    # Store end-effector poses (same structure as original)
                    endpose_grp = ep_grp.create_group("endpose")
                    for key, value in ep_data["endpose"].items():
                        endpose_grp.create_dataset(
                            key,
                            data=value,
                            compression="gzip",
                        )

                    # Store instructions in episode group
                    instr_grp = ep_grp.create_group("instructions")
                    instr_grp.create_dataset(
                        "seen",
                        data=ep_data["instructions_seen"],
                        dtype=str_dtype,
                    )
                    instr_grp.create_dataset(
                        "unseen",
                        data=ep_data["instructions_unseen"],
                        dtype=str_dtype,
                    )

                    # Store episode attributes
                    ep_grp.attrs["task"] = ep_info["task"]
                    ep_grp.attrs["variant"] = ep_info["variant"]
                    ep_grp.attrs["episode_idx"] = ep_info["episode_idx"]
                    ep_grp.attrs["num_frames"] = num_frames

                    total_frames += num_frames

        # Store root attributes
        out_f.attrs["robot_type"] = args.robot_type
        out_f.attrs["num_episodes"] = len(episodes)
        out_f.attrs["total_frames"] = total_frames
        out_f.attrs["dof"] = dof

    # Print summary
    file_size = output_path.stat().st_size / (1024 ** 3)
    print(f"\nDone!")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total frames: {total_frames}")
    print(f"  DOF per arm: {dof}")
    print(f"  File size: {file_size:.2f} GB")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
