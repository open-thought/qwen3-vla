#!/usr/bin/env python3
"""
Generate videos from RoboTwin training episodes using ONLY valid training frames.

Creates side-by-side videos showing all 3 camera views (left, right, head)
for selected episodes, filtering out idle/non-training frames based on
the valid_timesteps JSON file.
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_episode_images_filtered(
    zip_path: str,
    robot_type: str,
    variant: str,
    episode_idx: int,
    valid_timesteps: list[int],
):
    """Load only valid timestep images from an episode."""
    stem = f"{robot_type}_{variant}"
    episode_name = f"episode{episode_idx}"
    hdf5_path = f"{stem}/data/{episode_name}.hdf5"
    instruction_path = f"{stem}/instructions/{episode_name}.json"

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Load HDF5 data
        hdf5_bytes = zf.read(hdf5_path)
        hdf5_file = h5py.File(io.BytesIO(hdf5_bytes), "r")

        # Load instruction
        instruction_bytes = zf.read(instruction_path)
        instruction_data = json.loads(instruction_bytes.decode("utf-8"))
        instruction = instruction_data["seen"][0]  # First instruction variant

        # Get total number of timesteps
        total_timesteps = len(hdf5_file["observation"]["head_camera"]["rgb"])

        # Load only valid timestep images
        left_images = []
        right_images = []
        head_images = []
        original_indices = []

        # Sort valid_timesteps to maintain temporal order
        sorted_valid = sorted(valid_timesteps)

        for t in sorted_valid:
            if t >= total_timesteps:
                continue

            # Decompress images
            left_bytes = hdf5_file["observation"]["left_camera"]["rgb"][t]
            right_bytes = hdf5_file["observation"]["right_camera"]["rgb"][t]
            head_bytes = hdf5_file["observation"]["head_camera"]["rgb"][t]

            left_img = np.array(Image.open(io.BytesIO(left_bytes)).convert("RGB"))
            right_img = np.array(Image.open(io.BytesIO(right_bytes)).convert("RGB"))
            head_img = np.array(Image.open(io.BytesIO(head_bytes)).convert("RGB"))

            left_images.append(left_img)
            right_images.append(right_img)
            head_images.append(head_img)
            original_indices.append(t)

        hdf5_file.close()

    return {
        "left": left_images,
        "right": right_images,
        "head": head_images,
        "instruction": instruction,
        "num_timesteps": len(left_images),
        "total_timesteps": total_timesteps,
        "original_indices": original_indices,
    }


def create_video(
    episode_data: dict,
    output_path: str,
    fps: int = 10,
    add_text: bool = True,
    task_name: str = "",
    episode_idx: int = 0,
):
    """Create a video with side-by-side camera views."""
    left_images = episode_data["left"]
    right_images = episode_data["right"]
    head_images = episode_data["head"]
    instruction = episode_data["instruction"]
    num_frames = len(left_images)
    total_timesteps = episode_data["total_timesteps"]
    original_indices = episode_data["original_indices"]

    if num_frames == 0:
        print(f"Warning: No valid frames for {output_path}, skipping")
        return

    # Get image dimensions
    h, w = left_images[0].shape[:2]

    # Create combined frame: [left | head | right] stacked, with text area at bottom
    text_height = 80 if add_text else 0
    combined_w = w * 3
    combined_h = h + text_height

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))

    for i in range(num_frames):
        # Create combined frame
        frame = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

        # Place images: left | head | right
        frame[:h, :w] = left_images[i]
        frame[:h, w:2*w] = head_images[i]
        frame[:h, 2*w:3*w] = right_images[i]

        # Add labels
        if add_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)
            thickness = 1

            # Camera labels
            cv2.putText(frame, "Left Camera", (10, h - 10), font, font_scale, color, thickness)
            cv2.putText(frame, "Head Camera", (w + 10, h - 10), font, font_scale, color, thickness)
            cv2.putText(frame, "Right Camera", (2*w + 10, h - 10), font, font_scale, color, thickness)

            # Task and frame info at bottom
            cv2.putText(frame, f"Task: {task_name}", (10, h + 20), font, font_scale, color, thickness)
            cv2.putText(frame, f"Valid Frame: {i+1}/{num_frames} (orig t={original_indices[i]}/{total_timesteps})",
                       (10, h + 40), font, font_scale, color, thickness)
            cv2.putText(frame, f"Episode: {episode_idx}", (10, h + 60), font, font_scale, color, thickness)

            # Instruction (truncated if too long)
            instr_text = instruction[:80] + "..." if len(instruction) > 80 else instruction
            cv2.putText(frame, f"Instruction: {instr_text}", (300, h + 60), font, 0.4, color, thickness)

        out.write(frame)

    out.release()
    print(f"Video saved to {output_path} ({num_frames} valid frames / {total_timesteps} total)")


def main():
    parser = argparse.ArgumentParser(description="Generate filtered videos from RoboTwin episodes")
    parser.add_argument("--dataset_root", type=str, default="/mnt/robotwin/dataset",
                        help="Path to RoboTwin dataset")
    parser.add_argument("--valid_timesteps_path", type=str,
                        default="data/robotwin_valid_timesteps_stack_blocks_two.json",
                        help="Path to valid timesteps JSON file")
    parser.add_argument("--output_dir", type=str, default="episode_videos_filtered",
                        help="Output directory for videos")
    parser.add_argument("--robot_type", type=str, default="aloha-agilex",
                        help="Robot type")
    parser.add_argument("--variant", type=str, default="clean_1k",
                        help="Variant")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name to filter by (optional, e.g., 'stack_blocks_two')")
    parser.add_argument("--num_videos", type=int, default=5,
                        help="Number of videos to generate")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video FPS")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Specific episode indices to render, e.g., '247,270,561'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for episode selection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load valid timesteps
    print(f"Loading valid timesteps from {args.valid_timesteps_path}...")
    with open(args.valid_timesteps_path) as f:
        all_valid_timesteps = json.load(f)
    print(f"  Loaded valid timesteps for {len(all_valid_timesteps)} episodes")

    # Filter episodes that match our robot_type, variant, and optionally task
    matching_episodes = {}
    for episode_key, timesteps in all_valid_timesteps.items():
        # episode_key format: "stack_blocks_two/aloha-agilex_clean_1k/episode247"
        parts = episode_key.split("/")
        if len(parts) != 3:
            continue

        task_name = parts[0]
        robot_variant = parts[1]  # e.g., "aloha-agilex_clean_1k"
        episode_str = parts[2]  # e.g., "episode247"

        # Check robot_type and variant match
        expected_prefix = f"{args.robot_type}_{args.variant}"
        if robot_variant != expected_prefix:
            continue

        # Check task filter if specified
        if args.task is not None and task_name != args.task:
            continue

        episode_idx = int(episode_str.replace("episode", ""))
        matching_episodes[episode_idx] = {
            "task": task_name,
            "episode_idx": episode_idx,
            "valid_timesteps": timesteps,
            "episode_key": episode_key,
        }

    filter_desc = f"{args.robot_type}_{args.variant}"
    if args.task:
        filter_desc = f"{args.task}/{filter_desc}"
    print(f"  Found {len(matching_episodes)} episodes matching {filter_desc}")

    if len(matching_episodes) == 0:
        print("No matching episodes found. Check robot_type, variant, and task parameters.")
        return

    # Select episodes to render
    if args.episodes:
        # Parse specific episodes
        indices = [int(i) for i in args.episodes.split(",")]
        selected = [matching_episodes[idx] for idx in indices if idx in matching_episodes]
    else:
        # Random sample
        np.random.seed(args.seed)
        all_indices = list(matching_episodes.keys())
        sample_indices = np.random.choice(
            all_indices,
            min(args.num_videos, len(all_indices)),
            replace=False
        )
        selected = [matching_episodes[idx] for idx in sample_indices]

    print(f"\nGenerating {len(selected)} videos...")

    # Get task name from first episode
    task_name = selected[0]["task"] if selected else "unknown"
    zip_path = Path(args.dataset_root) / task_name / f"{args.robot_type}_{args.variant}.zip"

    if not zip_path.exists():
        print(f"Error: Zip file not found at {zip_path}")
        return

    for ep_info in tqdm(selected, desc="Generating videos"):
        ep_idx = ep_info["episode_idx"]
        valid_timesteps = ep_info["valid_timesteps"]

        print(f"\nProcessing {task_name} episode {ep_idx} ({len(valid_timesteps)} valid frames)...")

        # Load episode with only valid timesteps
        episode_data = load_episode_images_filtered(
            str(zip_path),
            args.robot_type,
            args.variant,
            ep_idx,
            valid_timesteps,
        )

        # Create video
        output_path = output_dir / f"{task_name}_ep{ep_idx}_filtered.mp4"
        create_video(
            episode_data,
            str(output_path),
            fps=args.fps,
            task_name=task_name,
            episode_idx=ep_idx,
        )

    print(f"\nDone! Videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
