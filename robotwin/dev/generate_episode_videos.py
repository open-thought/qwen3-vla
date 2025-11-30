#!/usr/bin/env python3
"""
Generate videos from RoboTwin training episodes.

Creates side-by-side videos showing all 3 camera views (left, right, head)
for selected episodes.
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


def load_episode_images(zip_path: str, robot_type: str, variant: str, episode_idx: int):
    """Load all images from an episode."""
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

        # Get number of timesteps
        num_timesteps = len(hdf5_file["observation"]["head_camera"]["rgb"])

        # Load all images
        left_images = []
        right_images = []
        head_images = []

        for t in range(num_timesteps):
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

        hdf5_file.close()

    return {
        "left": left_images,
        "right": right_images,
        "head": head_images,
        "instruction": instruction,
        "num_timesteps": num_timesteps,
    }


def create_video(
    episode_data: dict,
    output_path: str,
    fps: int = 10,
    add_text: bool = True,
    task_name: str = "",
):
    """Create a video with side-by-side camera views."""
    left_images = episode_data["left"]
    right_images = episode_data["right"]
    head_images = episode_data["head"]
    instruction = episode_data["instruction"]
    num_frames = len(left_images)

    # Get image dimensions
    h, w = left_images[0].shape[:2]

    # Create combined frame: [left | head | right] stacked, with text area at bottom
    text_height = 60 if add_text else 0
    combined_w = w * 3
    combined_h = h + text_height

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_w, combined_h))

    for i in range(num_frames):
        # Create combined frame
        frame = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

        # Place images: left | head | right
        # Images are already in RGB format from PIL, use directly
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
            cv2.putText(frame, f"Frame: {i+1}/{num_frames}", (10, h + 40), font, font_scale, color, thickness)

            # Instruction (truncated if too long)
            instr_text = instruction[:100] + "..." if len(instruction) > 100 else instruction
            cv2.putText(frame, f"Instruction: {instr_text}", (300, h + 30), font, 0.4, color, thickness)

        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")


def find_episodes(dataset_root: str, task: str = None, robot_type: str = "aloha-agilex", variant: str = "clean_50"):
    """Find available episodes."""
    dataset_path = Path(dataset_root)
    episodes = []

    if task:
        tasks = [task]
    else:
        tasks = [d.name for d in dataset_path.iterdir() if d.is_dir()]

    for task_name in tasks:
        task_path = dataset_path / task_name
        zip_name = f"{robot_type}_{variant}.zip"
        zip_path = task_path / zip_name

        if zip_path.exists():
            # Count episodes in zip
            with zipfile.ZipFile(zip_path, 'r') as zf:
                stem = f"{robot_type}_{variant}"
                data_files = [f for f in zf.namelist() if f.startswith(f"{stem}/data/episode") and f.endswith(".hdf5")]
                num_episodes = len(data_files)

            for ep_idx in range(num_episodes):
                episodes.append({
                    "task": task_name,
                    "robot_type": robot_type,
                    "variant": variant,
                    "episode_idx": ep_idx,
                    "zip_path": str(zip_path),
                })

    return episodes


def main():
    parser = argparse.ArgumentParser(description="Generate videos from RoboTwin episodes")
    parser.add_argument("--dataset_root", type=str, default="/mnt/robotwin/dataset",
                        help="Path to RoboTwin dataset")
    parser.add_argument("--output_dir", type=str, default="episode_videos",
                        help="Output directory for videos")
    parser.add_argument("--task", type=str, default=None,
                        help="Specific task to generate videos for (default: sample from all)")
    parser.add_argument("--robot_type", type=str, default="aloha-agilex",
                        help="Robot type")
    parser.add_argument("--variant", type=str, default="clean_50",
                        help="Variant")
    parser.add_argument("--num_videos", type=int, default=5,
                        help="Number of videos to generate")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video FPS")
    parser.add_argument("--episodes", type=str, default=None,
                        help="Specific episodes to render, e.g., 'adjust_bottle:0,1,2' or '0,1,2' with --task")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find episodes
    print(f"Scanning dataset at {args.dataset_root}...")
    all_episodes = find_episodes(
        args.dataset_root,
        task=args.task,
        robot_type=args.robot_type,
        variant=args.variant,
    )
    print(f"Found {len(all_episodes)} episodes")

    # Select episodes to render
    if args.episodes:
        # Parse specific episodes
        selected = []
        if ":" in args.episodes:
            task, indices = args.episodes.split(":")
            indices = [int(i) for i in indices.split(",")]
            for ep in all_episodes:
                if ep["task"] == task and ep["episode_idx"] in indices:
                    selected.append(ep)
        else:
            indices = [int(i) for i in args.episodes.split(",")]
            for ep in all_episodes:
                if ep["episode_idx"] in indices:
                    selected.append(ep)
                    if len(selected) >= len(indices):
                        break
    else:
        # Random sample
        np.random.seed(42)
        indices = np.random.choice(len(all_episodes), min(args.num_videos, len(all_episodes)), replace=False)
        selected = [all_episodes[i] for i in indices]

    print(f"\nGenerating {len(selected)} videos...")

    for ep_info in tqdm(selected, desc="Generating videos"):
        task = ep_info["task"]
        ep_idx = ep_info["episode_idx"]
        zip_path = ep_info["zip_path"]

        print(f"\nProcessing {task} episode {ep_idx}...")

        # Load episode
        episode_data = load_episode_images(
            zip_path,
            ep_info["robot_type"],
            ep_info["variant"],
            ep_idx,
        )

        # Create video
        output_path = output_dir / f"{task}_ep{ep_idx}.mp4"
        create_video(
            episode_data,
            str(output_path),
            fps=args.fps,
            task_name=task,
        )

    print(f"\nDone! Videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
