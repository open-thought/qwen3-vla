#!/usr/bin/env python3
"""
Standalone evaluation script for Qwen3-VLA on RoboTwin.

This script runs evaluation without modifying the RoboTwin repository.
It sets up the environment, loads the model, and runs evaluation on specified tasks.

Usage:
    python eval/run_eval.py --task place_a2b_left --num_episodes 10
    python eval/run_eval.py --task place_a2b_left place_a2b_right --num_episodes 5
"""

import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
ROBOTWIN_DIR = Path("/home/koepf/robotics/RoboTwin")

# Change to RoboTwin directory (required for relative path imports in RoboTwin)
os.chdir(ROBOTWIN_DIR)

# Add paths for imports
sys.path.insert(0, str(ROBOTWIN_DIR))
sys.path.insert(0, str(ROBOTWIN_DIR / "policy"))
sys.path.insert(0, str(ROBOTWIN_DIR / "description" / "utils"))
sys.path.insert(0, str(PROJECT_DIR))

import argparse
import yaml
import numpy as np
import importlib
from datetime import datetime

# RoboTwin imports
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions

# Local imports
from eval.qwen3_vla_policy import Qwen3VLAPolicy, get_model, eval as policy_eval, reset_model
from eval.video_recorder import MultiCameraRecorder, get_observer_rgb


def class_decorator(task_name: str):
    """Load task environment class."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception as e:
        raise SystemExit(f"Failed to load task '{task_name}': {e}")
    return env_instance


def load_task_config(config_name: str = "demo_clean") -> dict:
    """Load task configuration from RoboTwin."""
    config_path = ROBOTWIN_DIR / "task_config" / f"{config_name}.yml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_embodiment_config(robot_file: str) -> dict:
    """Load robot embodiment configuration."""
    config_path = Path(robot_file) / "config.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_environment_config(
    task_name: str,
    task_config_name: str = "demo_clean",
) -> dict:
    """Setup complete environment configuration."""
    config = load_task_config(task_config_name)
    config["task_name"] = task_name
    config["task_config"] = task_config_name

    # Load embodiment configuration
    embodiment_config_path = ROBOTWIN_DIR / "task_config" / "_embodiment_config.yml"
    with open(embodiment_config_path, "r") as f:
        embodiment_types = yaml.safe_load(f)

    embodiment_type = config.get("embodiment", ["aloha-agilex"])

    def get_embodiment_file(etype):
        return embodiment_types[etype]["file_path"]

    if len(embodiment_type) == 1:
        config["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        config["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        config["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        config["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        config["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        config["embodiment_dis"] = embodiment_type[2]
        config["dual_arm_embodied"] = False

    config["left_embodiment_config"] = load_embodiment_config(config["left_robot_file"])
    config["right_embodiment_config"] = load_embodiment_config(config["right_robot_file"])

    # Load camera configuration
    camera_config_path = ROBOTWIN_DIR / "task_config" / "_camera_config.yml"
    with open(camera_config_path, "r") as f:
        camera_config = yaml.safe_load(f)

    head_camera_type = config["camera"]["head_camera_type"]
    config["head_camera_h"] = camera_config[head_camera_type]["h"]
    config["head_camera_w"] = camera_config[head_camera_type]["w"]

    return config


def run_evaluation(
    task_name: str,
    model: Qwen3VLAPolicy,
    num_episodes: int = 10,
    task_config_name: str = "demo_clean",
    instruction_type: str = "unseen",
    record_video: bool = True,
    video_output_dir: str = "eval_videos",
    seed: int = 42,
):
    """
    Run evaluation on a single task.

    Args:
        task_name: Name of the task to evaluate
        model: Qwen3VLAPolicy instance
        num_episodes: Number of episodes to evaluate
        task_config_name: Task configuration name
        instruction_type: Type of instruction ("seen" or "unseen")
        record_video: Whether to record videos
        video_output_dir: Directory for video output
        seed: Random seed

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating task: {task_name}")
    print(f"{'='*60}")

    # Setup configuration
    config = setup_environment_config(task_name, task_config_name)
    config["eval_mode"] = True
    config["render_freq"] = 0  # No GUI rendering

    # Data type configuration for observations
    config["data_type"] = {
        "rgb": True,
        "third_view": True,  # Enable observer camera
        "depth": False,
        "pointcloud": False,
        "endpose": True,
        "qpos": True,
        "mesh_segmentation": False,
        "actor_segmentation": False,
    }

    # Load task environment
    TASK_ENV = class_decorator(task_name)

    # Video recorder
    recorder = None
    if record_video:
        recorder = MultiCameraRecorder(
            output_dir=f"{video_output_dir}/{task_name}",
            fps=10,
        )

    # Load step limit
    step_limit_path = ROBOTWIN_DIR / "task_config" / "_eval_step_limit.yml"
    with open(step_limit_path, "r") as f:
        step_limits = yaml.safe_load(f)
    step_limit = step_limits.get(task_name, 1000)

    # Evaluation loop
    st_seed = 100000 * (1 + seed)
    success_count = 0
    episode_count = 0
    tested_seeds = []

    while episode_count < num_episodes:
        now_seed = st_seed + episode_count

        # Expert check - verify seed produces valid scenario
        config["render_freq"] = 0
        try:
            TASK_ENV.setup_demo(now_ep_num=episode_count, seed=now_seed, is_test=True, **config)
            episode_info = TASK_ENV.play_once()
            TASK_ENV.close_env()
        except (UnStableError, Exception) as e:
            print(f"  Seed {now_seed} failed expert check: {e}")
            st_seed += 1
            continue

        if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
            print(f"  Seed {now_seed} failed expert validation")
            st_seed += 1
            continue

        # Valid seed found, run actual evaluation
        tested_seeds.append(now_seed)

        # Setup environment for policy evaluation
        TASK_ENV.setup_demo(now_ep_num=episode_count, seed=now_seed, is_test=True, **config)

        # Generate instruction
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, num_episodes)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction)

        # Start video recording
        if recorder:
            recorder.start_episode(episode_count, task_name)

        # Reset model state
        reset_model(model)

        # Set step limit
        TASK_ENV.step_lim = step_limit
        TASK_ENV.take_action_cnt = 0
        TASK_ENV.eval_success = False

        # Initial observation
        observation = TASK_ENV.get_obs()
        step = 0

        print(f"\n  Episode {episode_count + 1}/{num_episodes} (seed={now_seed})")
        print(f"  Instruction: {instruction[:80]}...")

        # Run policy
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and not TASK_ENV.eval_success:
            # Record frame before action
            if recorder:
                obs_data = observation["observation"]
                observer_rgb = None
                if "third_view_rgb" in observation:
                    observer_rgb = observation["third_view_rgb"]
                else:
                    observer_rgb = get_observer_rgb(TASK_ENV)

                recorder.add_frame(
                    head_rgb=obs_data["head_camera"]["rgb"],
                    left_rgb=obs_data["left_camera"]["rgb"],
                    right_rgb=obs_data["right_camera"]["rgb"],
                    observer_rgb=observer_rgb,
                    step=TASK_ENV.take_action_cnt,
                )

            # Get action from policy
            observation = policy_eval(TASK_ENV, model, observation)
            step += 1

            # Print progress
            print(f"    Step {TASK_ENV.take_action_cnt}/{step_limit}", end="\r")

        # Episode complete
        success = TASK_ENV.eval_success or TASK_ENV.check_success()
        if success:
            success_count += 1
            print(f"\n  Result: \033[92mSUCCESS\033[0m")
        else:
            print(f"\n  Result: \033[91mFAILED\033[0m")

        # End video recording
        if recorder:
            # Add final frame with success indicator
            obs_data = observation["observation"]
            observer_rgb = get_observer_rgb(TASK_ENV) if hasattr(TASK_ENV, 'cameras') else None
            recorder.add_frame(
                head_rgb=obs_data["head_camera"]["rgb"],
                left_rgb=obs_data["left_camera"]["rgb"],
                right_rgb=obs_data["right_camera"]["rgb"],
                observer_rgb=observer_rgb,
                step=TASK_ENV.take_action_cnt,
                success=success,
            )
            recorder.end_episode(success)

        TASK_ENV.close_env()
        episode_count += 1

        # Print running stats
        print(f"  Success rate: {success_count}/{episode_count} ({100*success_count/episode_count:.1f}%)")

    # Cleanup
    if recorder:
        recorder.close()

    results = {
        "task_name": task_name,
        "num_episodes": num_episodes,
        "success_count": success_count,
        "success_rate": success_count / num_episodes,
        "tested_seeds": tested_seeds,
    }

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Success Rate: {success_count}/{num_episodes} ({100*results['success_rate']:.1f}%)")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VLA on RoboTwin")

    # Task arguments
    parser.add_argument(
        "--task", "-t",
        type=str,
        nargs="+",
        default=["place_a2b_left"],
        help="Task name(s) to evaluate"
    )
    parser.add_argument(
        "--num_episodes", "-n",
        type=int,
        default=10,
        help="Number of episodes per task"
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="demo_clean",
        help="Task configuration name"
    )
    parser.add_argument(
        "--instruction_type",
        type=str,
        default="unseen",
        choices=["seen", "unseen"],
        help="Instruction type"
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/robotwin/checkpoints/qwen3-vla-robotwin-full-h16/step_20000",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--norm_stats",
        type=str,
        default=str(PROJECT_DIR / "data" / "robotwin_norm_stats_h16.json"),
        help="Path to normalization statistics"
    )
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=16,
        help="Action prediction horizon"
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="aloha-agilex",
        help="Robot type"
    )

    # Video arguments
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=True,
        help="Record evaluation videos"
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Disable video recording"
    )
    parser.add_argument(
        "--video_output_dir",
        type=str,
        default="eval_videos",
        help="Video output directory"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (e.g., cuda:0, cuda:1)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Handle video flag
    record_video = args.record_video and not args.no_video

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SAPIEN test (required by RoboTwin)
    print("Running SAPIEN render test...")
    from script.test_render import Sapien_TEST
    Sapien_TEST()

    # Load model
    print(f"\nLoading model on {args.device}...")
    model_args = {
        "checkpoint_path": args.checkpoint,
        "norm_stats_path": args.norm_stats,
        "action_horizon": args.action_horizon,
        "robot_type": args.robot_type,
        "device": args.device,
    }
    model = get_model(model_args)

    # Run evaluation on each task
    all_results = {}
    for task_name in args.task:
        results = run_evaluation(
            task_name=task_name,
            model=model,
            num_episodes=args.num_episodes,
            task_config_name=args.task_config,
            instruction_type=args.instruction_type,
            record_video=record_video,
            video_output_dir=args.video_output_dir,
            seed=args.seed,
        )
        all_results[task_name] = results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.yaml"

    with open(results_file, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for task_name, results in all_results.items():
        sr = results["success_rate"] * 100
        print(f"  {task_name}: {results['success_count']}/{results['num_episodes']} ({sr:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
