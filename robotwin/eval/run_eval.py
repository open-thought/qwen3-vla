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
import subprocess
from datetime import datetime

# RoboTwin imports
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions

# Local imports
from eval.qwen3_vla_policy import Qwen3VLAPolicy, get_model, reset_model
from eval.video_recorder import MultiCameraRecorder, get_observer_rgb


def get_camera_config(camera_type: str) -> dict:
    """Load camera configuration from RoboTwin."""
    camera_config_path = ROBOTWIN_DIR / "task_config" / "_camera_config.yml"
    with open(camera_config_path, "r") as f:
        camera_configs = yaml.safe_load(f)
    return camera_configs.get(camera_type, {"w": 320, "h": 240})


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
    execute_steps: int = 1,
    use_builtin_video: bool = False,
    max_steps: int = None,
):
    """
    Run evaluation on a single task.

    Args:
        task_name: Name of the task to evaluate
        model: Qwen3VLAPolicy instance
        num_episodes: Number of episodes to evaluate
        task_config_name: Task configuration name
        instruction_type: Type of instruction ("seen" or "unseen")
        record_video: Whether to record videos (multi-camera)
        video_output_dir: Directory for video output
        seed: Random seed
        execute_steps: Number of action steps to execute before re-predicting
        use_builtin_video: Use RoboTwin's built-in video recording (head camera only,
                           records every simulation step inside take_action)
        max_steps: Maximum simulation steps per episode (overrides task-specific limit)

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

    # Video recorder (multi-camera)
    recorder = None
    if record_video and not use_builtin_video:
        recorder = MultiCameraRecorder(
            output_dir=f"{video_output_dir}/{task_name}",
            fps=10,
        )

    # Built-in RoboTwin video recording setup
    builtin_video_dir = None
    video_size = None
    if use_builtin_video:
        builtin_video_dir = Path(video_output_dir) / task_name
        builtin_video_dir.mkdir(parents=True, exist_ok=True)
        # Get camera resolution for ffmpeg
        head_camera_type = config.get("camera", {}).get("head_camera_type", "default")
        camera_cfg = get_camera_config(head_camera_type)
        video_size = f"{camera_cfg['w']}x{camera_cfg['h']}"
        # Set in config so TASK_ENV knows about it
        config["eval_video_save_dir"] = str(builtin_video_dir)

    # Load step limit (use max_steps override if provided)
    if max_steps is not None:
        step_limit = max_steps
    else:
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

        # Start video recording (multi-camera)
        if recorder:
            recorder.start_episode(episode_count, task_name)

        # Start built-in video recording (head camera, high frame rate)
        ffmpeg_process = None
        if use_builtin_video and builtin_video_dir:
            video_path = builtin_video_dir / f"episode{episode_count}.mp4"
            ffmpeg_process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "error",
                    "-f", "rawvideo",
                    "-pixel_format", "rgb24",
                    "-video_size", video_size,
                    "-framerate", "10",
                    "-i", "-",
                    "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264",
                    "-crf", "23",
                    str(video_path),
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg_process)

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

            # Get action from policy and execute
            # Note: policy_eval executes `execute_steps` actions internally
            # For accurate video recording with execute_steps > 1, we record after each action
            instruction = TASK_ENV.get_instruction()
            actions = model.get_action(observation, instruction, execute_steps=execute_steps)

            steps_to_execute = min(execute_steps, len(actions))
            for i in range(steps_to_execute):
                TASK_ENV.take_action(actions[i], action_type='qpos')
                observation = TASK_ENV.get_obs()

                # Record frame after each action (not just at policy call)
                if recorder and i < steps_to_execute - 1:  # Skip last frame, it's recorded at loop start
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

                if TASK_ENV.eval_success:
                    break

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

        # End video recording (multi-camera)
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

        # End built-in video recording
        if ffmpeg_process:
            TASK_ENV._del_eval_video_ffmpeg()
            status = "success" if success else "fail"
            # Rename video file to include success/fail status
            old_path = builtin_video_dir / f"episode{episode_count}.mp4"
            new_path = builtin_video_dir / f"episode{episode_count}_{status}.mp4"
            if old_path.exists():
                old_path.rename(new_path)
                print(f"  Video saved: {new_path}")

        TASK_ENV.close_env()
        episode_count += 1

        # Print running stats
        print(f"  Success rate: {success_count}/{episode_count} ({100*success_count/episode_count:.1f}%)")

    # Cleanup
    if recorder:
        recorder.close()

    # Print timing statistics
    model.print_timing_stats()

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
        choices=["demo_clean", "demo_randomized"],
        help="Task configuration name: 'demo_clean' (white table, no distractors) or "
             "'demo_randomized' (random backgrounds, textures, lighting, table clutter)"
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
        "--builtin_video",
        action="store_true",
        help="Use RoboTwin's built-in video recording (head camera only, records every sim step)"
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
        "--execute_steps", "-e",
        type=int,
        default=1,
        help="Number of action steps to execute before re-predicting (1=closed-loop, action_horizon=open-loop)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum simulation steps per episode (overrides task-specific limit from _eval_step_limit.yml)"
    )
    parser.add_argument(
        "--debug_actions",
        action="store_true",
        help="Print decoded action values for debugging"
    )
    parser.add_argument(
        "--binarize_gripper",
        action="store_true",
        help="Binarize gripper actions to 0 (closed) or 1 (open) using threshold"
    )
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=0.5,
        help="Threshold for gripper binarization (default: 0.5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (0 for greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p value"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="bspline",
        choices=["bspline", "bin"],
        help="Action tokenizer type: 'bspline' (smooth trajectory) or 'bin' (OpenVLA-style bins)"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=255,
        help="Number of bins for quantization (default: 255 for exact zero with symmetric bounds)"
    )
    parser.add_argument(
        "--symmetric_delta_norm",
        action="store_true",
        help="Use symmetric normalization for delta actions (0 maps to 0)"
    )
    # B-spline tokenizer parameters
    parser.add_argument(
        "--bspline_n_control_points",
        type=int,
        default=8,
        help="Number of B-spline control points per DoF (default: 8)"
    )
    parser.add_argument(
        "--bspline_degree",
        type=int,
        default=4,
        help="B-spline polynomial degree (default: 4)"
    )
    parser.add_argument(
        "--bspline_bounds",
        type=float,
        nargs=2,
        default=[-1.5, 1.5],
        help="Bounds for B-spline control point values (default: -1.5 1.5)"
    )
    parser.add_argument(
        "--bspline_token_order",
        type=str,
        default="basis_first",
        choices=["basis_first", "joint_first"],
        help="B-spline token ordering mode (default: basis_first)"
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
        "debug_actions": args.debug_actions,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tokenizer_type": args.tokenizer_type,
        "n_bins": args.n_bins,
        "symmetric_delta_norm": args.symmetric_delta_norm,
        "binarize_gripper": args.binarize_gripper,
        "gripper_threshold": args.gripper_threshold,
        "bspline_n_control_points": args.bspline_n_control_points,
        "bspline_degree": args.bspline_degree,
        "bspline_bounds": tuple(args.bspline_bounds),
        "bspline_token_order": args.bspline_token_order,
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
            execute_steps=args.execute_steps,
            use_builtin_video=args.builtin_video,
            max_steps=args.max_steps,
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
