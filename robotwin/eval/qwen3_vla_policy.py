"""
Qwen3-VLA Policy for RoboTwin Evaluation.

Implements the RoboTwin policy interface for evaluating fine-tuned Qwen3-VL models
with FAST action tokenization.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import time
from PIL import Image
from typing import Optional
from transformers import AutoModelForImageTextToText, AutoProcessor

from action_tokenizer import ActionTokenizer
from normalization import MultiRobotNormalizer, discretize_normalized_values


class Qwen3VLAPolicy:
    """
    Qwen3-VLA policy for RoboTwin evaluation.

    Handles:
    - Model loading from checkpoint
    - Observation encoding (images + state → prompt)
    - Action generation via autoregressive decoding
    - Action decoding (FAST tokens → denormalized delta actions)
    - Delta-to-absolute action conversion
    """

    # Token range for FAST tokens
    FAST_TOKEN_START = 151936
    FAST_TOKEN_END = 153983
    # EOT token to mark end of action sequence (Qwen3 <|im_end|>)
    EOT_TOKEN_ID = 151645

    def __init__(
        self,
        checkpoint_path: str,
        norm_stats_path: str,
        action_horizon: int = 16,
        robot_type: str = "aloha-agilex",
        device: str = "cuda:0",
        max_new_tokens: int = 150,  # Typical FAST sequences are 91-99 tokens
        debug_timing: bool = True,  # Print generation timing info
        debug_actions: bool = False,  # Print decoded action values
        temperature: float = 0.6,  # Sampling temperature (0 for greedy)
        top_p: float = 0.95,  # Nucleus sampling top-p
    ):
        """
        Initialize the Qwen3-VLA policy.

        Args:
            checkpoint_path: Path to fine-tuned model checkpoint
            norm_stats_path: Path to normalization statistics JSON
            action_horizon: Number of action steps to predict
            robot_type: Robot type for normalization (e.g., "aloha-agilex")
            device: Device to run model on
            max_new_tokens: Maximum tokens to generate
            debug_timing: Print generation timing info
            debug_actions: Print decoded action values
            temperature: Sampling temperature (0 for greedy decoding)
            top_p: Nucleus sampling top-p value
        """
        self.checkpoint_path = checkpoint_path
        self.action_horizon = action_horizon
        self.robot_type = robot_type
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.debug_timing = debug_timing
        self.debug_actions = debug_actions
        self.temperature = temperature
        self.top_p = top_p

        # Timing statistics
        self.total_generate_time = 0.0
        self.total_tokens_generated = 0
        self.num_generate_calls = 0

        print(f"Loading Qwen3-VLA from {checkpoint_path}...")

        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Load processor from base model (checkpoint tokenizer files may be incomplete)
        # Use the base Qwen3-VL-2B-Instruct processor
        base_model_name = "Qwen/Qwen3-VL-2B-Instruct"
        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )

        # Load action tokenizer (FAST)
        print("Loading FAST action tokenizer...")
        self.action_tokenizer = ActionTokenizer()

        # Load normalizer
        print(f"Loading normalization stats from {norm_stats_path}...")
        self.normalizer = MultiRobotNormalizer(norm_stats_path)

        # Get robot metadata
        metadata = self.normalizer.get_robot_metadata(robot_type)
        self.dof = metadata["dof"]
        # Action dim includes joints (2*dof) + grippers (2)
        self.action_dim = 2 * self.dof + 2  # Both arms + both grippers

        print(f"Robot: {robot_type}, DoF per arm: {self.dof}")
        print(f"Action dim: {self.action_dim} (joints: {2*self.dof}, grippers: 2), Horizon: {action_horizon}")

        # State for tracking
        self.current_qpos = None
        self.step_count = 0

    def reset(self):
        """Reset policy state at the start of a new episode."""
        self.current_qpos = None
        self.step_count = 0

    def print_timing_stats(self):
        """Print timing statistics summary."""
        if self.num_generate_calls > 0:
            avg_time = self.total_generate_time / self.num_generate_calls * 1000
            avg_tokens = self.total_tokens_generated / self.num_generate_calls
            avg_tok_per_sec = self.total_tokens_generated / self.total_generate_time if self.total_generate_time > 0 else 0
            print(f"\n  Timing Summary:")
            print(f"    Total generate calls: {self.num_generate_calls}")
            print(f"    Total tokens generated: {self.total_tokens_generated}")
            print(f"    Total generate time: {self.total_generate_time:.2f}s")
            print(f"    Avg time per call: {avg_time:.1f}ms")
            print(f"    Avg tokens per call: {avg_tokens:.1f}")
            print(f"    Avg throughput: {avg_tok_per_sec:.1f} tok/s")

    def reset_timing_stats(self):
        """Reset timing statistics."""
        self.total_generate_time = 0.0
        self.total_tokens_generated = 0
        self.num_generate_calls = 0

    def _prepare_images(self, observation: dict) -> list:
        """
        Prepare images from observation for model input.

        Args:
            observation: RoboTwin observation dict

        Returns:
            List of PIL Images [left, right, head]
        """
        images = []

        for cam_name in ["left_camera", "right_camera", "head_camera"]:
            rgb = observation["observation"][cam_name]["rgb"]

            # Ensure uint8
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

            # Convert to PIL
            pil_img = Image.fromarray(rgb)
            images.append(pil_img)

        return images

    def _get_state_vector(self, observation: dict) -> np.ndarray:
        """
        Extract state vector from observation.

        Args:
            observation: RoboTwin observation dict

        Returns:
            State vector (2*dof,) - both arms' joint positions
        """
        joint_action = observation["joint_action"]

        # Concatenate: left_arm + right_arm (without grippers for state)
        left_arm = np.array(joint_action["left_arm"])
        right_arm = np.array(joint_action["right_arm"])

        state = np.concatenate([left_arm, right_arm])
        return state

    def _get_gripper_state(self, observation: dict) -> np.ndarray:
        """
        Extract gripper states from observation.

        Args:
            observation: RoboTwin observation dict

        Returns:
            Gripper state (2,) - [left_gripper, right_gripper]
        """
        joint_action = observation["joint_action"]
        return np.array([
            joint_action["left_gripper"],
            joint_action["right_gripper"]
        ])

    def _build_prompt(
        self,
        instruction: str,
        discretized_state: np.ndarray,
    ) -> list:
        """
        Build the conversation prompt for the model.

        Args:
            instruction: Task instruction text
            discretized_state: State discretized to [0, 255]

        Returns:
            Conversation list for processor
        """
        # Format state as comma-separated integers
        state_str = ", ".join([str(int(s)) for s in discretized_state])

        prompt_text = f"""Task: {instruction}
Robot: {self.robot_type}
State: [{state_str}]"""

        return prompt_text

    def get_action(
        self,
        observation: dict,
        instruction: str,
    ) -> np.ndarray:
        """
        Get action from observation and instruction.

        Args:
            observation: RoboTwin observation dict
            instruction: Task instruction text

        Returns:
            Actions array (action_horizon, 14) - full qpos including grippers
        """
        # Prepare images
        images = self._prepare_images(observation)

        # Get current state (joints only) and grippers
        state = self._get_state_vector(observation)
        grippers = self._get_gripper_state(observation)

        # Store current qpos for delta conversion
        self.current_qpos = np.concatenate([
            observation["joint_action"]["left_arm"],
            [observation["joint_action"]["left_gripper"]],
            observation["joint_action"]["right_arm"],
            [observation["joint_action"]["right_gripper"]],
        ])

        # Normalize state and grippers for prompt (must match training format)
        normalized_state = self.normalizer.normalize_state(state, self.robot_type)
        normalized_grippers = self.normalizer.normalize_grippers(grippers, self.robot_type)

        # Concatenate normalized state and grippers: (2*dof + 2,) to match training
        normalized_state_with_grippers = np.concatenate([normalized_state, normalized_grippers])

        # Discretize to [0, 255] for prompt
        discretized_state = discretize_normalized_values(normalized_state_with_grippers, num_bins=256)

        # Build prompt text
        prompt_text = self._build_prompt(instruction, discretized_state)

        # Build conversation with images
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left camera:"},
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": "Right camera:"},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": "Head camera:"},
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate action tokens
        with torch.no_grad():
            torch.cuda.synchronize() if self.device.startswith("cuda") else None
            start_time = time.perf_counter()

            # Use sampling if temperature > 0, otherwise greedy decoding
            do_sample = self.temperature > 0
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                top_k=None,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=[self.EOT_TOKEN_ID],  # Stop at EOT token
                use_cache=True,  # Enable KV caching (should be default, but explicit)
            )

            torch.cuda.synchronize() if self.device.startswith("cuda") else None
            generate_time = time.perf_counter() - start_time

        # Extract generated tokens (remove input tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0, input_len:].cpu().tolist()
        num_tokens = len(generated_tokens)

        # Update timing statistics
        self.total_generate_time += generate_time
        self.total_tokens_generated += num_tokens
        self.num_generate_calls += 1

        if self.debug_timing:
            tokens_per_sec = num_tokens / generate_time if generate_time > 0 else 0
            avg_tokens_per_sec = self.total_tokens_generated / self.total_generate_time if self.total_generate_time > 0 else 0
            # Count FAST tokens vs other tokens
            num_fast = sum(1 for t in generated_tokens if self.FAST_TOKEN_START <= t <= self.FAST_TOKEN_END)
            num_eot = sum(1 for t in generated_tokens if t == self.EOT_TOKEN_ID)
            print(f"    Generate: {input_len} prompt + {num_tokens} output ({num_fast} FAST, {num_eot} EOT) "
                  f"in {generate_time*1000:.0f}ms ({tokens_per_sec:.1f} tok/s)")

        # Filter to FAST token range, stopping at EOT token
        fast_tokens = []
        for t in generated_tokens:
            if t == self.EOT_TOKEN_ID:
                break  # Stop at EOT token
            if self.FAST_TOKEN_START <= t <= self.FAST_TOKEN_END:
                fast_tokens.append(t)

        if self.debug_actions:
            # Print state info to verify it's changing between calls
            print(f"    State (first 6): [{', '.join(f'{x:.4f}' for x in state[:6])}]")
            print(f"    Discretized (first 6): [{', '.join(str(int(x)) for x in discretized_state[:6])}]")
            tokens_no_offset = [t - self.FAST_TOKEN_START for t in fast_tokens]
            print(f"    FAST tokens ({len(fast_tokens)}): {tokens_no_offset}")
            if self.num_generate_calls <= 2:
                # Print full prompt on first few calls
                print(f"    Prompt: {prompt_text}")

        if len(fast_tokens) == 0:
            print("Warning: No FAST tokens generated, returning zero actions")
            return np.zeros((self.action_horizon, 14))

        # Decode FAST tokens to normalized delta actions
        try:
            normalized_deltas = self.action_tokenizer.decode(
                [fast_tokens],
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
            )[0]  # Remove batch dim
        except Exception as e:
            # Remove offset for debugging
            tokens_no_offset = [t - self.FAST_TOKEN_START for t in fast_tokens]
            print(f"Error decoding tokens: {e}")
            print(f"Tokens: {tokens_no_offset[:100]}{'...' if len(tokens_no_offset) > 100 else ''}")
            # Return zero deltas on error
            return np.zeros((self.action_horizon, 14))

        # Split normalized_deltas into joint deltas and gripper values
        # normalized_deltas shape: (action_horizon, 2*dof + 2) = (action_horizon, 14)
        joint_dim = 2 * self.dof  # 12 for 6-DoF dual-arm
        normalized_joint_deltas = normalized_deltas[:, :joint_dim]
        normalized_grippers = normalized_deltas[:, joint_dim:]

        # Denormalize joint deltas and gripper values separately
        delta_joints = self.normalizer.denormalize_delta_actions(
            normalized_joint_deltas,
            robot_type=self.robot_type,
        )
        future_grippers = self.normalizer.denormalize_grippers(
            normalized_grippers,
            robot_type=self.robot_type,
        )

        # Convert delta joint actions to absolute qpos, with predicted grippers
        actions = self._delta_to_absolute(delta_joints, future_grippers)

        if self.debug_actions:
            print(f"    Delta joints shape: {delta_joints.shape}, range: [{delta_joints.min():.4f}, {delta_joints.max():.4f}]")
            print(f"    Future grippers shape: {future_grippers.shape}, range: [{future_grippers.min():.4f}, {future_grippers.max():.4f}]")
            print(f"    Delta joints (decoded from model):")
            for i, delta in enumerate(delta_joints):
                print(f"      Delta[{i:2d}]: [{', '.join(f'{x:8.4f}' for x in delta)}]")

        self.step_count += 1

        return actions

    def _delta_to_absolute(
        self,
        delta_joints: np.ndarray,
        future_grippers: np.ndarray,
    ) -> np.ndarray:
        """
        Convert delta joint actions to absolute joint positions.

        Args:
            delta_joints: Delta actions for joints (action_horizon, 2*dof)
            future_grippers: Predicted gripper positions (action_horizon, 2)

        Returns:
            Absolute actions (action_horizon, 14) with grippers
        """
        # Start from current qpos (without grippers)
        current_arm_pos = np.concatenate([
            self.current_qpos[:self.dof],  # left arm
            self.current_qpos[self.dof + 1:self.dof + 1 + self.dof],  # right arm
        ])

        # Accumulate deltas for joint positions
        absolute_arm = np.zeros_like(delta_joints)
        for t in range(delta_joints.shape[0]):
            if t == 0:
                absolute_arm[t] = current_arm_pos + delta_joints[t]
            else:
                absolute_arm[t] = absolute_arm[t-1] + delta_joints[t]

        # Combine joint positions with predicted gripper values
        actions = np.zeros((delta_joints.shape[0], 14))

        for t in range(delta_joints.shape[0]):
            # Left arm (6) + left gripper (1) + right arm (6) + right gripper (1)
            actions[t, :self.dof] = absolute_arm[t, :self.dof]
            actions[t, self.dof] = future_grippers[t, 0]  # left gripper (predicted)
            actions[t, self.dof + 1:self.dof + 1 + self.dof] = absolute_arm[t, self.dof:]
            actions[t, -1] = future_grippers[t, 1]  # right gripper (predicted)

        return actions


# ============================================================================
# RoboTwin Policy Interface Functions
# ============================================================================

_policy_instance: Optional[Qwen3VLAPolicy] = None


def get_model(usr_args: dict) -> Qwen3VLAPolicy:
    """
    Load the Qwen3-VLA model.

    Args:
        usr_args: Configuration dictionary with:
            - checkpoint_path: Path to model checkpoint
            - norm_stats_path: Path to normalization stats
            - action_horizon: Action prediction horizon (default: 16)
            - robot_type: Robot type (default: "aloha-agilex")
            - device: CUDA device (default: "cuda:0")
            - temperature: Sampling temperature (default: 0.6, 0 for greedy)
            - top_p: Nucleus sampling top-p (default: 0.95)

    Returns:
        Qwen3VLAPolicy instance
    """
    global _policy_instance

    checkpoint_path = usr_args.get(
        "checkpoint_path",
        "/mnt/robotwin/checkpoints/qwen3-vla-robotwin-full-h16/step_20000"
    )
    norm_stats_path = usr_args.get(
        "norm_stats_path",
        str(SCRIPT_DIR / "data" / "robotwin_norm_stats_h16.json")
    )
    action_horizon = usr_args.get("action_horizon", 16)
    robot_type = usr_args.get("robot_type", "aloha-agilex")
    device = usr_args.get("device", "cuda:0")
    debug_actions = usr_args.get("debug_actions", False)
    temperature = usr_args.get("temperature", 0.6)
    top_p = usr_args.get("top_p", 0.95)

    _policy_instance = Qwen3VLAPolicy(
        checkpoint_path=checkpoint_path,
        norm_stats_path=norm_stats_path,
        action_horizon=action_horizon,
        robot_type=robot_type,
        device=device,
        debug_actions=debug_actions,
        temperature=temperature,
        top_p=top_p,
    )

    return _policy_instance


def eval(TASK_ENV, model: Qwen3VLAPolicy, observation: dict, execute_steps: int = 1):
    """
    Run one evaluation step.

    Args:
        TASK_ENV: RoboTwin task environment
        model: Qwen3VLAPolicy instance
        observation: Current observation
        execute_steps: Number of action steps to execute before re-predicting.
                       1 = closed-loop (re-predict after each step)
                       action_horizon = open-loop (execute full chunk)
                       Default: 1 (closed-loop for better accuracy)

    Returns:
        Updated observation after executing actions
    """
    # Get instruction from environment
    instruction = TASK_ENV.get_instruction()

    # Get actions from model
    actions = model.get_action(observation, instruction)

    # Execute only the first `execute_steps` actions, then return for re-prediction
    steps_to_execute = min(execute_steps, len(actions))

    for i in range(steps_to_execute):
        TASK_ENV.take_action(actions[i], action_type='qpos')
        observation = TASK_ENV.get_obs()

        # Check for early success
        if TASK_ENV.eval_success:
            break

    return observation


def reset_model(model: Qwen3VLAPolicy):
    """
    Reset model state at the start of a new episode.

    Args:
        model: Qwen3VLAPolicy instance
    """
    model.reset()
