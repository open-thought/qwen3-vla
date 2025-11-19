import numpy as np
import sapien as sapien
import torch
import math
import base64
import io
import os
import re
import json
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional
import requests

# Optional pygame for human control mode
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available - human control mode disabled")

from mani_skill.agents.robots import Xlerobot
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import REGISTERED_SCENE_BUILDERS
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.building import actors
from mani_skill.utils import common, sapien_utils
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs.pose import Pose
import gymnasium as gym

# Import the fixed controller
from robot_controller_fixed import FixedRobotController


class VLMClient:
    """Client for interacting with VLM via OpenRouter API"""

    def __init__(self, api_key: str, model: str = "qwen/qwen3-vl-235b-a22b-instruct"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def encode_image_to_base64(self, image_array: np.ndarray) -> str:
        """Convert numpy image array to base64 string"""
        # Convert tensor to numpy if needed
        if torch.is_tensor(image_array):
            image_array = image_array.cpu().numpy()

        # Convert float32 [0, 1] to uint8 [0, 255]
        if image_array.dtype == np.float32 or image_array.dtype == np.float64:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)

        # Remove alpha channel if present (RGBA -> RGB)
        if image_array.shape[-1] == 4:
            image_array = image_array[..., :3]

        # Convert to PIL Image and encode
        image = Image.fromarray(image_array)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def query(self, prompt: str, images: List[np.ndarray], history_images: List[List[np.ndarray]] = None) -> str:
        """
        Query the VLM with a prompt and images, optionally including history

        Args:
            prompt: Text prompt for the VLM
            images: Current observation images [head_img, right_wrist_img]
            history_images: List of previous observation images, each element is [head_img, right_wrist_img]
        """
        # Build message content with images first, then prompt
        content = []

        # Add historical images if provided
        if history_images:
            for hist_idx, hist_imgs in enumerate(history_images):
                # Add text marker for this historical step
                content.append({
                    "type": "text",
                    "text": f"[Previous Step {hist_idx + 1}]"
                })
                # Add the historical images
                for img in hist_imgs:
                    img_base64 = self.encode_image_to_base64(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img_base64}
                    })

        # Add current step marker
        content.append({
            "type": "text",
            "text": "[Current Observation]"
        })

        # Add current images
        for img in images:
            img_base64 = self.encode_image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_base64}
            })

        # Add the prompt text at the end
        content.append({
            "type": "text",
            "text": prompt
        })

        # Make API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2048,  # Increased for thinking models
            "temperature": 0.2,
            # Prefer specific providers for Qwen models
            "provider": {
                "order": ["parasail"]  # These providers support Qwen models well
            }
        }

        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)

            # Log response status
            print(f"API Response Status: {response.status_code}")

            response.raise_for_status()
            result = response.json()

            # Log full response for debugging
            print(f"API Response Keys: {result.keys()}")

            # Check if response has expected structure
            if "choices" not in result:
                print(f"ERROR: No 'choices' in response. Full response: {result}")
                return "COMMAND: hold"

            if len(result["choices"]) == 0:
                print(f"ERROR: Empty choices array. Full response: {result}")
                return "COMMAND: hold"

            if "message" not in result["choices"][0]:
                print(f"ERROR: No 'message' in choice. Full response: {result}")
                return "COMMAND: hold"

            if "content" not in result["choices"][0]["message"]:
                print(f"ERROR: No 'content' in message. Full response: {result}")
                return "COMMAND: hold"

            content = result["choices"][0]["message"]["content"]

            # Check for empty content - may happen with thinking models
            if content is None or content == "":
                print(f"WARNING: Empty content in response.")

                # Log finish_reason if available
                if "finish_reason" in result["choices"][0]:
                    finish_reason = result['choices'][0]['finish_reason']
                    print(f"Finish reason: {finish_reason}")

                    if finish_reason == "length":
                        print("ERROR: Model hit token limit during thinking phase!")

                # Try to extract from reasoning field for thinking models
                message = result["choices"][0]["message"]
                if "reasoning" in message and message["reasoning"]:
                    print("Attempting to extract command from reasoning field...")
                    reasoning_text = message["reasoning"]

                    # Look for COMMAND: pattern in reasoning
                    if "COMMAND:" in reasoning_text:
                        # Extract the command
                        cmd_start = reasoning_text.find("COMMAND:")
                        cmd_text = reasoning_text[cmd_start:cmd_start+200]  # Get some context
                        print(f"Found command in reasoning: {cmd_text[:100]}")
                        return cmd_text
                    else:
                        print("No COMMAND: found in reasoning. Returning hold.")
                        return "COMMAND: hold"

                return ""

            return content if content is not None else ""

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error querying VLM: {e}")
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
            return "COMMAND: hold"
        except requests.exceptions.RequestException as e:
            print(f"Request Error querying VLM: {e}")
            return "COMMAND: hold"
        except KeyError as e:
            print(f"KeyError parsing VLM response: {e}")
            print(f"Full response: {result}")
            return "COMMAND: hold"
        except Exception as e:
            print(f"Unexpected error querying VLM: {e}")
            print(f"Error type: {type(e).__name__}")
            return "COMMAND: hold"


class CommandParser:
    """Parse natural language commands into robot actions"""

    @staticmethod
    def parse_command(command_text: str) -> Dict[str, Any]:
        """
        Parse natural language command into structured action dict.

        Expected commands:
        - "move ee forward 0.05" / "move ee backward 0.03"
        - "move ee up 0.02" / "move ee down 0.02"
        - "move ee right 0.02" / "move ee left 0.02"
        - "pitch up 0.1" / "pitch down 0.1"
        - "wrist roll 0.2" / "wrist roll -0.2"
        - "open gripper" / "close gripper" / "hold"

        Returns:
            dict with keys: ee_delta (x, y), pitch_delta, wrist_roll_delta, gripper
        """
        command_text = command_text.lower().strip()

        result = {
            "ee_delta": [0.0, 0.0],  # [x, y] in meters
            "pitch_delta": 0.0,
            "wrist_roll_delta": 0.0,
            "gripper": "hold"  # "open", "close", or "hold"
        }

        # Parse end-effector movement
        ee_forward_match = re.search(r'move\s+ee\s+forward\s+([\d.]+)', command_text)
        ee_backward_match = re.search(r'move\s+ee\s+backward\s+([\d.]+)', command_text)
        ee_up_match = re.search(r'move\s+ee\s+up\s+([\d.]+)', command_text)
        ee_down_match = re.search(r'move\s+ee\s+down\s+([\d.]+)', command_text)

        if ee_forward_match:
            result["ee_delta"][0] = float(ee_forward_match.group(1))
        if ee_backward_match:
            result["ee_delta"][0] = -float(ee_backward_match.group(1))
        if ee_up_match:
            result["ee_delta"][1] = float(ee_up_match.group(1))
        if ee_down_match:
            result["ee_delta"][1] = -float(ee_down_match.group(1))

        # Parse pitch
        pitch_up_match = re.search(r'pitch\s+up\s+([\d.]+)', command_text)
        pitch_down_match = re.search(r'pitch\s+down\s+([\d.]+)', command_text)
        if pitch_up_match:
            result["pitch_delta"] = float(pitch_up_match.group(1))
        if pitch_down_match:
            result["pitch_delta"] = -float(pitch_down_match.group(1))

        # Parse wrist roll
        wrist_match = re.search(r'wrist\s+roll\s+([-\d.]+)', command_text)
        if wrist_match:
            result["wrist_roll_delta"] = float(wrist_match.group(1))

        # Parse gripper
        if "open gripper" in command_text:
            result["gripper"] = "open"
        elif "close gripper" in command_text:
            result["gripper"] = "close"

        return result


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for a 2-link robotic arm (from keyboard demo)

    Parameters:
        x: End effector x coordinate
        y: End effector y coordinate
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)

    Returns:
        joint2, joint3: Joint angles in radians as defined in the URDF file
    """
    # Calculate joint2 and joint3 offsets in theta1 and theta2
    theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0

    # Calculate distance from origin to target point
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2  # Maximum reachable distance

    # If target point is beyond maximum workspace, scale it to the boundary
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max

    # If target point is less than minimum workspace (|l1-l2|), scale it
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min

    # Use law of cosines to calculate theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)

    # Clamp to valid range
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)

    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    # Convert theta1 and theta2 to joint2 and joint3 angles
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset

    # Ensure angles are within URDF limits
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))

    return joint2, joint3


class RobotController:
    """Controller for translating high-level commands to robot actions"""

    def __init__(self, action_dim: int = 6, initial_qpos=None):
        self.action_dim = action_dim
        self.tip_length = 0.108  # Length from wrist to end effector tip

        # Initial end effector position
        self.ee_pos = np.array([0.162, 0.118])
        self.pitch = 0.0
        self.wrist_roll = 1.57

        # Target joints (for P-controller) - initialize to current robot position if provided
        self.target_joints = np.zeros(16)  # Full 16D for dual arm mode

        if initial_qpos is not None:
            # Map initial qpos to target joints
            # The controller uses a simplified index mapping
            # Robot qpos has 17 joints, we track targets for the right arm subset
            qpos = initial_qpos.squeeze() if len(initial_qpos.shape) > 1 else initial_qpos

            # Initialize targets to match current position for right arm
            # This prevents unwanted motion when only gripper commands are issued
            if len(qpos) >= 17:
                # Right arm joint indices in robot: [3,6,9,11,13]
                # Map to controller target indices: [3,4,5,6,7] (we use different indexing)
                self.target_joints[3] = qpos[3]   # Rotation
                self.target_joints[4] = qpos[6]   # Pitch
                self.target_joints[5] = qpos[11]  # Wrist Pitch (was incorrectly mapped)
                self.target_joints[6] = qpos[13]  # Wrist Roll
                self.target_joints[12] = qpos[15] # Gripper

                # Set wrist_roll to match initial position
                self.wrist_roll = qpos[13]

                # Also update ee_pos to match initial robot configuration
                # This prevents IK from immediately trying to move to default position
                # We'll keep the default for now as IK computation is complex
        else:
            self.target_joints[6] = 1.57  # Default wrist roll

        # P gains
        self.p_gain = np.ones(16)
        self.p_gain[2:7] = 1.0   # First arm joints
        self.p_gain[12] = 0.05   # Gripper

    def update_from_command(self, parsed_command: Dict[str, Any]):
        """Update controller state based on parsed command"""
        # Only update joints that are actually commanded to change

        # Check if we need to update end effector position
        if parsed_command["ee_delta"][0] != 0.0 or parsed_command["ee_delta"][1] != 0.0:
            # Update end effector position
            self.ee_pos[0] += parsed_command["ee_delta"][0]
            self.ee_pos[1] += parsed_command["ee_delta"][1]

            # Calculate IK for arm joints
            compensated_y = self.ee_pos[1] + self.tip_length * math.sin(self.pitch)
            try:
                self.target_joints[3], self.target_joints[4] = inverse_kinematics(
                    self.ee_pos[0], compensated_y
                )
                # Apply pitch adjustment to joint 5
                self.target_joints[5] = self.target_joints[3] - self.target_joints[4] + self.pitch
            except Exception as e:
                print(f"IK error: {e}")

        # Update pitch if commanded
        if parsed_command["pitch_delta"] != 0.0:
            self.pitch += parsed_command["pitch_delta"]
            # Recalculate joint 5 with new pitch
            self.target_joints[5] = self.target_joints[3] - self.target_joints[4] + self.pitch

        # Update wrist roll if commanded
        if parsed_command["wrist_roll_delta"] != 0.0:
            self.wrist_roll += parsed_command["wrist_roll_delta"]
            self.target_joints[6] = self.wrist_roll

        # Update gripper only if commanded
        if parsed_command["gripper"] == "open":
            self.target_joints[12] = 2.5
        elif parsed_command["gripper"] == "close":
            self.target_joints[12] = 0.1
        # If gripper is "hold", don't change target_joints[12]

    def compute_action(self, current_qpos: np.ndarray) -> np.ndarray:
        """
        Compute action using P-controller.

        Args:
            current_qpos: Current joint positions (17D from robot)

        Returns:
            action: 6D action for single arm control [arm(5D), gripper(1D)]
        """
        # Map full qpos to action space
        # For single right arm control:
        # qpos indices [3,6,9,11,13] -> arm joints [0,1,2,3,4]
        # qpos index [15] -> gripper [5]

        current_mapped = np.zeros(6)
        if len(current_qpos.shape) > 1:
            current_qpos = current_qpos.squeeze()

        if len(current_qpos) >= 16:
            current_mapped[0] = current_qpos[3]   # Rotation
            current_mapped[1] = current_qpos[6]   # Pitch
            current_mapped[2] = current_qpos[9]   # Elbow
            current_mapped[3] = current_qpos[11]  # Wrist Pitch
            current_mapped[4] = current_qpos[13]  # Wrist Roll
            current_mapped[5] = current_qpos[15]  # Gripper

        # Compute delta using P-controller
        target_mapped = np.array([
            self.target_joints[3],
            self.target_joints[4],
            0.0,  # Joint not controlled via IK
            self.target_joints[5],
            self.target_joints[6],
            self.target_joints[12]
        ])

        action = self.p_gain[2:8] * (target_mapped - current_mapped)

        # Clip to reasonable bounds
        action = np.clip(action, -0.5, 0.5)

        return action[:6]  # Return 6D action


class KeyboardController:
    """Pygame-based keyboard controller for human debugging mode"""

    def __init__(self, window_width: int = 1600, window_height: int = 600):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is required for human control mode. Install with: pip install pygame")

        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Robot Control - Human Debug Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # Control state
        self.movement_delta = 0.03  # Default movement size
        self.pitch_delta = 0.1
        self.roll_delta = 0.2

        # Display areas for 3 camera views
        self.cam_width = window_width // 3
        self.cam_height = 480

        # Command history
        self.command_history = []
        self.max_history = 10

    def display_observations(self, images: List[np.ndarray], labels: List[str]):
        """Display camera images and controls"""
        # Clear screen
        self.screen.fill((30, 30, 30))

        # Display camera images
        for i, (img, label) in enumerate(zip(images, labels)):
            # Convert tensor to numpy if needed
            if torch.is_tensor(img):
                img = img.cpu().numpy()

            # Ensure image is in correct format
            # Remove batch dimension if present
            if len(img.shape) == 4:
                img = img.squeeze(0)

            # Convert to uint8 if needed
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Handle RGBA to RGB conversion if needed
            if img.shape[2] == 4:
                img = img[:, :, :3]

            # Ensure array is contiguous
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)

            # Debug shape if there are issues
            if len(img.shape) != 3:
                print(f"Warning: Image shape is {img.shape}, expected (H, W, 3)")
                continue

            # Convert to pygame surface - pygame expects (width, height, channels)
            # ManiSkill images are (height, width, channels), so we transpose
            try:
                img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            except ValueError as e:
                print(f"Error creating surface for {label}: {e}")
                print(f"Image shape: {img.shape}, dtype: {img.dtype}")
                continue
            img_surface = pygame.transform.scale(img_surface, (self.cam_width - 10, self.cam_height))

            # Display image
            x_pos = i * self.cam_width + 5
            self.screen.blit(img_surface, (x_pos, 5))

            # Draw label
            label_text = self.font.render(label, True, (255, 255, 255))
            self.screen.blit(label_text, (x_pos + 10, 10))

        # Display controls help
        y_offset = self.cam_height + 20
        help_texts = [
            "=== KEYBOARD CONTROLS ===",
            "Movement: W/S (forward/back), A/D (left/right), Q/E (up/down)",
            "Pitch: R/F (up/down)  |  Wrist Roll: T/G  |  Gripper: Space (open), X (close)",
            "Step Size: 1-5 (0.01m to 0.05m)  |  Hold: H  |  Quit: ESC",
            "",
            f"Current step size: {self.movement_delta:.3f}m",
            "",
            "=== COMMAND HISTORY ==="
        ]

        for text in help_texts:
            rendered = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(rendered, (10, y_offset))
            y_offset += 25

        # Display command history
        for cmd in self.command_history[-self.max_history:]:
            rendered = self.font.render(f"  {cmd}", True, (150, 255, 150))
            self.screen.blit(rendered, (10, y_offset))
            y_offset += 20

        pygame.display.flip()

    def get_command(self) -> Optional[str]:
        """Get keyboard input and return command string"""
        command = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"

            if event.type == pygame.KEYDOWN:
                # Movement commands
                if event.key == pygame.K_w:
                    command = f"move ee forward {self.movement_delta}"
                elif event.key == pygame.K_s:
                    command = f"move ee backward {self.movement_delta}"
                elif event.key == pygame.K_q:
                    command = f"move ee up {self.movement_delta}"
                elif event.key == pygame.K_e:
                    command = f"move ee down {self.movement_delta}"
                elif event.key == pygame.K_a:
                    command = f"move left {self.movement_delta}"
                elif event.key == pygame.K_d:
                    command = f"move right {self.movement_delta}"

                # Pitch commands
                elif event.key == pygame.K_r:
                    command = f"pitch up {self.pitch_delta}"
                elif event.key == pygame.K_f:
                    command = f"pitch down {self.pitch_delta}"

                # Wrist roll
                elif event.key == pygame.K_t:
                    command = f"wrist roll {self.roll_delta}"
                elif event.key == pygame.K_g:
                    command = f"wrist roll {-self.roll_delta}"

                # Gripper
                elif event.key == pygame.K_SPACE:
                    command = "open gripper"
                elif event.key == pygame.K_x:
                    command = "close gripper"

                # Hold
                elif event.key == pygame.K_h:
                    command = "hold"

                # Step size adjustment
                elif event.key == pygame.K_1:
                    self.movement_delta = 0.01
                    print(f"Step size set to {self.movement_delta}m")
                elif event.key == pygame.K_2:
                    self.movement_delta = 0.02
                    print(f"Step size set to {self.movement_delta}m")
                elif event.key == pygame.K_3:
                    self.movement_delta = 0.03
                    print(f"Step size set to {self.movement_delta}m")
                elif event.key == pygame.K_4:
                    self.movement_delta = 0.04
                    print(f"Step size set to {self.movement_delta}m")
                elif event.key == pygame.K_5:
                    self.movement_delta = 0.05
                    print(f"Step size set to {self.movement_delta}m")

                # Quit
                elif event.key == pygame.K_ESCAPE:
                    return "QUIT"

        if command:
            self.command_history.append(command)

        return command

    def cleanup(self):
        """Clean up pygame resources"""
        pygame.quit()


@register_env("KitchenStack-v1", max_episode_steps=2000)
class KitchenStackEnv(BaseEnv):
    """
    """

    SUPPORTED_ROBOTS = ["xlerobot"]
    agent: Xlerobot

    def __init__(
        self,
        *args,
        num_envs=1,
        **kwargs
    ):
        scene_builder_cls = REGISTERED_SCENE_BUILDERS["ReplicaCAD"].scene_builder_cls
        self.scene_builder: SceneBuilder = scene_builder_cls(self)
        self.build_config_idxs = [1]
        self.init_config_idxs = None
        if num_envs == 1:
            reconfiguration_freq = 1
        else:
            reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids="xlerobot",
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=50,
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**21,
                max_rigid_contact_count=2**23,
            ),
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict(reconfigure=False)
        self._set_episode_rng(seed, options.get("env_idx", torch.arange(self.num_envs)))
        if "reconfigure" in options and options["reconfigure"]:
            self.build_config_idxs = options.get(
                "build_config_idxs", self.build_config_idxs
            )
            self.init_config_idxs = options.get("init_config_idxs", None)
        else:
            assert (
                "build_config_idxs" not in options
            ), "options dict cannot contain build_config_idxs without reconfigure=True"
            self.init_config_idxs = options.get(
                "init_config_idxs", self.init_config_idxs
            )
        if isinstance(self.build_config_idxs, int):
            self.build_config_idxs = [self.build_config_idxs]
        if isinstance(self.init_config_idxs, int):
            self.init_config_idxs = [self.init_config_idxs]
        return super().reset(seed, options)

    def _load_lighting(self, options: dict):
        if self.scene_builder.builds_lighting:
            return
        return super()._load_lighting(options)

    def _load_agent(self, options: dict):
        robot_initial_pose = self.scene_builder.robot_initial_pose
        super()._load_agent(
            options,
            initial_agent_poses=robot_initial_pose,
        )

    def _load_scene(self, options: dict):
        if self.scene_builder.build_configs is not None:
            self.scene_builder.build(
                self.build_config_idxs
                if self.build_config_idxs is not None
                else self.scene_builder.sample_build_config_idxs()
            )
        else:
            self.scene_builder.build()

        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            if self.scene_builder.init_configs is not None:
                self.scene_builder.initialize(
                    env_idx,
                    (
                        self.init_config_idxs
                        if self.init_config_idxs is not None
                        else self.scene_builder.sample_init_config_idxs()
                    ),
                )
            else:
                self.scene_builder.initialize(env_idx)

            # override initial pose
            robot_initial_pose = sapien.Pose(p=[1.874, -5.114, 0.02], q=[1, 0, 0, 0])
            self.scene_builder.env.agent.robot.set_pose(robot_initial_pose)

            # look down to see grippers and table
            head_tilt_joint_index = self.agent.robot.active_joints_map['head_tilt_joint'].active_index
            qpos = self.agent.robot.get_qpos()
            qpos[:, head_tilt_joint_index] = 0.5
            self.agent.robot.set_qpos(qpos)
            self.agent.controller.controllers['body'].reset()

            # drop cube on table
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.9
            xy = torch.rand((b, 2)) * 0.01 - 0.01 + torch.tensor([2.08, -5.114])
            region = [[-0.05, -0.05], [0.05, 0.05]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02]))
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            pA = Pose.create_from_pq(p=xyz.clone(), q=qs)
            self.cubeA.set_pose(pA)

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))


    def evaluate(self) -> dict:
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }
    
    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda

        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        robot_camera_pose = sapien_utils.look_at([1, 0, 0.6], [0, 0, 0.3])
        robot_camera_config = CameraConfig(
            "render_camera",  
            robot_camera_pose,
            640,
            480,
            1,
            0.01,
            100,
            mount=self.agent.top_base_link,
        )
        return [robot_camera_config]


def build_vlm_prompt(history: List[Dict[str, str]], history_size: int = 3) -> str:
    """Build prompt for VLM including task description and command interface"""

    base_prompt = """**TASK:** Your goal is to pick up the RED cube and stack it on top of the GREEN cube, then let go of the cube without it falling.

**ROBOT CONTROL:** You are controlling the RIGHT ARM of a dual-arm robot. The robot has:
- A head camera (overhead view) - for scene understanding
- A right wrist camera (eye-in-hand) - this is YOUR gripper's view

**STRATEGY:**
1. Use the head camera to locate the red cube
2. Move the right arm so you can see the red cube clearly in YOUR right wrist camera
3. Approach the red cube with the gripper OPEN
4. Position the gripper around the cube
5. CLOSE the gripper to grasp the cube
6. Lift and move to place it on the green cube
7. OPEN the gripper to release

**AVAILABLE COMMANDS:** Output ONE command per step using this format:

Command examples:
- "move ee forward 0.05" - move YOUR right arm's end-effector forward (away from robot base) by 0.05 meters
- "move ee backward 0.03" - move YOUR right arm's end-effector backward (toward robot base) by 0.03 meters
- "move ee up 0.02" - move YOUR right arm's end-effector up by 0.02 meters
- "move ee down 0.02" - move YOUR right arm's end-effector down by 0.02 meters
- "pitch up 0.1" - tilt YOUR right wrist up by 0.1 radians
- "pitch down 0.1" - tilt YOUR right wrist down by 0.1 radians
- "wrist roll 0.2" - roll YOUR right wrist by 0.2 radians
- "open gripper" - OPEN YOUR right gripper
- "close gripper" - CLOSE YOUR right gripper to grasp
- "hold" - maintain current position

**IMPORTANT RULES:**
- Output ONLY ONE command per step
- Use small movements (0.01-0.05 meters recommended)
- ALWAYS open the gripper BEFORE approaching the cube
- Make sure you can see the target in YOUR wrist camera before grasping
- Start your response with "COMMAND:" followed by the command
- Example: "COMMAND: move ee forward 0.03"

"""

    # Add text history if available (last N commands only)
    if history:
        base_prompt += "\n**RECENT ACTIONS:**\n"
        recent_history = history[-history_size:]  # Keep only last N items
        for entry in recent_history:
            base_prompt += f"Query {entry['vlm_query']}: {entry['command']}\n"
        base_prompt += "\n"

    base_prompt += "**CURRENT STEP:** Based on the images above, what command should the robot execute now?\n"

    return base_prompt


def human_control_loop(
    seed: int = 42,
    max_steps: int = 2000,
    log_dir: Optional[str] = None,
    steps_per_command: int = 10
):
    """
    Human control loop using pygame keyboard interface for debugging

    Args:
        seed: Random seed for environment (default: 42)
        max_steps: Maximum simulation steps (default: 2000)
        log_dir: Optional directory to log observations (images and joint states)
        steps_per_command: Number of simulation steps to execute per command (default: 10)
    """
    if not PYGAME_AVAILABLE:
        print("ERROR: pygame is not installed. Install with: pip install pygame")
        return

    # Configuration
    shader = "rt-fast"
    num_envs = 1
    render_mode = "rgb_array"
    env_id = "KitchenStack-v1"
    obs_mode = "sensor_data"
    control_mode = "pd_joint_delta_pos_dual_arm"

    # Setup logging directory if requested
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        print(f"Logging observations to: {log_path}")
    else:
        log_path = None

    # Create environment
    env_kwargs = dict(
        obs_mode=obs_mode,
        reward_mode=None,
        control_mode=control_mode,
        render_mode=render_mode,
        sensor_configs=dict(shader_pack=shader),
        human_render_camera_configs=dict(shader_pack=shader),
        viewer_camera_configs=dict(shader_pack=shader),
        num_envs=num_envs,
        sim_backend="auto",
        render_backend="gpu",
        enable_shadow=True,
        parallel_in_single_scene=False,
    )

    env: BaseEnv = gym.make(env_id, **env_kwargs)

    # Reset environment first to get initial robot position
    obs, _ = env.reset(seed=seed, options=dict(reconfigure=True))

    # Get initial robot position
    initial_qpos = env.unwrapped.agent.robot.qpos

    # Initialize controllers - using the fixed controller with correct joint mappings
    controller = FixedRobotController(initial_qpos)
    keyboard = KeyboardController()

    print("=" * 80)
    print("Human Control Loop Started")
    print("=" * 80)
    print(f"Control mode: {control_mode}")
    print(f"Random seed: {seed}")
    print(f"Max steps: {max_steps}")
    print(f"Steps per command: {steps_per_command}")
    if log_path:
        print(f"Logging enabled: {log_path}")
    print("=" * 80)
    print("\nSee pygame window for controls\n")

    step_count = 0
    command_count = 0

    try:
        while step_count < max_steps:
            # Extract camera images
            head_img = obs['sensor_data']['fetch_head']['Color'][0]
            right_wrist_img = obs['sensor_data']['fetch_right_arm_camera']['Color'][0]
            left_wrist_img = obs['sensor_data']['fetch_left_arm_camera']['Color'][0]

            # Display in pygame window
            images = [head_img, right_wrist_img, left_wrist_img]
            labels = ["Head Camera (Overhead)", "Right Wrist Camera", "Left Wrist Camera"]
            keyboard.display_observations(images, labels)

            # Get command from keyboard
            command_text = keyboard.get_command()

            if command_text == "QUIT":
                print("\nUser quit")
                break

            if command_text:
                command_count += 1
                print(f"\n[Command {command_count}] {command_text}")

                # Parse and execute command using fixed controller
                parsed = controller.parse_command(command_text)
                print(f"[Command {command_count}] Parsed: {parsed}")

                # Update controller targets
                controller.update_targets(parsed)

                # Get current joint positions for logging
                current_qpos = env.unwrapped.agent.robot.qpos

                # Log if enabled
                if log_path:
                    # Log images
                    for i, (img, label) in enumerate(zip(images, ["head", "right_wrist", "left_wrist"])):
                        img_path = log_path / f"{command_count:05d}_{label}.png"
                        if torch.is_tensor(img):
                            img_np = img.cpu().numpy()
                        else:
                            img_np = img
                        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                            img_np = (img_np * 255).astype(np.uint8)
                        Image.fromarray(img_np).save(img_path)

                    # Log state
                    # Convert qpos to list properly
                    if torch.is_tensor(current_qpos):
                        qpos_list = current_qpos.cpu().numpy().tolist()
                    elif hasattr(current_qpos, 'tolist'):
                        qpos_list = current_qpos.tolist()
                    else:
                        qpos_list = list(current_qpos)

                    state_data = {
                        "command": command_count,
                        "command_text": command_text,
                        "parsed": parsed,
                        "qpos": qpos_list,
                        "target_positions": controller.target_positions.tolist(),
                        "right_arm_targets": {
                            "shoulder_rotation": float(controller.target_positions[3]),
                            "shoulder_pitch": float(controller.target_positions[6]),
                            "elbow": float(controller.target_positions[9]),
                            "wrist_pitch": float(controller.target_positions[11]),
                            "wrist_roll": float(controller.target_positions[13]),
                            "gripper": float(controller.target_positions[15])
                        }
                    }
                    state_path = log_path / f"{command_count:05d}_state.json"
                    with open(state_path, 'w') as f:
                        json.dump(state_data, f, indent=2)

                # Execute action for multiple steps
                total_reward = 0.0
                for substep in range(steps_per_command):
                    step_count += 1

                    # Compute action
                    current_qpos = env.unwrapped.agent.robot.qpos
                    action = controller.compute_action(current_qpos)  # Already returns 16D action

                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    reward_scalar = reward.item() if torch.is_tensor(reward) else reward
                    total_reward += reward_scalar

                    if terminated or truncated:
                        break

                # Print status
                avg_reward = total_reward / steps_per_command
                print(f"[Command {command_count}] Avg Reward: {avg_reward:.4f}, Total Reward: {total_reward:.4f}")
                print(f"[Command {command_count}] Info: {info}")

                if terminated or truncated:
                    print("\nEpisode ended")
                    print(f"Final info: {info}")
                    break
            else:
                # No command, just display current state
                keyboard.clock.tick(30)  # 30 FPS when idle

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError in control loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        keyboard.cleanup()
        env.close()
        print("\nHuman Control Loop Complete!")


def vlm_control_loop(
    history_size: int = 2,
    model: str = "qwen/qwen3-vl-235b-a22b-instruct",
    seed: int = 42,
    max_steps: int = 200,
    log_dir: Optional[str] = None,
    steps_per_command: int = 10
):
    """
    VLM control loop

    Args:
        history_size: Number of previous steps to include in VLM prompt (default: 2)
        model: OpenRouter model to use (default: qwen/qwen3-vl-235b-a22b-instruct)
        seed: Random seed for environment (default: 42)
        max_steps: Maximum VLM queries per episode (default: 200)
        log_dir: Optional directory to log observations (images and joint states)
        steps_per_command: Number of simulation steps to execute per VLM command (default: 10)
    """
    # Configuration
    shader = "rt-fast"
    num_envs = 1
    render_mode = "rgb_array"
    env_id = "KitchenStack-v1"
    obs_mode = "sensor_data"
    control_mode = "pd_joint_delta_pos_dual_arm"  # Using dual arm mode but only control right arm

    # Get API key from environment - required
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables.")
        print("Set it with: export OPENROUTER_API_KEY='your_key_here'")
        return

    # Setup logging directory if requested
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        print(f"Logging observations to: {log_path}")
    else:
        log_path = None

    # Create environment
    env_kwargs = dict(
        obs_mode=obs_mode,
        reward_mode=None,
        control_mode=control_mode,
        render_mode=render_mode,
        sensor_configs=dict(shader_pack=shader),
        human_render_camera_configs=dict(shader_pack=shader),
        viewer_camera_configs=dict(shader_pack=shader),
        num_envs=num_envs,
        sim_backend="auto",
        render_backend="gpu",
        enable_shadow=True,
        parallel_in_single_scene=False,
    )

    env: BaseEnv = gym.make(env_id, **env_kwargs)

    # Reset environment first to get initial robot position
    obs, _ = env.reset(seed=seed, options=dict(reconfigure=True))

    # Get initial robot position
    initial_qpos = env.unwrapped.agent.robot.qpos

    # Initialize VLM client and controller with current robot position
    vlm_client = VLMClient(api_key, model=model)
    controller = FixedRobotController(initial_qpos)  # Using fixed controller

    # History tracking - stores both commands and images
    history = []  # List of dicts with 'vlm_query', 'command', etc.
    history_images = []  # List of image lists for VLM

    print("=" * 80)
    print("VLM Control Loop Started")
    print("=" * 80)
    print(f"Action space: {env.action_space}")
    print(f"Control mode: {control_mode}")
    print(f"VLM model: {model}")
    print(f"History size: {history_size}")
    print(f"Random seed: {seed}")
    print(f"Max VLM queries: {max_steps}")
    print(f"Simulation steps per command: {steps_per_command}")
    print(f"Max simulation steps: {max_steps * steps_per_command}")
    if log_path:
        print(f"Logging enabled: {log_path}")
    print("=" * 80)

    vlm_query_count = 0
    sim_step_count = 0

    while vlm_query_count < max_steps:
        vlm_query_count += 1

        # Extract camera images (only head and right wrist)
        head_img = obs['sensor_data']['fetch_head']['Color'][0]  # Remove batch dim
        right_wrist_img = obs['sensor_data']['fetch_right_arm_camera']['Color'][0]
        left_wrist_img = obs['sensor_data']['fetch_left_arm_camera']['Color'][0]  # Still log it

        # Only send head and right wrist to VLM
        images = [head_img, right_wrist_img]

        # Get current joint positions for logging
        current_qpos = env.unwrapped.agent.robot.qpos

        # Log observations if enabled (only on VLM query steps)
        if log_path:
            step_str = f"{vlm_query_count:05d}"  # Zero-padded to 5 digits

            # Save images
            for img_name, img_data in [
                ("head", head_img),
                ("right_wrist", right_wrist_img),
                ("left_wrist", left_wrist_img)
            ]:
                img_to_save = img_data

                # Convert tensor to numpy if needed
                if torch.is_tensor(img_to_save):
                    img_to_save = img_to_save.cpu().numpy()

                # Convert float32 [0, 1] to uint8 [0, 255]
                if img_to_save.dtype == np.float32 or img_to_save.dtype == np.float64:
                    if img_to_save.max() <= 1.0:
                        img_to_save = (img_to_save * 255).astype(np.uint8)
                    else:
                        img_to_save = img_to_save.astype(np.uint8)

                # Remove alpha channel if present
                if img_to_save.shape[-1] == 4:
                    img_to_save = img_to_save[..., :3]

                img_pil = Image.fromarray(img_to_save)
                img_pil.save(log_path / f"{step_str}_{img_name}.png")

            # Save joint state as JSON
            joint_state = {
                "vlm_query": vlm_query_count,
                "sim_step": sim_step_count,
                "qpos": current_qpos.cpu().numpy().tolist() if torch.is_tensor(current_qpos) else current_qpos.tolist(),
                "ee_pos": controller.ee_pos.tolist(),
                "pitch": float(controller.pitch),
                "wrist_roll": float(controller.wrist_roll),
                "target_joints": controller.target_joints.tolist()
            }
            with open(log_path / f"{step_str}_state.json", "w") as f:
                json.dump(joint_state, f, indent=2)

        # Query VLM for command with visual history
        prompt = build_vlm_prompt(history, history_size)

        # Get recent image history (last N steps)
        recent_history_images = history_images[-history_size:] if history_images else None

        print(f"\n[VLM Query {vlm_query_count}, Sim Step {sim_step_count}] Querying VLM (with {len(recent_history_images) if recent_history_images else 0} historical image sets)...")
        vlm_response = vlm_client.query(prompt, images, history_images=recent_history_images)
        print(f"VLM Response: {repr(vlm_response)}")  # Use repr to show whitespace

        # Check if response is empty
        if not vlm_response or not vlm_response.strip():
            print(f"WARNING: VLM returned empty response at query {vlm_query_count}")
            command_text = "hold"  # Default to hold command
        elif "COMMAND:" in vlm_response:
            command_text = vlm_response.split("COMMAND:")[1].strip().split("\n")[0]
        else:
            command_text = vlm_response.strip()

        print(f"[VLM Query {vlm_query_count}] Command: {command_text}")

        # Log prompt and response if enabled
        if log_path:
            step_str = f"{vlm_query_count:05d}"
            prompt_log = {
                "vlm_query": vlm_query_count,
                "sim_step": sim_step_count,
                "prompt": prompt,
                "vlm_response": vlm_response,
                "vlm_response_length": len(vlm_response),
                "vlm_response_empty": not vlm_response or not vlm_response.strip(),
                "extracted_command": command_text
            }
            with open(log_path / f"{step_str}_prompt.json", "w") as f:
                json.dump(prompt_log, f, indent=2)

        # Parse command using fixed controller
        parsed_command = controller.parse_command(command_text)
        print(f"[VLM Query {vlm_query_count}] Parsed: {parsed_command}")

        # Update controller targets
        controller.update_targets(parsed_command)

        # Compute 16D action directly from fixed controller
        action = controller.compute_action(current_qpos)

        # Log only non-zero action values
        non_zero_actions = [(i, v) for i, v in enumerate(action) if abs(v) > 0.001]
        print(f"[VLM Query {vlm_query_count}] Action (non-zero): {non_zero_actions}")
        print(f"[VLM Query {vlm_query_count}] Executing action for {steps_per_command} steps...")

        # Execute the same action for N steps
        total_reward = 0.0
        for substep in range(steps_per_command):
            sim_step_count += 1
            obs, reward, terminated, truncated, info = env.step(action)

            # Accumulate reward
            reward_val = reward.item() if torch.is_tensor(reward) else reward
            total_reward += reward_val

            if terminated or truncated:
                print(f"\n{'='*80}")
                print(f"Episode ended at VLM query {vlm_query_count}, sim step {sim_step_count}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"{'='*80}")
                break

        # Update history with command and images (after executing action)
        history.append({
            "vlm_query": vlm_query_count,
            "sim_step": sim_step_count,
            "command": command_text,
            "parsed": parsed_command
        })

        # Store images for next iteration's history (only head and right wrist)
        # Use the latest observation after executing all substeps
        final_head_img = obs['sensor_data']['fetch_head']['Color'][0]
        final_right_wrist_img = obs['sensor_data']['fetch_right_arm_camera']['Color'][0]
        history_images.append([
            final_head_img.cpu().numpy() if torch.is_tensor(final_head_img) else final_head_img,
            final_right_wrist_img.cpu().numpy() if torch.is_tensor(final_right_wrist_img) else final_right_wrist_img
        ])

        # Print status
        avg_reward = total_reward / steps_per_command
        print(f"[VLM Query {vlm_query_count}] Avg Reward: {avg_reward:.4f}, Total Reward: {total_reward:.4f}")
        print(f"[VLM Query {vlm_query_count}] Info: {info}")

        if terminated or truncated:
            break

    env.close()
    print("\nVLM Control Loop Complete!")


def main():
    """Main entry point that dispatches to VLM or human control mode"""
    import argparse

    parser = argparse.ArgumentParser(description="Robot control in ManiSkill (VLM or human mode)")
    parser.add_argument("--human", action="store_true",
                        help="Enable human control mode with pygame keyboard interface")
    parser.add_argument("--history-size", type=int, default=2,
                        help="Number of previous steps to include in VLM prompt (default: 2, VLM mode only)")
    parser.add_argument("--model", type=str, default="qwen/qwen3-vl-235b-a22b-instruct",
                        help="OpenRouter model to use (default: qwen/qwen3-vl-235b-a22b-instruct, VLM mode only)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for environment (default: 42)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum VLM queries (VLM mode) or simulation steps/100 (human mode) per episode (default: 200)")
    parser.add_argument("--steps-per-command", type=int, default=10,
                        help="Number of simulation steps to execute per command (default: 10)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Optional directory to log observations (images and joint states)")

    args = parser.parse_args()

    if args.human:
        # Human control mode
        human_control_loop(
            seed=args.seed,
            max_steps=args.max_steps * 10,  # Convert to simulation steps
            log_dir=args.log_dir,
            steps_per_command=args.steps_per_command
        )
    else:
        # VLM control mode
        vlm_control_loop(
            history_size=args.history_size,
            model=args.model,
            seed=args.seed,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
            steps_per_command=args.steps_per_command
        )


if __name__ == '__main__':
    main()