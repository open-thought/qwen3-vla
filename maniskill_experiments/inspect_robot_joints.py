#!/usr/bin/env python3
"""
Inspect XLerobot joint configuration and indices in ManiSkill
"""

import numpy as np
import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv

# Import to register the KitchenStack-v1 environment
import sys
sys.path.append('/home/koepf/robotics/qwen3-vla/maniskill_experiments')
from vlm_control_loop import KitchenStackEnv

def inspect_robot_joints():
    """Inspect the robot's joint configuration"""
    # Environment setup
    env_kwargs = dict(
        obs_mode="sensor_data",
        reward_mode=None,
        control_mode="pd_joint_delta_pos_dual_arm",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="auto",
    )

    # Create environment
    env_id = "KitchenStack-v1"
    env: BaseEnv = gym.make(env_id, **env_kwargs)

    # Reset to initialize robot
    obs, _ = env.reset(seed=42, options=dict(reconfigure=True))

    # Get robot reference
    robot = env.unwrapped.agent.robot

    print("=" * 80)
    print("ROBOT JOINT CONFIGURATION INSPECTION")
    print("=" * 80)

    # Basic robot info
    print(f"\nRobot type: {type(robot).__name__}")
    print(f"Robot name: {robot.name if hasattr(robot, 'name') else 'N/A'}")

    # Get articulation info
    articulation = robot.articulation if hasattr(robot, 'articulation') else robot

    print(f"\n--- ARTICULATION INFO ---")
    print(f"Total joints: {articulation.dof if hasattr(articulation, 'dof') else 'N/A'}")

    # Get active joints
    if hasattr(articulation, 'get_active_joints'):
        active_joints = articulation.get_active_joints()
        print(f"\nActive joints count: {len(active_joints)}")
        print("\n--- ACTIVE JOINTS ---")
        for i, joint in enumerate(active_joints):
            print(f"  [{i:2d}] {joint.name}")

    # Get qpos (current joint positions)
    qpos = robot.qpos
    print(f"\n--- QPOS (Current Joint Positions) ---")
    print(f"Shape: {qpos.shape}")
    print(f"Values:")
    for i, val in enumerate(qpos.squeeze()):
        print(f"  [{i:2d}] = {val:8.4f}")

    # Get qlimits (joint limits)
    if hasattr(robot, 'qlimits'):
        qlimits = robot.qlimits
        print(f"\n--- JOINT LIMITS ---")
        print(f"Shape: {qlimits.shape}")
        # qlimits is typically shape [n_joints, 2] where [:, 0] is lower limits and [:, 1] is upper limits
        if len(qlimits.shape) == 2 and qlimits.shape[1] == 2:
            for i in range(qlimits.shape[0]):
                low = qlimits[i, 0].item() if hasattr(qlimits[i, 0], 'item') else qlimits[i, 0]
                high = qlimits[i, 1].item() if hasattr(qlimits[i, 1], 'item') else qlimits[i, 1]
                print(f"  [{i:2d}] Range: [{low:8.4f}, {high:8.4f}]")
        else:
            print(f"  Unexpected shape for qlimits: {qlimits}")

    # Check for active_joints_map
    if hasattr(robot, 'active_joints_map'):
        print(f"\n--- ACTIVE JOINTS MAP ---")
        print(robot.active_joints_map)

    # Check control joints
    if hasattr(robot, 'control_joints'):
        print(f"\n--- CONTROL JOINTS ---")
        for i, joint in enumerate(robot.control_joints):
            print(f"  [{i:2d}] {joint}")

    # Check for arm-specific attributes
    if hasattr(robot, 'arm_joint_names'):
        print(f"\n--- ARM JOINT NAMES ---")
        print(f"Left arm: {robot.arm_joint_names.get('left', 'N/A')}")
        print(f"Right arm: {robot.arm_joint_names.get('right', 'N/A')}")

    # Check for end effector info
    if hasattr(robot, 'ee_link_names'):
        print(f"\n--- END EFFECTOR LINKS ---")
        print(f"End effector links: {robot.ee_link_names}")

    # Try to get specific arm joint indices
    print(f"\n--- ATTEMPTING TO MAP ARM JOINTS ---")
    joint_names = [joint.name for joint in active_joints] if 'active_joints' in locals() else []

    # Look for right arm joints
    right_arm_keywords = ['right', 'r_', 'fetch_right']
    left_arm_keywords = ['left', 'l_', 'fetch_left']

    right_joints = {}
    left_joints = {}
    base_joints = {}

    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        if any(kw in name_lower for kw in right_arm_keywords):
            right_joints[i] = name
        elif any(kw in name_lower for kw in left_arm_keywords):
            left_joints[i] = name
        elif 'base' in name_lower or 'wheel' in name_lower:
            base_joints[i] = name

    print("\nRight arm joints:")
    for idx, name in sorted(right_joints.items()):
        print(f"  [{idx:2d}] {name}")

    print("\nLeft arm joints:")
    for idx, name in sorted(left_joints.items()):
        print(f"  [{idx:2d}] {name}")

    print("\nBase/wheel joints:")
    for idx, name in sorted(base_joints.items()):
        print(f"  [{idx:2d}] {name}")

    # Check action space
    print(f"\n--- ACTION SPACE ---")
    print(f"Action space: {env.action_space}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Action low: {env.action_space.low}")
    print(f"Action high: {env.action_space.high}")

    # Try to understand control mapping
    if hasattr(env.unwrapped.agent, 'controller'):
        controller = env.unwrapped.agent.controller
        print(f"\n--- CONTROLLER INFO ---")
        print(f"Controller type: {type(controller).__name__}")
        if hasattr(controller, 'control_mode'):
            print(f"Control mode: {controller.control_mode}")
        if hasattr(controller, 'action_dim'):
            print(f"Action dimension: {controller.action_dim}")

        # Check for action_mapping attribute
        if hasattr(controller, 'action_mapping'):
            print(f"\n--- ACTION MAPPING ---")
            print(f"controller.action_mapping = {controller.action_mapping}")
            print(f"Type: {type(controller.action_mapping)}")
            if hasattr(controller.action_mapping, 'shape'):
                print(f"Shape: {controller.action_mapping.shape}")
            print(f"Content: {controller.action_mapping}")

        # Check for CombinedController with Controllers dict
        if hasattr(controller, 'controllers'):
            print(f"\n--- COMBINED CONTROLLER SUBCONTROLLERS ---")
            print(f"Number of subcontrollers: {len(controller.controllers)}")

            # Use the action_mapping if available
            action_mapping = controller.action_mapping if hasattr(controller, 'action_mapping') else {}

            for name, subctrl in controller.controllers.items():
                print(f"\n  Subcontroller: {name}")
                print(f"    Type: {type(subctrl).__name__}")
                if hasattr(subctrl, 'control_mode'):
                    print(f"    Control mode: {subctrl.control_mode}")
                if hasattr(subctrl, 'action_dim'):
                    print(f"    Action dimension: {subctrl.action_dim}")
                if hasattr(subctrl, 'joint_indices'):
                    print(f"    Joint indices: {subctrl.joint_indices}")
                if hasattr(subctrl, 'control_joint_indices'):
                    print(f"    Control joint indices: {subctrl.control_joint_indices}")
                if hasattr(subctrl, 'action_joint_indices'):
                    print(f"    Action joint indices: {subctrl.action_joint_indices}")
                if hasattr(subctrl, 'joints'):
                    print(f"    Number of joints: {len(subctrl.joints)}")
                    print(f"    Joint details:")

                    # Get the action range for this controller
                    if name in action_mapping:
                        action_start, _ = action_mapping[name]
                    else:
                        action_start = 0

                    for joint_idx, j in enumerate(subctrl.joints):
                        if hasattr(j, 'name') and hasattr(j, 'index') and hasattr(j, 'active_index'):
                            # Convert tensor to int for display
                            index_val = j.index.item() if hasattr(j.index, 'item') else j.index
                            active_index_val = j.active_index.item() if hasattr(j.active_index, 'item') else j.active_index
                            # Calculate the global action index for this joint
                            global_action_idx = action_start + joint_idx
                            print(f"      - {j.name}: action_idx={global_action_idx}, index={index_val}, active_index={active_index_val}")
                        else:
                            print(f"      - {j}")

                # Check the action mapping
                if hasattr(subctrl, 'compute_ik'):
                    print(f"    Has IK solver")
                if hasattr(subctrl, '_action_space'):
                    print(f"    Action space: {subctrl._action_space}")

                # Use action_mapping to show the action range
                if name in action_mapping:
                    start_idx, end_idx = action_mapping[name]
                    print(f"    Action range (from action_mapping): [{start_idx}:{end_idx})")
                    print(f"    Action dimensions: {end_idx - start_idx}")

        # Check action-to-joint mapping
        if hasattr(controller, 'get_action_space_dim'):
            print(f"\nTotal action space dim from controller: {controller.get_action_space_dim()}")

        if hasattr(controller, '_action_to_joints'):
            print(f"\nAction to joints mapping: {controller._action_to_joints}")

    env.close()
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    inspect_robot_joints()