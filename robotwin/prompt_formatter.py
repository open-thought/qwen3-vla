"""
Unified prompt formatter for Qwen3-VLA training and evaluation.

This module provides a single PromptFormatter class that ensures consistent
prompt construction between training and inference. Using the same formatter
guarantees that the model sees identical input formats during both phases.
"""

import numpy as np
import torch
from typing import Union, Optional
from PIL import Image


class PromptFormatter:
    """
    Unified prompt formatter for Qwen3-VLA.

    Handles:
    - State vector formatting (discretization to [0, 255])
    - Prompt text construction (task, robot type, state)
    - Conversation structure with 3 camera images

    Use this class in both training (data_collator.py) and evaluation
    (qwen3_vla_policy.py) to ensure consistency.
    """

    def __init__(
        self,
        num_state_bins: int = 256,
    ):
        """
        Initialize the prompt formatter.

        Args:
            num_state_bins: Number of bins for state discretization (default: 256)
        """
        self.num_state_bins = num_state_bins

    def format_state(self, discretized_state: np.ndarray, dropout_mask: Optional[np.ndarray] = None) -> str:
        """
        Format discretized state as a comma-separated string.

        Args:
            discretized_state: State array with values in [0, num_state_bins-1]
            dropout_mask: Optional boolean mask where True means "drop out this value"
                         Dropped values are replaced with "?" in the output

        Returns:
            Formatted string like "128, 64, 200, ..." or "?, 64, ?, ..." with dropout
        """
        if dropout_mask is None:
            return ", ".join([str(int(s)) for s in discretized_state])
        else:
            return ", ".join([
                "?" if dropout_mask[i] else str(int(s))
                for i, s in enumerate(discretized_state)
            ])

    def build_prompt_text(
        self,
        task_description: str,
        robot_type: str,
        discretized_state: np.ndarray,
        state_dropout_mask: Optional[np.ndarray] = None,
        include_text_state: bool = True,
    ) -> str:
        """
        Build the text portion of the prompt.

        Args:
            task_description: Natural language task instruction
            robot_type: Robot type identifier (e.g., "aloha-agilex")
            discretized_state: State array with values in [0, num_state_bins-1]
            state_dropout_mask: Optional boolean mask for state dropout
            include_text_state: Whether to include the state as text in the prompt

        Returns:
            Formatted prompt text
        """
        prompt_text = f"""Task: {task_description}
Robot: {robot_type}"""

        if include_text_state:
            state_str = self.format_state(discretized_state, state_dropout_mask)
            prompt_text += f"\nState: [{state_str}]"

        return prompt_text

    def build_conversation(
        self,
        left_camera: Union[torch.Tensor, Image.Image, np.ndarray],
        right_camera: Union[torch.Tensor, Image.Image, np.ndarray],
        head_camera: Union[torch.Tensor, Image.Image, np.ndarray],
        task_description: str,
        robot_type: str,
        discretized_state: np.ndarray,
        state_dropout_mask: Optional[np.ndarray] = None,
        include_text_state: bool = True,
        image_dropout_mask: Optional[tuple[bool, bool, bool]] = None,
    ) -> list:
        """
        Build the full conversation structure for the model.

        This creates a conversation list compatible with Qwen3-VL's
        processor.apply_chat_template() method.

        Args:
            left_camera: Left camera image (tensor, PIL Image, or numpy array)
            right_camera: Right camera image
            head_camera: Head camera image
            task_description: Natural language task instruction
            robot_type: Robot type identifier
            discretized_state: State array with values in [0, num_state_bins-1]
            state_dropout_mask: Optional boolean mask for state dropout
            include_text_state: Whether to include the state as text in the prompt
            image_dropout_mask: Optional tuple of 3 booleans (left, right, head) where
                True means the image should be dropped/omitted from the prompt

        Returns:
            Conversation list for processor.apply_chat_template()
        """
        prompt_text = self.build_prompt_text(
            task_description=task_description,
            robot_type=robot_type,
            discretized_state=discretized_state,
            state_dropout_mask=state_dropout_mask,
            include_text_state=include_text_state,
        )

        # Build content list, optionally omitting images based on dropout mask
        content = []
        cameras = [
            ("Left camera:", left_camera),
            ("Right camera:", right_camera),
            ("Head camera:", head_camera),
        ]

        drop_left, drop_right, drop_head = image_dropout_mask or (False, False, False)
        drop_flags = [drop_left, drop_right, drop_head]

        for i, (label, image) in enumerate(cameras):
            if not drop_flags[i]:
                content.append({"type": "text", "text": label})
                content.append({"type": "image", "image": image})

        content.append({"type": "text", "text": prompt_text})

        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]

        return conversation

    def build_batch_conversations(
        self,
        samples: list[dict],
    ) -> list[list]:
        """
        Build conversations for a batch of samples.

        Args:
            samples: List of sample dictionaries with keys:
                - left_camera, right_camera, head_camera: Image tensors
                - task_description: Task instruction string
                - robot_type: Robot type string
                - discretized_state: State array

        Returns:
            List of conversations for batch processing
        """
        conversations = []

        for sample in samples:
            conversation = self.build_conversation(
                left_camera=sample["left_camera"],
                right_camera=sample["right_camera"],
                head_camera=sample["head_camera"],
                task_description=sample["task_description"],
                robot_type=sample["robot_type"],
                discretized_state=sample["discretized_state"],
            )
            conversations.append(conversation)

        return conversations


# Singleton instance for convenience
_default_formatter: Optional[PromptFormatter] = None


def get_prompt_formatter(num_state_bins: int = 256) -> PromptFormatter:
    """
    Get the default PromptFormatter instance.

    Args:
        num_state_bins: Number of bins for state discretization

    Returns:
        PromptFormatter instance
    """
    global _default_formatter

    if _default_formatter is None or _default_formatter.num_state_bins != num_state_bins:
        _default_formatter = PromptFormatter(num_state_bins=num_state_bins)

    return _default_formatter


def test_prompt_formatter():
    """Test the prompt formatter."""
    print("Testing PromptFormatter...")
    print("=" * 60)

    formatter = PromptFormatter()

    # Test state formatting
    state = np.array([128, 64, 200, 100, 150, 80, 128, 64, 200, 100, 150, 80, 127, 127])
    state_str = formatter.format_state(state)
    print(f"State string: {state_str}")

    # Test prompt text building
    prompt_text = formatter.build_prompt_text(
        task_description="Pick up the red block and place it on the table",
        robot_type="aloha-agilex",
        discretized_state=state,
    )
    print(f"\nPrompt text:\n{prompt_text}")

    # Test conversation building with dummy images
    dummy_img = torch.rand(3, 240, 320)

    conversation = formatter.build_conversation(
        left_camera=dummy_img,
        right_camera=dummy_img,
        head_camera=dummy_img,
        task_description="Pick up the red block",
        robot_type="aloha-agilex",
        discretized_state=state,
    )

    print(f"\nConversation structure:")
    print(f"  Num messages: {len(conversation)}")
    print(f"  Message role: {conversation[0]['role']}")
    print(f"  Content items: {len(conversation[0]['content'])}")
    for i, item in enumerate(conversation[0]['content']):
        if item['type'] == 'text':
            text_preview = item['text'][:50] + "..." if len(item['text']) > 50 else item['text']
            print(f"    [{i}] text: {text_preview}")
        else:
            print(f"    [{i}] {item['type']}: shape={item['image'].shape}")

    print("\n" + "=" * 60)
    print("PromptFormatter test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_prompt_formatter()
