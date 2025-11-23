"""
Data collator for VLA training with Qwen3-VL.

Prepares batches for multi-modal VLM training with images, text prompts,
and action token targets.
"""

import torch
from typing import Dict, List


class VLADataCollator:
    """
    Collate samples into batches for VLA training.

    Handles:
    - Multi-image input (3 cameras)
    - Text prompt with task, robot type, and discretized state
    - Action token targets for training
    - Proper masking to compute loss only on action tokens
    """

    def __init__(
        self,
        processor,
        action_token_start: int = 151936,  # First FAST token ID
        action_token_end: int = 153983,    # Last FAST token ID
    ):
        """
        Args:
            processor: Qwen3VL processor for tokenizing and preparing inputs
            action_token_start: First token ID in FAST vocabulary
            action_token_end: Last token ID in FAST vocabulary
        """
        self.processor = processor
        self.action_token_start = action_token_start
        self.action_token_end = action_token_end

    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            samples: List of dictionaries from RoboTwinVLADataset

        Returns:
            Dictionary with batched tensors ready for model input
        """
        batch_size = len(samples)

        # Prepare conversations for Qwen processor
        # Format: List of conversations, each with images and text
        prompts = []
        action_token_sequences = []

        for sample in samples:
            # Build text prompt
            state_str = ", ".join([str(int(s)) for s in sample["discretized_state"]])

            prompt_text = f"""Task: {sample['task_description']}
Robot: {sample['robot_type']}
State: [{state_str}]"""

            # Prepare conversation with 3 images
            # Images are already tensors in (C, H, W) format with values in [0, 1]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Left camera:"},
                        {"type": "image", "image": sample["left_camera"]},
                        {"type": "text", "text": "Right camera:"},
                        {"type": "image", "image": sample["right_camera"]},
                        {"type": "text", "text": "Head camera:"},
                        {"type": "image", "image": sample["head_camera"]},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            prompts.append(conversation)
            action_token_sequences.append(sample["action_tokens"])

        # Use processor.apply_chat_template to prepare inputs
        # Images are already float tensors in [0, 1] range, so set do_rescale=False
        batch_inputs = self.processor.apply_chat_template(
            prompts,
            tokenize=True,
            add_generation_prompt=True,  # Add assistant response start
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            do_rescale=False,  # Pixel values are already in 0-1 range
        )

        # Now we need to append the action tokens to the input_ids
        # and create labels for training

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for i in range(batch_size):
            # Get the prompt input_ids (without action tokens)
            prompt_ids = batch_inputs["input_ids"][i]
            prompt_mask = batch_inputs["attention_mask"][i]

            # Get action tokens for this sample
            action_tokens = torch.tensor(action_token_sequences[i], dtype=torch.long)

            # Concatenate prompt + action tokens
            full_input_ids = torch.cat([prompt_ids, action_tokens])
            full_attention_mask = torch.cat([prompt_mask, torch.ones_like(action_tokens)])

            # Create labels: -100 for prompt (don't compute loss), actual tokens for actions
            prompt_labels = torch.full_like(prompt_ids, -100)
            full_labels = torch.cat([prompt_labels, action_tokens])

            input_ids_list.append(full_input_ids)
            attention_mask_list.append(full_attention_mask)
            labels_list.append(full_labels)

        # Pad to same length within batch
        max_len = max(len(ids) for ids in input_ids_list)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_len - len(input_ids)

            if pad_len > 0:
                # Pad with processor's pad token
                pad_token_id = self.processor.tokenizer.pad_token_id or 0
                padded_input_ids.append(
                    torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                )
                padded_attention_mask.append(
                    torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                )
                padded_labels.append(
                    torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
                )
            else:
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attention_mask)
                padded_labels.append(labels)

        # Stack into batch tensors
        batch_inputs["input_ids"] = torch.stack(padded_input_ids)
        batch_inputs["attention_mask"] = torch.stack(padded_attention_mask)
        batch_inputs["labels"] = torch.stack(padded_labels)

        return batch_inputs

    def decode_action_tokens(
        self,
        token_ids: torch.Tensor,
        action_horizon: int,
        action_dim: int,
        normalizer,
        robot_type: str
    ) -> torch.Tensor:
        """
        Decode predicted action tokens back to robot actions.

        Args:
            token_ids: Predicted token IDs (batch, seq_len)
            action_horizon: Action prediction horizon
            action_dim: Action dimension (2*dof)
            normalizer: MultiRobotNormalizer instance
            robot_type: Robot type for denormalization

        Returns:
            Decoded actions (batch, action_horizon, action_dim)
        """
        from action_tokenizer import ActionTokenizer

        # Initialize tokenizer
        tokenizer = ActionTokenizer()

        # Filter to only FAST tokens
        batch_size = token_ids.shape[0]
        action_token_sequences = []

        for i in range(batch_size):
            # Get tokens in FAST range
            tokens = token_ids[i]
            fast_tokens = tokens[
                (tokens >= self.action_token_start) & (tokens <= self.action_token_end)
            ].tolist()
            action_token_sequences.append(fast_tokens)

        # Decode using FAST tokenizer
        normalized_deltas = tokenizer.decode(
            action_token_sequences,
            action_horizon=action_horizon,
            action_dim=action_dim
        )

        # Denormalize
        denormalized_deltas = normalizer.denormalize_delta_actions(
            normalized_deltas,
            robot_type=robot_type
        )

        return torch.from_numpy(denormalized_deltas).float()


def test_collator():
    """Test the data collator."""
    print("Testing VLA Data Collator...")
    print("=" * 60)

    # Load processor
    print("Loading Qwen3-VL processor...")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True
    )

    # Create collator
    collator = VLADataCollator(processor=processor)

    # Create dummy samples
    print("\nCreating dummy samples...")
    dummy_samples = []
    for i in range(4):
        sample = {
            "left_camera": torch.rand(3, 256, 256),
            "right_camera": torch.rand(3, 256, 256),
            "head_camera": torch.rand(3, 256, 256),
            "task_description": f"Task {i}: Pick up the object",
            "robot_type": "aloha-agilex",
            "discretized_state": torch.randint(0, 256, (12,)).numpy(),
            "action_tokens": torch.randint(151936, 153984, (30 + i * 5,)).tolist(),
        }
        dummy_samples.append(sample)

    # Collate batch
    print("Collating batch...")
    batch = collator(dummy_samples)

    print(f"\nBatch contents:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")

    # Check that labels are properly masked
    print(f"\nLabel statistics:")
    for i in range(len(dummy_samples)):
        labels_i = batch['labels'][i]
        num_prompt_tokens = (labels_i == -100).sum().item()
        num_action_tokens = (labels_i != -100).sum().item()
        print(f"  Sample {i}: {num_prompt_tokens} prompt tokens (masked), {num_action_tokens} action tokens")

    print("\n" + "=" * 60)
    print("✓ Data collator test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_collator()
