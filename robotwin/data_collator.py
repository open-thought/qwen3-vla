"""
Data collator for VLA training with Qwen3-VL.

Prepares batches for multi-modal VLM training with images, text prompts,
and action token targets.
"""

import torch
from typing import Dict, List

from prompt_formatter import PromptFormatter


class VLADataCollator:
    """
    Collate samples into batches for VLA training.

    Handles:
    - Multi-image input (3 cameras)
    - Text prompt with task, robot type, and discretized state
    - Action token targets for training
    - Proper masking to compute loss only on action tokens
    """

    # Base offset for action tokens in extended vocabulary
    ACTION_TOKEN_OFFSET = 151936

    def __init__(
        self,
        processor,
        action_token_start: int = 151936,  # First action token ID
        action_token_end: int = 152190,    # Last action token ID (default: 151936 + 255 - 1)
        add_eot_token: bool = True,        # Add EOT token after action sequence
        tokenizer_type: str = "bspline",   # "bspline" or "bin"
        n_bins: int = 255,                 # Number of bins for quantization
        state_reconstruction: bool = False,  # Enable state reconstruction auxiliary task
        state_reconstruction_only_on_dropout: bool = True,  # Only add reconstruction when state was dropped
    ):
        """
        Args:
            processor: Qwen3VL processor for tokenizing and preparing inputs
            action_token_start: First token ID in action vocabulary
            action_token_end: Last token ID in action vocabulary
            add_eot_token: Whether to append EOT token after action sequence
            tokenizer_type: Type of action tokenizer ("bspline" or "bin")
            n_bins: Number of bins for quantization (default: 255 for exact zero)
            state_reconstruction: Enable state reconstruction auxiliary task. When True,
                appends tokenized state after actions: [actions] -> [EOT] -> [state] -> [EOT]
            state_reconstruction_only_on_dropout: Only add state reconstruction when
                state dropout was applied to this sample (default True)
        """
        self.processor = processor
        self.tokenizer_type = tokenizer_type
        self.n_bins = n_bins
        self.state_reconstruction = state_reconstruction
        self.state_reconstruction_only_on_dropout = state_reconstruction_only_on_dropout

        # Set token range based on n_bins (both bspline and bin use n_bins tokens)
        action_token_end = self.ACTION_TOKEN_OFFSET + n_bins - 1

        self.action_token_start = action_token_start
        self.action_token_end = action_token_end
        self.add_eot_token = add_eot_token
        # Use <|im_end|> as EOT token (ID 151645 in Qwen3)
        self.eot_token_id = processor.tokenizer.eos_token_id

        # Unified prompt formatter (ensures consistency with eval)
        self.prompt_formatter = PromptFormatter()

    def _tokenize_state_for_reconstruction(self, discretized_state) -> torch.Tensor:
        """
        Tokenize discretized state values for reconstruction task.

        Converts state to text like "State: [128, 64, 200, ...]" and tokenizes it.
        This allows the model to predict the full state from images.

        Args:
            discretized_state: Array of discretized state values [0-255]

        Returns:
            Tensor of token IDs for the state reconstruction target
        """
        # Format state as text (same format as in prompt, but without masking)
        state_str = ", ".join([str(int(s)) for s in discretized_state])
        state_text = f"State: [{state_str}]"

        # Tokenize (without special tokens - we'll add EOT separately)
        tokens = self.processor.tokenizer.encode(state_text, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long)

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
            # Build conversation using unified prompt formatter
            # This ensures consistency with eval (qwen3_vla_policy.py)
            conversation = self.prompt_formatter.build_conversation(
                left_camera=sample["left_camera"],
                right_camera=sample["right_camera"],
                head_camera=sample["head_camera"],
                task_description=sample["task_description"],
                robot_type=sample["robot_type"],
                discretized_state=sample["discretized_state"],
                state_dropout_mask=sample.get("state_dropout_mask"),  # May be None
            )

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

        # EOT token tensor (reused across samples)
        eot_token = torch.tensor([self.eot_token_id], dtype=torch.long)

        for i in range(batch_size):
            # Get the prompt input_ids (without action tokens)
            prompt_ids = batch_inputs["input_ids"][i]
            prompt_mask = batch_inputs["attention_mask"][i]

            # Get action tokens for this sample
            action_tokens = torch.tensor(action_token_sequences[i], dtype=torch.long)

            # Optionally append EOT token after action sequence
            if self.add_eot_token:
                action_tokens = torch.cat([action_tokens, eot_token])

            # Check if we should add state reconstruction for this sample
            sample = samples[i]
            has_dropout = sample.get("state_dropout_mask") is not None
            add_state_recon = (
                self.state_reconstruction and
                (not self.state_reconstruction_only_on_dropout or has_dropout)
            )

            if add_state_recon:
                # Tokenize the full (uncorrupted) state for reconstruction
                state_tokens = self._tokenize_state_for_reconstruction(sample["discretized_state"])
                # Add EOT after state reconstruction
                state_with_eot = torch.cat([state_tokens, eot_token])

                # Full sequence: prompt + actions + EOT + state + EOT
                full_input_ids = torch.cat([prompt_ids, action_tokens, state_with_eot])
                full_attention_mask = torch.cat([
                    prompt_mask,
                    torch.ones_like(action_tokens),
                    torch.ones_like(state_with_eot)
                ])

                # Labels: -100 for prompt, actual tokens for actions and state reconstruction
                prompt_labels = torch.full_like(prompt_ids, -100)
                full_labels = torch.cat([prompt_labels, action_tokens, state_with_eot])
            else:
                # Standard sequence: prompt + actions (+ EOT)
                full_input_ids = torch.cat([prompt_ids, action_tokens])
                full_attention_mask = torch.cat([prompt_mask, torch.ones_like(action_tokens)])

                # Create labels: -100 for prompt (don't compute loss), actual tokens for actions (+ EOT)
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

        # Handle state history if present
        # Check if any sample has state_history
        if any(sample.get("state_history") is not None for sample in samples):
            state_histories = []
            for sample in samples:
                state_hist = sample.get("state_history")
                if state_hist is not None:
                    state_histories.append(torch.tensor(state_hist, dtype=torch.float32))
                else:
                    # This shouldn't happen if state_history is consistently enabled/disabled
                    raise ValueError(
                        "Inconsistent state_history: some samples have it, others don't. "
                        "Ensure state_history_len is set consistently in the dataset."
                    )
            batch_inputs["state_history"] = torch.stack(state_histories)  # (batch, K, state_dim)

        return batch_inputs


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
            "discretized_state": torch.randint(0, 256, (14,)).numpy(),  # 12 joints + 2 grippers
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
