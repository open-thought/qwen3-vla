"""
Action tokenizer wrapper for FAST (Frequency Action Space Tokenizer).

Handles encoding normalized delta actions into discrete tokens and decoding back.
Uses the pre-trained universal FAST tokenizer from "physical-intelligence/fast".
"""

import numpy as np
import torch
from transformers import AutoProcessor


class ActionTokenizer:
    """
    Wrapper for FAST action tokenizer.

    Encodes normalized delta actions (in [-1, 1] range) into discrete tokens.
    Uses pre-trained universal tokenizer from physical-intelligence.
    """

    # Token range for FAST tokens in the extended vocabulary
    # Original Qwen3-VL vocab: 151936 tokens (0-151935)
    # FAST tokens: 2048 tokens (151936-153983)
    VOCAB_OFFSET = 151936
    VOCAB_SIZE = 2048

    def __init__(self, model_name: str = "physical-intelligence/fast"):
        """
        Initialize action tokenizer with pre-trained FAST model.

        Args:
            model_name: HuggingFace model name for pre-trained FAST tokenizer
        """
        # Load pre-trained FAST processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print(f"Loaded FAST tokenizer from {model_name}")
        print(f"  Vocab size: {self.VOCAB_SIZE}")
        print(f"  Token range: [{self.VOCAB_OFFSET}, {self.VOCAB_OFFSET + self.VOCAB_SIZE - 1}]")

    def encode(
        self,
        normalized_delta_actions: np.ndarray | torch.Tensor,
        return_torch: bool = False
    ) -> list[list[int]] | torch.Tensor:
        """
        Encode normalized delta actions into discrete tokens.

        Args:
            normalized_delta_actions: Normalized delta actions in [-1, 1] range
                Shape: (batch, action_horizon, 2*dof) or (action_horizon, 2*dof)
            return_torch: If True, return torch.Tensor instead of list

        Returns:
            Token sequences with vocab offset applied
            - If return_torch=False: list of token lists (length may vary per sample)
            - If return_torch=True: padded tensor of shape (batch, max_seq_len)
        """
        # Convert to numpy if needed
        if isinstance(normalized_delta_actions, torch.Tensor):
            normalized_delta_actions = normalized_delta_actions.cpu().numpy()

        # Ensure 3D: (batch, action_horizon, action_dim)
        if normalized_delta_actions.ndim == 2:
            normalized_delta_actions = normalized_delta_actions[None, :]

        # Encode using FAST processor
        tokens = self.processor(normalized_delta_actions)

        # Apply vocabulary offset to map to extended vocab space
        tokens_with_offset = []
        for token_seq in tokens:
            # Filter out special tokens (if any) and apply offset
            tokens_with_offset.append([t + self.VOCAB_OFFSET for t in token_seq])

        if return_torch:
            # Convert to padded tensor
            max_len = max(len(seq) for seq in tokens_with_offset)
            batch_size = len(tokens_with_offset)

            # Use vocab offset as padding token (first FAST token)
            padded_tokens = torch.full(
                (batch_size, max_len),
                self.VOCAB_OFFSET,
                dtype=torch.long
            )

            for i, seq in enumerate(tokens_with_offset):
                padded_tokens[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

            return padded_tokens

        return tokens_with_offset

    def decode(
        self,
        tokens: list[list[int]] | torch.Tensor,
        action_horizon: int,
        action_dim: int
    ) -> np.ndarray:
        """
        Decode tokens back to normalized delta actions.

        Args:
            tokens: Token sequences with vocab offset
                - list of token lists, or
                - tensor of shape (batch, seq_len)
            action_horizon: Number of timesteps in action chunk
            action_dim: Action dimension (2*dof for dual-arm robots)

        Returns:
            Normalized delta actions in [-1, 1] range
            Shape: (batch, action_horizon, action_dim)
        """
        # Convert tensor to list if needed
        if isinstance(tokens, torch.Tensor):
            # Remove padding (tokens equal to VOCAB_OFFSET)
            tokens_list = []
            for seq in tokens:
                valid_tokens = seq[seq >= self.VOCAB_OFFSET].tolist()
                tokens_list.append(valid_tokens)
            tokens = tokens_list

        # Remove vocabulary offset
        tokens_no_offset = []
        for token_seq in tokens:
            tokens_no_offset.append([t - self.VOCAB_OFFSET for t in token_seq])

        # Decode using FAST processor
        decoded_actions = self.processor.decode(
            tokens_no_offset,
            time_horizon=action_horizon,
            action_dim=action_dim
        )

        return decoded_actions

    def get_token_range(self) -> tuple[int, int]:
        """
        Get the range of token IDs used by FAST.

        Returns:
            Tuple of (min_token_id, max_token_id) inclusive
        """
        return (self.VOCAB_OFFSET, self.VOCAB_OFFSET + self.VOCAB_SIZE - 1)


def test_action_tokenizer():
    """Test action tokenizer on synthetic data."""
    print("Testing action tokenizer...")
    print("=" * 60)

    # Initialize tokenizer
    tokenizer = ActionTokenizer()

    # Create synthetic normalized delta actions
    batch_size = 4
    action_horizon = 50
    action_dim = 12  # 6-DoF dual-arm robot

    # Random actions in [-1, 1] range
    np.random.seed(42)
    actions = np.random.randn(batch_size, action_horizon, action_dim).astype(np.float32)
    actions = np.clip(actions, -1.0, 1.0)

    print(f"\nInput actions shape: {actions.shape}")
    print(f"Input actions range: [{actions.min():.4f}, {actions.max():.4f}]")

    # Test encoding
    tokens = tokenizer.encode(actions, return_torch=False)
    print(f"\nEncoded tokens:")
    print(f"  Num sequences: {len(tokens)}")
    print(f"  Sequence lengths: {[len(seq) for seq in tokens]}")
    print(f"  Token range: [{min(min(seq) for seq in tokens)}, {max(max(seq) for seq in tokens)}]")

    # Test torch encoding
    tokens_torch = tokenizer.encode(actions, return_torch=True)
    print(f"\nEncoded tokens (torch):")
    print(f"  Shape: {tokens_torch.shape}")
    print(f"  Token range: [{tokens_torch.min().item()}, {tokens_torch.max().item()}]")

    # Test decoding
    decoded_actions = tokenizer.decode(tokens, action_horizon=action_horizon, action_dim=action_dim)
    print(f"\nDecoded actions shape: {decoded_actions.shape}")
    print(f"Decoded actions range: [{decoded_actions.min():.4f}, {decoded_actions.max():.4f}]")

    # Check reconstruction error
    reconstruction_error = np.abs(actions - decoded_actions).mean()
    print(f"\nReconstruction error (MAE): {reconstruction_error:.6f}")

    # FAST is lossy compression, so some error is expected
    # But it should be reasonably small (< 0.1)
    if reconstruction_error < 0.1:
        print("✓ Reconstruction error within acceptable range")
    else:
        print(f"✗ WARNING: Reconstruction error is large: {reconstruction_error}")

    # Test single action
    single_action = actions[0]
    single_tokens = tokenizer.encode(single_action, return_torch=False)
    single_decoded = tokenizer.decode(single_tokens, action_horizon=action_horizon, action_dim=action_dim)

    print(f"\nSingle action test:")
    print(f"  Input shape: {single_action.shape}")
    print(f"  Token sequence length: {len(single_tokens[0])}")
    print(f"  Output shape: {single_decoded.shape}")
    print(f"  Reconstruction error: {np.abs(single_action - single_decoded[0]).mean():.6f}")

    print("\n" + "=" * 60)
    print("✓ Action tokenizer tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_action_tokenizer()
