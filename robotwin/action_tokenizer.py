"""
Action tokenizers for VLA training.

Supports two tokenization methods:
1. FAST (Frequency Action Space Tokenizer) - compressed representation using pre-trained tokenizer
2. Bin (OpenVLA-style) - simple uniform discretization, one token per action dimension

Both tokenizers map to extended vocabulary tokens (starting at VOCAB_OFFSET).
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch


class BaseActionTokenizer(ABC):
    """Abstract base class for action tokenizers."""

    # Token range for action tokens in the extended vocabulary
    # Original Qwen3-VL vocab: 151936 tokens (0-151935)
    VOCAB_OFFSET = 151936

    @abstractmethod
    def encode(
        self,
        normalized_actions: np.ndarray | torch.Tensor,
        return_torch: bool = False
    ) -> list[list[int]] | torch.Tensor:
        """Encode normalized actions into discrete tokens."""
        pass

    @abstractmethod
    def decode(
        self,
        tokens: list[list[int]] | torch.Tensor,
        action_horizon: int,
        action_dim: int
    ) -> np.ndarray:
        """Decode tokens back to normalized actions."""
        pass

    @abstractmethod
    def get_token_range(self) -> tuple[int, int]:
        """Get the range of token IDs used by this tokenizer."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get the vocabulary size of this tokenizer."""
        pass


class BinTokenizer(BaseActionTokenizer):
    """
    Simple bin-based action tokenizer (OpenVLA-style).

    Discretizes each action dimension independently into N bins.
    One token per action dimension, so action_horizon * action_dim tokens total.

    This is simpler than FAST but produces more tokens.
    """

    def __init__(
        self,
        n_bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        """
        Initialize bin tokenizer.

        Args:
            n_bins: Number of bins for discretization (default: 256)
            min_action: Minimum action value (default: -1.0)
            max_action: Maximum action value (default: 1.0)
        """
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action

        # Create uniform bins and bin centers
        self.bins = np.linspace(min_action, max_action, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Vocabulary size is the number of bins
        self._vocab_size = n_bins

        print(f"Initialized BinTokenizer")
        print(f"  Bins: {n_bins}")
        print(f"  Action range: [{min_action}, {max_action}]")
        print(f"  Token range: [{self.VOCAB_OFFSET}, {self.VOCAB_OFFSET + n_bins - 1}]")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(
        self,
        normalized_actions: np.ndarray | torch.Tensor,
        return_torch: bool = False
    ) -> list[list[int]] | torch.Tensor:
        """
        Encode normalized actions into discrete tokens.

        Each action value becomes one token, so output length = action_horizon * action_dim.

        Args:
            normalized_actions: Normalized actions in [-1, 1] range
                Shape: (batch, action_horizon, action_dim) or (action_horizon, action_dim)
            return_torch: If True, return torch.Tensor instead of list

        Returns:
            Token sequences with vocab offset applied
        """
        # Convert to numpy if needed
        if isinstance(normalized_actions, torch.Tensor):
            normalized_actions = normalized_actions.cpu().numpy()

        # Ensure 3D: (batch, action_horizon, action_dim)
        if normalized_actions.ndim == 2:
            normalized_actions = normalized_actions[None, :]

        batch_size, action_horizon, action_dim = normalized_actions.shape

        # Clip to valid range
        clipped = np.clip(normalized_actions, self.min_action, self.max_action)

        # Digitize: map continuous values to bin indices (1 to n_bins)
        # np.digitize returns indices in [1, n_bins] for values in [min, max]
        discretized = np.digitize(clipped, self.bins)

        # Clip to valid bin range [1, n_bins] and convert to 0-indexed [0, n_bins-1]
        discretized = np.clip(discretized, 1, self.n_bins) - 1

        # Flatten to (batch, action_horizon * action_dim) and add vocab offset
        tokens_flat = discretized.reshape(batch_size, -1) + self.VOCAB_OFFSET

        # Convert to list of lists
        tokens_list = [list(seq) for seq in tokens_flat]

        if return_torch:
            return torch.tensor(tokens_flat, dtype=torch.long)

        return tokens_list

    def decode(
        self,
        tokens: list[list[int]] | torch.Tensor,
        action_horizon: int,
        action_dim: int
    ) -> np.ndarray:
        """
        Decode tokens back to normalized actions.

        Args:
            tokens: Token sequences with vocab offset
            action_horizon: Number of timesteps in action chunk
            action_dim: Action dimension

        Returns:
            Normalized actions in [-1, 1] range
            Shape: (batch, action_horizon, action_dim)
        """
        # Convert to numpy array
        if isinstance(tokens, torch.Tensor):
            tokens_array = tokens.cpu().numpy()
        else:
            # Pad to same length if needed
            max_len = max(len(seq) for seq in tokens)
            tokens_array = np.zeros((len(tokens), max_len), dtype=np.int64)
            for i, seq in enumerate(tokens):
                tokens_array[i, :len(seq)] = seq

        batch_size = tokens_array.shape[0]
        expected_len = action_horizon * action_dim

        # Remove vocab offset to get bin indices
        bin_indices = tokens_array - self.VOCAB_OFFSET

        # Clip to valid bin range
        bin_indices = np.clip(bin_indices, 0, len(self.bin_centers) - 1)

        # Take only the expected number of tokens
        if bin_indices.shape[1] >= expected_len:
            bin_indices = bin_indices[:, :expected_len]
        else:
            # Pad with zeros (center bin) if not enough tokens
            padded = np.zeros((batch_size, expected_len), dtype=np.int64)
            padded[:, :bin_indices.shape[1]] = bin_indices
            bin_indices = padded

        # Map bin indices to bin centers
        actions_flat = self.bin_centers[bin_indices]

        # Reshape to (batch, action_horizon, action_dim)
        actions = actions_flat.reshape(batch_size, action_horizon, action_dim)

        return actions.astype(np.float32)

    def get_token_range(self) -> tuple[int, int]:
        """Get the range of token IDs used."""
        return (self.VOCAB_OFFSET, self.VOCAB_OFFSET + self.n_bins - 1)


class FASTTokenizer(BaseActionTokenizer):
    """
    Wrapper for FAST action tokenizer.

    Encodes normalized delta actions (in [-1, 1] range) into discrete tokens.
    Uses pre-trained universal tokenizer from physical-intelligence.

    FAST uses compression, so output length varies and is typically much shorter
    than action_horizon * action_dim.
    """

    # FAST vocab size
    FAST_VOCAB_SIZE = 2048

    def __init__(self, model_name: str = "physical-intelligence/fast"):
        """
        Initialize action tokenizer with pre-trained FAST model.

        Args:
            model_name: HuggingFace model name for pre-trained FAST tokenizer
        """
        from transformers import AutoProcessor

        # Load pre-trained FAST processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self._vocab_size = self.FAST_VOCAB_SIZE

        print(f"Loaded FAST tokenizer from {model_name}")
        print(f"  Vocab size: {self._vocab_size}")
        print(f"  Token range: [{self.VOCAB_OFFSET}, {self.VOCAB_OFFSET + self._vocab_size - 1}]")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

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
        return (self.VOCAB_OFFSET, self.VOCAB_OFFSET + self._vocab_size - 1)


# Backward compatibility alias
ActionTokenizer = FASTTokenizer


def create_action_tokenizer(
    tokenizer_type: Literal["fast", "bin"] = "fast",
    **kwargs
) -> BaseActionTokenizer:
    """
    Factory function to create an action tokenizer.

    Args:
        tokenizer_type: Type of tokenizer to create
            - "fast": FAST tokenizer (compressed, variable-length output)
            - "bin": Simple bin tokenizer (OpenVLA-style, fixed-length output)
        **kwargs: Additional arguments passed to the tokenizer constructor

    Returns:
        Action tokenizer instance
    """
    if tokenizer_type == "fast":
        return FASTTokenizer(**kwargs)
    elif tokenizer_type == "bin":
        return BinTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Must be 'fast' or 'bin'.")


def _test_tokenizer(tokenizer: BaseActionTokenizer, name: str, action_horizon: int = 16, action_dim: int = 14):
    """Test a single tokenizer."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")

    # Create synthetic normalized actions
    batch_size = 4
    np.random.seed(42)
    actions = np.random.rand(batch_size, action_horizon, action_dim).astype(np.float32) * 2 - 1  # [-1, 1]

    print(f"\nInput actions shape: {actions.shape}")
    print(f"Input actions range: [{actions.min():.4f}, {actions.max():.4f}]")

    # Test encoding
    tokens = tokenizer.encode(actions, return_torch=False)
    print(f"\nEncoded tokens:")
    print(f"  Num sequences: {len(tokens)}")
    print(f"  Sequence lengths: {[len(seq) for seq in tokens]}")
    print(f"  Token range: [{min(min(seq) for seq in tokens)}, {max(max(seq) for seq in tokens)}]")

    # Test decoding
    decoded_actions = tokenizer.decode(tokens, action_horizon=action_horizon, action_dim=action_dim)
    print(f"\nDecoded actions shape: {decoded_actions.shape}")
    print(f"Decoded actions range: [{decoded_actions.min():.4f}, {decoded_actions.max():.4f}]")

    # Check reconstruction error
    reconstruction_error = np.abs(actions - decoded_actions).mean()
    print(f"\nReconstruction error (MAE): {reconstruction_error:.6f}")

    # Expected error thresholds
    # BinTokenizer: ~0.004 (256 bins over [-1,1] = bin width 0.0078)
    # FASTTokenizer: varies, typically < 0.1
    threshold = 0.01 if name == "BinTokenizer" else 0.1

    if reconstruction_error < threshold:
        print(f"✓ Reconstruction error within acceptable range (< {threshold})")
    else:
        print(f"✗ WARNING: Reconstruction error is large: {reconstruction_error}")

    return reconstruction_error


def test_action_tokenizers():
    """Test both action tokenizers."""
    print("Testing Action Tokenizers")
    print("=" * 60)

    action_horizon = 16
    action_dim = 14  # 12 joints + 2 grippers

    # Test BinTokenizer
    bin_tokenizer = BinTokenizer(n_bins=256)
    bin_error = _test_tokenizer(bin_tokenizer, "BinTokenizer", action_horizon, action_dim)

    # Test FASTTokenizer
    fast_tokenizer = FASTTokenizer()
    fast_error = _test_tokenizer(fast_tokenizer, "FASTTokenizer", action_horizon, action_dim)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"BinTokenizer:  MAE = {bin_error:.6f}, tokens = {action_horizon * action_dim} (fixed)")
    print(f"FASTTokenizer: MAE = {fast_error:.6f}, tokens = variable (compressed)")
    print(f"\n✓ All tokenizer tests completed!")


if __name__ == "__main__":
    test_action_tokenizers()
