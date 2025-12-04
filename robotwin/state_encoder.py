"""
State History Encoder for VLA models.

Encodes K timesteps of robot state history into embeddings that can be
prepended to the transformer input sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class StateEncoderConfig:
    """Configuration for state history encoder."""
    encoder_type: str = "conv1d"  # "linear", "conv1d", "mlp", "transformer", "rnn"
    history_len: int = 10  # Number of past timesteps (K)
    state_dim: int = 14  # State dimension (2*6 DoF + 2 grippers)
    hidden_dim: int = 256  # Hidden dimension for encoder
    output_dim: int = 1536  # Output dim (must match Qwen3-VL hidden_size)
    n_output_tokens: int = 4  # Number of output embedding tokens
    dropout: float = 0.0

    # Output projection initialization: normal distribution with this std
    # Default 0.0 means use 1/sqrt(output_dim) which gives output norm ~1.0
    # Set to a positive value to override, or negative to use PyTorch default
    output_proj_init_std: float = 0.0

    # Conv1D specific
    conv_channels: List[int] = None  # Default: [64, 128, 256]
    conv_kernel_size: int = 3

    # Transformer specific
    n_heads: int = 4
    n_layers: int = 2

    # RNN specific
    rnn_type: str = "lstm"  # "lstm" or "gru"
    bidirectional: bool = True
    rnn_layers: int = 2

    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [64, 128, 256]


class Conv1DStateEncoder(nn.Module):
    """
    1D Convolutional encoder for state history.

    Processes state history (K, state_dim) with 1D convolutions over time,
    capturing local temporal patterns like velocity and acceleration.

    Input: (batch, K, state_dim) - K timesteps of normalized state
    Output: (batch, n_output_tokens, output_dim) - embeddings to prepend
    """

    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config

        # Build convolutional layers
        # Input shape: (batch, state_dim, K) - channels = state_dim, seq_len = K
        conv_layers = []
        in_channels = config.state_dim

        for out_channels in config.conv_channels:
            conv_layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=config.conv_kernel_size,
                    padding=config.conv_kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ])
            in_channels = out_channels

        self.convs = nn.Sequential(*conv_layers)

        # Adaptive pooling to get exactly n_output_tokens
        self.pool = nn.AdaptiveAvgPool1d(config.n_output_tokens)

        # Project to output dimension (no bias to avoid offset that increases norm)
        self.output_proj = nn.Linear(config.conv_channels[-1], config.output_dim, bias=False)

        # Custom Gaussian initialization for output projection to control embedding magnitude
        if config.output_proj_init_std >= 0:
            std = config.output_proj_init_std if config.output_proj_init_std > 0 else config.output_dim ** -0.5
            nn.init.normal_(self.output_proj.weight, mean=0.0, std=std)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

        # Layer norm on output
        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_history: (batch, K, state_dim) normalized state history in [-1, 1]

        Returns:
            (batch, n_output_tokens, output_dim) embeddings
        """
        # Transpose for conv1d: (batch, state_dim, K)
        x = state_history.transpose(1, 2)

        # Apply convolutions
        x = self.convs(x)  # (batch, channels[-1], K)

        # Pool to fixed number of tokens
        x = self.pool(x)  # (batch, channels[-1], n_output_tokens)

        # Transpose back: (batch, n_output_tokens, channels[-1])
        x = x.transpose(1, 2)

        # Project to output dimension
        x = self.output_proj(x)  # (batch, n_output_tokens, output_dim)

        # Apply dropout (layer norm disabled - let transformer's RMSNorm handle normalization)
        x = self.dropout(x)
        # x = self.output_norm(x)

        return x


class LinearStateEncoder(nn.Module):
    """
    Simple linear projection encoder for state history.

    Projects each timestep independently to the output dimension,
    letting the transformer handle temporal relationships via
    self-attention and positional embeddings.

    Similar to ViT's linear patch projection - minimal preprocessing,
    maximum use of transformer capabilities.

    Input: (batch, K, state_dim) - K timesteps of normalized state
    Output: (batch, K, output_dim) - one embedding per timestep
    """

    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config

        # Linear encoder outputs one token per timestep, so these must match
        assert config.history_len == config.n_output_tokens, (
            f"LinearStateEncoder requires history_len == n_output_tokens, "
            f"got {config.history_len} != {config.n_output_tokens}"
        )

        # Simple per-timestep projection (like ViT patch embedding)
        self.proj = nn.Linear(config.state_dim, config.output_dim)

        # Custom Gaussian initialization for output projection to control embedding magnitude
        if config.output_proj_init_std >= 0:
            std = config.output_proj_init_std if config.output_proj_init_std > 0 else config.output_dim ** -0.5
            nn.init.normal_(self.proj.weight, mean=0.0, std=std)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_history: (batch, K, state_dim) normalized state history

        Returns:
            (batch, K, output_dim) embeddings - one per timestep
        """
        # Project each timestep: (batch, K, state_dim) -> (batch, K, output_dim)
        return self.proj(state_history)


class MLPStateEncoder(nn.Module):
    """
    MLP encoder for state history.

    Flattens state history and processes through MLP layers.
    Simple but effective baseline.

    Input: (batch, K, state_dim) - K timesteps of normalized state
    Output: (batch, n_output_tokens, output_dim) - embeddings to prepend
    """

    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config

        input_dim = config.history_len * config.state_dim
        output_total = config.n_output_tokens * config.output_dim

        # Build MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, output_total),
        )

        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_history: (batch, K, state_dim) normalized state history

        Returns:
            (batch, n_output_tokens, output_dim) embeddings
        """
        batch_size = state_history.shape[0]

        # Flatten: (batch, K * state_dim)
        x = state_history.flatten(start_dim=1)

        # MLP
        x = self.mlp(x)  # (batch, n_output_tokens * output_dim)

        # Reshape
        x = x.view(batch_size, self.config.n_output_tokens, self.config.output_dim)

        # Layer norm
        x = self.output_norm(x)

        return x


class TransformerStateEncoder(nn.Module):
    """
    Transformer encoder for state history.

    Processes each timestep as a token, uses self-attention to capture
    temporal relationships, then uses learned query tokens to extract
    output embeddings.

    Input: (batch, K, state_dim) - K timesteps of normalized state
    Output: (batch, n_output_tokens, output_dim) - embeddings to prepend
    """

    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config

        # Project each timestep to embedding dimension
        self.input_proj = nn.Linear(config.state_dim, config.hidden_dim)

        # Learnable positional embeddings for history
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.history_len, config.hidden_dim) * 0.02
        )

        # Learnable query tokens (like DETR object queries)
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.n_output_tokens, config.hidden_dim) * 0.02
        )

        # Transformer encoder for self-attention over history
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # Cross-attention to extract output tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_history: (batch, K, state_dim) normalized state history

        Returns:
            (batch, n_output_tokens, output_dim) embeddings
        """
        batch_size = state_history.shape[0]

        # Project input: (batch, K, hidden_dim)
        x = self.input_proj(state_history)

        # Add positional embeddings
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Self-attention over history
        x = self.transformer_encoder(x)  # (batch, K, hidden_dim)

        # Cross-attention: queries attend to encoded history
        queries = self.query_tokens.expand(batch_size, -1, -1)
        out, _ = self.cross_attn(queries, x, x)  # (batch, n_output_tokens, hidden_dim)

        # Project to output dimension
        out = self.output_proj(out)  # (batch, n_output_tokens, output_dim)
        out = self.output_norm(out)

        return out


class RNNStateEncoder(nn.Module):
    """
    RNN (LSTM/GRU) encoder for state history.

    Processes state history sequentially, uses attention pooling
    to extract fixed number of output embeddings.

    Input: (batch, K, state_dim) - K timesteps of normalized state
    Output: (batch, n_output_tokens, output_dim) - embeddings to prepend
    """

    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.state_dim, config.hidden_dim)

        # RNN
        rnn_class = nn.LSTM if config.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.rnn_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.rnn_layers > 1 else 0,
            batch_first=True,
        )

        rnn_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        # Attention pooling to get n_output_tokens
        self.attention = nn.Sequential(
            nn.Linear(rnn_output_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.n_output_tokens),
        )

        # Output projection
        self.output_proj = nn.Linear(rnn_output_dim, config.output_dim)
        self.output_norm = nn.LayerNorm(config.output_dim)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_history: (batch, K, state_dim) normalized state history

        Returns:
            (batch, n_output_tokens, output_dim) embeddings
        """
        # Project input
        x = self.input_proj(state_history)  # (batch, K, hidden_dim)

        # RNN
        x, _ = self.rnn(x)  # (batch, K, rnn_output_dim)

        # Attention pooling: compute attention weights over timesteps
        attn_weights = self.attention(x)  # (batch, K, n_output_tokens)
        attn_weights = F.softmax(attn_weights, dim=1)  # normalize over K

        # Weighted sum: (batch, rnn_output_dim, K) @ (batch, K, n_output_tokens)
        # -> (batch, rnn_output_dim, n_output_tokens)
        out = torch.bmm(x.transpose(1, 2), attn_weights)
        out = out.transpose(1, 2)  # (batch, n_output_tokens, rnn_output_dim)

        # Project to output dimension
        out = self.output_proj(out)  # (batch, n_output_tokens, output_dim)
        out = self.output_norm(out)

        return out


def create_state_encoder(config: StateEncoderConfig) -> nn.Module:
    """
    Factory function to create state encoder based on config.

    Args:
        config: StateEncoderConfig with encoder settings

    Returns:
        State encoder module
    """
    encoder_classes = {
        "linear": LinearStateEncoder,
        "conv1d": Conv1DStateEncoder,
        "mlp": MLPStateEncoder,
        "transformer": TransformerStateEncoder,
        "rnn": RNNStateEncoder,
    }

    if config.encoder_type not in encoder_classes:
        raise ValueError(
            f"Unknown encoder type: {config.encoder_type}. "
            f"Available: {list(encoder_classes.keys())}"
        )

    return encoder_classes[config.encoder_type](config)


def test_encoders():
    """Test all encoder variants."""
    print("Testing State History Encoders")
    print("=" * 60)

    # Test configuration
    batch_size = 4
    history_len = 10
    state_dim = 14
    output_dim = 1536  # Qwen3-VL-2B hidden size
    n_output_tokens = 4

    # Create dummy input
    state_history = torch.randn(batch_size, history_len, state_dim)
    print(f"Input shape: {state_history.shape}")
    print(f"Expected output shape: ({batch_size}, {n_output_tokens}, {output_dim})")
    print()

    # Test each encoder type
    for encoder_type in ["conv1d", "mlp", "transformer", "rnn"]:
        print(f"Testing {encoder_type.upper()} encoder...")

        config = StateEncoderConfig(
            encoder_type=encoder_type,
            history_len=history_len,
            state_dim=state_dim,
            output_dim=output_dim,
            n_output_tokens=n_output_tokens,
        )

        encoder = create_state_encoder(config)

        # Count parameters
        num_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

        # Forward pass
        with torch.no_grad():
            output = encoder(state_history)

        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {num_params:,} ({trainable_params:,} trainable)")
        print(f"  Output dtype: {output.dtype}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Verify shape
        expected_shape = (batch_size, n_output_tokens, output_dim)
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
        print(f"  Shape check PASSED")
        print()

    print("=" * 60)
    print("All encoder tests passed!")


if __name__ == "__main__":
    test_encoders()
