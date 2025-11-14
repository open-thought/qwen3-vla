from typing import Optional
import torch
import torch.nn as nn
import math


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]  # even sub dimension
        x2 = x[..., 1::2]  # odd sub dimension
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]: 
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim > 0 and dim % num_heads == 0
        self.head_dim = dim // num_heads

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input features (B, T, C)
        Returns: output features (B, T, C)
        """
        B, T, C = x.shape
        device, dtype = x.device, x.dtype

        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q, B, T)  # (B, H, T, D)
        k = reshape_heads(k, B, T)
        v = reshape_heads(v, B, T)

        # Apply RoPE
        cos, sin = self.rope(seq_len=T, device=device, dtype=dtype)
        q, k = apply_rope(q, k, cos, sin)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        return output


class CrossAttention(nn.Module):
    """Multi-head cross-attention with RoPE."""

    def __init__(self, dim: int, src_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.src_dim = src_dim
        self.num_heads = num_heads
        assert dim > 0 and dim % num_heads == 0
        self.head_dim = dim // num_heads

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(src_dim, dim)
        self.v_proj = nn.Linear(src_dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: query features (B, T, C)
        context: key/value features (B, K, C_src)
        Returns: output features (B, T, C)
        """
        B, T, C = x.shape
        K = context.size(1)
        device, dtype = x.device, x.dtype

        # Compute Q from x, K and V from context
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Reshape for multi-head attention
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q, B, T)
        k = reshape_heads(k, B, K)
        v = reshape_heads(v, B, K)

        # Apply RoPE
        cos_q, sin_q = self.rope(seq_len=T, device=device, dtype=dtype)
        q, _ = apply_rope(q, q, cos_q, sin_q)
        cos_k, sin_k = self.rope(seq_len=K, device=device, dtype=dtype)
        _, k = apply_rope(k, k, cos_k, sin_k)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        return output


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with self-attention and cross-attention.

    Following the classic Transformer decoder architecture:
    1. Self-attention with residual connection
    2. Cross-attention to adapter hidden states with residual connection
    3. Feed-forward network with residual connection
    """

    def __init__(self, src_dim: int, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.dim = dim

        # LayerNorms (pre-normalization)
        self.ln_self_attn = nn.LayerNorm(dim)
        self.ln_cross_attn = nn.LayerNorm(dim)
        self.ln_ffn = nn.LayerNorm(dim)

        # Attention modules
        self.self_attn = SelfAttention(dim=dim, num_heads=num_heads)
        self.cross_attn = CrossAttention(dim=dim, src_dim=src_dim, num_heads=num_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        adapter_hidden_states: torch.Tensor,
        proprio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: input features (B, T, C)
        adapter_hidden_states: adapter hidden states from VLM (B, K_a, C_src)
        proprio_features: proprioceptive features (B, 1, C_src)
        """
        # Prepare context by concatenating proprioceptive features if provided
        if proprio_features is not None:
            context = torch.cat((adapter_hidden_states, proprio_features), dim=1)
        else:
            context = adapter_hidden_states

        # 1. Self-attention with residual
        x = x + self.self_attn(self.ln_self_attn(x))

        # 2. Cross-attention with residual
        x = x + self.cross_attn(self.ln_cross_attn(x), context)

        # 3. FFN with residual
        x = x + self.ffn(self.ln_ffn(x))

        return x


class ActionDecoder(nn.Module):
    """Transformer decoder for action prediction with adapter hidden state conditioning."""

    def __init__(
        self,
        num_blocks: int,
        hidden_dim: int,
        src_dim: int,
        output_dim: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim

        # Input and output projections
        self.in_layer_norm = nn.LayerNorm(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        # Stack of transformer decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.decoder_blocks.append(
                TransformerDecoderBlock(
                    src_dim=src_dim,
                    dim=hidden_dim,
                    num_heads=num_heads
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        adapter_hidden_states: torch.Tensor,
        proprio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: input features (B, T, hidden_dim)
        adapter_hidden_states: adapter hidden states from VLM (B, num_layers, num_embeddings, src_dim)
        proprio_features: proprioceptive features (B, 1, src_dim)
        Returns: output features (B, T, output_dim)
        """
        # Initial layer norm
        x = self.in_layer_norm(x)

        # Apply transformer decoder blocks
        for i, block in enumerate(self.decoder_blocks):
            x = block(
                x,
                adapter_hidden_states=adapter_hidden_states[:, i + 1, :],
                proprio_features=proprio_features,
            )

        # Final layer norm and projection
        x = self.layer_norm(x)
        x = self.out_proj(x)

        return x


class ActionHead(nn.Module):
    """Simplified action head that generates continuous actions using adapter hidden states."""

    def __init__(
        self,
        num_layers: int = 24,
        src_dim: int = 2048,  # cross attention source dimension
        hidden_dim: int = 512,
        action_dim: int = 7,
        action_chunk_len: int = 8,
    ):
        super().__init__()
        self.src_dim = src_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_chunk_len = action_chunk_len

        # Action token embeddings - learnable embeddings for each position in the action sequence
        self.action_token_embeddings = nn.Embedding(
            num_embeddings=self.action_chunk_len,
            embedding_dim=self.hidden_dim,
        )

        self.decoder = ActionDecoder(
            num_blocks=num_layers,
            hidden_dim=hidden_dim,
            src_dim=src_dim,
            output_dim=action_dim,
        )

    def predict_action(
        self,
        hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector=None,
    ):
        """
        hidden_states: Adapter hidden states from the VLM (B, num_layers, num_embeddings, src_dim)
        proprio: Proprioceptive state (B, proprio_dim)
        proprio_projector: Projector to map proprio to src_dim
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        if proprio is not None:
            proprio = proprio.reshape(batch_size, -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio).unsqueeze(
                dim=1
            )  # (bsz, 1, src_dim)
        else:
            proprio_features = None

        # Get action token embeddings for the sequence
        action_token_ids = torch.arange(self.action_chunk_len, device=device)
        action_token_features = self.action_token_embeddings(action_token_ids)  # (action_chunk_len, hidden_dim)
        action_token_features = action_token_features.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, action_chunk_len, hidden_dim)

        # Decode actions using adapter hidden states
        action = self.decoder(
            x=action_token_features,
            adapter_hidden_states=hidden_states,
            proprio_features=proprio_features,
        )

        return action


class ProprioProjector(nn.Module):
    """
    Projects proprioceptive inputs into the LLM's embedding space.
    """

    def __init__(self, proprio_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.proprio_dim = proprio_dim
        self.llm_dim = llm_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the simplified architecture
    num_layers = 28 + 1
    B = 1  # Batch size
    num_adapter_embeddings = 64  # Number of adapter embeddings/hidden states
    D = 2048  # Source dimension (VLM hidden dimension)

    # Create dummy adapter hidden states from VLM
    hidden_states = [torch.randn(B, num_adapter_embeddings, D) for _ in range(num_layers)]
    print("len(hidden_states):", len(hidden_states))
    print("hidden_states[0].shape:", hidden_states[0].shape)

    # Stack hidden states across layers
    combined_hidden_states = []
    for item in hidden_states:
        combined_hidden_states.append(item.unsqueeze(1))
    hidden_states = torch.cat(
        combined_hidden_states, dim=1
    )  # [batch_size, layers, num_embeddings, dim]
    print("combined hidden_states.shape:", hidden_states.shape)

    action_dim = 7
    proprio_dim = 8

    # Create proprioceptive state
    proprio_state = torch.randn(B, proprio_dim)

    # Create projector for proprioceptive features
    proprio_projector = ProprioProjector(proprio_dim=proprio_dim, llm_dim=D)

    # Create action head
    head = ActionHead(
        num_layers=num_layers - 1,  # -1 because we use layers for cross-attention
        src_dim=D,
        hidden_dim=512,
        action_dim=action_dim,
        action_chunk_len=8,
    )
    print("param count:", count_parameters(head))

    # Predict actions
    out = head.predict_action(
        hidden_states=hidden_states,
        proprio_projector=proprio_projector,
        proprio=proprio_state,
    )
    print("output shape:", out.shape)
    print("expected shape: (batch_size={}, action_chunk_len={}, action_dim={})".format(
        B, head.action_chunk_len, action_dim
    ))
