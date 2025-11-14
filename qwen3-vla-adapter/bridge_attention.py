from typing import Optional
import torch
import torch.nn as nn
import math
from torch.nn import init


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


class MLPResNetBlockPro(nn.Module):
    """MLP ResNet block with separate projections for self, adapter, task + RoPE.

    Residual MLP block with cross-attention conditioning.
    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    """

    def __init__(self, src_dim: int, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim > 0 and dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(src_dim, dim)
        self.v_adapter = nn.Linear(src_dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(src_dim, dim)
        self.v_task = nn.Linear(src_dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.empty(1))
        self.initialize_gating_factor()

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def initialize_gating_factor(self):
        self.gating_factor.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        p: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector (for FiLM)
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        if p is not None:
            # concat h_a and p
            h_adapter = torch.cat((h_a, p), dim=1)
        else:
            h_adapter = h_a

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1)
        K_t = h_task.size(1)

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)

        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(
            v_tokens, B, T
        )
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(
            v_adapter, B, K_a
        )
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        device, dtype = x.device, x.dtype
        cos_main, sin_main = self.rope(seq_len=T, device=device, dtype=dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=device, dtype=dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=device, dtype=dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens, v_adapter, v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # residual + FFN
        x = self.ffn(output + x)
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    def __init__(
        self,
        num_blocks: int,
        hidden_dim: int,
        src_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.in_layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(
                MLPResNetBlockPro(src_dim=src_dim, dim=hidden_dim)
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        h_a: torch.Tensor,  # action embedding
        h_t: torch.Tensor,  # task embedding
        p: Optional[torch.Tensor] = None,  # proprioception
    ) -> torch.Tensor:
        x = self.in_layer_norm(x)
        # x: (batch_size, input_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(
                x,
                h_t=h_t[:, i + 1, :],
                h_a=h_a[:, i + 1, :],
                p=p,
            )  # shape: (batch_size, hidden_dim)
        x = self.layer_norm(x)  # shape: (batch_size, hidden_dim)
        x = self.out_proj(x)  # shape: (batch_size, output_dim)
        return x


class ActionHead(nn.Module):
    """Simple action head that generates continuous actions."""

    def __init__(
        self,
        num_layers: int = 24,
        src_dim: int = 2048,  # cross attention source
        hidden_dim: int = 512,
        action_dim: int = 7,
        action_chunk_len: int = 8,
        num_task_tokens: int = 64,
    ):
        super().__init__()
        self.src_dim = src_dim
        self.hidden_dim = hidden_dim
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.action_chunk_len = action_chunk_len

        self.initial_hidden_state = nn.Parameter(
            torch.empty(1, self.action_chunk_len, self.hidden_dim)
        )
        self.init_hidden_state()

        self.model = MLPResNet(
            num_blocks=num_layers,
            hidden_dim=hidden_dim,
            src_dim=src_dim,
            output_dim=action_dim,
        )

    def init_hidden_state(self) -> None:
        init.uniform_(self.initial_hidden_state, -0.01, 0.01)

    def predict_action(
        self,
        hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector=None,
    ):
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        if proprio is not None:
            proprio = proprio.reshape(batch_size, -1)  # (bsz, proprio_dim)
            proprio_features = proprio_projector(proprio).unsqueeze(
                dim=1
            )  # (bsz, 1, hidden_dim)
        else:
            proprio_features = None

        task_hidden_states = hidden_states[:, :, :-self.num_task_tokens]
        actions_hidden_states = hidden_states[:, :, -self.num_task_tokens:]

        initial_action_hidden_state = self.initial_hidden_state.repeat(batch_size, 1, 1)

        action = self.model(
            initial_action_hidden_state,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
        )

        return action

    def get_gating_factors(self) -> torch.Tensor:
        return torch.cat(tuple(m.gating_factor for m in self.model.mlp_resnet_blocks))


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
    # create dummy hidden state
    num_layers = 28 + 1
    B, L, D = 1, 78, 2048
    hidden_states = [torch.randn(B, L, D) for _ in range(num_layers)]
    print("len(hidden_state)", len(hidden_states))
    print("hidden_state[0].shape", hidden_states[0].shape)

    combined_hidden_states = []
    for item in hidden_states:
        # batch_size, seq_len, dim = item.shape
        combined_hidden_states.append(item.unsqueeze(1))
    hidden_states = torch.cat(
        combined_hidden_states, dim=1
    )  # [batch_size, layers, seq_len, dim]

    action_dim = 7

    proprio_state = torch.randn(1, 8)

    proprio_projector = ProprioProjector(proprio_dim=8, llm_dim=2048)
    proprio_projector.to()
    head = ActionHead(
        num_layers=num_layers - 1,
        src_dim=D,
        hidden_dim=512,
        action_dim=action_dim,
        num_task_tokens=20,
    )
    print("param count:", count_parameters(head))
    out = head.predict_action(
        hidden_states=hidden_states,
        proprio_projector=proprio_projector,
        proprio=proprio_state,
    )
    print("out", out.shape)
