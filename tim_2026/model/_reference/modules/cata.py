from __future__ import annotations

import torch
import torch.nn as nn


class CATA(nn.Module):
    """Content-Aware Token Aggregator for transport tokens.

    CATA replaces the flat spatial token grid with a fixed number of
    attention-aggregated tokens. The learnable anchors are shared across
    images and episodes.
    """

    def __init__(
        self,
        token_dim: int,
        num_anchors: int,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(token_dim) <= 0:
            raise ValueError("token_dim must be positive")
        if int(num_anchors) <= 0:
            raise ValueError("num_anchors must be positive")
        if int(num_heads) <= 0:
            raise ValueError("num_heads must be positive")
        if int(token_dim) % int(num_heads) != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if not 0.0 <= float(attn_dropout) < 1.0:
            raise ValueError("attn_dropout must be in [0, 1)")

        self.token_dim = int(token_dim)
        self.num_anchors = int(num_anchors)
        self.num_heads = int(num_heads)

        self.anchors = nn.Parameter(torch.empty(self.num_anchors, self.token_dim))
        nn.init.trunc_normal_(self.anchors, std=0.02)

        self.q_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.attn_dropout = nn.Dropout(float(attn_dropout)) if float(attn_dropout) > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(self.token_dim)

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        if z_flat.dim() != 3:
            raise ValueError(f"z_flat must have shape (Batch, Tokens, Dim), got {tuple(z_flat.shape)}")
        batch_size, token_count, token_dim = z_flat.shape
        if token_dim != self.token_dim:
            raise ValueError(f"z_flat last dim {token_dim} does not match token_dim={self.token_dim}")

        head_dim = self.token_dim // self.num_heads
        q = self.q_proj(self.anchors).unsqueeze(0).expand(batch_size, -1, -1)
        k = self.k_proj(z_flat)
        v = self.v_proj(z_flat)

        q = q.reshape(batch_size, self.num_anchors, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(batch_size, token_count, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(batch_size, token_count, self.num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
        attn = self.attn_dropout(attn.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, self.num_anchors, self.token_dim)
        return self.norm(self.out_proj(out))


__all__ = ["CATA"]
