"""Transport-token projector for HROT / Ours (Tier-1 upgrades)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_hrot_transport_projector(
    input_dim: int,
    output_dim: int,
    *,
    use_mlp: bool = False,
    use_residual: bool = False,
    mlp_hidden_dim: int | None = None,
) -> nn.Module:
    """Build the backbone-to-transport projector.

    When both ``use_mlp`` and ``use_residual`` are false, returns the legacy
    ``LayerNorm -> Linear`` sequential module for checkpoint compatibility.
    """
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    if use_mlp or use_residual:
        return HROTTransportProjector(
            input_dim=input_dim,
            output_dim=output_dim,
            use_mlp=use_mlp,
            use_residual=use_residual,
            mlp_hidden_dim=mlp_hidden_dim,
        )
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, output_dim, bias=False),
    )


class HROTTransportProjector(nn.Module):
    """LayerNorm backbone tokens -> transport geometry with optional MLP + skip."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        use_mlp: bool = False,
        use_residual: bool = False,
        mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not use_mlp and not use_residual:
            raise ValueError("HROTTransportProjector requires use_mlp or use_residual")
        hidden_dim = int(mlp_hidden_dim or max(input_dim, output_dim * 2, 256))
        if hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_mlp = bool(use_mlp)
        self.use_residual = bool(use_residual)
        self.mlp_hidden_dim = hidden_dim

        self.input_norm = nn.LayerNorm(self.input_dim)
        if self.use_mlp:
            self.fc1 = nn.Linear(self.input_dim, hidden_dim, bias=False)
            self.fc2 = nn.Linear(hidden_dim, self.output_dim, bias=False)
            self.fc = None
        else:
            self.fc1 = None
            self.fc2 = None
            self.fc = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.skip = (
            nn.Linear(self.input_dim, self.output_dim, bias=False)
            if self.use_residual
            else None
        )

    def _main_branch(self, normed: torch.Tensor) -> torch.Tensor:
        if self.use_mlp:
            assert self.fc1 is not None and self.fc2 is not None
            return self.fc2(F.gelu(self.fc1(normed)))
        assert self.fc is not None
        return self.fc(normed)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.input_norm(tokens)
        projected = self._main_branch(normed)
        if self.skip is not None:
            projected = projected + self.skip(normed)
        return projected

    def forward_with_prenorm(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.input_norm(tokens)
        pre_proj_norms = normed.norm(p=2, dim=-1).clamp(min=1e-6)
        projected = self._main_branch(normed)
        if self.skip is not None:
            projected = projected + self.skip(normed)
        return projected, pre_proj_norms


__all__ = ["HROTTransportProjector", "build_hrot_transport_projector"]
