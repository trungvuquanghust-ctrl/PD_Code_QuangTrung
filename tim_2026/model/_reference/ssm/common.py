"""Common selective state-space utilities for hierarchical few-shot models."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveStateSpaceCell(nn.Module):
    """Input-conditioned diagonal state-space update."""

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.step_proj = nn.Linear(input_dim, state_dim)
        self.cond_step_proj = nn.Linear(input_dim, state_dim, bias=False)
        self.input_proj = nn.Linear(input_dim, state_dim)
        self.cond_input_proj = nn.Linear(input_dim, state_dim, bias=False)
        self.state_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.output_proj = nn.Linear(input_dim + state_dim, input_dim)
        self.output_norm = nn.LayerNorm(input_dim)
        self.log_decay = nn.Parameter(torch.zeros(state_dim))

    def init_state(
        self,
        batch_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(*batch_shape, self.state_dim, device=device, dtype=dtype)

    def forward(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        carry_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        step = F.softplus(self.step_proj(x_t))
        proposal = self.input_proj(x_t)
        if conditioning is not None:
            step = step + 0.1 * F.softplus(self.cond_step_proj(conditioning))
            proposal = proposal + self.cond_input_proj(conditioning)

        proposal = torch.tanh(proposal + self.state_proj(state))
        decay_rate = torch.exp(self.log_decay).to(device=x_t.device, dtype=x_t.dtype)
        decay = torch.exp(-step * decay_rate)
        if carry_scale is not None:
            decay = decay * carry_scale

        new_state = decay * state + (1.0 - decay) * proposal
        output = self.output_proj(torch.cat([x_t, new_state], dim=-1))
        output = self.output_norm(output + x_t)
        return output, new_state


def run_selective_scan(
    cell: SelectiveStateSpaceCell,
    inputs: torch.Tensor,
    conditioning: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    carry_scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a selective scan over a sequence tensor shaped `(B, T, D)`."""
    squeeze_batch = False
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)
        squeeze_batch = True
        if conditioning is not None and conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(0)
        if carry_scales is not None and carry_scales.dim() == 2:
            carry_scales = carry_scales.unsqueeze(0)

    if inputs.dim() != 3:
        raise ValueError(f"Expected inputs shaped (B, T, D), got {tuple(inputs.shape)}")

    batch_size = inputs.shape[0]
    state = initial_state
    if state is None:
        state = cell.init_state((batch_size,), device=inputs.device, dtype=inputs.dtype)

    outputs = []
    for step_idx in range(inputs.shape[1]):
        cond_t = conditioning[:, step_idx] if conditioning is not None else None
        carry_t = carry_scales[:, step_idx] if carry_scales is not None else None
        out_t, state = cell(inputs[:, step_idx], state, conditioning=cond_t, carry_scale=carry_t)
        outputs.append(out_t)

    outputs = torch.stack(outputs, dim=1)
    if squeeze_batch:
        return outputs.squeeze(0), state.squeeze(0)
    return outputs, state
