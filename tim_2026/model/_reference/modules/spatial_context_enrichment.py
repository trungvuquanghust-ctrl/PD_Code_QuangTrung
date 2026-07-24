"""Spatial Context Enrichment via multi-scale depthwise convolution.

Enriches each spatial token with neighbor context before OT matching.
When the residual gate is zero (initialization), output === input.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CONTEXT_FUSION_MODES = frozenset({"weighted_sum", "bottleneck"})


def parse_context_kernel_sizes(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        items = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        items = list(raw)
    sizes = []
    for item in items:
        k = int(item)
        if k < 3 or k % 2 == 0:
            raise ValueError(f"Context kernel size must be odd >= 3, got {k}")
        sizes.append(k)
    if not sizes:
        raise ValueError("context_kernel_sizes must contain at least one kernel size")
    return tuple(sorted(set(sizes)))


class SpatialContextEnrichment(nn.Module):
    """Multi-scale depthwise convolution with residual gating.

    Each branch applies a depthwise conv at a different kernel size,
    capturing progressively wider spatial neighborhoods.  A learnable
    residual gate (initialized near zero) controls how much enrichment
    is mixed into the original feature map.

    ``change_max`` hard-caps ``||delta|| / ||F||`` so the conv kernels
    cannot compensate for a small gate by producing outsized deltas.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...] = (3, 5),
        fusion: str = "weighted_sum",
        gate_max: float = 1.0,
        change_max: float = 0.0,
    ) -> None:
        super().__init__()
        fusion = str(fusion).strip().lower().replace("-", "_")
        if fusion not in CONTEXT_FUSION_MODES:
            raise ValueError(
                f"Unsupported context fusion mode: {fusion}. "
                f"Expected one of {sorted(CONTEXT_FUSION_MODES)}"
            )
        self.channels = int(channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.fusion_mode = fusion
        self.gate_max = float(gate_max)
        self.change_max = float(change_max)

        self.branches = nn.ModuleList()
        for k in self.kernel_sizes:
            self.branches.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=channels,
                    bias=False,
                )
            )

        num_branches = 1 + len(self.kernel_sizes)

        if fusion == "weighted_sum":
            self.branch_weights = nn.Parameter(torch.zeros(num_branches))
            self.raw_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(channels * num_branches, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
            )
            nn.init.zeros_(self.fusion_conv[-1].weight)
            nn.init.zeros_(self.fusion_conv[-1].bias)

    @property
    def gate_value(self) -> torch.Tensor:
        if self.fusion_mode == "weighted_sum":
            g = torch.sigmoid(self.raw_gate - 4.0)
            if self.gate_max < 1.0:
                g = g.clamp(max=self.gate_max)
            return g
        return self.raw_gate.new_tensor(1.0)

    def _cap_delta(self, feature_map: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.change_max <= 0.0:
            return delta
        f_norm = feature_map.detach().norm().clamp_min(1e-8)
        d_norm = delta.norm()
        ratio = d_norm / f_norm
        scale = (self.change_max / ratio).clamp(max=1.0)
        return delta * scale

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.dim() != 4:
            raise ValueError(
                f"SpatialContextEnrichment expects 4D input, got shape={tuple(feature_map.shape)}"
            )
        branch_outputs = [feature_map]
        for branch in self.branches:
            branch_outputs.append(branch(feature_map))

        if self.fusion_mode == "weighted_sum":
            weights = torch.softmax(
                self.branch_weights.to(device=feature_map.device, dtype=feature_map.dtype),
                dim=0,
            )
            fused = torch.zeros_like(feature_map)
            for w, b in zip(weights, branch_outputs):
                fused = fused + w * b
            gate = torch.sigmoid(
                self.raw_gate.to(device=feature_map.device, dtype=feature_map.dtype) - 4.0
            )
            if self.gate_max < 1.0:
                gate = gate.clamp(max=self.gate_max)
            delta = gate * (fused - feature_map)
            delta = self._cap_delta(feature_map, delta)
            enriched = feature_map + delta
        else:
            concat = torch.cat(branch_outputs, dim=1)
            delta = self.fusion_conv(concat)
            delta = self._cap_delta(feature_map, delta)
            enriched = feature_map + delta

        debug = getattr(self, "_debug_cache", None)
        if debug is not None:
            debug.append({
                "original": feature_map.detach(),
                "enriched": enriched.detach(),
                "delta": delta.detach(),
                "branches": [b.detach() for b in branch_outputs],
            })
        return enriched

    def diagnostics(self, feature_map: torch.Tensor, enriched: torch.Tensor) -> dict[str, torch.Tensor]:
        ref = feature_map.detach()
        diag: dict[str, torch.Tensor] = {}
        if self.fusion_mode == "weighted_sum":
            weights = torch.softmax(self.branch_weights.detach().float(), dim=0)
            diag["context/gate_value"] = self.gate_value.detach().float()
            diag["context/branch_weight_original"] = weights[0]
            for idx, k in enumerate(self.kernel_sizes):
                diag[f"context/branch_weight_conv{k}"] = weights[1 + idx]
        change = (enriched.detach().float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-8)
        diag["context/token_change_ratio"] = change
        return diag


def _to_spatial_heatmap(fmap: torch.Tensor) -> np.ndarray:
    """Channel-wise L2 norm → (H, W) heatmap, normalized to [0, 1]."""
    hmap = fmap.float().norm(dim=1).squeeze(0).cpu().numpy()
    lo, hi = hmap.min(), hmap.max()
    if hi - lo > 1e-8:
        hmap = (hmap - lo) / (hi - lo)
    return hmap


def _to_delta_heatmap(delta: torch.Tensor) -> np.ndarray:
    """Signed delta magnitude per position, normalized to [-1, 1]."""
    energy = delta.float().pow(2).sum(dim=1).squeeze(0).cpu()
    sign = delta.float().sum(dim=1).squeeze(0).cpu().sign()
    signed = sign * energy.sqrt()
    sn = signed.numpy()
    absmax = max(abs(sn.min()), abs(sn.max()), 1e-8)
    return sn / absmax


def _img_to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor → (H, W) or (H, W, 3) numpy, [0, 1]."""
    img = img_tensor.detach().float().cpu()
    if img.dim() == 3:
        img = img.permute(1, 2, 0)
    arr = img.numpy()
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    return arr


def export_context_debug_figure(
    *,
    debug_record: dict,
    input_image: torch.Tensor | None,
    kernel_sizes: tuple[int, ...],
    image_index: int,
    save_path: str,
    branch_weights: np.ndarray | None = None,
    gate_value: float | None = None,
) -> None:
    """Export one multi-panel figure showing what context enrichment did to a single image."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ImportError:
        return

    original = debug_record["original"]
    enriched = debug_record["enriched"]
    delta = debug_record["delta"]
    branches = debug_record["branches"]

    if original.dim() == 4 and original.shape[0] > 1:
        idx = min(image_index, original.shape[0] - 1)
        original = original[idx : idx + 1]
        enriched = enriched[idx : idx + 1]
        delta = delta[idx : idx + 1]
        branches = [b[idx : idx + 1] for b in branches]

    num_branches = len(branches)
    has_input = input_image is not None
    ncols = 3 + num_branches + (1 if has_input else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.0))
    fig.patch.set_facecolor("white")

    col = 0
    if has_input:
        inp_np = _img_to_numpy(input_image)
        axes[col].imshow(inp_np, cmap="gray" if inp_np.ndim == 2 else None)
        axes[col].set_title("Input image", fontsize=10, weight="bold")
        axes[col].axis("off")
        col += 1

    orig_hmap = _to_spatial_heatmap(original)
    im = axes[col].imshow(orig_hmap, cmap="inferno", vmin=0, vmax=1)
    axes[col].set_title("Original F\n||F||/channel", fontsize=10, weight="bold")
    axes[col].axis("off")
    _add_colorbar(fig, axes[col], im)
    col += 1

    branch_labels = ["identity"] + [f"conv{k}" for k in kernel_sizes]
    for b_idx, (branch_out, label) in enumerate(zip(branches, branch_labels)):
        b_hmap = _to_spatial_heatmap(branch_out)
        w_text = f" (w={branch_weights[b_idx]:.3f})" if branch_weights is not None else ""
        im = axes[col].imshow(b_hmap, cmap="inferno", vmin=0, vmax=1)
        axes[col].set_title(f"Branch: {label}{w_text}", fontsize=10)
        axes[col].axis("off")
        _add_colorbar(fig, axes[col], im)
        col += 1

    delta_hmap = _to_delta_heatmap(delta)
    change_ratio = float(delta.float().norm().item() / original.float().norm().clamp_min(1e-8).item())
    im = axes[col].imshow(delta_hmap, cmap="RdBu_r", vmin=-1, vmax=1)
    gate_text = f", gate={gate_value:.4f}" if gate_value is not None else ""
    axes[col].set_title(f"Delta (enriched-orig)\nchange={change_ratio:.4f}{gate_text}", fontsize=10, weight="bold")
    axes[col].axis("off")
    _add_colorbar(fig, axes[col], im)
    col += 1

    enr_hmap = _to_spatial_heatmap(enriched)
    im = axes[col].imshow(enr_hmap, cmap="inferno", vmin=0, vmax=1)
    axes[col].set_title("Enriched F+delta\n||F+delta||/channel", fontsize=10, weight="bold")
    axes[col].axis("off")
    _add_colorbar(fig, axes[col], im)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_baseline_debug_figure(
    *,
    feature_map: torch.Tensor,
    input_image: torch.Tensor | None,
    save_path: str,
) -> None:
    """Export a simple figure showing the raw backbone feature map (no enrichment)."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fmap = feature_map
    if fmap.dim() == 4 and fmap.shape[0] > 1:
        fmap = fmap[0:1]

    has_input = input_image is not None
    h, w = int(fmap.shape[-2]), int(fmap.shape[-1])

    energy = _to_spatial_heatmap(fmap)
    channel_var = fmap.float().squeeze(0).var(dim=0).cpu().numpy()
    cv_lo, cv_hi = channel_var.min(), channel_var.max()
    if cv_hi - cv_lo > 1e-8:
        channel_var = (channel_var - cv_lo) / (cv_hi - cv_lo)

    ncols = 3 if has_input else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.5))
    fig.patch.set_facecolor("white")

    col = 0
    if has_input:
        inp_np = _img_to_numpy(input_image)
        axes[col].imshow(inp_np, cmap="gray" if inp_np.ndim == 2 else None)
        axes[col].set_title("Input scalogram", fontsize=11, weight="bold")
        axes[col].axis("off")
        col += 1

    im = axes[col].imshow(energy, cmap="inferno", vmin=0, vmax=1)
    axes[col].set_title(f"Backbone F energy\n||F||/ch, grid={h}x{w}", fontsize=11, weight="bold")
    axes[col].axis("off")
    _add_colorbar(fig, axes[col], im)
    col += 1

    im = axes[col].imshow(channel_var, cmap="viridis", vmin=0, vmax=1)
    axes[col].set_title("Channel variance\n(high = discriminative)", fontsize=11, weight="bold")
    axes[col].axis("off")
    _add_colorbar(fig, axes[col], im)

    fig.suptitle("Baseline (no enrichment)", fontsize=13, weight="bold", y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _add_colorbar(fig, ax, im):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
