"""Few-shot-specific Mamba encoder with stable multi-scale orthogonal scan."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tim_2026.model._reference.ssm.intra_image_mamba import build_2d_position_grid

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - optional in some local shells
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


def _hw_to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def _resolve_group_count(num_channels: int, max_groups: int = 32) -> int:
    groups = min(int(max_groups), int(num_channels))
    while groups > 1 and num_channels % groups != 0:
        groups //= 2
    return max(groups, 1)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


class EpisodicGN(nn.GroupNorm):
    """GroupNorm variant that remains stable in low-shot episodic training."""

    def __init__(self, num_channels: int, eps: float = 1e-4) -> None:
        super().__init__(_resolve_group_count(num_channels), num_channels, eps=eps)


class EpisodicLayerNorm(nn.Module):
    """LayerNorm over the last dimension for token tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)


class LayerScale(nn.Module):
    """Small residual scaling to keep early optimization stable."""

    def __init__(self, dim: int, init_value: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            return x * self.gamma.view(1, -1, 1, 1)
        if x.dim() == 3:
            return x * self.gamma.view(1, 1, -1)
        raise ValueError(f"LayerScale expects 3D or 4D input, got shape={tuple(x.shape)}")


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * (mask / keep_prob)


class ResidualConvStage(nn.Module):
    """Lightweight residual conv stage used before the Mamba stages."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = EpisodicGN(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = EpisodicGN(out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = F.gelu(x + residual)
        return self.pool(x)


class TransitionDown(nn.Module):
    """Exact 2x spatial downsampling that preserves residual flow."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            EpisodicGN(out_channels),
        )
        self.skip_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.skip_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main(x)
        skip = self.skip_proj(self.skip_pool(x))
        return F.gelu(main + skip)


class ReferenceSelectiveScan1D(nn.Module):
    """Reference 1D selective scan used when fused Mamba kernels are unavailable."""

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        kernel_size: int = 3,
        dt_min: float = 0.02,
        dt_max: float = 0.20,
    ) -> None:
        super().__init__()
        if int(dim) <= 0 or int(dim) % 2 != 0:
            raise ValueError("dim must be a positive even integer")
        if int(d_state) <= 0:
            raise ValueError("d_state must be positive")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.dim = int(dim)
        self.inner_dim = self.dim // 2
        self.d_state = int(d_state)
        dt_rank = max(self.inner_dim // 8, 4)

        self.in_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dw_conv_x = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.inner_dim,
            bias=False,
        )
        self.dw_conv_z = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.inner_dim,
            bias=False,
        )
        self.x_proj = nn.Linear(self.inner_dim, dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.inner_dim, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.inner_dim, 1)))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dt_proj._fsl_mamba_keep_bias = True
        self.out_proj._fsl_mamba_zero_init = True

        dt_init = torch.linspace(math.log(float(dt_min)), math.log(float(dt_max)), self.inner_dim).exp()
        self.dt_proj.bias.data.copy_(torch.tensor([_inverse_softplus(v) for v in dt_init.tolist()], dtype=torch.float32))
        nn.init.zeros_(self.out_proj.weight)

    def _depthwise_token_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        x = conv(x.transpose(1, 2)).transpose(1, 2).contiguous()
        return F.silu(x)

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        if dim != self.inner_dim:
            raise ValueError(f"Expected dim={self.inner_dim}, got {dim}")

        dt_rank = self.dt_proj.in_features
        projected = self.x_proj(x)
        dt_raw, b_raw, c_raw = projected.split([dt_rank, self.d_state, self.d_state], dim=-1)

        x32 = x.float()
        delta = F.softplus(self.dt_proj(dt_raw).float()).clamp(min=1e-4, max=1.0)
        b_term = torch.tanh(b_raw.float())
        c_term = torch.tanh(c_raw.float())
        a = (F.softplus(self.A_log.float()) + 1e-4).unsqueeze(0)
        d = self.D.float().unsqueeze(0)

        state = torch.zeros(batch_size, self.inner_dim, self.d_state, device=x.device, dtype=torch.float32)
        outputs = []
        for token_idx in range(seq_len):
            delta_t = delta[:, token_idx].unsqueeze(-1)
            decay = torch.exp(-delta_t * a).clamp(min=1e-4, max=1.0)
            drive = (1.0 - decay) * (b_term[:, token_idx].unsqueeze(1) * x32[:, token_idx].unsqueeze(-1))
            state = decay * state + drive
            y_t = (state * c_term[:, token_idx].unsqueeze(1)).sum(dim=-1) + d * x32[:, token_idx]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_branch, z_branch = self.in_proj(x).chunk(2, dim=-1)
        x_branch = self._depthwise_token_conv(x_branch, self.dw_conv_x)
        z_branch = self._depthwise_token_conv(z_branch, self.dw_conv_z)
        y = self._scan(x_branch)
        return self.out_proj(torch.cat([y, z_branch], dim=-1))


class OfficialMambaScan1D(nn.Module):
    """CUDA-friendly bidirectional scan that delegates the sequence core to mamba-ssm."""

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        kernel_size: int = 3,
        expand: int = 1,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError("OfficialMambaScan1D requires mamba-ssm") from MAMBA_IMPORT_ERROR

        self.input_norm = EpisodicLayerNorm(dim)
        pos_hidden = max(dim // 4, 16)
        self.position_proj = nn.Sequential(
            nn.Linear(6, pos_hidden),
            nn.GELU(),
            nn.Linear(pos_hidden, dim),
        )
        d_conv = max(int(kernel_size), 4)
        self.forward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mix_proj = nn.Linear(dim * 3, dim, bias=False)
        self.output_norm = EpisodicLayerNorm(dim)

        for branch in (self.forward_mamba, self.backward_mamba):
            for module in branch.modules():
                module._fsl_mamba_skip_init = True

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        if position.shape[:2] != x.shape[:2]:
            raise ValueError(
                "position must have the same batch/token dimensions as x: "
                f"x={tuple(x.shape)} position={tuple(position.shape)}"
            )
        scan_inputs = self.input_norm(x) + self.position_proj(position)
        forward_outputs = self.forward_mamba(scan_inputs)
        backward_outputs = torch.flip(
            self.backward_mamba(torch.flip(scan_inputs, dims=[1]).contiguous()),
            dims=[1],
        ).contiguous()
        mixed = self.mix_proj(torch.cat([forward_outputs, backward_outputs, scan_inputs], dim=-1))
        return self.output_norm(mixed)


class StableBidirectionalScan1D(nn.Module):
    """Position-aware fallback scan that mirrors the official Mamba path."""

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_norm = EpisodicLayerNorm(dim)
        pos_hidden = max(dim // 4, 16)
        self.position_proj = nn.Sequential(
            nn.Linear(6, pos_hidden),
            nn.GELU(),
            nn.Linear(pos_hidden, dim),
        )
        self.forward_scan = ReferenceSelectiveScan1D(dim, d_state=d_state, kernel_size=kernel_size)
        self.backward_scan = ReferenceSelectiveScan1D(dim, d_state=d_state, kernel_size=kernel_size)
        self.mix_proj = nn.Linear(dim * 3, dim, bias=False)
        self.output_norm = EpisodicLayerNorm(dim)

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        if position.shape[:2] != x.shape[:2]:
            raise ValueError(
                "position must have the same batch/token dimensions as x: "
                f"x={tuple(x.shape)} position={tuple(position.shape)}"
            )
        scan_inputs = self.input_norm(x) + self.position_proj(position)
        forward_outputs = self.forward_scan(scan_inputs)
        backward_outputs = torch.flip(
            self.backward_scan(torch.flip(scan_inputs, dims=[1]).contiguous()),
            dims=[1],
        ).contiguous()
        mixed = self.mix_proj(torch.cat([forward_outputs, backward_outputs, scan_inputs], dim=-1))
        return self.output_norm(mixed)


class AxialMambaSequenceMixer1D(nn.Module):
    """Select the best available sequence backend while preserving one public interface."""

    def __init__(self, dim: int, d_state: int = 8, kernel_size: int = 3) -> None:
        super().__init__()
        if Mamba is not None:
            self.backend = OfficialMambaScan1D(
                dim=dim,
                d_state=d_state,
                kernel_size=kernel_size,
                expand=1,
            )
            self.backend_name = "mamba_ssm"
        else:
            self.backend = StableBidirectionalScan1D(
                dim=dim,
                d_state=d_state,
                kernel_size=kernel_size,
            )
            self.backend_name = "reference"

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        return self.backend(x, position=position)


class ConvFeedForward(nn.Module):
    """Small conv feed-forward used after the scan fusion."""

    def __init__(self, dim: int, expansion: float = 1.5) -> None:
        super().__init__()
        hidden_dim = max(int(dim * expansion), dim)
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.norm1 = EpisodicGN(hidden_dim)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        self.norm2 = EpisodicGN(hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.fc2._fsl_mamba_zero_init = True
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.fc1(x)))
        x = F.gelu(self.norm2(self.dw(x)))
        return self.fc2(x)


class CrossScaleOrthogonalMambaBlock(nn.Module):
    """Few-shot Mamba block with axial CUDA-friendly scan plus a pooled proxy scan."""

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        scan_kernel_size: int = 3,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.norm1 = EpisodicGN(dim)
        self.local_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            EpisodicGN(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.row_scan = AxialMambaSequenceMixer1D(dim, d_state=d_state, kernel_size=scan_kernel_size)
        self.col_scan = AxialMambaSequenceMixer1D(dim, d_state=d_state, kernel_size=scan_kernel_size)
        gate_hidden = max(dim // 8, 8)
        self.branch_gate = nn.Sequential(
            nn.Conv2d(dim, gate_hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(gate_hidden, 3, kernel_size=1, bias=True),
        )
        self.mix_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.mix_proj._fsl_mamba_zero_init = True
        nn.init.zeros_(self.mix_proj.weight)

        self.norm2 = EpisodicGN(dim)
        self.ffn = ConvFeedForward(dim, expansion=1.5)
        self.ls1 = LayerScale(dim, init_value=layer_scale_init)
        self.ls2 = LayerScale(dim, init_value=layer_scale_init)
        self.drop_path = DropPath(drop_path)
        self.scan_backend = self.row_scan.backend_name

    def _position_map(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = build_2d_position_grid(height, width, device, dtype)
        return position.reshape(1, height, width, 6).expand(batch_size, -1, -1, -1)

    def _scan_rows(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        position = self._position_map(batch_size, height, width, x.device, x.dtype)
        tokens = x.permute(0, 2, 3, 1).reshape(batch_size * height, width, channels)
        pos_tokens = position.reshape(batch_size * height, width, 6)
        mixed = self.row_scan(tokens, position=pos_tokens)
        return mixed.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _scan_cols(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        position = self._position_map(batch_size, height, width, x.device, x.dtype)
        tokens = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)
        pos_tokens = position.permute(0, 2, 1, 3).reshape(batch_size * width, height, 6)
        mixed = self.col_scan(tokens, position=pos_tokens)
        return mixed.reshape(batch_size, width, height, channels).permute(0, 3, 2, 1).contiguous()

    def _proxy_scan(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        proxy = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        proxy = 0.5 * (self._scan_rows(proxy) + self._scan_cols(proxy))
        return F.interpolate(proxy, size=(height, width), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        local_feat = self.local_branch(x_norm)
        axial_feat = 0.5 * (self._scan_rows(x_norm) + self._scan_cols(x_norm))
        proxy_feat = self._proxy_scan(x_norm)

        branch_logits = self.branch_gate(x_norm)
        branch_weights = torch.softmax(branch_logits, dim=1)
        fused = (
            branch_weights[:, 0:1] * local_feat
            + branch_weights[:, 1:2] * axial_feat
            + branch_weights[:, 2:3] * proxy_feat
        )
        x = x + self.drop_path(self.ls1(self.mix_proj(fused)))
        x = x + self.drop_path(self.ls2(self.ffn(self.norm2(x))))
        return x


class FeaturePerturbation(nn.Module):
    """Optional small Gaussian feature perturbation for regularization."""

    def __init__(self, sigma: float = 0.0) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sigma <= 0.0:
            return x
        return x + torch.randn_like(x) * self.sigma


def _stage4_spatial(image_size: int) -> int:
    spatial = max(1, int(image_size))
    for _ in range(4):
        spatial = max(1, spatial // 2)
    return spatial


class FSLMambaEncoder(nn.Module):
    """Lightweight few-shot backbone with late-stage Mamba mixing."""

    def __init__(
        self,
        image_size: int = 84,
        pool_output: bool = False,
        in_channels: int = 3,
        base_dim: int = 48,
        output_dim: int = 320,
        d_state: int = 8,
        drop_path: float = 0.02,
        perturb_sigma: float = 0.0,
        scan_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.pool_output = bool(pool_output)
        self.out_channels = int(output_dim)
        self.out_dim = int(output_dim)
        self.out_spatial = _stage4_spatial(image_size)
        self.feat_dim = [self.out_channels, self.out_spatial, self.out_spatial]

        dim1 = int(base_dim)
        dim2 = int(base_dim) * 2
        dim3 = int(base_dim) * 4
        dim4 = int(output_dim)

        self.stage1 = ResidualConvStage(in_channels, dim1, downsample=True)
        self.stage2 = ResidualConvStage(dim1, dim2, downsample=True)
        self.stage3_down = TransitionDown(dim2, dim3)
        self.stage3 = CrossScaleOrthogonalMambaBlock(
            dim3,
            d_state=d_state,
            scan_kernel_size=scan_kernel_size,
            drop_path=float(drop_path) * 0.5,
        )
        self.stage4_down = TransitionDown(dim3, dim4)
        self.stage4 = CrossScaleOrthogonalMambaBlock(
            dim4,
            d_state=d_state,
            scan_kernel_size=scan_kernel_size,
            drop_path=float(drop_path),
        )
        self.token_norm = EpisodicLayerNorm(dim4)
        self.perturb = FeaturePerturbation(perturb_sigma)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if getattr(module, "_fsl_mamba_skip_init", False):
                continue
            if isinstance(module, nn.Conv2d):
                if getattr(module, "_fsl_mamba_zero_init", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                if getattr(module, "_fsl_mamba_zero_init", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    if not getattr(module, "_fsl_mamba_keep_bias", False):
                        nn.init.zeros_(module.bias)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(self.stage3_down(x))
        x = self.stage4(self.stage4_down(x))
        return self.perturb(x)

    def forward_dual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.forward_feature_map(x)
        local_feat = self.token_norm(_hw_to_tokens(feature_map))
        global_feat = local_feat.mean(dim=1)
        return feature_map, local_feat, global_feat

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_feature_map(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map, _, global_feat = self.forward_dual(x)
        if not self.pool_output:
            return feature_map
        return global_feat


SlimMambaEncoder = FSLMambaEncoder

__all__ = ["FSLMambaEncoder", "SlimMambaEncoder"]
