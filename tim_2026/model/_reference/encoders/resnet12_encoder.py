"""Official-style ResNet-12 encoder variants for few-shot benchmarks."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def _pool_out_size(size, stride):
    if stride <= 1:
        return size
    return ((size - stride) // stride) + 1


class DropBlock(nn.Module):
    """Device-safe DropBlock used by few-shot ResNet-12 backbones."""

    def __init__(self, block_size):
        super().__init__()
        self.block_size = int(block_size)

    def forward(self, x, gamma):
        if (not self.training) or gamma <= 0:
            return x

        batch_size, channels, height, width = x.shape
        if height < self.block_size or width < self.block_size:
            return x

        sample_h = height - self.block_size + 1
        sample_w = width - self.block_size + 1
        sampling_mask = torch.rand(
            batch_size,
            channels,
            sample_h,
            sample_w,
            device=x.device,
            dtype=x.dtype,
        )
        sampling_mask = (sampling_mask < gamma).to(x.dtype)

        left_pad = (self.block_size - 1) // 2
        right_pad = self.block_size // 2
        block_mask = F.pad(sampling_mask, (left_pad, right_pad, left_pad, right_pad))
        block_mask = F.max_pool2d(block_mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        keep_mask = 1.0 - block_mask

        keep_count = keep_mask.sum().clamp_min(1.0)
        return keep_mask * x * (keep_mask.numel() / keep_count)


class BasicBlock(nn.Module):
    """ResNet-12 basic block used in FEAT/FRN/DeepEMD-style backbones."""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        drop_rate=0.0,
        drop_block=False,
        block_size=5,
        use_pool=True,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride) if use_pool and stride > 1 else nn.Identity()
        self.drop_rate = float(drop_rate)
        self.drop_block = bool(drop_block)
        self.block_size = int(block_size)
        self.num_batches_tracked = 0
        self.dropblock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        out = self.pool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size(-1)
                keep_rate = max(
                    1.0 - self.drop_rate / (20 * 2000) * self.num_batches_tracked,
                    1.0 - self.drop_rate,
                )
                gamma = (1.0 - keep_rate) / (self.block_size ** 2)
                gamma *= (feat_size ** 2) / max(1, (feat_size - self.block_size + 1) ** 2)
                out = self.dropblock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet12Encoder(nn.Module):
    """Flexible official-style ResNet-12 encoder for different benchmark families."""

    def __init__(
        self,
        image_size=64,
        pool_output=True,
        variant="fewshot",
        drop_rate=0.0,
        dropblock_size=5,
    ):
        super().__init__()
        self.pool_output = bool(pool_output)
        self.variant = variant
        self.inplanes = 3

        if variant == "deepbdc":
            stage_strides = [2, 2, 2, 1]
        else:
            stage_strides = [2, 2, 2, 2]

        self.layer1 = self._make_layer(64, stride=stage_strides[0], drop_rate=drop_rate)
        self.layer2 = self._make_layer(160, stride=stage_strides[1], drop_rate=drop_rate)
        self.layer3 = self._make_layer(320, stride=stage_strides[2], drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(640, stride=stage_strides[3], drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.out_channels = 640
        self.out_dim = 640

        spatial = int(image_size)
        for stride in stage_strides:
            spatial = _pool_out_size(spatial, stride)
        self.out_spatial = max(1, spatial)
        self.feat_dim = [self.out_channels, self.out_spatial, self.out_spatial]

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=5):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )

        layer = BasicBlock(
            self.inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            drop_rate=drop_rate,
            drop_block=drop_block,
            block_size=block_size,
            use_pool=True,
        )
        self.inplanes = planes
        return nn.Sequential(layer)

    def forward_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        if not self.pool_output:
            return features
        return F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
