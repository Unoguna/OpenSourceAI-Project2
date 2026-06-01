"""Small StyleGAN-inspired generator implemented from scratch.

This intentionally keeps the training loop simple while adding the core
StyleGAN ideas: a mapping network, learned constant input, per-layer style
modulation, noise injection, and progressive synthesis blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_channels(channels: dict[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in channels.items()}


@dataclass
class StyleGeneratorConfig:
    z_dim: int
    resolutions: list[int]
    channels: dict[int, int]
    architecture: str = "stylegan_lite"
    w_dim: int = 512
    mapping_layers: int = 4

    def __post_init__(self) -> None:
        self.resolutions = [int(r) for r in self.resolutions]
        self.channels = _normalize_channels(self.channels)
        if self.architecture != "stylegan_lite":
            raise ValueError(f"Unsupported architecture: {self.architecture!r}")
        if self.resolutions[0] != 4:
            raise ValueError("StyleGenerator must start at resolution 4")
        for r in self.resolutions:
            if r not in self.channels:
                raise ValueError(f"channels missing entry for resolution {r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StyleGeneratorConfig":
        return cls(**d)


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = [PixelNorm()]
        for i in range(num_layers):
            layers.append(nn.Linear(z_dim if i == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


class StyledConv(nn.Module):
    """ONNX-friendly style modulation followed by a regular convolution."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int):
        super().__init__()
        self.affine = nn.Linear(w_dim, in_ch * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.noise_strength = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(w).chunk(2, dim=1)
        x = x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = self.conv(x)
        if self.training:
            noise = torch.randn(
                x.shape[0], 1, x.shape[2], x.shape[3],
                device=x.device, dtype=x.dtype,
            )
            x = x + self.noise_strength * noise
        x = x + self.bias[None, :, None, None]
        return F.leaky_relu(x, 0.2)


class StyleSynthesisBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, upsample: bool):
        super().__init__()
        self.upsample = upsample
        self.conv1 = StyledConv(in_ch, out_ch, w_dim)
        self.conv2 = StyledConv(out_ch, out_ch, w_dim)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv1(x, w)
        return self.conv2(x, w)


class StyleGenerator(nn.Module):
    def __init__(self, cfg: StyleGeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.z_dim
        first_ch = cfg.channels[4]
        self.mapping = MappingNetwork(cfg.z_dim, cfg.w_dim, cfg.mapping_layers)
        self.constant = nn.Parameter(torch.randn(1, first_ch, 4, 4))

        blocks: list[nn.Module] = []
        for i, res in enumerate(cfg.resolutions):
            in_ch = cfg.channels[cfg.resolutions[max(0, i - 1)]]
            out_ch = cfg.channels[res]
            blocks.append(StyleSynthesisBlock(in_ch, out_ch, cfg.w_dim, upsample=i > 0))
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.Conv2d(cfg.channels[cfg.resolutions[-1]], 3, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.constant.expand(z.shape[0], -1, -1, -1)
        for block in self.blocks:
            x = block(x, w)
        return torch.tanh(self.to_rgb(x))
