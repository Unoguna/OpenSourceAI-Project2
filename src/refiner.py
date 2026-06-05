"""Residual image refiners for the project submission path.

The refiner path keeps the provided 256 generator frozen and learns small
residual image-to-image modules at 512 and optionally 1024. This is much easier
to stabilize than retraining the full generator trunk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import build_baseline_256_generator


class RefinerResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.conv2(F.silu(self.conv1(x)))


class ResidualRefiner(nn.Module):
    def __init__(self, channels: int = 32, blocks: int = 6, max_delta: float = 0.35):
        super().__init__()
        self.channels = channels
        self.blocks_n = blocks
        self.max_delta = max_delta
        self.head = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[RefinerResBlock(channels) for _ in range(blocks)])
        self.tail = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.head(x))
        h = self.blocks(h)
        delta = torch.tanh(self.tail(h)) * self.max_delta
        return (x + delta).clamp(-1.0, 1.0)


@dataclass(frozen=True)
class RefinerConfig:
    channels: int = 32
    blocks: int = 6
    max_delta: float = 0.35


class RefinerChain(nn.Module):
    """Frozen 256 generator followed by residual refiners."""

    z_dim = 512

    def __init__(
        self,
        G256: nn.Module,
        *,
        refiner512: nn.Module | None = None,
        refiner1024: nn.Module | None = None,
        target_resolution: int = 512,
    ):
        super().__init__()
        self.G256 = G256
        self.refiner512 = refiner512
        self.refiner1024 = refiner1024
        self.target_resolution = int(target_resolution)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G256(z)
        if self.target_resolution >= 512:
            x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
            if self.refiner512 is not None:
                x = self.refiner512(x)
        if self.target_resolution >= 1024:
            x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
            if self.refiner1024 is not None:
                x = self.refiner1024(x)
        return x


def refiner_param_count(*modules: nn.Module | None) -> int:
    return sum(sum(p.numel() for p in m.parameters()) for m in modules if m is not None)


def total_submission_params(chain: RefinerChain) -> int:
    return sum(p.numel() for p in chain.parameters())


def build_refiner_from_meta(meta: dict) -> ResidualRefiner:
    cfg = meta.get("refiner_config", {})
    return ResidualRefiner(
        channels=int(cfg.get("channels", 32)),
        blocks=int(cfg.get("blocks", 6)),
        max_delta=float(cfg.get("max_delta", 0.35)),
    )


def load_frozen_g256_from_state(state: dict, device: str | torch.device = "cpu") -> nn.Module:
    G = build_baseline_256_generator().to(device).eval()
    G.load_state_dict(state)
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def load_refiner_chain_from_ckpt(
    ckpt_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> RefinerChain:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("model_type") != "refiner_chain":
        raise RuntimeError(f"Not a refiner checkpoint: {ckpt_path}")

    meta = ckpt.get("meta", {})
    target_resolution = int(meta.get("target_resolution", 512))
    G = load_frozen_g256_from_state(ckpt["G256_state"], device=device)

    R512 = None
    if "refiner512_state" in ckpt or "refiner512_ema_state" in ckpt:
        R512 = build_refiner_from_meta(meta).to(device).eval()
        key = "refiner512_ema_state" if use_ema and "refiner512_ema_state" in ckpt else "refiner512_state"
        R512.load_state_dict(ckpt[key])

    R1024 = None
    if "refiner1024_state" in ckpt or "refiner1024_ema_state" in ckpt:
        R1024 = build_refiner_from_meta(meta).to(device).eval()
        key = "refiner1024_ema_state" if use_ema and "refiner1024_ema_state" in ckpt else "refiner1024_state"
        R1024.load_state_dict(ckpt[key])

    chain = RefinerChain(
        G,
        refiner512=R512,
        refiner1024=R1024,
        target_resolution=target_resolution,
    ).to(device).eval()
    return chain
