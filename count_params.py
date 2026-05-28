"""Print generator/discriminator parameter counts for a config or checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.model import Discriminator, DiscriminatorConfig, Generator, GeneratorConfig


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    args = parser.parse_args()

    if (args.config is None) == (args.ckpt is None):
        raise SystemExit("Pass exactly one of --config or --ckpt")

    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        g_cfg = GeneratorConfig.from_dict(cfg["generator"])
        d_cfg = DiscriminatorConfig.from_dict(cfg["discriminator"])
    else:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        meta = ckpt.get("meta", {})
        if "generator_config" not in meta or "discriminator_config" not in meta:
            raise SystemExit("Checkpoint does not contain meta.generator_config/discriminator_config")
        g_cfg = GeneratorConfig.from_dict(meta["generator_config"])
        d_cfg = DiscriminatorConfig.from_dict(meta["discriminator_config"])

    G = Generator(g_cfg)
    D = Discriminator(d_cfg)
    g_params = count(G)
    d_params = count(D)
    print(f"Generator:     {g_params:,} ({g_params / 1e6:.2f}M)")
    print(f"Discriminator: {d_params:,} ({d_params / 1e6:.2f}M)")
    print(f"G under 40M:   {g_params < 40_000_000}")


if __name__ == "__main__":
    main()
