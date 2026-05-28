"""Export a Generator to ONNX with the fixed leaderboard interface.

Submission contract:
    input  z      shape (B, 512), dtype float32
    output image  shape (B, 3, 1024, 1024), dtype float32, range [-1, 1]

The wrapper always resizes G(z) to 1024x1024, so baseline 256 checkpoints,
512 checkpoints, and native 1024 checkpoints share the same ONNX contract.

Examples:
    python export_onnx.py --ckpt ckpt/ffhq256_baseline.pt --out submission.onnx
    python export_onnx.py --ckpt runs/pg_1024/final.pt --out submission.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import Generator, GeneratorConfig, build_baseline_256_generator


TARGET_RESOLUTION = 1024


class SubmissionWrapper(nn.Module):
    """Run G(z) and resize the image to 1024x1024 with bilinear interpolation."""

    def __init__(self, G: nn.Module):
        super().__init__()
        self.G = G

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G(z)
        return F.interpolate(
            x,
            size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
            mode="bilinear",
            align_corners=False,
        )


def export_to_onnx(
    G: nn.Module,
    out_path: str | Path,
    *,
    opset: int = 17,
    batch_size: int = 1,
) -> None:
    """Export `G` wrapped to (B, 512) -> (B, 3, 1024, 1024)."""
    if getattr(G, "z_dim", None) != 512:
        raise ValueError(
            f"G.z_dim must be 512 (assignment spec). Got {getattr(G, 'z_dim', None)!r}."
        )

    G.eval()
    wrapper = SubmissionWrapper(G).eval()

    dummy_z = torch.randn(batch_size, 512)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_z,
        str(out_path),
        input_names=["z"],
        output_names=["image"],
        opset_version=opset,
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        dynamo=False,
    )

    with torch.no_grad():
        ref_out = wrapper(dummy_z)
    print(f"Saved ONNX -> {out_path}")
    print("  input  z      (B, 512)")
    print(
        f"  output image  {tuple(ref_out.shape)} (B dynamic), range "
        f"[{ref_out.min():.3f}, {ref_out.max():.3f}]"
    )


def _load_generator_from_ckpt(ckpt_path: Path) -> nn.Module:
    """Load G_ema from either the 256 baseline or a train.py checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "meta" in ckpt and isinstance(ckpt["meta"], dict) and "generator_config" in ckpt["meta"]:
        g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
        G = Generator(g_cfg)
        print(f"Architecture: checkpoint meta (max_res={g_cfg.resolutions[-1]})")
    else:
        G = build_baseline_256_generator()
        print("Architecture: baseline 256 (no generator_config metadata)")

    state = ckpt.get("G_ema_state") or ckpt.get("G_state")
    if state is None:
        raise RuntimeError("Checkpoint has neither G_ema_state nor G_state")
    G.load_state_dict(state)
    print(f"Generator params: {sum(p.numel() for p in G.parameters())/1e6:.2f}M")
    return G


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to a baseline ckpt or a train.py checkpoint with meta.generator_config.",
    )
    parser.add_argument("--out", type=Path, default=Path("submission.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    G = _load_generator_from_ckpt(args.ckpt)
    export_to_onnx(G, args.out, opset=args.opset, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
