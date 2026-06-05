"""Train residual refiners on top of the frozen 256 baseline generator.

Examples:
    python train_refiner.py --target-res 512 --train-zip data/train_50k_512.zip \
        --g-ckpt ckpt/ffhq256_baseline.pt --run-dir runs/refiner512

    python train_refiner.py --target-res 1024 --train-zip data/train_50k_1024.zip \
        --g-ckpt ckpt/ffhq256_baseline.pt --init-refiner runs/refiner512/final.pt \
        --run-dir runs/refiner1024
"""
from __future__ import annotations

import argparse
import copy
import functools
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

from src.augment import diff_augment
from src.dataset import ZipImageDataset, infinite_loader
from src.model import Discriminator, DiscriminatorConfig
from src.refiner import (
    RefinerChain,
    ResidualRefiner,
    load_frozen_g256_from_state,
    load_refiner_chain_from_ckpt,
    refiner_param_count,
    total_submission_params,
)

print = functools.partial(print, flush=True)


def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for eb, b in zip(ema.buffers(), model.buffers()):
            eb.copy_(b)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (
        x[:, :, :, 1:] - x[:, :, :, :-1]
    ).abs().mean()


def discriminator_config(resolution: int) -> DiscriminatorConfig:
    if resolution == 512:
        return DiscriminatorConfig(
            resolutions=[512, 256, 128, 64, 32, 16, 8, 4],
            channels={512: 32, 256: 64, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512},
            use_spectral_norm=True,
            minibatch_std_group=4,
            attention_resolutions=[32],
        )
    if resolution == 1024:
        return DiscriminatorConfig(
            resolutions=[1024, 512, 256, 128, 64, 32, 16, 8, 4],
            channels={
                1024: 32,
                512: 32,
                256: 64,
                128: 128,
                64: 256,
                32: 512,
                16: 512,
                8: 512,
                4: 512,
            },
            use_spectral_norm=True,
            minibatch_std_group=4,
            attention_resolutions=[32],
        )
    raise ValueError(f"target resolution must be 512 or 1024, got {resolution}")


@torch.no_grad()
def make_input_image(
    G256: torch.nn.Module,
    R512: torch.nn.Module | None,
    z: torch.Tensor,
    *,
    target_resolution: int,
) -> torch.Tensor:
    x = G256(z)
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
    if R512 is not None:
        x = R512(x)
    if target_resolution == 1024:
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def save_preview(
    G256: torch.nn.Module,
    R512: torch.nn.Module | None,
    R_target: torch.nn.Module,
    z: torch.Tensor,
    out_path: Path,
    *,
    target_resolution: int,
) -> None:
    G256.eval()
    if R512 is not None:
        R512.eval()
    R_target.eval()
    base = make_input_image(G256, R512, z, target_resolution=target_resolution)
    refined = R_target(base)
    grid = torch.cat([base[:8], refined[:8]], dim=0)
    grid = ((grid + 1.0) / 2.0).clamp(0.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, out_path, nrow=8)


def save_checkpoint(
    path: Path,
    *,
    G256_state: dict,
    R512: torch.nn.Module | None,
    R512_ema: torch.nn.Module | None,
    R1024: torch.nn.Module | None,
    R1024_ema: torch.nn.Module | None,
    D: torch.nn.Module,
    optR: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    state = {
        "model_type": "refiner_chain",
        "step": step,
        "G256_state": G256_state,
        "args": vars(args),
        "D_state": D.state_dict(),
        "optR_state": optR.state_dict(),
        "optD_state": optD.state_dict(),
        "meta": {
            "target_resolution": args.target_res,
            "refiner_config": {
                "channels": args.channels,
                "blocks": args.blocks,
                "max_delta": args.max_delta,
            },
        },
    }
    if R512 is not None:
        state["refiner512_state"] = R512.state_dict()
    if R512_ema is not None:
        state["refiner512_ema_state"] = R512_ema.state_dict()
    if R1024 is not None:
        state["refiner1024_state"] = R1024.state_dict()
    if R1024_ema is not None:
        state["refiner1024_ema_state"] = R1024_ema.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_base_and_refiners(args: argparse.Namespace, device: str):
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if resume_ckpt.get("model_type") != "refiner_chain":
            raise RuntimeError(f"Not a refiner checkpoint: {args.resume}")
        G256 = load_frozen_g256_from_state(resume_ckpt["G256_state"], device=device)
        if args.target_res == 512:
            return G256, resume_ckpt["G256_state"], None, None

        R512 = None
        R512_ema = None
        if "refiner512_state" in resume_ckpt:
            R512 = ResidualRefiner(
                channels=args.channels,
                blocks=args.blocks,
                max_delta=args.max_delta,
            ).to(device)
            R512.load_state_dict(resume_ckpt["refiner512_state"])
            R512.eval()
            for p in R512.parameters():
                p.requires_grad_(False)
        if "refiner512_ema_state" in resume_ckpt:
            R512_ema = ResidualRefiner(
                channels=args.channels,
                blocks=args.blocks,
                max_delta=args.max_delta,
            ).to(device).eval()
            R512_ema.load_state_dict(resume_ckpt["refiner512_ema_state"])
        return G256, resume_ckpt["G256_state"], R512, R512_ema

    g_ckpt = torch.load(args.g_ckpt, map_location=device, weights_only=False)
    g_state = g_ckpt.get("G_ema_state") or g_ckpt.get("G_state")
    if g_state is None:
        raise RuntimeError(f"{args.g_ckpt} has neither G_ema_state nor G_state")
    G256 = load_frozen_g256_from_state(g_state, device=device)

    R512 = None
    R512_ema = None
    if args.target_res == 1024:
        if args.init_refiner is None:
            raise SystemExit("--init-refiner is required when --target-res 1024")
        init_chain = load_refiner_chain_from_ckpt(args.init_refiner, device=device, use_ema=True)
        R512 = init_chain.refiner512
        if R512 is None:
            raise RuntimeError("init refiner checkpoint does not contain refiner512")
        R512.eval()
        for p in R512.parameters():
            p.requires_grad_(False)
        R512_ema = copy.deepcopy(R512).eval()

    return G256, g_state, R512, R512_ema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-res", type=int, choices=[512, 1024], required=True)
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--g-ckpt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-refiner", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr-r", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--max-delta", type=float, default=0.35)
    parser.add_argument("--id-weight", type=float, default=0.03)
    parser.add_argument("--tv-weight", type=float, default=0.005)
    parser.add_argument("--r1-gamma", type=float, default=1.0)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--augment", default="color,translation")
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    if args.batch is None:
        args.batch = 8 if args.target_res == 512 else 4
    if args.wandb_name is None:
        args.wandb_name = f"refiner{args.target_res}"

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    G256, G256_state, R512, R512_ema = load_base_and_refiners(args, device)
    R_target = ResidualRefiner(
        channels=args.channels,
        blocks=args.blocks,
        max_delta=args.max_delta,
    ).to(device)
    R_target_ema = copy.deepcopy(R_target).eval()
    for p in R_target_ema.parameters():
            p.requires_grad_(False)

    start_step = 0
    resume_state = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        start_step = int(resume_state.get("step", 0))
        if args.target_res == 512:
            R_target.load_state_dict(resume_state["refiner512_state"])
            R_target_ema.load_state_dict(resume_state.get("refiner512_ema_state", resume_state["refiner512_state"]))
        else:
            R_target.load_state_dict(resume_state["refiner1024_state"])
            R_target_ema.load_state_dict(resume_state.get("refiner1024_ema_state", resume_state["refiner1024_state"]))

    if args.target_res == 512:
        chain_for_count = RefinerChain(G256, refiner512=R_target, target_resolution=512)
    else:
        chain_for_count = RefinerChain(
            G256,
            refiner512=R512,
            refiner1024=R_target,
            target_resolution=1024,
        )
    print(f"Target resolution: {args.target_res}")
    print(f"Refiner params: {refiner_param_count(R_target)/1e6:.3f}M")
    print(f"Submission params: {total_submission_params(chain_for_count)/1e6:.3f}M")

    D = Discriminator(discriminator_config(args.target_res)).to(device)
    optR = torch.optim.Adam(R_target.parameters(), lr=args.lr_r, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    if resume_state is not None:
        if "D_state" in resume_state:
            D.load_state_dict(resume_state["D_state"])
        if "optR_state" in resume_state:
            optR.load_state_dict(resume_state["optR_state"])
        if "optD_state" in resume_state:
            optD.load_state_dict(resume_state["optD_state"])
        print(f"Resumed from {args.resume} at step={start_step}")

    ds = ZipImageDataset(args.train_zip, flip=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    it = infinite_loader(loader)

    run = None
    if _HAS_WANDB and args.wandb_mode != "disabled":
        run = wandb.init(
            project="ffhqgen-student",
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    fixed_z = torch.randn(8, 512, device=device)
    sample_dir = args.run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for step in range(start_step + 1, args.steps + 1):
        real = next(it).to(device)
        if real.shape[-1] != args.target_res:
            real = F.interpolate(
                real,
                size=(args.target_res, args.target_res),
                mode="bilinear",
                align_corners=False,
            )

        z = torch.randn(args.batch, 512, device=device)
        base = make_input_image(G256, R512, z, target_resolution=args.target_res)
        fake = R_target(base).detach()

        d_real = D(diff_augment(real, args.augment))
        d_fake = D(diff_augment(fake, args.augment))
        loss_d = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
        optD.zero_grad(set_to_none=True)
        loss_d.backward()
        optD.step()

        loss_r1 = torch.tensor(0.0, device=device)
        if step % args.r1_every == 0:
            real_r1 = real.detach().requires_grad_(True)
            r1_out = D(diff_augment(real_r1, args.augment)).sum()
            grad = torch.autograd.grad(r1_out, real_r1, create_graph=True)[0]
            loss_r1 = (
                grad.pow(2).flatten(1).sum(1).mean()
                * (args.r1_gamma * 0.5 * args.r1_every)
            )
            optD.zero_grad(set_to_none=True)
            loss_r1.backward()
            optD.step()

        z = torch.randn(args.batch, 512, device=device)
        base = make_input_image(G256, R512, z, target_resolution=args.target_res)
        refined = R_target(base)
        adv = F.softplus(-D(diff_augment(refined, args.augment))).mean()
        identity = (refined - base).abs().mean()
        tv = tv_loss(refined)
        loss_r = adv + args.id_weight * identity + args.tv_weight * tv

        optR.zero_grad(set_to_none=True)
        loss_r.backward()
        optR.step()
        ema_update(R_target_ema, R_target, decay=0.999)

        if step % 100 == 0:
            print(
                f"step={step} d={loss_d.item():.4f} r={loss_r.item():.4f} "
                f"adv={adv.item():.4f} id={identity.item():.4f} "
                f"tv={tv.item():.4f} r1={loss_r1.item():.4f}"
            )
            if run is not None:
                wandb.log(
                    {
                        "loss/D": loss_d.item(),
                        "loss/R": loss_r.item(),
                        "loss/adv": adv.item(),
                        "loss/identity": identity.item(),
                        "loss/tv": tv.item(),
                        "loss/r1": loss_r1.item(),
                    },
                    step=step,
                )

        if step % args.save_every == 0:
            R512_save = R_target if args.target_res == 512 else R512
            R512_ema_save = R_target_ema if args.target_res == 512 else R512_ema
            R1024_save = R_target if args.target_res == 1024 else None
            R1024_ema_save = R_target_ema if args.target_res == 1024 else None
            save_checkpoint(
                args.run_dir / f"refiner{args.target_res}_{step:06d}.pt",
                G256_state=G256_state,
                R512=R512_save,
                R512_ema=R512_ema_save,
                R1024=R1024_save,
                R1024_ema=R1024_ema_save,
                D=D,
                optR=optR,
                optD=optD,
                step=step,
                args=args,
            )
            save_checkpoint(
                args.run_dir / "final.pt",
                G256_state=G256_state,
                R512=R512_save,
                R512_ema=R512_ema_save,
                R1024=R1024_save,
                R1024_ema=R1024_ema_save,
                D=D,
                optR=optR,
                optD=optD,
                step=step,
                args=args,
            )
            save_preview(
                G256,
                R512,
                R_target_ema,
                fixed_z,
                sample_dir / f"grid_{step:06d}.png",
                target_resolution=args.target_res,
            )

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
