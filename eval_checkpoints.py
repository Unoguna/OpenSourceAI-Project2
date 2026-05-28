"""Generate samples and optionally compute FID for many checkpoints.

Example:
    python eval_checkpoints.py --ckpts runs/pg_1024/ckpt_*.pt \
        --real-dir data/valid_1024 --n 5000 --batch-size 4

If pytorch-fid is not installed or --real-dir is omitted, the script still
creates sample grids/directories and writes a CSV with parameter counts.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import torch
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image

from generate import load_generator


def slugify(path: Path) -> str:
    text = str(path).replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


@torch.no_grad()
def dump_samples(
    ckpt: Path,
    out_dir: Path,
    grid_path: Path,
    *,
    n: int,
    batch_size: int,
    seed: int,
    device: str,
) -> int:
    G = load_generator(ckpt, device=device, use_ema=True)
    n_params = sum(p.numel() for p in G.parameters())
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path.parent.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    grid_samples: list[torch.Tensor] = []
    made = 0
    while made < n:
        b = min(batch_size, n - made)
        z = torch.randn(b, G.z_dim, generator=gen).to(device)
        fake = G(z)
        imgs = ((fake + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
        if len(grid_samples) * batch_size < 64:
            grid_samples.append(imgs)
        for i, img in enumerate(imgs):
            to_pil_image(img).save(out_dir / f"{made + i:06d}.png")
        made += b

    grid = vutils.make_grid(torch.cat(grid_samples, dim=0)[:64], nrow=8, padding=2)
    vutils.save_image(grid, grid_path)
    return n_params


def compute_fid(real_dir: Path, fake_dir: Path, device: str) -> float | None:
    cmd = [
        sys.executable,
        "-m",
        "pytorch_fid",
        str(real_dir),
        str(fake_dir),
        "--device",
        "cuda" if device == "cuda" else "cpu",
    ]
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"FID failed for {fake_dir}: {exc}")
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stdout)
            print(exc.stderr)
        return None

    text = proc.stdout + "\n" + proc.stderr
    match = re.search(r"FID:\s*([0-9.]+)", text)
    if not match:
        print(text)
        return None
    return float(match.group(1))


def prepare_real_dir(real_dir: Path | None, real_zip: Path | None, out_dir: Path) -> Path | None:
    if real_dir is not None:
        return real_dir
    if real_zip is None:
        return None

    extract_dir = out_dir / real_zip.stem
    marker = extract_dir / ".extracted"
    if marker.exists():
        return extract_dir

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting real validation zip: {real_zip} -> {extract_dir}")
    with zipfile.ZipFile(real_zip) as zf:
        zf.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, default=None)
    parser.add_argument("--real-zip", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_runs"))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default=None)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    real_dir = prepare_real_dir(args.real_dir, args.real_zip, args.out_dir)
    rows: list[dict[str, str | int | float | None]] = []

    for ckpt in args.ckpts:
        if not ckpt.exists():
            print(f"skip missing checkpoint: {ckpt}")
            continue
        name = slugify(ckpt.with_suffix(""))
        sample_dir = args.out_dir / name / "samples"
        grid_path = args.out_dir / name / "grid.png"
        if args.clean and sample_dir.exists():
            shutil.rmtree(sample_dir)

        print(f"\n== {ckpt} ==")
        n_params = dump_samples(
            ckpt,
            sample_dir,
            grid_path,
            n=args.n,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )
        fid = compute_fid(real_dir, sample_dir, device) if real_dir else None
        rows.append(
            {
                "checkpoint": str(ckpt),
                "samples": args.n,
                "params": n_params,
                "params_m": round(n_params / 1e6, 3),
                "fid": fid,
                "sample_dir": str(sample_dir),
                "grid": str(grid_path),
            }
        )
        print(f"params={n_params/1e6:.2f}M fid={fid} grid={grid_path}")

    if real_dir:
        rows.sort(key=lambda r: float("inf") if r["fid"] is None else float(r["fid"]))

    csv_path = args.out_dir / "fid_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "samples", "params", "params_m", "fid", "sample_dir", "grid"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
