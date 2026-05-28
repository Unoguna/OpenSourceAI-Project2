# FFHQ-256 baseline — student package

This package contains everything you need to load the distributed 256×256
baseline, sample from it, and fine-tune it up to 512 or 1024.

```
ffhqgen_student/
├── README.md
├── requirements.txt
├── train.py                       training loop (fine-tune the baseline, or resume)
├── generate.py                    sample grid from any ckpt
├── export_onnx.py                 leaderboard submission: (B,512) → (B,3,1024,1024)
├── ckpt/
│   └── ffhq256_baseline.pt        239 MB — G + D + G_ema state_dicts (slim)
├── configs/
│   └── baseline_256.yaml          starting config (matches the distributed ckpt)
└── src/
    ├── __init__.py
    ├── model.py                   Generator / Discriminator / EMA / baseline builders
    ├── losses.py                  non-saturating logistic + R1
    ├── augment.py                 DiffAug (color / translation / cutout)
    └── dataset.py                 zip-backed image dataset
```

The baseline was trained on FFHQ (50k images) for 5.0M images at 256×256. It's
intentionally **higher quality than "mediocre"** — your fine-tune starts from
a non-trivial starting point so the work focuses on upscaling and refinement,
not basic face structure.

## Quick start

Run everything from the package root (`ffhqgen_student/`):

```bash
pip install -r requirements.txt

# 1. Verify the baseline loads and samples
python generate.py --ckpt ckpt/ffhq256_baseline.pt \
                           --out sample_256.png --n 64

# 2. Fine-tune at 256 to verify your training loop works
python train.py --config configs/baseline_256.yaml \
                       --init-from ckpt/ffhq256_baseline.pt

# 3. Progressive scale-up. This package includes 512 / 1024 configs and
#    train.py can warm-start the shared trunk from the 256 baseline.
python train.py --config configs/baseline_512.yaml \
                       --init-from ckpt/ffhq256_baseline.pt
python train.py --config configs/baseline_1024.yaml \
                       --init-from runs/pg_512/final.pt
```

## Recommended Colab workflow

Use GitHub for code and Google Drive for large files / experiment outputs.

```text
code    -> GitHub
data    -> Google Drive
execute -> Colab GPU
edit    -> VS Code
```

1. Edit locally in VS Code.
2. Commit and push to GitHub.
3. Open `project02_colab.ipynb` in Colab.
4. Set `REPO_URL`, `DATA_ROOT`, and run the notebook.
5. Checkpoints, samples, FID results, and `submission.onnx` are written to
   `MyDrive/project2_outputs`.

Recommended Drive data layout:

```text
MyDrive/project2_data/
  train_50k_256.zip
  train_50k_512.zip
  train_50k_1024.zip
  valid_10k_256.zip
  valid_10k_512.zip
  valid_10k_1024.zip
```

The training configs use the 512 and 1024 training zips directly. FID evaluation
uses `valid_10k_1024.zip`; `eval_checkpoints.py --real-zip ...` extracts it
under the evaluation output directory before running `pytorch-fid`.

## Architecture (`src/model.py`)

Config-driven ResNet GAN:
- **Generator** 21.2M params: `z(512) → Linear → 4×4 → ResBlockUp×6 → 256×256`,
  Group Norm, self-attention at 32×32, tanh output.
- **Discriminator** 20.2M params: mirror of G with Spectral Norm everywhere,
  MinibatchStd, no normalization layer in residual blocks.
- **EMA**: half-life 10k images. `G_ema_state` is what you sample from for FID.

Extending to 512 or 1024 is a config change only — see below. This solution
uses a conservative channel schedule 256:64, 512:32, 1024:16, which keeps the
generator comfortably under the 40M-parameter hard threshold.

## Scaling to 512 / 1024

This is the core of the assignment — designing the additional up-block(s)
that take 256→512 (and 512→1024), and the matching down-block(s) on D.
Decide on your own:

- **Block design.** ResBlockUp-style (NN-upsample + Conv + Conv)? Sub-pixel
  conv? Transposed conv? Something else? `model.py` ships the baseline's
  ResBlockUp / ResBlockDown, but you're not required to reuse them — the
  assignment grades the resulting FID, not the architecture.
- **Channels.** How many channels at 512 / 1024? Halving each step
  (...256:64, 512:32, 1024:16) is a sensible default but not the only choice.


`train.py --init-from ...` warm-starts from any compatible checkpoint. It copies
only tensors whose names and shapes match the target generator, and remaps the
discriminator's shared 256->4 stages when higher-resolution discriminator blocks
are prepended. Newly-added 512 / 1024 blocks are randomly initialized.

Suggested training order:

```bash
# Stage 0: sanity-check the distributed baseline.
python generate.py --ckpt ckpt/ffhq256_baseline.pt --n 64 --out sample_256.png

# Stage 1: train 512 from the provided 256 baseline.
python train.py --config configs/baseline_512.yaml \
                --init-from ckpt/ffhq256_baseline.pt

# Stage 2: train 1024 from the best 512 checkpoint.
python train.py --config configs/baseline_1024.yaml \
                --init-from runs/pg_512/final.pt
```

For the leaderboard submission, your trained model only has to satisfy the
ONNX interface in `export_onnx.py` (input: `(B, 512)` z, output:
`(B, 3, 1024, 1024)` image). Anything in between is up to you.

## Training recipe (and why)

The settings in `baseline_256.yaml` were arrived at after **three divergences**
during the baseline run. Lessons:

| Setting | Value | Lesson |
|---|---|---|
| `beta2` | **0.9** | 0.99 averages too long — when a gradient spike hits, Adam takes too long to adapt and the run blows up. |
| `lr_g`, `lr_d` | both **1e-3** | TTUR (lower D lr) caused D under-training and mode collapse. Symmetric lr was stable. |
| `r1_gamma` | **10** | Higher γ (20, 30) suppressed D learning too much. |
| `augment` | `color,translation` | DiffAug **cutout 50%** was too aggressive — masked-out regions starved D. |
| `precision` | **fp32** | bf16 trained fine for ~3M images then a late spike was easier to diagnose in fp32. Either works. |
| `grad_clip_d` | 100 (effectively off) | D has Spectral Norm — already bounded; clipping is a no-op. |
| `grad_clip_g` | 10 | Real protection on G — has caught grad spikes without distorting training. |

### Measuring FID

The leaderboard ranks by FID, so you may want a number to track. 

- Dump a few thousand samples from your model (via `generate.py` in
  a loop, or directly from the ONNX session) into a directory.
- Use `pytorch-fid` (`pip install pytorch-fid`) on that directory vs a
  directory of real images at the same resolution: `python -m pytorch_fid
  <samples_dir> <real_dir>`. Cache the real-side Inception statistics with
  `--save-stats` so subsequent FID runs only re-extract the fake side.
- The leaderboard uses the same `pytorch-fid` Inception features, so this is
  your honest self-check before submission.


## Inference / sampling

```bash
# from the slim baseline ckpt
python generate.py --ckpt ckpt/ffhq256_baseline.pt --n 64 --out grid.png

# from your own fine-tune ckpt (auto-detects architecture from meta)
python generate.py --ckpt runs/my_run/ckpt_001000000.pt --n 64

# without EMA (raw G — usually noticeably worse)
python generate.py --ckpt ckpt/ffhq256_baseline.pt --no-ema --n 64
```

## Leaderboard submission (ONNX export)

Every leaderboard entry exports a single ONNX file with this fixed interface:

```
input  z      shape (B, 512), dtype float32
output image  shape (B, 3, 1024, 1024), dtype float32, range [-1, 1]
```

The `SubmissionWrapper` in `export_onnx.py` runs your Generator and
resizes the output to 1024×1024 with bilinear interpolation — so 256-, 512-,
and 1024-native models all submit through the same contract. The grader
doesn't need to know your architecture.

For the baseline 256 (sanity check the pipeline):

```bash
python export_onnx.py --ckpt ckpt/ffhq256_baseline.pt \
                              --out submission.onnx
```

For your own fine-tuned model, the CLI reads `meta.generator_config` from
checkpoints saved by `train.py`:

```bash
python export_onnx.py --ckpt runs/pg_1024/final.pt --out submission.onnx
```

Verify locally with onnxruntime before submitting:

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession("submission.onnx")
out = sess.run(None, {"z": np.random.randn(4, 512).astype(np.float32)})[0]
assert out.shape == (4, 3, 1024, 1024)
```

## Checkpoint selection for a higher score

GAN quality is not monotonic, so compare several checkpoints instead of blindly
submitting the final one.

```bash
# Check that a config stays under the 40M generator limit.
python count_params.py --config configs/baseline_1024.yaml
python count_params.py --config configs/quality_1024.yaml

# Generate individual PNGs for visual inspection or FID.
python generate.py --ckpt runs/pg_1024/final.pt \
                   --out sample_grid.png \
                   --out-dir eval_samples/pg_1024_final \
                   --n 128 --batch-size 4

# Compare checkpoints. Use a real validation image directory when available.
python eval_checkpoints.py --ckpts runs/pg_1024/ckpt_*.pt runs/quality_1024/ckpt_*.pt \
                           --real-zip data/valid_10k_1024.zip \
                           --out-dir eval_runs \
                           --n 5000 --batch-size 4
```

`eval_runs/fid_results.csv` is sorted by FID when `--real-dir` is supplied.
Use the best FID together with the saved sample grids for the final checkpoint
choice.

## Resuming your own run

`train.py --resume` restores G/D/G_ema/optimizers/RNG/wandb run id, so an
interrupted run continues bit-for-bit:

```bash
python train.py --config configs/baseline_256.yaml \
                       --resume runs/my_run/ckpt_001000000.pt
```

Do not mix `--init-from` and `--resume` — `--init-from` is for the *first*
launch of a fine-tune, `--resume` is for continuing an in-progress one.
