# Project 02 - FFHQ 얼굴 생성 모델

이 저장소는 Open Source AI Practice Project 02를 위한 얼굴 생성 모델 코드입니다.
최종 모델은 제공된 `ffhq256_baseline.pt`의 256x256 generator를 고정하고, 512 및 1024 해상도용 residual refiner를 추가로 학습하는 방식입니다.

최종 생성기는 다음 조건을 만족합니다.

- 입력: `z` shape `(B, 512)`
- 출력: image shape `(B, 3, 1024, 1024)`
- 최종 checkpoint: `refiner1024_040000.pt`
- 최종 FID: `56.137947`
- generator parameter 수: 약 `21.42M`으로 40M 미만

## 1. 의존성 설치 방법

Python 3.10 이상과 CUDA 사용 가능한 PyTorch 환경을 권장합니다. Colab GPU 환경에서 실행하는 경우 아래 명령을 사용합니다.

```bash
pip install -r requirements.txt
pip install pytorch-fid scipy
```

또는 제출 zip에 포함된 `pyproject.toml`을 사용해 설치할 수 있습니다.

```bash
pip install .
```

ONNX export와 검증을 위해 `onnx`, `onnxruntime`도 필요합니다. `requirements.txt`에 포함되어 있지만, 누락된 경우 아래처럼 설치할 수 있습니다.

```bash
pip install onnx onnxruntime
```

## 2. 필요한 데이터 및 checkpoint

대용량 데이터와 baseline checkpoint는 GitHub에 올리지 않고 Google Drive 또는 로컬 `data/`, `ckpt/` 폴더에 둡니다.

권장 구조:

```text
project2/
  ckpt/
    ffhq256_baseline.pt
  data/
    train_50k_512.zip
    train_50k_1024.zip
    valid_10k_512.zip
    valid_10k_1024.zip
```

Colab에서는 다음과 같이 Drive 경로를 사용했습니다.

```text
/content/drive/MyDrive/project2_data/
  ffhq256_baseline.pt
  train_50k_512.zip
  train_50k_1024.zip
  valid_10k_512.zip
  valid_10k_1024.zip
```

## 3. 모델 학습 방법

최종 방식은 두 단계로 학습합니다.

1. 256 baseline generator를 고정하고 512 residual refiner 학습
2. 512 refiner를 고정하고 1024 residual refiner 학습

### 3.1 Refiner 512 학습

```bash
python train_refiner.py \
  --target-res 512 \
  --train-zip data/train_50k_512.zip \
  --g-ckpt ckpt/ffhq256_baseline.pt \
  --run-dir runs/refiner512 \
  --steps 150000 \
  --batch 8 \
  --lr-r 2e-4 \
  --lr-d 2e-4 \
  --save-every 5000 \
  --wandb-name refiner512 \
  --wandb-mode online
```

### 3.2 Refiner 1024 학습

512 refiner 학습이 끝난 뒤, 가장 좋은 512 checkpoint를 `--init-refiner`로 넣어 1024 refiner를 학습합니다.

```bash
python train_refiner.py \
  --target-res 1024 \
  --train-zip data/train_50k_1024.zip \
  --g-ckpt ckpt/ffhq256_baseline.pt \
  --init-refiner runs/refiner512/final.pt \
  --run-dir runs/refiner1024 \
  --steps 80000 \
  --batch 4 \
  --lr-r 1e-4 \
  --lr-d 1e-4 \
  --save-every 5000 \
  --wandb-name refiner1024 \
  --wandb-mode online
```

학습을 중단했다가 이어서 실행하려면 `--resume` 옵션을 사용합니다.

```bash
python train_refiner.py \
  --target-res 1024 \
  --train-zip data/train_50k_1024.zip \
  --g-ckpt ckpt/ffhq256_baseline.pt \
  --init-refiner runs/refiner512/final.pt \
  --run-dir runs/refiner1024 \
  --resume runs/refiner1024/refiner1024_040000.pt \
  --steps 80000
```

## 4. 노이즈에서 이미지 생성 방법

`generate.py`는 512차원 Gaussian noise `z`를 샘플링한 뒤 checkpoint를 통해 이미지를 생성합니다.

최종 checkpoint에서 16장 샘플 이미지를 생성하는 명령은 다음과 같습니다.

```bash
python generate.py \
  --ckpt runs/refiner1024/refiner1024_040000.pt \
  --out sample_best_refiner.png \
  --n 16 \
  --nrow 4 \
  --batch-size 4
```

FID 계산용으로 개별 PNG 파일을 저장하려면 `--out-dir`을 추가합니다.

```bash
python generate.py \
  --ckpt runs/refiner1024/refiner1024_040000.pt \
  --out sample_grid.png \
  --out-dir eval_samples/refiner1024_040000 \
  --n 10000 \
  --batch-size 4
```

## 5. FID 평가 방법

여러 checkpoint를 비교하려면 `eval_checkpoints.py`를 사용합니다.

```bash
python eval_checkpoints.py \
  --ckpts runs/refiner1024/refiner1024_010000.pt runs/refiner1024/refiner1024_040000.pt \
  --real-zip data/valid_10k_1024.zip \
  --out-dir eval_refiner_final \
  --n 10000 \
  --batch-size 4 \
  --clean
```

평가 결과는 아래 파일에 저장됩니다.

```text
eval_refiner_final/fid_results.csv
```

최종 제출에 사용한 checkpoint는 다음입니다.

```text
runs/refiner1024/refiner1024_040000.pt
```

## 6. ONNX export 방법

최종 제출용 ONNX 파일은 다음 명령으로 생성합니다.

```bash
python export_onnx.py \
  --ckpt runs/refiner1024/refiner1024_040000.pt \
  --out submission.onnx \
  --batch-size 1
```

export 결과는 다음 조건을 만족해야 합니다.

```text
input  z      (B, 512)
output image  (B, 3, 1024, 1024)
```

ONNX Runtime으로 간단히 검증하는 예시는 다음과 같습니다.

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("submission.onnx", providers=["CPUExecutionProvider"])
z = np.random.randn(1, 512).astype(np.float32)
out = sess.run(None, {"z": z})[0]

print(out.shape, out.dtype, out.min(), out.max())
assert out.shape == (1, 3, 1024, 1024)
```

## 7. 코드 구성

```text
src/model.py       기본 FFHQ-256 generator/discriminator 구조
src/refiner.py     residual refiner 및 refiner chain 구현
train_refiner.py   512/1024 refiner 학습 코드
generate.py        noise z에서 이미지 생성
eval_checkpoints.py checkpoint별 샘플 생성 및 FID 평가
export_onnx.py     최종 ONNX export
count_params.py    generator parameter 수 확인
```

## 8. 최종 모델 요약

최종 모델은 다음 흐름으로 이미지를 생성합니다.

```text
z noise (B, 512)
  -> frozen FFHQ-256 generator
  -> 256x256 image
  -> bilinear upsample to 512
  -> residual refiner 512
  -> bilinear upsample to 1024
  -> residual refiner 1024
  -> output image (B, 3, 1024, 1024)
```

baseline generator는 고정하고, trainable generator component인 residual refiner와 discriminator를 함께 adversarial하게 학습했습니다. 이 방식은 baseline의 얼굴 구조를 보존하면서 고해상도 질감만 보정하므로 1024 단계에서의 불안정성을 줄일 수 있었습니다.
