# Vision Transformer for CIFAR-10 Image Classification

A clean, reproducible TensorFlow/Keras implementation of a **Vision Transformer (ViT)** for **CIFAR-10**.

## Pipeline

```text
CIFAR-10 image
→ Normalization
→ Resize to 72 × 72
→ Data augmentation
→ 6 × 6 patch extraction
→ Linear projection + positional embeddings
→ 8 Transformer encoder blocks
→ Flatten + MLP head
→ 10-class logits
```

## Main Configuration

| Parameter | Value |
|---|---:|
| Input shape | 32 × 32 × 3 |
| Resized image | 72 × 72 |
| Patch size | 6 × 6 |
| Number of patches | 144 |
| Projection dimension | 64 |
| Attention heads | 4 |
| Transformer blocks | 8 |
| Transformer MLP | 128 → 64 |
| Classifier MLP | 2048 → 1024 |
| Classes | 10 |
| Batch size | 256 |
| Epochs | 40 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --smoke-test
```

For full training:

```bash
python main.py
```

## Outputs

The training workflow writes:

```text
outputs/best.weights.h5
outputs/metrics.json
outputs/accuracy_curve.png
outputs/loss_curve.png
```

The training pipeline reports Top-1 accuracy, Top-5 accuracy, and test loss.
## Repository Structure

```text
.
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── src/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── plots.py
├── tests/
├── notebooks/
├── outputs/
└── original/
    └── vision_transformer_original.py
```

## Fixes Applied

The original Colab-exported script contained issues including notebook-only `!pip` commands, an early MLP return, an invalid Dense expression, a misplaced `PatchEncoder.call()`, duplicate model definitions, inconsistent patch counts, a missing-argument training call, and inconsistent softmax/from-logits handling.

The refactored version fixes these while preserving the intended ViT architecture.

## Reproducibility

CIFAR-10 downloads automatically through Keras on first run. Full training is compute-intensive; a GPU-enabled TensorFlow environment is recommended.Full training was not rerun during repository cleanup due to CIFAR-10 download instability; the code is structured for reproducible training when the dataset is available.

## Author

**Meghana Baladari**  
M.S. Data Science, Analytics and Engineering  
Arizona State University
