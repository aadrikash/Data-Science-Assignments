#  Autoencoder for Image Denoising — MNIST
> **Week 6 Assessment | Deep Learning Project**

A **Convolutional Autoencoder** that removes Gaussian noise from MNIST handwritten-digit images, built with PyTorch.

---

## Project Structure

```
week5/
├── autoencoder_mnist_denoising.ipynb   ← Main notebook (run this!)
├── autoencoder_denoising.py            ← Standalone Python script
├── requirements.txt                    ← Dependencies
├── README.md                           ← This file
│
└── (generated after training)
    ├── data/                           ← MNIST dataset (auto-downloaded)
    ├── best_autoencoder.pth            ← Best model weights
    ├── loss_curves.png                 ← Train/val loss plot
    ├── clean_vs_noisy.png              ← Before noise injection
    ├── denoising_results.png           ← Original | Noisy | Denoised
    ├── noise_levels.png                ← Robustness across σ values
    └── latent_space_pca.png            ← PCA of bottleneck features
```

---

## Model Architecture

```
Input  (1, 28, 28)
         │
    ┌────▼────────────────────────┐
    │         ENCODER              │
    │  Conv(1→32) → BN → ReLU     │
    │  MaxPool(2)   [28×28→14×14] │
    │  Conv(32→64) → BN → ReLU    │
    │  MaxPool(2)   [14×14→7×7]   │
    └────┬────────────────────────┘
         │
    Bottleneck (64, 7, 7) = 3136 features
         │
    ┌────▼────────────────────────┐
    │         DECODER              │
    │  ConvT(64→32) → BN → ReLU   │
    │  Upsample     [7×7→14×14]   │
    │  ConvT(32→1) → Sigmoid      │
    │  Upsample     [14×14→28×28] │
    └────┬────────────────────────┘
         │
Output (1, 28, 28) — denoised image
```

| Property | Value |
|----------|-------|
| Trainable parameters | ~113K |
| Loss function | MSE |
| Optimizer | Adam (lr=1e-3) |
| LR Scheduler | ReduceLROnPlateau |
| Noise model | Gaussian (σ=0.4) |

---

##  Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Open the notebook
```bash
jupyter notebook autoencoder_mnist_denoising.ipynb
```

### 3. Or run as a script
```bash
python autoencoder_denoising.py
```

> **MNIST** (~11 MB) downloads automatically on first run.

---

##  Results

| Metric | Value |
|--------|-------|
| PSNR (Noisy vs Clean) | ~20 dB |
| PSNR (Denoised vs Clean) | ~27–30 dB |
| **Improvement** | **+7–10 dB** |

---

##  Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Autoencoder** | Neural net that compresses input to latent space then reconstructs it |
| **Encoder** | Compresses 28×28 image → compact 7×7 feature map |
| **Decoder** | Reconstructs 28×28 image from compressed representation |
| **Bottleneck** | Forces network to learn only essential, noise-free features |
| **PSNR** | Quality metric; higher = better reconstruction |
| **MSE Loss** | Pixel-wise squared error between denoised and clean images |

---
