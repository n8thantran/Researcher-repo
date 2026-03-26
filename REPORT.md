# Replication Report: "Inspect Transfer Learning Architecture with Dilated Convolution"

## Summary

This report documents the replication of the paper "Inspect Transfer Learning Architecture with Dilated Convolution" which proposes modified VGG-16 and VGG-19 architectures using dilated convolutions and transfer learning for CIFAR-10 and CIFAR-100 image classification.

## What Was Implemented

### Architecture (models.py)
All four model variants from the paper were implemented:

1. **VGG-16 Basic**: Standard VGG-16 with ImageNet pretrained weights, all layers trainable, custom classifier (FC 512 → FC 256 → num_classes).
2. **VGG-16 Proposed**: Frozen blocks 1-2 (pretrained), dilated block 3 (rate=2, 3 conv), dilated block 4 (rate=4, 3 conv), block 5 with two parallel branches (rate=4 and rate=8, 3 conv each), concatenated and pooled, then FC classifier.
3. **VGG-19 Basic**: Standard VGG-19 with same setup as VGG-16 Basic.
4. **VGG-19 Proposed**: Frozen blocks 1-2, dilated block 3 (rate=2, 4 conv), dilated block 4 (rate=2, 4 conv), block 5 with two parallel branches (rate=2 and rate=4, 4 conv each), same classifier.

### Data Pipeline (data.py)
- CIFAR-10 and CIFAR-100 datasets loaded via torchvision
- Split: 40,000 train / 10,000 validation / 10,000 test (as specified in paper)
- Augmentation via Albumentations matching Keras ImageDataGenerator:
  - Horizontal and vertical flips
  - Rotation range: 30°
  - Width/height shift: 0.3
  - Zoom range: 0.3
  - Fill mode: nearest (border replicate)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Training (train.py)
- Optimizer: Adam with lr=1e-5
- LR Scheduler: ReduceLROnPlateau(monitor='val_loss', patience=7, factor=√0.05 ≈ 0.2236)
- Loss: Cross-Entropy
- Epochs: 250 (with early stopping when LR < 1e-8 and no improvement for 15 epochs)
- Batch size: 64
- Best model saved based on validation accuracy

## Commands Run

```bash
# Train all 8 experiments (4 models × 2 datasets)
python train.py --model vgg16_basic --dataset cifar10 --epochs 250
python train.py --model vgg16_basic --dataset cifar100 --epochs 250
python train.py --model vgg16_proposed --dataset cifar10 --epochs 250
python train.py --model vgg16_proposed --dataset cifar100 --epochs 250
python train.py --model vgg19_basic --dataset cifar10 --epochs 250
python train.py --model vgg19_basic --dataset cifar100 --epochs 250
python train.py --model vgg19_proposed --dataset cifar10 --epochs 250
python train.py --model vgg19_proposed --dataset cifar100 --epochs 250

# Generate plots and tables
python plot_results.py

# Or run everything at once:
bash reproduce.sh
```

## Results

### Table I Replication: Test Accuracy (%)

| Model | CIFAR-10 (Ours) | CIFAR-10 (Paper) | CIFAR-100 (Ours) | CIFAR-100 (Paper) |
|-------|-----------------|------------------|------------------|-------------------|
| VGG-16 Basic | 86.75% | 88.47% | 60.52% | 84.62% |
| VGG-16 Proposed | 81.01% | 93.75% | 52.37% | 88.28% |
| VGG-19 Basic | 87.83% | 87.40% | 61.12% | 81.09% |
| VGG-19 Proposed | 80.67% | 90.85% | 53.94% | 86.38% |

### Analysis

**CIFAR-10 Basic Models**: Our VGG-19 Basic (87.83%) closely matches the paper's 87.40%, and VGG-16 Basic (86.75%) is within ~2% of the paper's 88.47%. This validates our basic training pipeline.

**CIFAR-10 Proposed Models**: Our proposed models (81.01% and 80.67%) fall short of the paper's claims (93.75% and 90.85%). The proposed models actually perform worse than the basic models in our implementation.

**CIFAR-100**: All our CIFAR-100 results are significantly below the paper's claims. Even the basic models achieve only ~60% vs the paper's 81-84%.

### Possible Reasons for Discrepancy

1. **Framework differences (Keras vs PyTorch)**: The paper uses Keras with TensorFlow backend. Keras's `same` padding for dilated convolutions may handle boundary effects differently than PyTorch, especially on small 8×8 feature maps with large dilation rates (4, 8).

2. **Dilated convolutions on small feature maps**: With 32×32 CIFAR input, after 2 maxpool layers the feature maps are 8×8. Dilation rate 8 on an 8×8 feature map means the effective receptive field extends far beyond the feature map boundaries, making most of the kernel weights operate on padding values. Keras's same-padding may handle this more gracefully.

3. **CIFAR-100 gap**: Standard VGG architectures on 32×32 images typically achieve 70-75% on CIFAR-100 with careful tuning. The paper's claim of 84.62% for basic VGG-16 on CIFAR-100 is unusually high and may involve additional techniques not described in the paper.

4. **Transfer learning effectiveness**: VGG models pretrained on 224×224 ImageNet images may not transfer as effectively to 32×32 CIFAR images. The paper's Keras implementation may handle the resolution mismatch differently.

5. **No maxpool in dilated blocks**: The paper text mentions maxpool after each block, but applying maxpool after dilated blocks would reduce 8×8 → 4×4 → 2×2 → 1×1, making large dilation rates ineffective. We chose to omit maxpool in dilated blocks (the more architecturally sensible choice), but this may differ from the paper's actual implementation.

## Important File Paths

| File | Description |
|------|-------------|
| `models.py` | All 4 model architectures |
| `data.py` | Data loading and augmentation pipeline |
| `train.py` | Training loop with LR scheduling |
| `plot_results.py` | Results visualization |
| `reproduce.sh` | Script to reproduce all experiments |
| `results/table1.md` | Results comparison table |
| `results/training_curves.png` | Training curves for all experiments |
| `results/comparison_bar.png` | Bar chart comparing results vs paper |
| `results/*/results.json` | Per-experiment test metrics |
| `results/*/history.json` | Per-experiment training history |
| `results/*/best_model.pth` | Best model checkpoints |

## What Is Still Incomplete or Approximate

1. **Accuracy gap**: Our results do not match the paper's claimed accuracies, particularly for proposed models and CIFAR-100. The basic VGG-19 on CIFAR-10 is the closest match (87.83% vs 87.40%).

2. **Maxpool ambiguity**: The paper's description of maxpool in dilated blocks is ambiguous. We tried both with and without maxpool; without maxpool gave better results.

3. **Keras-specific behavior**: Some Keras-specific behaviors (e.g., same-padding implementation, weight initialization details, data augmentation pipeline specifics) may not be perfectly replicated in PyTorch.

4. **Batch normalization**: The paper doesn't mention batch normalization, so we didn't include it. Adding BN to dilated blocks might improve results.

5. **Training duration**: All experiments ran for the full 250 epochs (or until early stopping). Some experiments may benefit from longer training or different hyperparameters.
