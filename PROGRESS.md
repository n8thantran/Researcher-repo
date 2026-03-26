# Progress Tracking

## Current Phase: Running proposed model experiments (v2 - no maxpool in dilated blocks)

## Paper Summary
**Title**: "Inspect Transfer Learning Architecture with Dilated Convolution"
Modified VGG-16 and VGG-19 with dilated convolutions and transfer learning for CIFAR-10/100.

### Target Results (Table I):
| Model | CIFAR-10 | CIFAR-100 |
|-------|----------|-----------|
| VGG-16 basic | 88.47% | 84.62% |
| VGG-16 proposed | 93.75% | 88.28% |
| VGG-19 basic | 87.40% | 81.09% |
| VGG-19 proposed | 90.85% | 86.38% |

### Training Details:
- Epochs: 250, LR: 1e-5, Adam optimizer, batch_size=128
- LR scheduler: ReduceLROnPlateau(val_loss, patience=7, factor=sqrt(0.05)≈0.2236)
- Loss: Cross-Entropy
- Data split: 40k train, 10k val, 10k test
- Augmentation: horizontal/vertical flips, rotation 30, shift 0.3, zoom 0.3
- Input: 32×32 CIFAR images, ImageNet normalization
- Keras implementation with TensorFlow backend

## Architecture Details:
**VGG-16 Proposed:**
- Block 1: Freeze (pretrained), 2 conv+relu+maxpool → 16x16
- Block 2: Freeze (pretrained), 2 conv+relu+maxpool → 8x8
- Block 3: Dilation=2, 3 conv+relu (no maxpool) → 8x8
- Block 4: Dilation=4, 3 conv+relu (no maxpool) → 8x8
- Block 5: Concat(dilation=4, dilation=8), 3 conv+relu each (no maxpool) → 8x8
- AdaptiveAvgPool → 1x1, Concat → 1024
- FC 512 → FC 256 → FC num_classes

**VGG-19 Proposed:**
- Block 1&2: Same as VGG-16
- Block 3: Dilation=2, 4 conv+relu (no maxpool) → 8x8
- Block 4: Dilation=2, 4 conv+relu (no maxpool) → 8x8
- Block 5: Concat(dilation=2, dilation=4), 4 conv+relu each (no maxpool) → 8x8
- Rest same as VGG-16

## Implementation Plan
- [x] 1. Data Pipeline (data.py) - Albumentations augmentation, CIFAR loading
- [x] 2. Model Architecture (models.py) - VGG-16/19 basic and proposed
- [x] 3. Training script (train.py) - Adam, LR schedule, checkpointing, early stopping
- [x] 4. Run all 4 basic experiments
- [x] 5. Fix proposed models - remove maxpool from dilated blocks
- [ ] 6. Run 4 proposed model experiments (IN PROGRESS - run_proposed_v2.py)
- [ ] 7. Generate results tables and plots (plot_results.py)
- [ ] 8. Write reproduce.sh
- [ ] 9. Write REPORT.md
- [ ] 10. Final commit and push

## Current Results
| Model | CIFAR-10 | CIFAR-100 | Paper C10 | Paper C100 |
|-------|----------|-----------|-----------|------------|
| VGG-16 basic | 86.75% | 60.52% | 88.47% | 84.62% |
| VGG-16 proposed | 81.01% | running... | 93.75% | 88.28% |
| VGG-19 basic | 87.83% | 61.12% | 87.40% | 81.09% |
| VGG-19 proposed | pending | pending | 90.85% | 86.38% |

## Key Issues / Failed Approaches
1. **First proposed run (with maxpool in dilated blocks)**: VGG-16 proposed got 77.26% on CIFAR-10. Maxpool in dilated blocks shrinks spatial dims too much - dilation=8 on 2x2 maps is pointless.
2. **Second proposed run (no maxpool, no BN)**: VGG-16 proposed got 81.01% on CIFAR-10. Better but still below paper.
3. **CIFAR-100 basic models very low** (~60% vs 81-84% paper): Standard VGG on 32x32 CIFAR-100 typically gets 60-73%. Paper used Keras which may handle VGG feature extraction differently.
4. **Paper's numbers seem unrealistically high**: 93.75% CIFAR-10, 88.28% CIFAR-100 for VGG-16. Most VGG papers on 32x32 CIFAR don't reach these numbers without significant modifications (BN, dropout, etc.).
5. **Proposed < Basic on CIFAR-10**: Our proposed (81.01%) < basic (86.75%). Paper claims proposed should be higher. The frozen pretrained blocks help but dilated convs with random-ish classifier aren't converging as well.

## Possible Improvements (if time):
- Higher LR or different LR schedule
- Add dropout to classifier  
- Try with BN in dilated blocks
- Resize images to 64x64 or 224x224

## Key Files:
- models.py: All 4 model architectures
- data.py: Data loading with augmentation
- train.py: Training loop with LR scheduling
- run_proposed_v2.py: Script to run 4 proposed experiments
- results/: Contains all experiment results
- logs/: Training logs

## Background Process:
- PID file: logs/proposed_v2.pid
- Log: logs/proposed_v2.log
- Running: vgg16_proposed_cifar100 (started), then vgg19_proposed_cifar10, vgg19_proposed_cifar100
