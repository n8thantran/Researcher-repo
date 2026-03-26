#!/bin/bash
# reproduce.sh - Reproduce all experiments from:
# "Inspect Transfer Learning Architecture with Dilated Convolution"
#
# This script trains all 8 model/dataset combinations and generates
# the results table and plots.
#
# Usage: bash reproduce.sh
# Estimated time: ~8-12 hours on a single GPU (250 epochs each, early stopping)
#
# To run a single experiment:
#   python train.py --model vgg16_basic --dataset cifar10 --epochs 250
#
# To just regenerate plots from existing results:
#   python plot_results.py

set -e

echo "=============================================="
echo "Reproducing: Inspect Transfer Learning Architecture"
echo "              with Dilated Convolution"
echo "=============================================="
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision albumentations opencv-python-headless matplotlib numpy --quiet
echo "Dependencies installed."
echo ""

# Create results directory
mkdir -p results

# Define all experiments
MODELS=("vgg16_basic" "vgg16_proposed" "vgg19_basic" "vgg19_proposed")
DATASETS=("cifar10" "cifar100")

# Run all 8 experiments
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        exp_name="${model}_${dataset}"
        echo "=============================================="
        echo "Running experiment: ${exp_name}"
        echo "=============================================="
        
        # Skip if results already exist
        if [ -f "results/${exp_name}/results.json" ]; then
            echo "Results already exist for ${exp_name}, skipping..."
            echo "(Delete results/${exp_name}/ to re-run)"
            echo ""
            continue
        fi
        
        python train.py \
            --model "$model" \
            --dataset "$dataset" \
            --epochs 250 \
            --batch_size 64 \
            --lr 1e-5 \
            --patience 7 \
            --output_dir results \
            --num_workers 4
        
        echo ""
    done
done

echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
echo ""

# Generate results table and plots
echo "Generating results table and plots..."
python plot_results.py

echo ""
echo "=============================================="
echo "Results Summary"
echo "=============================================="
cat results/table1.md

echo ""
echo "Output files:"
echo "  results/table1.md          - Results comparison table"
echo "  results/training_curves.png - Training curves for all experiments"
echo "  results/comparison_bar.png  - Bar chart comparing our results vs paper"
echo "  results/*/results.json     - Per-experiment metrics"
echo "  results/*/history.json     - Per-experiment training history"
echo ""
echo "Done!"
