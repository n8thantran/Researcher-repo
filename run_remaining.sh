#!/bin/bash
cd /workspace

# Remaining experiments
experiments=(
    "vgg16_proposed cifar100"
    "vgg19_basic cifar100" 
    "vgg19_proposed cifar10"
    "vgg19_proposed cifar100"
)

for exp in "${experiments[@]}"; do
    model=$(echo $exp | cut -d' ' -f1)
    dataset=$(echo $exp | cut -d' ' -f2)
    echo "=== Starting ${model}_${dataset} at $(date) ==="
    
    # Skip if already done
    if [ -f "results/${model}_${dataset}/results.json" ]; then
        echo "Already done, skipping"
        continue
    fi
    
    python3 train.py --model $model --dataset $dataset --epochs 250 --batch_size 128 --lr 1e-5 --patience 7 --output_dir results --num_workers 4
    echo "=== Finished ${model}_${dataset} at $(date) ==="
done

echo "=== ALL DONE ==="
