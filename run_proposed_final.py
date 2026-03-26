#!/usr/bin/env python3
"""Run all 4 proposed model experiments sequentially."""
import subprocess
import sys
import os

experiments = [
    ('vgg16_proposed', 'cifar10'),
    ('vgg16_proposed', 'cifar100'),
    ('vgg19_proposed', 'cifar10'),
    ('vgg19_proposed', 'cifar100'),
]

for model, dataset in experiments:
    exp_dir = f"results/{model}_{dataset}"
    results_file = os.path.join(exp_dir, "results.json")
    
    # Skip if already done
    if os.path.exists(results_file):
        print(f"SKIP: {model} {dataset} already done")
        continue
    
    print(f"\n{'='*80}")
    print(f"Running: {model} on {dataset}")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--dataset", dataset,
        "--epochs", "250",
        "--batch_size", "128",
        "--lr", "1e-5",
        "--patience", "7",
        "--output_dir", "results",
        "--num_workers", "4",
    ]
    
    result = subprocess.run(cmd, cwd="/workspace")
    if result.returncode != 0:
        print(f"FAILED: {model} {dataset}")
    else:
        print(f"DONE: {model} {dataset}")
    sys.stdout.flush()

print("\nAll proposed experiments complete!")
