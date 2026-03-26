#!/usr/bin/env python3
"""Run all 4 proposed model experiments sequentially."""
import subprocess
import sys
import os

experiments = [
    ("vgg16_proposed", "cifar10"),
    ("vgg16_proposed", "cifar100"),
    ("vgg19_proposed", "cifar10"),
    ("vgg19_proposed", "cifar100"),
]

for model, dataset in experiments:
    result_dir = f"results/{model}_{dataset}"
    result_file = os.path.join(result_dir, "results.json")
    
    if os.path.exists(result_file):
        print(f"SKIP (already done): {model} {dataset}")
        continue
    
    print(f"\n{'='*80}")
    print(f"Running: {model} on {dataset}")
    print(f"{'='*80}")
    
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
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"DONE: {model} {dataset}")
    else:
        print(f"FAILED: {model} {dataset} (exit code {result.returncode})")

print("\nAll proposed experiments complete!")
