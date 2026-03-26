#!/usr/bin/env python3
"""
Run all 8 experiments sequentially.
"""
import subprocess
import sys
import time
import os
import json

EXPERIMENTS = [
    # CIFAR-10 experiments (already have basic, need proposed)
    ('vgg16_proposed', 'cifar10'),
    ('vgg19_proposed', 'cifar10'),
    # All CIFAR-100 experiments
    ('vgg16_basic', 'cifar100'),
    ('vgg16_proposed', 'cifar100'),
    ('vgg19_basic', 'cifar100'),
    ('vgg19_proposed', 'cifar100'),
]

def run_experiment(model, dataset):
    exp_name = f"{model}_{dataset}"
    results_file = f"results/{exp_name}/results.json"
    
    # Skip if already completed
    if os.path.exists(results_file):
        with open(results_file) as f:
            r = json.load(f)
        print(f"SKIP {exp_name}: already done (acc={r['test_accuracy']:.2f}%)")
        return r['test_accuracy']
    
    print(f"\n{'='*80}")
    print(f"RUNNING: {exp_name}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, 'train.py',
        '--model', model,
        '--dataset', dataset,
        '--epochs', '250',
        '--batch_size', '64',
        '--lr', '1e-5',
        '--patience', '7',
        '--output_dir', 'results',
        '--num_workers', '4',
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"FAILED: {exp_name} (return code {result.returncode})")
        return None
    
    # Read result
    if os.path.exists(results_file):
        with open(results_file) as f:
            r = json.load(f)
        print(f"DONE: {exp_name} = {r['test_accuracy']:.2f}% ({elapsed/60:.1f} min)")
        return r['test_accuracy']
    return None

def main():
    total_start = time.time()
    results = {}
    
    for model, dataset in EXPERIMENTS:
        acc = run_experiment(model, dataset)
        results[f"{model}_{dataset}"] = acc
    
    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    all_models = ['vgg16_basic', 'vgg16_proposed', 'vgg19_basic', 'vgg19_proposed']
    all_datasets = ['cifar10', 'cifar100']
    
    header = f"{'Model':<20}"
    for ds in all_datasets:
        header += f" {ds:>12}"
    print(header)
    print("-" * 50)
    
    for model in all_models:
        row = f"{model:<20}"
        for ds in all_datasets:
            key = f"{model}_{ds}"
            results_file = f"results/{key}/results.json"
            if os.path.exists(results_file):
                with open(results_file) as f:
                    r = json.load(f)
                row += f" {r['test_accuracy']:>11.2f}%"
            else:
                row += f" {'N/A':>12}"
        print(row)
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")

if __name__ == '__main__':
    main()
