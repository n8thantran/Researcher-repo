#!/usr/bin/env python3
"""
Run all 8 experiments sequentially.
Saves results to /workspace/results/<model>_<dataset>/
"""
import subprocess
import sys
import time
import os
import json

experiments = [
    # Run in order: basic models first (faster), then proposed
    ('vgg16_basic', 'cifar10'),
    ('vgg16_basic', 'cifar100'),
    ('vgg19_basic', 'cifar10'),
    ('vgg19_basic', 'cifar100'),
    ('vgg16_proposed', 'cifar10'),
    ('vgg16_proposed', 'cifar100'),
    ('vgg19_proposed', 'cifar10'),
    ('vgg19_proposed', 'cifar100'),
]

def run_experiment(model, dataset):
    exp_name = f"{model}_{dataset}"
    print(f"\n{'='*80}")
    print(f"Starting: {exp_name}")
    print(f"{'='*80}")
    
    start = time.time()
    
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
    
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    
    print(f"\n{exp_name} completed in {elapsed/60:.1f} minutes (exit code: {result.returncode})")
    
    # Read and print results
    results_file = f"results/{exp_name}/results.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            r = json.load(f)
        print(f"  Test accuracy: {r['test_accuracy']:.2f}%")
        print(f"  Best val accuracy: {r['best_val_accuracy']:.2f}% (epoch {r['best_epoch']})")
        print(f"  Epochs run: {r['epochs_run']}")
    
    return result.returncode

def main():
    total_start = time.time()
    
    # Check which experiments already have results
    remaining = []
    for model, dataset in experiments:
        exp_name = f"{model}_{dataset}"
        results_file = f"results/{exp_name}/results.json"
        if os.path.exists(results_file):
            with open(results_file) as f:
                r = json.load(f)
            print(f"SKIP {exp_name}: already done (test_acc={r['test_accuracy']:.2f}%)")
        else:
            remaining.append((model, dataset))
    
    if not remaining:
        print("\nAll experiments already completed!")
    else:
        print(f"\nRunning {len(remaining)} remaining experiments...")
        for model, dataset in remaining:
            run_experiment(model, dataset)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'CIFAR-10':>10} {'CIFAR-100':>10}")
    print("-"*42)
    
    for model_type in ['vgg16_basic', 'vgg16_proposed', 'vgg19_basic', 'vgg19_proposed']:
        row = f"{model_type:<20}"
        for ds in ['cifar10', 'cifar100']:
            exp_name = f"{model_type}_{ds}"
            results_file = f"results/{exp_name}/results.json"
            if os.path.exists(results_file):
                with open(results_file) as f:
                    r = json.load(f)
                row += f" {r['test_accuracy']:>9.2f}%"
            else:
                row += f" {'N/A':>9}"
        print(row)
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")

if __name__ == '__main__':
    main()
