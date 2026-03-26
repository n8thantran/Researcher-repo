"""
Run all 8 experiments sequentially, skipping already completed ones.
"""
import subprocess
import sys
import os
import json
import time

EXPERIMENTS = [
    # Run CIFAR-10 first (faster), then CIFAR-100
    ('vgg16_basic', 'cifar10'),
    ('vgg19_basic', 'cifar10'),
    ('vgg16_proposed', 'cifar10'),
    ('vgg19_proposed', 'cifar10'),
    ('vgg16_basic', 'cifar100'),
    ('vgg19_basic', 'cifar100'),
    ('vgg16_proposed', 'cifar100'),
    ('vgg19_proposed', 'cifar100'),
]

def run_experiment(model, dataset):
    exp_name = f"{model}_{dataset}"
    result_file = f"results/{exp_name}/results.json"
    
    # Skip if already completed
    if os.path.exists(result_file):
        with open(result_file) as f:
            r = json.load(f)
        print(f"SKIP {exp_name}: already done (test_acc={r['test_accuracy']:.2f}%)")
        return r['test_accuracy']
    
    print(f"\n{'='*80}")
    print(f"STARTING: {exp_name} at {time.strftime('%H:%M:%S')}")
    print(f"{'='*80}", flush=True)
    
    cmd = [
        sys.executable, 'train.py',
        '--model', model,
        '--dataset', dataset,
        '--epochs', '250',
        '--batch_size', '128',
        '--lr', '1e-5',
        '--patience', '7',
        '--output_dir', 'results',
        '--num_workers', '4',
    ]
    
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"FAILED: {exp_name} (return code {result.returncode})")
        return None
    
    # Read result
    if os.path.exists(result_file):
        with open(result_file) as f:
            r = json.load(f)
        print(f"DONE: {exp_name} -> test_acc={r['test_accuracy']:.2f}% in {elapsed:.0f}s")
        return r['test_accuracy']
    else:
        print(f"WARNING: No results file for {exp_name}")
        return None

def main():
    print(f"Running all experiments sequentially")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    total_start = time.time()
    for model, dataset in EXPERIMENTS:
        acc = run_experiment(model, dataset)
        results[f"{model}_{dataset}"] = acc
    
    total_time = time.time() - total_start
    
    print("\n" + "="*80)
    print("SUMMARY OF ALL RESULTS")
    print("="*80)
    
    # Paper targets for comparison
    targets = {
        'vgg16_basic_cifar10': 88.47, 'vgg16_basic_cifar100': 84.62,
        'vgg16_proposed_cifar10': 93.75, 'vgg16_proposed_cifar100': 88.28,
        'vgg19_basic_cifar10': 87.40, 'vgg19_basic_cifar100': 81.09,
        'vgg19_proposed_cifar10': 90.85, 'vgg19_proposed_cifar100': 86.38,
    }
    
    print(f"{'Experiment':<30} {'Our Acc':>10} {'Paper':>10} {'Diff':>10}")
    print("-"*62)
    for name, acc in results.items():
        target = targets.get(name, 0)
        if acc is not None:
            print(f"{name:<30} {acc:>9.2f}% {target:>9.2f}% {acc-target:>+9.2f}%")
        else:
            print(f"{name:<30} {'FAILED':>10} {target:>9.2f}%")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")

if __name__ == '__main__':
    main()
