"""
Run all 8 experiments sequentially.
Paper: "Inspect Transfer Learning Architecture with Dilated Convolution"

8 experiments = 4 models × 2 datasets
"""
import os
import sys
import time
import json
from train import train

# All experiments to run
experiments = [
    # (model_name, dataset_name)
    ('vgg16_basic', 'cifar10'),
    ('vgg16_basic', 'cifar100'),
    ('vgg16_proposed', 'cifar10'),
    ('vgg16_proposed', 'cifar100'),
    ('vgg19_basic', 'cifar10'),
    ('vgg19_basic', 'cifar100'),
    ('vgg19_proposed', 'cifar10'),
    ('vgg19_proposed', 'cifar100'),
]

def main():
    total_start = time.time()
    all_results = {}
    
    for i, (model_name, dataset_name) in enumerate(experiments):
        exp_name = f"{model_name}_{dataset_name}"
        results_file = f"results/{exp_name}/results.json"
        
        # Skip if already completed
        if os.path.exists(results_file):
            print(f"\n{'='*80}")
            print(f"[{i+1}/8] SKIPPING {exp_name} (already completed)")
            with open(results_file) as f:
                all_results[exp_name] = json.load(f)
            print(f"  Test accuracy: {all_results[exp_name]['test_accuracy']:.2f}%")
            continue
        
        print(f"\n{'='*80}")
        print(f"[{i+1}/8] Running {exp_name}")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        start = time.time()
        results = train(
            model_name=model_name,
            dataset_name=dataset_name,
            epochs=250,
            batch_size=64,
            lr=1e-5,
            patience=7,
            output_dir='results',
            num_workers=4,
        )
        elapsed = time.time() - start
        
        all_results[exp_name] = results
        print(f"\n{exp_name} completed in {elapsed/60:.1f} minutes")
        print(f"Test accuracy: {results['test_accuracy']:.2f}%")
        sys.stdout.flush()
    
    # Print summary table
    total_elapsed = time.time() - total_start
    print(f"\n\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_elapsed/60:.1f} minutes")
    print(f"{'='*80}")
    
    # Paper's target results
    paper_results = {
        'vgg16_basic_cifar10': 88.47,
        'vgg16_basic_cifar100': 84.62,
        'vgg16_proposed_cifar10': 93.75,
        'vgg16_proposed_cifar100': 88.28,
        'vgg19_basic_cifar10': 87.40,
        'vgg19_basic_cifar100': 81.09,
        'vgg19_proposed_cifar10': 90.85,
        'vgg19_proposed_cifar100': 86.38,
    }
    
    print(f"\n{'Model':<25} {'Dataset':<10} {'Our Acc(%)':<12} {'Paper Acc(%)':<12} {'Diff':<8}")
    print("-" * 70)
    
    for model_name, dataset_name in experiments:
        exp_name = f"{model_name}_{dataset_name}"
        our_acc = all_results.get(exp_name, {}).get('test_accuracy', 0)
        paper_acc = paper_results.get(exp_name, 0)
        diff = our_acc - paper_acc
        print(f"{model_name:<25} {dataset_name:<10} {our_acc:<12.2f} {paper_acc:<12.2f} {diff:+.2f}")
    
    # Save summary
    summary = {
        'our_results': {k: v.get('test_accuracy', 0) for k, v in all_results.items()},
        'paper_results': paper_results,
        'total_time_minutes': total_elapsed / 60,
    }
    
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to results/summary.json")


if __name__ == '__main__':
    main()
