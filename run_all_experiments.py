#!/usr/bin/env python3
"""
Run all 8 experiments sequentially.
Saves results after each experiment completes.
Skips experiments that already have results.json.
"""
import sys
import os
import json
import time

sys.path.insert(0, '/workspace')
os.chdir('/workspace')

from train import train

EXPERIMENTS = [
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
    all_results = {}
    
    for model_name, dataset_name in EXPERIMENTS:
        exp_name = f"{model_name}_{dataset_name}"
        results_file = f'results/{exp_name}/results.json'
        
        # Check if already complete
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)
            # Only skip if accuracy is reasonable
            min_acc = 80 if dataset_name == 'cifar10' else 60
            if results.get('test_accuracy', 0) > min_acc and results.get('best_epoch', 0) > 0:
                print(f"\n{'='*80}")
                print(f"SKIP {exp_name}: already complete (test_acc={results['test_accuracy']:.2f}%)")
                print(f"{'='*80}")
                all_results[exp_name] = results
                continue
            else:
                print(f"\n{'='*80}")
                print(f"RE-RUNNING {exp_name}: previous result insufficient (test_acc={results.get('test_accuracy', 0):.2f}%)")
                print(f"{'='*80}")
                # Remove old results to force retraining
                os.remove(results_file)
                best_model = f'results/{exp_name}/best_model.pth'
                if os.path.exists(best_model):
                    os.remove(best_model)
        
        print(f"\n{'='*80}")
        print(f"STARTING: {exp_name}")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        start_time = time.time()
        
        try:
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
            elapsed = time.time() - start_time
            print(f"\nCOMPLETED {exp_name} in {elapsed/60:.1f} min: test_acc={results['test_accuracy']:.2f}%")
            all_results[exp_name] = results
        except Exception as e:
            print(f"\nFAILED {exp_name}: {e}")
            import traceback
            traceback.print_exc()
        
        sys.stdout.flush()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Dataset':<12} {'Test Acc':>10} {'Best Val Acc':>12} {'Best Epoch':>12}")
    print("-" * 75)
    for exp_name, results in all_results.items():
        print(f"{results.get('model_name','?'):<25} {results.get('dataset_name','?'):<12} "
              f"{results.get('test_accuracy',0):>9.2f}% {results.get('best_val_accuracy',0):>11.2f}% "
              f"{results.get('best_epoch','?'):>12}")
    
    # Save summary
    with open('results/summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll experiments complete! Summary saved to results/summary.json")
    
    # Write completion marker
    with open('results/DONE', 'w') as f:
        f.write('done\n')


if __name__ == '__main__':
    main()
