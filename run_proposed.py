"""Run the 4 proposed model experiments sequentially."""
import subprocess
import sys
import time

experiments = [
    ('vgg16_proposed', 'cifar10', 128),
    ('vgg16_proposed', 'cifar100', 128),
    ('vgg19_proposed', 'cifar10', 128),
    ('vgg19_proposed', 'cifar100', 128),
]

for model, dataset, bs in experiments:
    print(f"\n{'='*80}")
    print(f"Starting: {model} on {dataset} (batch_size={bs})")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    start = time.time()
    result = subprocess.run([
        sys.executable, 'train.py',
        '--model', model,
        '--dataset', dataset,
        '--epochs', '250',
        '--batch_size', str(bs),
        '--lr', '1e-5',
        '--patience', '7',
        '--output_dir', 'results',
        '--num_workers', '4',
    ], capture_output=False)
    elapsed = time.time() - start
    print(f"\nFinished {model}_{dataset} in {elapsed/60:.1f} minutes (exit code: {result.returncode})")
    sys.stdout.flush()

print("\n\nAll proposed experiments complete!")
