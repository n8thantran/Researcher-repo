"""
Generate plots and tables from experiment results.
Matches the paper's figures/tables:
- Table I: Test accuracy comparison
- Training curves (loss and accuracy vs epochs)
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir='results'):
    """Load all experiment results."""
    results = {}
    histories = {}
    
    for exp_name in os.listdir(results_dir):
        exp_dir = os.path.join(results_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        
        results_path = os.path.join(exp_dir, 'results.json')
        history_path = os.path.join(exp_dir, 'history.json')
        
        if os.path.exists(results_path):
            with open(results_path) as f:
                results[exp_name] = json.load(f)
        
        if os.path.exists(history_path):
            with open(history_path) as f:
                histories[exp_name] = json.load(f)
    
    return results, histories


def generate_table(results, output_dir='results'):
    """Generate Table I: Test accuracy comparison."""
    
    # Paper's target results
    paper_results = {
        'vgg16_basic': {'cifar10': 88.47, 'cifar100': 84.62},
        'vgg16_proposed': {'cifar10': 93.75, 'cifar100': 88.28},
        'vgg19_basic': {'cifar10': 87.40, 'cifar100': 81.09},
        'vgg19_proposed': {'cifar10': 90.85, 'cifar100': 86.38},
    }
    
    lines = []
    lines.append("=" * 80)
    lines.append("TABLE I: Test Accuracy Comparison")
    lines.append("=" * 80)
    lines.append(f"{'Model':<20} {'CIFAR-10 (ours)':>15} {'CIFAR-10 (paper)':>16} {'CIFAR-100 (ours)':>16} {'CIFAR-100 (paper)':>17}")
    lines.append("-" * 86)
    
    for model_base in ['vgg16_basic', 'vgg16_proposed', 'vgg19_basic', 'vgg19_proposed']:
        c10_key = f"{model_base}_cifar10"
        c100_key = f"{model_base}_cifar100"
        
        c10_ours = results.get(c10_key, {}).get('test_accuracy', None)
        c100_ours = results.get(c100_key, {}).get('test_accuracy', None)
        c10_paper = paper_results.get(model_base, {}).get('cifar10', None)
        c100_paper = paper_results.get(model_base, {}).get('cifar100', None)
        
        c10_ours_str = f"{c10_ours:.2f}%" if c10_ours is not None else "N/A"
        c100_ours_str = f"{c100_ours:.2f}%" if c100_ours is not None else "N/A"
        c10_paper_str = f"{c10_paper:.2f}%" if c10_paper is not None else "N/A"
        c100_paper_str = f"{c100_paper:.2f}%" if c100_paper is not None else "N/A"
        
        lines.append(f"{model_base:<20} {c10_ours_str:>15} {c10_paper_str:>16} {c100_ours_str:>16} {c100_paper_str:>17}")
    
    lines.append("=" * 86)
    
    table_text = "\n".join(lines)
    print(table_text)
    
    with open(os.path.join(output_dir, 'table_results.txt'), 'w') as f:
        f.write(table_text + "\n")
    
    return table_text


def plot_training_curves(histories, output_dir='results'):
    """Plot training curves for all experiments."""
    
    datasets = ['cifar10', 'cifar100']
    model_names = ['vgg16_basic', 'vgg16_proposed', 'vgg19_basic', 'vgg19_proposed']
    
    for dataset in datasets:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Curves - {dataset.upper()}', fontsize=16)
        
        for idx, model_name in enumerate(model_names):
            exp_name = f"{model_name}_{dataset}"
            if exp_name not in histories:
                continue
            
            h = histories[exp_name]
            epochs = list(range(1, len(h['train_loss']) + 1))
            
            # Loss plot
            ax_loss = axes[0][idx % 2] if idx < 2 else axes[1][idx % 2]
            # Actually, let's use a different layout
        
        plt.close(fig)
        
        # Better: individual plots per experiment
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle(f'Training Curves - {dataset.upper()}', fontsize=16)
        
        for idx, model_name in enumerate(model_names):
            exp_name = f"{model_name}_{dataset}"
            if exp_name not in histories:
                continue
            
            h = histories[exp_name]
            epochs = list(range(1, len(h['train_loss']) + 1))
            
            # Loss
            axes[0][idx].plot(epochs, h['train_loss'], label='Train', alpha=0.8)
            axes[0][idx].plot(epochs, h['val_loss'], label='Val', alpha=0.8)
            axes[0][idx].set_title(model_name.replace('_', ' ').title())
            axes[0][idx].set_xlabel('Epoch')
            axes[0][idx].set_ylabel('Loss')
            axes[0][idx].legend()
            axes[0][idx].grid(True, alpha=0.3)
            
            # Accuracy
            axes[1][idx].plot(epochs, h['train_acc'], label='Train', alpha=0.8)
            axes[1][idx].plot(epochs, h['val_acc'], label='Val', alpha=0.8)
            axes[1][idx].set_xlabel('Epoch')
            axes[1][idx].set_ylabel('Accuracy (%)')
            axes[1][idx].legend()
            axes[1][idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_curves_{dataset}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training_curves_{dataset}.png")
    
    # Also plot basic vs proposed comparison 
    for dataset in datasets:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Basic vs Proposed - {dataset.upper()}', fontsize=14)
        
        colors = {'vgg16_basic': 'blue', 'vgg16_proposed': 'red', 
                  'vgg19_basic': 'green', 'vgg19_proposed': 'orange'}
        labels = {'vgg16_basic': 'VGG-16 Basic', 'vgg16_proposed': 'VGG-16 Proposed',
                  'vgg19_basic': 'VGG-19 Basic', 'vgg19_proposed': 'VGG-19 Proposed'}
        
        for model_name in model_names:
            exp_name = f"{model_name}_{dataset}"
            if exp_name not in histories:
                continue
            
            h = histories[exp_name]
            epochs = list(range(1, len(h['val_loss']) + 1))
            
            axes[0].plot(epochs, h['val_loss'], label=labels[model_name], 
                        color=colors[model_name], alpha=0.8)
            axes[1].plot(epochs, h['val_acc'], label=labels[model_name],
                        color=colors[model_name], alpha=0.8)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Accuracy (%)')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{dataset}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison_{dataset}.png")


def main():
    results_dir = 'results'
    results, histories = load_results(results_dir)
    
    if not results:
        print("No results found! Run experiments first.")
        return
    
    print(f"Found {len(results)} experiments\n")
    
    generate_table(results, results_dir)
    plot_training_curves(histories, results_dir)
    
    print("\nAll plots and tables generated!")


if __name__ == '__main__':
    main()
