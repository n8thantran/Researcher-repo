#!/usr/bin/env python3
"""
Generate results tables and training plots for the paper replication.
Reads results from results/ directory and generates:
1. Summary table (Table I replication)
2. Training curves (accuracy and loss)
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
OUTPUT_DIR = "results"

# Paper's target values
PAPER_RESULTS = {
    ("vgg16_basic", "cifar10"): 88.47,
    ("vgg16_basic", "cifar100"): 84.62,
    ("vgg16_proposed", "cifar10"): 93.75,
    ("vgg16_proposed", "cifar100"): 88.28,
    ("vgg19_basic", "cifar10"): 87.40,
    ("vgg19_basic", "cifar100"): 81.09,
    ("vgg19_proposed", "cifar10"): 90.85,
    ("vgg19_proposed", "cifar100"): 86.38,
}

EXPERIMENTS = [
    ("vgg16_basic", "cifar10"),
    ("vgg16_basic", "cifar100"),
    ("vgg16_proposed", "cifar10"),
    ("vgg16_proposed", "cifar100"),
    ("vgg19_basic", "cifar10"),
    ("vgg19_basic", "cifar100"),
    ("vgg19_proposed", "cifar10"),
    ("vgg19_proposed", "cifar100"),
]


def load_results():
    """Load all experiment results."""
    results = {}
    for model, dataset in EXPERIMENTS:
        key = f"{model}_{dataset}"
        result_file = os.path.join(RESULTS_DIR, key, "results.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                results[(model, dataset)] = json.load(f)
    return results


def load_history(model, dataset):
    """Load training history."""
    key = f"{model}_{dataset}"
    hist_file = os.path.join(RESULTS_DIR, key, "history.json")
    if os.path.exists(hist_file):
        with open(hist_file) as f:
            return json.load(f)
    return None


def generate_table(results):
    """Generate Table I comparison."""
    print("\n" + "=" * 90)
    print("TABLE I: Effect of Different Combination of Dilation Rate for VGG-16 and VGG-19")
    print("=" * 90)
    print(f"{'Model':<25} {'CIFAR-10 (Ours)':<18} {'CIFAR-10 (Paper)':<18} {'CIFAR-100 (Ours)':<18} {'CIFAR-100 (Paper)':<18}")
    print("-" * 90)
    
    rows = [
        ("VGG-16 Basic", "vgg16_basic"),
        ("VGG-16 Proposed", "vgg16_proposed"),
        ("VGG-19 Basic", "vgg19_basic"),
        ("VGG-19 Proposed", "vgg19_proposed"),
    ]
    
    table_data = []
    
    for label, model in rows:
        c10 = results.get((model, "cifar10"), {})
        c100 = results.get((model, "cifar100"), {})
        
        c10_acc = c10.get("test_accuracy", None)
        c100_acc = c100.get("test_accuracy", None)
        
        c10_paper = PAPER_RESULTS.get((model, "cifar10"), None)
        c100_paper = PAPER_RESULTS.get((model, "cifar100"), None)
        
        c10_str = f"{c10_acc:.2f}%" if c10_acc else "N/A"
        c100_str = f"{c100_acc:.2f}%" if c100_acc else "N/A"
        c10p_str = f"{c10_paper:.2f}%" if c10_paper else "N/A"
        c100p_str = f"{c100_paper:.2f}%" if c100_paper else "N/A"
        
        print(f"{label:<25} {c10_str:<18} {c10p_str:<18} {c100_str:<18} {c100p_str:<18}")
        
        table_data.append((label, c10_acc, c10_paper, c100_acc, c100_paper))
    
    print("=" * 90)
    
    # Save as markdown
    with open(os.path.join(OUTPUT_DIR, "table1.md"), "w") as f:
        f.write("# Table I: Effect of Different Combination of Dilation Rate\n\n")
        f.write("| Model | CIFAR-10 (Ours) | CIFAR-10 (Paper) | CIFAR-100 (Ours) | CIFAR-100 (Paper) |\n")
        f.write("|-------|-----------------|------------------|------------------|-------------------|\n")
        for label, c10, c10p, c100, c100p in table_data:
            c10_s = f"{c10:.2f}%" if c10 else "N/A"
            c100_s = f"{c100:.2f}%" if c100 else "N/A"
            c10p_s = f"{c10p:.2f}%" if c10p else "N/A"
            c100p_s = f"{c100p:.2f}%" if c100p else "N/A"
            f.write(f"| {label} | {c10_s} | {c10p_s} | {c100_s} | {c100p_s} |\n")
    
    print(f"\nTable saved to {os.path.join(OUTPUT_DIR, 'table1.md')}")
    return table_data


def plot_training_curves(results):
    """Generate training curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = [
        ("CIFAR-10 Accuracy", "cifar10", "accuracy", axes[0, 0]),
        ("CIFAR-100 Accuracy", "cifar100", "accuracy", axes[0, 1]),
        ("CIFAR-10 Loss", "cifar10", "loss", axes[1, 0]),
        ("CIFAR-100 Loss", "cifar100", "loss", axes[1, 1]),
    ]
    
    colors = {
        "vgg16_basic": "blue",
        "vgg16_proposed": "red",
        "vgg19_basic": "green",
        "vgg19_proposed": "orange",
    }
    
    labels = {
        "vgg16_basic": "VGG-16 Basic",
        "vgg16_proposed": "VGG-16 Proposed",
        "vgg19_basic": "VGG-19 Basic",
        "vgg19_proposed": "VGG-19 Proposed",
    }
    
    for title, dataset, metric, ax in configs:
        for model in ["vgg16_basic", "vgg16_proposed", "vgg19_basic", "vgg19_proposed"]:
            history = load_history(model, dataset)
            if history is None:
                continue
            
            epochs = range(1, len(history["train_loss"]) + 1)
            
            if metric == "accuracy":
                ax.plot(epochs, history["val_acc"], color=colors[model],
                       label=labels[model], linewidth=1.5)
            else:
                ax.plot(epochs, history["val_loss"], color=colors[model],
                       label=labels[model], linewidth=1.5)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)" if metric == "accuracy" else "Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {os.path.join(OUTPUT_DIR, 'training_curves.png')}")
    plt.close()


def plot_comparison_bar():
    """Generate bar chart comparing our results vs paper."""
    results = load_results()
    
    models = ["VGG-16\nBasic", "VGG-16\nProposed", "VGG-19\nBasic", "VGG-19\nProposed"]
    model_keys = ["vgg16_basic", "vgg16_proposed", "vgg19_basic", "vgg19_proposed"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, dataset in enumerate(["cifar10", "cifar100"]):
        ax = axes[idx]
        ours = []
        paper = []
        
        for mk in model_keys:
            r = results.get((mk, dataset), {})
            ours.append(r.get("test_accuracy", 0))
            paper.append(PAPER_RESULTS.get((mk, dataset), 0))
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ours, width, label='Ours', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, paper, width, label='Paper', color='coral', alpha=0.8)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{"CIFAR-10" if dataset == "cifar10" else "CIFAR-100"} Results')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(50, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_bar.png"), dpi=150, bbox_inches='tight')
    print(f"Comparison bar chart saved to {os.path.join(OUTPUT_DIR, 'comparison_bar.png')}")
    plt.close()


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = load_results()
    
    print(f"Found {len(results)} experiment results")
    
    table_data = generate_table(results)
    plot_training_curves(results)
    plot_comparison_bar()
    
    print("\nDone! All results generated.")
