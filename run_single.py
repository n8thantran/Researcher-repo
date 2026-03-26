#!/usr/bin/env python3
"""
Run a single experiment. Designed to be called from the shell.
Handles the case where a best_model.pth exists but results.json doesn't.
"""
import sys
import os
import json
import torch
import torch.nn as nn

# Add workspace to path
sys.path.insert(0, '/workspace')

from data import get_dataloaders
from models import get_model
from train import train, evaluate


def run_single(model_name, dataset_name):
    exp_dir = f'results/{model_name}_{dataset_name}'
    results_file = os.path.join(exp_dir, 'results.json')
    
    # Check if already complete
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"SKIP {model_name} {dataset_name}: already complete (test_acc={results['test_accuracy']:.2f}%)")
        return results
    
    # Check if we have a best model but no results (interrupted run)
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"Found existing best_model.pth for {model_name} {dataset_name}, evaluating...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes = 10 if dataset_name == 'cifar10' else 100
        _, val_loader, test_loader, _ = get_dataloaders(dataset_name, batch_size=64, num_workers=4)
        model = get_model(model_name, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        criterion = nn.CrossEntropyLoss()
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"  Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # If accuracy is reasonable, save and skip retraining
        min_acc = 70 if dataset_name == 'cifar10' else 50
        if test_acc > min_acc:
            results = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'best_val_accuracy': val_acc,
                'best_epoch': -1,
                'note': 'Recovered from interrupted run'
            }
            os.makedirs(exp_dir, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"  Saved recovered results.")
            return results
        else:
            print(f"  Accuracy too low ({test_acc:.2f}%), retraining...")
            os.remove(best_model_path)
    
    # Full training
    print(f"\nTraining {model_name} on {dataset_name}...")
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
    return results


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python run_single.py <model_name> <dataset_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    run_single(model_name, dataset_name)
