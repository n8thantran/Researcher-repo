"""
Training script for the paper:
"Inspect Transfer Learning Architecture with Dilated Convolution"

Training details from paper:
- Optimizer: Adam
- Learning rate: 1e-5
- LR scheduler: ReduceLROnPlateau(monitor='val_loss', patience=7, factor=sqrt(0.05)≈0.2236)
- Loss: Cross-Entropy
- Epochs: 250
- Batch size: 64 (not specified in paper; reasonable default)
"""

import os
import json
import time
import argparse
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import get_dataloaders
from models import get_model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train(model_name, dataset_name, epochs=250, batch_size=64, lr=1e-5, 
          patience=7, output_dir='results', num_workers=4):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    exp_name = f"{model_name}_{dataset_name}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Data
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        dataset_name, batch_size=batch_size, num_workers=num_workers
    )
    
    # Model
    model = get_model(model_name, num_classes=num_classes).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    print(f"  Classes: {num_classes}")
    print(f"  Total params: {total_params/1e6:.2f}M, Trainable: {trainable_params/1e6:.2f}M")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # factor = sqrt(0.05) ≈ 0.2236
    # Keras ReduceLROnPlateau defaults: min_delta=0, min_lr=0
    factor = math.sqrt(0.05)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, 
        verbose=False, threshold=0, threshold_mode='abs',
        min_lr=0
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [], 'epoch_time': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"LR: {lr}, Patience: {patience}, Factor: {factor:.4f}")
    print("-" * 80)
    sys.stdout.flush()
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # LR scheduler step
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
        
        # Print progress
        if epoch % 10 == 0 or epoch <= 5 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s | "
                  f"Best: {best_val_acc:.2f}% (ep {best_epoch})")
            sys.stdout.flush()
        
        # Early stopping: stop if LR is very small and no improvement for 15 epochs
        if current_lr < 1e-8 and (epoch - best_epoch) > 15:
            print(f"\nEarly stopping at epoch {epoch}: LR={current_lr:.2e}, "
                  f"no improvement for {epoch - best_epoch} epochs")
            break
    
    # Evaluate on test set using best model
    print("\n" + "=" * 80)
    print("Loading best model and evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pth'), weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"TEST ACCURACY: {test_acc:.2f}%")
    print(f"TEST LOSS: {test_loss:.4f}")
    print(f"Best validation accuracy was {best_val_acc:.2f}% at epoch {best_epoch}")
    sys.stdout.flush()
    
    # Save results
    results = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'epochs_run': len(history['train_loss']),
        'epochs_max': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'patience': patience,
        'factor': factor,
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to {exp_dir}/")
    sys.stdout.flush()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VGG models with dilated convolutions')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['vgg16_basic', 'vgg16_proposed', 'vgg19_basic', 'vgg19_proposed'])
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
