"""Quick test to see if upscaling helps CIFAR-100 accuracy."""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torchvision.models as models
import time
import math

device = torch.device('cuda')

# Test with different resolutions
for img_size in [32, 64, 224]:
    print(f"\n{'='*60}")
    print(f"Testing with {img_size}x{img_size} resolution")
    print(f"{'='*60}")
    
    # Simple transforms
    train_transform = transforms.Compose([
        transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(img_size) if img_size != 32 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_full = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    
    # Use 40k train, 10k val  
    rng = np.random.RandomState(42)
    indices = rng.permutation(50000)
    train_set = Subset(train_full, indices[:40000])
    val_set = Subset(train_full, indices[40000:])
    
    bs = 64 if img_size <= 64 else 32
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    
    # VGG-16 basic
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model = nn.Sequential(
        vgg.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 100),
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=math.sqrt(0.05), patience=7, 
        threshold=1e-4, threshold_mode='abs', min_lr=0
    )
    
    # Train 30 epochs
    best_val_acc = 0
    for epoch in range(1, 31):
        model.train()
        t0 = time.time()
        correct, total = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        train_acc = 100.0 * correct / total
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                val_total += targets.size(0)
                val_correct += pred.eq(targets).sum().item()
        val_acc = 100.0 * val_correct / val_total
        val_loss /= val_total
        scheduler.step(val_loss)
        best_val_acc = max(best_val_acc, val_acc)
        
        lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% (best: {best_val_acc:.1f}%) | LR: {lr:.2e} | {dt:.1f}s")
    
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

print("\nDone!")
