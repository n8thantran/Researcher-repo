"""
Data pipeline for CIFAR-10 and CIFAR-100 datasets.
- Loads CIFAR-10 / CIFAR-100
- Splits 50k training into 40k train + 10k validation
- Applies preprocessing (normalization using ImageNet stats)
- Data augmentation matching Keras ImageDataGenerator:
    horizontal/vertical flips, rotation_range=30, shift_range=0.3, zoom_range=0.3
    fill_mode='nearest' (Keras default)
- Returns DataLoaders
"""

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationsDataset(Dataset):
    """Wrapper to apply albumentations transforms to a torchvision dataset."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img is a PIL Image, convert to numpy
        img = np.array(img)  # H, W, C
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
        return img, label


def get_train_transform():
    """
    Training augmentation matching Keras ImageDataGenerator:
    - rotation_range=30
    - width_shift_range=0.3 
    - height_shift_range=0.3
    - zoom_range=0.3
    - horizontal_flip=True
    - vertical_flip=True
    - fill_mode='nearest' (Keras default)
    
    In Keras ImageDataGenerator, each augmentation is applied independently.
    We use albumentations to match this behavior.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.3,        # width_shift_range=0.3, height_shift_range=0.3
            scale_limit=0.3,        # zoom_range=0.3 -> scale in [0.7, 1.3]
            rotate_limit=30,        # rotation_range=30
            border_mode=cv2.BORDER_REPLICATE,  # fill_mode='nearest'
            p=0.8,  # Apply most of the time but not always
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return transform


def get_test_transform():
    """Test/validation transform: just normalize."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return transform


def get_dataloaders(dataset_name='cifar10', batch_size=64, num_workers=4):
    """
    Get train, val, test dataloaders for CIFAR-10 or CIFAR-100.
    
    Paper specifies: 40,000 training, 10,000 validation, 10,000 test.
    """
    assert dataset_name in ['cifar10', 'cifar100'], f"Unknown dataset: {dataset_name}"
    
    DatasetClass = datasets.CIFAR10 if dataset_name == 'cifar10' else datasets.CIFAR100
    
    # Load raw datasets (no torchvision transforms)
    full_train_raw = DatasetClass(root='./data', train=True, download=True, transform=None)
    test_raw = DatasetClass(root='./data', train=False, download=True, transform=None)
    
    # Split training into 40k train + 10k validation
    rng = np.random.RandomState(42)
    indices = rng.permutation(50000)
    train_indices = indices[:40000]
    val_indices = indices[40000:]
    
    train_subset = Subset(full_train_raw, train_indices)
    val_subset = Subset(full_train_raw, val_indices)
    
    # Wrap with albumentations
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    
    train_dataset = AlbumentationsDataset(train_subset, train_transform)
    val_dataset = AlbumentationsDataset(val_subset, test_transform)
    test_dataset = AlbumentationsDataset(test_raw, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    num_classes = 10 if dataset_name == 'cifar10' else 100
    
    print(f"Dataset: {dataset_name}")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print(f"  Classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes


if __name__ == '__main__':
    # Quick sanity check
    for ds in ['cifar10', 'cifar100']:
        train_loader, val_loader, test_loader, num_classes = get_dataloaders(ds, batch_size=64)
        # Check a batch
        images, labels = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
        print()
