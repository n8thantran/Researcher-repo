"""
Model architectures for the paper:
"Inspect Transfer Learning Architecture with Dilated Convolution"

Implements:
1. VGG-16 Basic (standard VGG-16 with transfer learning, all blocks trainable)
2. VGG-16 Proposed (frozen blocks 1-2, dilated blocks 3-5 with concatenation)
3. VGG-19 Basic (standard VGG-19 with transfer learning, all blocks trainable)
4. VGG-19 Proposed (frozen blocks 1-2, dilated blocks 3-5 with concatenation)

Key: Paper does NOT mention dropout. Classifier is FC 512 -> 256 -> num_classes.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def make_dilated_block(in_channels, out_channels, num_layers, dilation_rate, use_maxpool=True):
    """Create a block of dilated convolutions with batch norm, ReLU, and optional max-pooling."""
    layers = []
    for i in range(num_layers):
        ic = in_channels if i == 0 else out_channels
        layers.append(nn.Conv2d(ic, out_channels, kernel_size=3, 
                                padding=dilation_rate, dilation=dilation_rate))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
    if use_maxpool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def init_dilated_block_from_pretrained(dilated_block, pretrained_convs):
    """Initialize dilated conv block weights from pretrained VGG conv layers."""
    conv_idx = 0
    for layer in dilated_block:
        if isinstance(layer, nn.Conv2d) and conv_idx < len(pretrained_convs):
            pretrained_conv = pretrained_convs[conv_idx]
            # Copy weights if shapes match
            if layer.weight.shape == pretrained_conv.weight.shape:
                layer.weight.data.copy_(pretrained_conv.weight.data)
                if layer.bias is not None and pretrained_conv.bias is not None:
                    layer.bias.data.copy_(pretrained_conv.bias.data)
            conv_idx += 1


class VGG16Basic(nn.Module):
    """
    VGG-16 Basic: Standard VGG-16 with pretrained weights.
    All blocks are trainable (no freezing).
    FC: 512 -> 256 -> num_classes (NO dropout - paper doesn't mention it).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16Proposed(nn.Module):
    """
    VGG-16 Proposed:
    - Block 1: Freeze (pretrained) - 2 conv layers, 64 filters, maxpool
    - Block 2: Freeze (pretrained) - 2 conv layers, 128 filters, maxpool
    - Block 3: Dilation rate 2, three 3x3 conv layers, 256 filters, maxpool
    - Block 4: Dilation rate 4, three 3x3 conv layers, 512 filters, maxpool
    - Block 5: Concatenate two parallel branches (dilation rate 4 and 8), 512 filters each
    - AdaptiveAvgPool -> FC 512 -> FC 256 -> FC num_classes
    """
    def __init__(self, num_classes=10):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())
        
        # Block 1: layers 0-4 (conv, relu, conv, relu, maxpool)
        self.block1 = nn.Sequential(*features[:5])
        # Block 2: layers 5-9
        self.block2 = nn.Sequential(*features[5:10])
        
        # Freeze blocks 1 and 2
        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.block2.parameters():
            param.requires_grad = False
        
        # Get pretrained conv layers for initialization
        block3_pretrained = [features[10], features[12], features[14]]
        block4_pretrained = [features[17], features[19], features[21]]
        block5_pretrained = [features[24], features[26], features[28]]
        
        # Block 3: dilation=2, 3 conv layers, 256 filters + maxpool
        self.block3 = make_dilated_block(128, 256, num_layers=3, dilation_rate=2, use_maxpool=True)
        init_dilated_block_from_pretrained(self.block3, block3_pretrained)
        
        # Block 4: dilation=4, 3 conv layers, 512 filters + maxpool
        self.block4 = make_dilated_block(256, 512, num_layers=3, dilation_rate=4, use_maxpool=True)
        init_dilated_block_from_pretrained(self.block4, block4_pretrained)
        
        # Block 5: Two parallel branches, concatenated (no maxpool)
        self.block5_branch1 = make_dilated_block(512, 512, num_layers=3, dilation_rate=4, use_maxpool=False)
        init_dilated_block_from_pretrained(self.block5_branch1, block5_pretrained)
        self.block5_branch2 = make_dilated_block(512, 512, num_layers=3, dilation_rate=8, use_maxpool=False)
        init_dilated_block_from_pretrained(self.block5_branch2, block5_pretrained)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # After concatenation: 512 + 512 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        b1 = self.block5_branch1(x)
        b2 = self.block5_branch2(x)
        x = torch.cat([b1, b2], dim=1)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG19Basic(nn.Module):
    """
    VGG-19 Basic: Standard VGG-19 with pretrained weights.
    All blocks are trainable.
    FC: 512 -> 256 -> num_classes (NO dropout).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG19Proposed(nn.Module):
    """
    VGG-19 Proposed:
    - Block 1: Freeze (pretrained)
    - Block 2: Freeze (pretrained)
    - Block 3: Dilation rate 2, four 3x3 conv layers, 256 filters, maxpool
    - Block 4: Dilation rate 2, four 3x3 conv layers, 512 filters, maxpool
    - Block 5: Concatenate two parallel branches (dilation rate 2 and 4),
               four 3x3 conv layers, 512 filters each
    - AdaptiveAvgPool -> FC 512 -> FC 256 -> FC num_classes
    """
    def __init__(self, num_classes=10):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())
        
        self.block1 = nn.Sequential(*features[:5])
        self.block2 = nn.Sequential(*features[5:10])
        
        # Freeze blocks 1 and 2
        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.block2.parameters():
            param.requires_grad = False
        
        # Get pretrained conv layers for initialization
        block3_pretrained = [features[10], features[12], features[14], features[16]]
        block4_pretrained = [features[19], features[21], features[23], features[25]]
        block5_pretrained = [features[28], features[30], features[32], features[34]]
        
        # Block 3: dilation=2, 4 conv layers, 256 filters + maxpool
        self.block3 = make_dilated_block(128, 256, num_layers=4, dilation_rate=2, use_maxpool=True)
        init_dilated_block_from_pretrained(self.block3, block3_pretrained)
        
        # Block 4: dilation=2, 4 conv layers, 512 filters + maxpool
        self.block4 = make_dilated_block(256, 512, num_layers=4, dilation_rate=2, use_maxpool=True)
        init_dilated_block_from_pretrained(self.block4, block4_pretrained)
        
        # Block 5: Two parallel branches, concatenated (no maxpool)
        self.block5_branch1 = make_dilated_block(512, 512, num_layers=4, dilation_rate=2, use_maxpool=False)
        init_dilated_block_from_pretrained(self.block5_branch1, block5_pretrained)
        self.block5_branch2 = make_dilated_block(512, 512, num_layers=4, dilation_rate=4, use_maxpool=False)
        init_dilated_block_from_pretrained(self.block5_branch2, block5_pretrained)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # After concatenation: 512 + 512 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        b1 = self.block5_branch1(x)
        b2 = self.block5_branch2(x)
        x = torch.cat([b1, b2], dim=1)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(model_name, num_classes=10):
    """Factory function to get model by name."""
    models_dict = {
        'vgg16_basic': VGG16Basic,
        'vgg16_proposed': VGG16Proposed,
        'vgg19_basic': VGG19Basic,
        'vgg19_proposed': VGG19Proposed,
    }
    assert model_name in models_dict, f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}"
    return models_dict[model_name](num_classes=num_classes)


if __name__ == '__main__':
    # Sanity check all models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for model_name in ['vgg16_basic', 'vgg16_proposed', 'vgg19_basic', 'vgg19_proposed']:
        for num_classes in [10, 100]:
            model = get_model(model_name, num_classes=num_classes).to(device)
            x = torch.randn(2, 3, 32, 32).to(device)
            out = model(x)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"{model_name} (classes={num_classes}): "
                  f"output={out.shape}, "
                  f"total_params={total_params/1e6:.1f}M, "
                  f"trainable={trainable_params/1e6:.1f}M")
    
    print("\nAll models passed sanity check!")
