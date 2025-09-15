"""
Basic Vision Transformer Usage Examples
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224


def basic_model_usage():
    """Demonstrate basic model creation and inference"""
    print("=== Basic ViT Model Usage ===")
    
    # Create different model sizes
    models = {
        'tiny': vit_tiny_patch16_224(num_classes=1000),
        'small': vit_small_patch16_224(num_classes=1000),
        'base': vit_base_patch16_224(num_classes=1000),
        'large': vit_large_patch16_224(num_classes=1000)
    }
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    
    for name, model in models.items():
        # Set to evaluation mode
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"ViT-{name.capitalize()}: {num_params:,} parameters, Output shape: {output.shape}")


def feature_extraction_example():
    """Demonstrate feature extraction"""
    print("\n=== Feature Extraction Example ===")
    
    model = vit_base_patch16_224(num_classes=1000)
    model.eval()
    
    x = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        # Extract features before classification head
        features = model.forward_features(x)  # [1, 196, 768]
        
        # Global average pooling
        pooled_features = features.mean(dim=1)  # [1, 768]
        
        # Final classification
        logits = model.head(pooled_features)  # [1, 1000]
    
    print(f"Patch features shape: {features.shape}")
    print(f"Pooled features shape: {pooled_features.shape}")
    print(f"Final logits shape: {logits.shape}")


def custom_image_size_example():
    """Demonstrate using different image sizes"""
    print("\n=== Custom Image Size Example ===")
    
    # Different image sizes (must be divisible by patch_size=16)
    image_sizes = [224, 256, 384]
    
    for img_size in image_sizes:
        model = vit_base_patch16_224(num_classes=1000)
        model.eval()
        
        x = torch.randn(1, 3, img_size, img_size)
        
        with torch.no_grad():
            output = model(x)
        
        num_patches = (img_size // 16) ** 2
        print(f"Image size {img_size}x{img_size}: {num_patches} patches, Output: {output.shape}")


def transfer_learning_example():
    """Demonstrate transfer learning setup"""
    print("\n=== Transfer Learning Example ===")
    
    # Load pretrained model
    model = vit_base_patch16_224(num_classes=1000)
    
    # Modify for new task (e.g., 10 classes)
    num_classes = 10
    model.head = torch.nn.Linear(model.dim, num_classes)
    
    # Freeze backbone (optional)
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classification head
    for param in model.head.parameters():
        param.requires_grad = True
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Output shape for {num_classes} classes: {output.shape}")


if __name__ == "__main__":
    basic_model_usage()
    feature_extraction_example()
    custom_image_size_example()
    transfer_learning_example()
    
    print("\n✅ All examples completed successfully!")
