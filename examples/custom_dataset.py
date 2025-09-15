"""
Custom Dataset Training Example for Vision Transformers
Shows how to train ViT on custom datasets with proper data loading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model import vit_small_patch16_224


class CustomImageDataset(Dataset):
    """
    Custom dataset class for loading images with labels
    Supports CSV annotation files and directory-based organization
    """
    
    def __init__(self, data_dir, annotations_file=None, transform=None, target_transform=None):
        """
        Args:
            data_dir (str): Directory with all the images
            annotations_file (str): Path to CSV file with annotations (optional)
            transform (callable): Optional transform to be applied on images
            target_transform (callable): Optional transform to be applied on labels
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        if annotations_file:
            # Load from CSV file
            self.annotations = pd.read_csv(annotations_file)
            self.image_paths = [self.data_dir / fname for fname in self.annotations.iloc[:, 0]]
            self.labels = self.annotations.iloc[:, 1].tolist()
        else:
            # Load from directory structure (folder names as labels)
            self.image_paths = []
            self.labels = []
            self.class_names = []
            
            for class_dir in sorted(self.data_dir.iterdir()):
                if class_dir.is_dir():
                    class_idx = len(self.class_names)
                    self.class_names.append(class_dir.name)
                    
                    for img_path in class_dir.glob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            self.image_paths.append(img_path)
                            self.labels.append(class_idx)
        
        self.num_classes = len(set(self.labels))
        print(f"Loaded {len(self.image_paths)} images with {self.num_classes} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def create_data_transforms():
    """Create train and validation transforms"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir, batch_size=32, validation_split=0.2, num_workers=4):
    """Create train and validation data loaders"""
    train_transform, val_transform = create_data_transforms()
    
    # Load full dataset
    full_dataset = CustomImageDataset(data_dir, transform=train_transform)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms to validation set
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.num_classes


class CustomViTTrainer:
    """Custom trainer for Vision Transformer"""
    
    def __init__(self, num_classes, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Create model
        self.model = vit_small_patch16_224(num_classes=num_classes)
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with proper weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.05,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Training on device: {self.device}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50):
        """Full training loop"""
        best_val_acc = 0.0
        
        print("Starting training...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                }, 'best_custom_vit_model.pth')
                print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
            
            print('-' * 60)
        
        print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')
        return best_val_acc


def main():
    """Main training function"""
    # Configuration
    DATA_DIR = "./custom_dataset"  # Change this to your dataset path
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    NUM_WORKERS = 4
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory '{DATA_DIR}' not found!")
        print("Please organize your dataset as follows:")
        print("custom_dataset/")
        print("├── class1/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("├── class2/")
        print("│   ├── image3.jpg")
        print("│   └── image4.jpg")
        print("└── ...")
        return
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, num_classes = create_data_loaders(
            DATA_DIR, BATCH_SIZE, VALIDATION_SPLIT, NUM_WORKERS
        )
        
        # Create trainer
        trainer = CustomViTTrainer(num_classes=num_classes)
        
        # Start training
        best_accuracy = trainer.train(train_loader, val_loader, NUM_EPOCHS)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Best validation accuracy: {best_accuracy:.2f}%")
        print(f"Model saved as: best_custom_vit_model.pth")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


def create_sample_dataset():
    """Create a sample dataset for testing (optional)"""
    import shutil
    from pathlib import Path
    
    # Create sample dataset structure
    sample_dir = Path("sample_dataset")
    sample_dir.mkdir(exist_ok=True)
    
    # Create class directories
    for class_name in ["cats", "dogs", "birds"]:
        class_dir = sample_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    print("Sample dataset structure created in 'sample_dataset/'")
    print("Add your images to the respective class folders and run the training script.")


if __name__ == "__main__":
    main()
