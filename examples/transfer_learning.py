"""
Transfer Learning Example with Vision Transformers
Shows how to fine-tune pre-trained ViT models on new tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model import vit_base_patch16_224


class TransferLearningViT(nn.Module):
    """ViT model adapted for transfer learning"""
    
    def __init__(self, num_classes, pretrained_path=None, freeze_backbone=True):
        super().__init__()
        
        # Load base ViT model
        self.backbone = vit_base_patch16_224(num_classes=1000)  # ImageNet classes
        
        # Load pretrained weights if provided
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                self.backbone.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.backbone.load_state_dict(checkpoint)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen - only training classification head")
        else:
            print("Backbone unfrozen - fine-tuning entire model")
        
        # Replace classification head
        self.backbone.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.backbone.dim, num_classes)
        )
        
        # Always allow gradient flow through new head
        for param in self.backbone.head.parameters():
            param.requires_grad = True
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for fine-tuning")
    
    def get_trainable_params(self):
        """Get only trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]


def create_cifar10_loaders(batch_size=64, num_workers=4):
    """Create CIFAR-10 data loaders with appropriate transforms"""
    
    # Transforms for training
    train_transform = transforms.Compose([
        transforms.Resize(224),  # ViT expects 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transforms for validation
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


class TransferLearningTrainer:
    """Trainer for transfer learning scenarios"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Different learning rates for different stages
        self.head_lr = 1e-3
        self.backbone_lr = 1e-5  # Much smaller for fine-tuning
        
        print(f"Training on device: {self.device}")
    
    def create_optimizer(self, stage='head_only'):
        """Create optimizer based on training stage"""
        trainable_params = self.model.get_trainable_params()
        
        if stage == 'head_only':
            # Only train the head
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.head_lr,
                weight_decay=0.01
            )
        else:
            # Fine-tune entire model with different learning rates
            head_params = list(self.model.backbone.head.parameters())
            backbone_params = [p for p in trainable_params if p not in head_params]
            
            optimizer = optim.AdamW([
                {'params': head_params, 'lr': self.head_lr},
                {'params': backbone_params, 'lr': self.backbone_lr}
            ], weight_decay=0.01)
        
        return optimizer
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
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
    
    def two_stage_training(self, train_loader, val_loader, 
                          head_epochs=10, finetune_epochs=20):
        """Two-stage training: head-only then fine-tuning"""
        best_val_acc = 0.0
        
        print("=" * 60)
        print("STAGE 1: Training classification head only")
        print("=" * 60)
        
        # Stage 1: Train head only
        optimizer = self.create_optimizer('head_only')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=head_epochs)
        
        for epoch in range(head_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch)
            val_loss, val_acc = self.validate(val_loader)
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{head_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'stage': 'head_only'
                }, 'transfer_learning_stage1.pth')
        
        print(f"Stage 1 completed. Best accuracy: {best_val_acc:.2f}%")
        
        print("\n" + "=" * 60)
        print("STAGE 2: Fine-tuning entire model")
        print("=" * 60)
        
        # Stage 2: Fine-tune entire model
        self.model.unfreeze_backbone()
        optimizer = self.create_optimizer('finetune')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
        
        for epoch in range(finetune_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch)
            val_loss, val_acc = self.validate(val_loader)
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{finetune_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'stage': 'finetune'
                }, 'transfer_learning_final.pth')
        
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc


def main():
    """Main transfer learning example"""
    print("Vision Transformer Transfer Learning Example")
    print("=" * 50)
    
    # Configuration
    NUM_CLASSES = 10  # CIFAR-10
    BATCH_SIZE = 32   # Smaller batch size for ViT
    PRETRAINED_PATH = None  # Set to your pretrained model path if available
    
    try:
        # Create data loaders
        print("Loading CIFAR-10 dataset...")
        train_loader, val_loader = create_cifar10_loaders(BATCH_SIZE)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        # Create transfer learning model
        print("Creating transfer learning model...")
        model = TransferLearningViT(
            num_classes=NUM_CLASSES,
            pretrained_path=PRETRAINED_PATH,
            freeze_backbone=True
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.get_trainable_params())
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        
        # Create trainer
        trainer = TransferLearningTrainer(model)
        
        # Run two-stage training
        best_acc = trainer.two_stage_training(
            train_loader, val_loader,
            head_epochs=10,
            finetune_epochs=20
        )
        
        print(f"\n🎉 Transfer learning completed!")
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print(f"Models saved:")
        print(f"  - transfer_learning_stage1.pth (head-only training)")
        print(f"  - transfer_learning_final.pth (final fine-tuned model)")
        
    except Exception as e:
        print(f"Error during transfer learning: {str(e)}")
        raise


def compare_with_from_scratch():
    """Compare transfer learning vs training from scratch"""
    print("\nComparison: Transfer Learning vs From Scratch")
    print("=" * 50)
    
    # This would require running both scenarios
    # For now, just show the concept
    print("To compare transfer learning effectiveness:")
    print("1. Run this script with pretrained weights")
    print("2. Run with pretrained_path=None (random initialization)")
    print("3. Compare final accuracies and training time")


if __name__ == "__main__":
    main()
    compare_with_from_scratch()
