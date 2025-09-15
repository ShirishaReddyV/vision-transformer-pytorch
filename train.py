"""
Vision Transformer Training Script
Comprehensive training pipeline with logging, checkpoints, and validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import os
import time
from tqdm import tqdm
import json

from model import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224


class ViTTrainer:
    """Vision Transformer Trainer Class"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Logging
        self.writer = SummaryWriter(args.log_dir)
        
        # Training state
        self.start_epoch = 0
        self.best_acc = 0.0
        
        # Load checkpoint if exists
        if args.resume:
            self._load_checkpoint()
    
    def _create_model(self):
        """Create model based on architecture"""
        model_dict = {
            'tiny': vit_tiny_patch16_224,
            'small': vit_small_patch16_224,
            'base': vit_base_patch16_224,
            'large': vit_large_patch16_224
        }
        
        if self.args.arch not in model_dict:
            raise ValueError(f"Architecture {self.args.arch} not supported")
        
        model = model_dict[self.args.arch](num_classes=self.args.num_classes)
        
        print(f"Created {self.args.arch} model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def _create_optimizer(self):
        """Create optimizer with proper weight decay"""
        # Separate parameters for weight decay
        decay = []
        no_decay = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        
        params = [
            {'params': no_decay, 'weight_decay': 0.0},
            {'params': decay, 'weight_decay': self.args.weight_decay}
        ]
        
        optimizer = optim.AdamW(params, lr=self.args.lr, betas=(0.9, 0.999))
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs, eta_min=self.args.lr * 0.01
        )
        return scheduler
    
    def _create_data_loaders(self):
        """Create train and validation data loaders"""
        # Data transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Datasets
        if self.args.dataset == 'imagenet':
            train_dataset = datasets.ImageFolder(
                os.path.join(self.args.data_path, 'train'),
                transform=train_transform
            )
            val_dataset = datasets.ImageFolder(
                os.path.join(self.args.data_path, 'val'),
                transform=val_transform
            )
        elif self.args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root=self.args.data_path,
                train=True,
                download=True,
                transform=train_transform
            )
            val_dataset = datasets.CIFAR10(
                root=self.args.data_path,
                train=False,
                download=True,
                transform=val_transform
            )
        else:
            raise ValueError(f"Dataset {self.args.dataset} not supported")
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = []
        accuracies = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item() / len(data)
            
            losses.append(loss.item())
            accuracies.append(acc)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.4f}',
                'LR': f'{self.optimizer.param_groups["lr"]:.6f}'
            })
            
            # Log to tensorboard
            if batch_idx % self.args.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy', acc, step)
                self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups['lr'], step)
        
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        losses = []
        accuracies = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(target.view_as(pred)).sum().item() / len(data)
                
                losses.append(loss.item())
                accuracies.append(acc)
        
        val_loss = sum(losses) / len(losses)
        val_acc = sum(accuracies) / len(accuracies)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'arch': self.args.arch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'args': self.args
        }
        
        filename = os.path.join(self.args.checkpoint_dir, 'checkpoint.pth')
        torch.save(state, filename)
        
        if is_best:
            best_filename = os.path.join(self.args.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_filename)
    
    def _load_checkpoint(self):
        """Load checkpoint"""
        checkpoint_path = os.path.join(self.args.checkpoint_dir, 'checkpoint.pth')
        
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_acc = checkpoint['best_acc']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, best_acc {self.best_acc:.4f})")
        else:
            print(f"No checkpoint found at '{checkpoint_path}'")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Check if best model
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Print epoch results
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Best Acc: {self.best_acc:.4f}')
        
        print(f'Training completed! Best validation accuracy: {self.best_acc:.4f}')
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer Training')
    
    # Model arguments
    parser.add_argument('--arch', default='base', choices=['tiny', 'small', 'base', 'large'],
                        help='model architecture')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='number of classes')
    
    # Dataset arguments
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'cifar10'],
                        help='dataset name')
    parser.add_argument('--data-path', default='./data', type=str,
                        help='path to dataset')
    
    # Training arguments
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=0.05, type=float,
                        help='weight decay')
    
    # System arguments
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', type=str,
                        help='path to save checkpoints')
    parser.add_argument('--log-dir', default='./logs', type=str,
                        help='path to save logs')
    parser.add_argument('--log-interval', default=100, type=int,
                        help='how many batches to wait before logging')
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = ViTTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
