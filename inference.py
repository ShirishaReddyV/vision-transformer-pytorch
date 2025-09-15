"""
Vision Transformer Inference Script
Examples for using trained ViT models for prediction.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import argparse
import time
from pathlib import Path

from model import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224


class ViTPredictor:
    """Vision Transformer Predictor"""
    
    def __init__(self, model_path, arch='base', num_classes=1000, device=None):
        """
        Initialize ViT predictor
        
        Args:
            model_path: Path to trained model checkpoint
            arch: Model architecture (tiny, small, base, large)
            num_classes: Number of output classes
            device: Device to run inference on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.arch = arch
        self.num_classes = num_classes
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {arch} model on {self.device}")
    
    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        model_dict = {
            'tiny': vit_tiny_patch16_224,
            'small': vit_small_patch16_224,
            'base': vit_base_patch16_224,
            'large': vit_large_patch16_224
        }
        
        # Create model
        model = model_dict[self.arch](num_classes=self.num_classes)
        model.to(self.device)
        
        # Load weights
        if Path(model_path).exists():
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Model path {model_path} not found. Using randomly initialized model.")
        
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess single image"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        return self.transform(image).unsqueeze(0)  # Add batch dimension
    
    def predict_single(self, image_path, return_features=False):
        """
        Predict single image
        
        Args:
            image_path: Path to image or PIL Image
            return_features: Whether to return feature embeddings
        
        Returns:
            predictions: Class probabilities
            features: Feature embeddings (if return_features=True)
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        with torch.no_grad():
            if return_features:
                features = self.model.forward_features(image_tensor)
                features = features.mean(dim=1)  # Global average pooling
                logits = self.model.head(features)
                predictions = F.softmax(logits, dim=1)
                return predictions.cpu(), features.cpu()
            else:
                logits = self.model(image_tensor)
                predictions = F.softmax(logits, dim=1)
                return predictions.cpu()
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict batch of images
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference
        
        Returns:
            predictions: Batch predictions
        """
        all_predictions = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for path in batch_paths:
                tensor = self.preprocess_image(path)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(batch_tensor)
                predictions = F.softmax(logits, dim=1)
                all_predictions.append(predictions.cpu())
        
        return torch.cat(all_predictions, dim=0)
    
    def get_top_predictions(self, predictions, top_k=5, class_names=None):
        """
        Get top-k predictions with class names
        
        Args:
            predictions: Model predictions
            top_k: Number of top predictions to return
            class_names: List of class names
        
        Returns:
            top_predictions: List of (class, probability) tuples
        """
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze(0)
        
        top_probs, top_indices = torch.topk(predictions, top_k)
        
        top_predictions = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = class_names[idx] if class_names else f"Class_{idx}"
            top_predictions.append((class_name, prob.item()))
        
        return top_predictions
    
    def benchmark_inference(self, image_path, num_runs=100):
        """
        Benchmark inference speed
        
        Args:
            image_path: Path to test image
            num_runs: Number of inference runs
        
        Returns:
            avg_time: Average inference time in milliseconds
        """
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(image_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(image_tensor)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
        return avg_time


def load_imagenet_classes():
    """Load ImageNet class names"""
    try:
        # Try to load from local file
        with open('imagenet_classes.json', 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        # Return generic class names
        return [f"class_{i}" for i in range(1000)]


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer Inference')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--image-path', required=True, help='Path to input image')
    parser.add_argument('--arch', default='base', choices=['tiny', 'small', 'base', 'large'],
                        help='Model architecture')
    parser.add_argument('--num-classes', default=1000, type=int, help='Number of classes')
    parser.add_argument('--top-k', default=5, type=int, help='Top-k predictions to show')
    parser.add_argument('--benchmark', action='store_true', help='Run inference benchmark')
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = ViTPredictor(
        model_path=args.model_path,
        arch=args.arch,
        num_classes=args.num_classes
    )
    
    # Load class names
    class_names = load_imagenet_classes() if args.num_classes == 1000 else None
    
    # Make prediction
    print(f"Predicting image: {args.image_path}")
    predictions = predictor.predict_single(args.image_path)
    
    # Get top predictions
    top_preds = predictor.get_top_predictions(predictions, args.top_k, class_names)
    
    print(f"\nTop {args.top_k} predictions:")
    for i, (class_name, prob) in enumerate(top_preds, 1):
        print(f"{i}. {class_name}: {prob:.4f}")
    
    # Benchmark if requested
    if args.benchmark:
        print("\nRunning inference benchmark...")
        avg_time = predictor.benchmark_inference(args.image_path)
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Throughput: {1000/avg_time:.1f} images/sec")


if __name__ == '__main__':
    main()
