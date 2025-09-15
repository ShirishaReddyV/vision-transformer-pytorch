"""
Performance Comparison Script for Vision Transformers
Compare different ViT variants against each other and CNNs
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from memory_profiler import profile
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224


class ModelBenchmark:
    """Comprehensive model benchmarking suite"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Benchmarking on: {self.device}")
    
    def count_parameters(self, model):
        """Count model parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    def measure_inference_time(self, model, input_tensor, num_runs=100, warmup_runs=10):
        """Measure inference time"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize for accurate timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        return avg_time, output.shape
    
    def measure_memory_usage(self, model, input_tensor):
        """Measure memory usage"""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                output = model(input_tensor)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            return memory_used
        else:
            return 0  # Cannot measure CPU memory easily
    
    def benchmark_vit_models(self, batch_size=1, image_size=224):
        """Benchmark all ViT model variants"""
        print(f"\n=== ViT Model Benchmark (batch_size={batch_size}, image_size={image_size}) ===")
        
        models = {
            'ViT-Tiny': vit_tiny_patch16_224(num_classes=1000),
            'ViT-Small': vit_small_patch16_224(num_classes=1000),
            'ViT-Base': vit_base_patch16_224(num_classes=1000),
            'ViT-Large': vit_large_patch16_224(num_classes=1000)
        }
        
        input_tensor = torch.randn(batch_size, 3, image_size, image_size).to(self.device)
        results = []
        
        for name, model in models.items():
            model.to(self.device)
            
            # Count parameters
            total_params, trainable_params = self.count_parameters(model)
            
            # Measure inference time
            avg_time, output_shape = self.measure_inference_time(model, input_tensor)
            
            # Measure memory usage
            memory_mb = self.measure_memory_usage(model, input_tensor)
            
            # Calculate throughput
            throughput = batch_size / avg_time
            
            results.append({
                'Model': name,
                'Parameters (M)': total_params / 1e6,
                'Inference Time (ms)': avg_time * 1000,
                'Throughput (img/s)': throughput,
                'Memory (MB)': memory_mb,
                'Output Shape': str(output_shape)
            })
            
            print(f"{name}: {total_params/1e6:.1f}M params, {avg_time*1000:.2f}ms, {throughput:.1f} img/s")
        
        return pd.DataFrame(results)
    
    def compare_batch_sizes(self, model_name='base'):
        """Compare performance across different batch sizes"""
        print(f"\n=== Batch Size Comparison for ViT-{model_name} ===")
        
        model_dict = {
            'tiny': vit_tiny_patch16_224,
            'small': vit_small_patch16_224,
            'base': vit_base_patch16_224,
            'large': vit_large_patch16_224
        }
        
        model = model_dict[model_name](num_classes=1000).to(self.device)
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        results = []
        
        for batch_size in batch_sizes:
            try:
                input_tensor = torch.randn(batch_size, 3, 224, 224).to(self.device)
                avg_time, _ = self.measure_inference_time(model, input_tensor)
                throughput = batch_size / avg_time
                
                results.append({
                    'Batch Size': batch_size,
                    'Time per Sample (ms)': (avg_time / batch_size) * 1000,
                    'Total Time (ms)': avg_time * 1000,
                    'Throughput (img/s)': throughput
                })
                
                print(f"Batch {batch_size}: {avg_time*1000:.2f}ms total, {throughput:.1f} img/s")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Batch {batch_size}: Out of memory")
                    break
                else:
                    raise e
        
        return pd.DataFrame(results)
    
    def compare_image_sizes(self, model_name='base'):
        """Compare performance across different image sizes"""
        print(f"\n=== Image Size Comparison for ViT-{model_name} ===")
        
        model_dict = {
            'tiny': vit_tiny_patch16_224,
            'small': vit_small_patch16_224,
            'base': vit_base_patch16_224,
            'large': vit_large_patch16_224
        }
        
        model = model_dict[model_name](num_classes=1000).to(self.device)
        image_sizes = [224, 256, 320, 384, 448]  # Must be divisible by 16
        results = []
        
        for img_size in image_sizes:
            try:
                input_tensor = torch.randn(1, 3, img_size, img_size).to(self.device)
                avg_time, output_shape = self.measure_inference_time(model, input_tensor)
                throughput = 1 / avg_time
                num_patches = (img_size // 16) ** 2
                
                results.append({
                    'Image Size': f"{img_size}x{img_size}",
                    'Patches': num_patches,
                    'Inference Time (ms)': avg_time * 1000,
                    'Throughput (img/s)': throughput
                })
                
                print(f"{img_size}x{img_size}: {num_patches} patches, {avg_time*1000:.2f}ms, {throughput:.1f} img/s")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Image size {img_size}: Out of memory")
                    break
                else:
                    raise e
        
        return pd.DataFrame(results)
    
    def create_visualizations(self, df_models, df_batch, df_image):
        """Create performance visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Vision Transformer Performance Analysis', fontsize=16)
        
        # Model comparison
        axes[0, 0].bar(df_models['Model'], df_models['Parameters (M)'])
        axes[0, 0].set_title('Model Size Comparison')
        axes[0, 0].set_ylabel('Parameters (M)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        axes[0, 1].bar(df_models['Model'], df_models['Inference Time (ms)'])
        axes[0, 1].set_title('Inference Time Comparison')
        axes[0, 1].set_ylabel('Time (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Batch size scaling
        if not df_batch.empty:
            axes[1, 0].plot(df_batch['Batch Size'], df_batch['Throughput (img/s)'], 'o-')
            axes[1, 0].set_title('Throughput vs Batch Size')
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('Throughput (img/s)')
            axes[1, 0].grid(True)
        
        # Image size scaling
        if not df_image.empty:
            axes[1, 1].plot(df_image['Patches'], df_image['Inference Time (ms)'], 'o-')
            axes[1, 1].set_title('Inference Time vs Number of Patches')
            axes[1, 1].set_xlabel('Number of Patches')
            axes[1, 1].set_ylabel('Time (ms)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('vit_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_benchmark():
    """Run comprehensive benchmarking suite"""
    benchmark = ModelBenchmark()
    
    # Benchmark all models
    df_models = benchmark.benchmark_vit_models(batch_size=1)
    print(f"\nModel Comparison Results:")
    print(df_models.to_string(index=False))
    
    # Batch size comparison
    df_batch = benchmark.compare_batch_sizes('base')
    
    # Image size comparison  
    df_image = benchmark.compare_image_sizes('base')
    
    # Save results
    df_models.to_csv('vit_model_comparison.csv', index=False)
    df_batch.to_csv('vit_batch_comparison.csv', index=False)
    df_image.to_csv('vit_image_size_comparison.csv', index=False)
    
    # Create visualizations
    benchmark.create_visualizations(df_models, df_batch, df_image)
    
    print("\n✅ Comprehensive benchmark completed!")
    print("Results saved to CSV files and visualization saved as PNG")


if __name__ == "__main__":
    run_comprehensive_benchmark()
