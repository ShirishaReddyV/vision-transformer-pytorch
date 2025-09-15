"""
Speed Benchmark Script for Vision Transformers
Comprehensive performance testing across different configurations
"""

import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile
import gc
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224


class SpeedBenchmark:
    """Comprehensive speed benchmarking for Vision Transformers"""
    
    def __init__(self, device=None, precision='fp32'):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = precision
        print(f"Benchmarking on: {self.device}")
        print(f"Precision: {precision}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def warmup_gpu(self):
        """Warm up GPU for accurate benchmarking"""
        if self.device.type == 'cuda':
            dummy = torch.randn(100, 100).to(self.device)
            for _ in range(100):
                _ = torch.mm(dummy, dummy)
            torch.cuda.synchronize()
            del dummy
            gc.collect()
    
    def measure_inference_time(self, model, input_tensor, num_runs=100, warmup_runs=10):
        """Measure inference time with proper GPU synchronization"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize for accurate timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                output = model(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        
        times = np.array(times)
        return {
            'mean': times.mean(),
            'std': times.std(),
            'min': times.min(),
            'max': times.max(),
            'median': np.median(times)
        }
    
    def measure_memory_usage(self, model, input_tensor):
        """Measure peak GPU memory usage"""
        if self.device.type != 'cuda':
            return 0
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        return peak_memory
    
    def benchmark_model_variants(self, batch_size=1, image_size=224, num_runs=100):
        """Benchmark different ViT model variants"""
        print(f"\n=== Model Variants Benchmark (batch={batch_size}, image={image_size}) ===")
        
        models = {
            'ViT-Tiny': vit_tiny_patch16_224(num_classes=1000),
            'ViT-Small': vit_small_patch16_224(num_classes=1000),
            'ViT-Base': vit_base_patch16_224(num_classes=1000),
            'ViT-Large': vit_large_patch16_224(num_classes=1000)
        }
        
        input_tensor = torch.randn(batch_size, 3, image_size, image_size).to(self.device)
        if self.precision == 'fp16':
            input_tensor = input_tensor.half()
        
        results = []
        
        for name, model in models.items():
            try:
                model.to(self.device)
                if self.precision == 'fp16':
                    model = model.half()
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                
                # Measure timing
                timing_stats = self.measure_inference_time(model, input_tensor, num_runs)
                
                # Measure memory
                memory_mb = self.measure_memory_usage(model, input_tensor)
                
                # Calculate throughput
                throughput = batch_size / timing_stats['mean']
                
                result = {
                    'Model': name,
                    'Parameters (M)': total_params / 1e6,
                    'Inference Time (ms)': timing_stats['mean'] * 1000,
                    'Std (ms)': timing_stats['std'] * 1000,
                    'Throughput (img/s)': throughput,
                    'Memory (MB)': memory_mb,
                }
                
                results.append(result)
                print(f"{name}: {timing_stats['mean']*1000:.2f}±{timing_stats['std']*1000:.2f}ms, "
                      f"{throughput:.1f} img/s, {memory_mb:.1f}MB")
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{name}: Out of memory")
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e
        
        return pd.DataFrame(results)
    
    def benchmark_batch_sizes(self, model_name='small', image_size=224):
        """Benchmark different batch sizes"""
        print(f"\n=== Batch Size Benchmark for ViT-{model_name.capitalize()} ===")
        
        model_dict = {
            'tiny': vit_tiny_patch16_224,
            'small': vit_small_patch16_224,
            'base': vit_base_patch16_224,
            'large': vit_large_patch16_224
        }
        
        model = model_dict[model_name](num_classes=1000).to(self.device)
        if self.precision == 'fp16':
            model = model.half()
        
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        results = []
        
        for batch_size in batch_sizes:
            try:
                input_tensor = torch.randn(batch_size, 3, image_size, image_size).to(self.device)
                if self.precision == 'fp16':
                    input_tensor = input_tensor.half()
                
                timing_stats = self.measure_inference_time(model, input_tensor, num_runs=50)
                memory_mb = self.measure_memory_usage(model, input_tensor)
                
                throughput = batch_size / timing_stats['mean']
                time_per_image = timing_stats['mean'] / batch_size * 1000
                
                result = {
                    'Batch Size': batch_size,
                    'Total Time (ms)': timing_stats['mean'] * 1000,
                    'Time per Image (ms)': time_per_image,
                    'Throughput (img/s)': throughput,
                    'Memory (MB)': memory_mb
                }
                
                results.append(result)
                print(f"Batch {batch_size}: {time_per_image:.2f}ms/img, {throughput:.1f} img/s, {memory_mb:.1f}MB")
                
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Batch {batch_size}: Out of memory")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e
        
        return pd.DataFrame(results)
    
    def benchmark_image_sizes(self, model_name='small', batch_size=1):
        """Benchmark different image sizes"""
        print(f"\n=== Image Size Benchmark for ViT-{model_name.capitalize()} ===")
        
        model_dict = {
            'tiny': vit_tiny_patch16_224,
            'small': vit_small_patch16_224,
            'base': vit_base_patch16_224,
            'large': vit_large_patch16_224
        }
        
        model = model_dict[model_name](num_classes=1000).to(self.device)
        if self.precision == 'fp16':
            model = model.half()
        
        image_sizes = [224, 256, 288, 320, 352, 384, 416, 448]
        results = []
        
        for img_size in image_sizes:
            try:
                input_tensor = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
                if self.precision == 'fp16':
                    input_tensor = input_tensor.half()
                
                timing_stats = self.measure_inference_time(model, input_tensor, num_runs=50)
                memory_mb = self.measure_memory_usage(model, input_tensor)
                
                num_patches = (img_size // 16) ** 2
                throughput = batch_size / timing_stats['mean']
                
                result = {
                    'Image Size': f"{img_size}x{img_size}",
                    'Patches': num_patches,
                    'Inference Time (ms)': timing_stats['mean'] * 1000,
                    'Throughput (img/s)': throughput,
                    'Memory (MB)': memory_mb
                }
                
                results.append(result)
                print(f"{img_size}x{img_size} ({num_patches} patches): {timing_stats['mean']*1000:.2f}ms, "
                      f"{throughput:.1f} img/s, {memory_mb:.1f}MB")
                
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Image size {img_size}: Out of memory")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e
        
        return pd.DataFrame(results)
    
    def benchmark_precision_comparison(self, model_name='small'):
        """Compare FP32 vs FP16 performance"""
        print(f"\n=== Precision Comparison for ViT-{model_name.capitalize()} ===")
        
        model_dict = {
            'tiny': vit_tiny_patch16_224,
            'small': vit_small_patch16_224,
            'base': vit_base_patch16_224,
            'large': vit_large_patch16_224
        }
        
        input_tensor_fp32 = torch.randn(1, 3, 224, 224).to(self.device)
        input_tensor_fp16 = input_tensor_fp32.half()
        
        results = []
        
        for precision, input_tensor in [('FP32', input_tensor_fp32), ('FP16', input_tensor_fp16)]:
            model = model_dict[model_name](num_classes=1000).to(self.device)
            
            if precision == 'FP16':
                model = model.half()
            
            timing_stats = self.measure_inference_time(model, input_tensor, num_runs=100)
            memory_mb = self.measure_memory_usage(model, input_tensor)
            
            result = {
                'Precision': precision,
                'Inference Time (ms)': timing_stats['mean'] * 1000,
                'Throughput (img/s)': 1 / timing_stats['mean'],
                'Memory (MB)': memory_mb
            }
            
            results.append(result)
            print(f"{precision}: {timing_stats['mean']*1000:.2f}ms, "
                  f"{1/timing_stats['mean']:.1f} img/s, {memory_mb:.1f}MB")
            
            del model
            torch.cuda.empty_cache()
        
        return pd.DataFrame(results)
    
    def create_benchmark_plots(self, df_models, df_batch, df_image):
        """Create comprehensive benchmark visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Vision Transformer Performance Benchmark ({self.device})', fontsize=16)
        
        # Model comparison - Parameters
        if not df_models.empty:
            axes[0, 0].bar(df_models['Model'], df_models['Parameters (M)'], color='skyblue')
            axes[0, 0].set_title('Model Size (Parameters)')
            axes[0, 0].set_ylabel('Parameters (M)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Model comparison - Inference time
        if not df_models.empty:
            axes[0, 1].bar(df_models['Model'], df_models['Inference Time (ms)'], color='lightcoral')
            axes[0, 1].set_title('Inference Time')
            axes[0, 1].set_ylabel('Time (ms)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Model comparison - Throughput
        if not df_models.empty:
            axes[0, 2].bar(df_models['Model'], df_models['Throughput (img/s)'], color='lightgreen')
            axes[0, 2].set_title('Throughput')
            axes[0, 2].set_ylabel('Images/second')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Batch size scaling
        if not df_batch.empty:
            axes[1, 0].plot(df_batch['Batch Size'], df_batch['Throughput (img/s)'], 'o-', color='purple')
            axes[1, 0].set_title('Throughput vs Batch Size')
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('Throughput (img/s)')
            axes[1, 0].grid(True)
            axes[1, 0].set_xscale('log', base=2)
        
        # Image size scaling
        if not df_image.empty:
            axes[1, 1].plot(df_image['Patches'], df_image['Inference Time (ms)'], 'o-', color='orange')
            axes[1, 1].set_title('Inference Time vs Patches')
            axes[1, 1].set_xlabel('Number of Patches')
            axes[1, 1].set_ylabel('Time (ms)')
            axes[1, 1].grid(True)
        
        # Memory usage
        if not df_models.empty:
            axes[1, 2].bar(df_models['Model'], df_models['Memory (MB)'], color='gold')
            axes[1, 2].set_title('Memory Usage')
            axes[1, 2].set_ylabel('Memory (MB)')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'vit_speed_benchmark_{self.device.type}.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_speed_benchmark():
    """Run comprehensive speed benchmarking"""
    # Test both FP32 and FP16 if CUDA is available
    precisions = ['fp32']
    if torch.cuda.is_available():
        precisions.append('fp16')
    
    all_results = {}
    
    for precision in precisions:
        print(f"\n{'='*80}")
        print(f"BENCHMARKING WITH {precision.upper()}")
        print(f"{'='*80}")
        
        benchmark = SpeedBenchmark(precision=precision)
        benchmark.warmup_gpu()
        
        # Model variants benchmark
        df_models = benchmark.benchmark_model_variants(batch_size=1)
        
        # Batch size benchmark (using smaller model to avoid OOM)
        df_batch = benchmark.benchmark_batch_sizes('small')
        
        # Image size benchmark
        df_image = benchmark.benchmark_image_sizes('small')
        
        # Precision comparison (only for first run)
        if precision == 'fp32' and torch.cuda.is_available():
            df_precision = benchmark.benchmark_precision_comparison('small')
            print(f"\nPrecision Comparison:")
            print(df_precision.to_string(index=False))
        
        # Save results
        df_models.to_csv(f'vit_model_benchmark_{precision}.csv', index=False)
        df_batch.to_csv(f'vit_batch_benchmark_{precision}.csv', index=False)
        df_image.to_csv(f'vit_image_benchmark_{precision}.csv', index=False)
        
        # Create visualizations
        benchmark.create_benchmark_plots(df_models, df_batch, df_image)
        
        all_results[precision] = {
            'models': df_models,
            'batch': df_batch,
            'image': df_image
        }
        
        print(f"\n{precision.upper()} Results Summary:")
        print("Model Variants:")
        print(df_models[['Model', 'Inference Time (ms)', 'Throughput (img/s)', 'Memory (MB)']].to_string(index=False))
    
    print(f"\n✅ Comprehensive speed benchmark completed!")
    print(f"Results saved to CSV files and visualizations saved as PNG files")
    
    return all_results


if __name__ == "__main__":
    run_comprehensive_speed_benchmark()
