"""
Comprehensive test suite for Vision Transformer models
Tests model architecture, forward pass, parameter counts, and edge cases
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    OptimizedViT,
    TransformerBlock,
    Attention,
    MLP,
    DropPath,
    vit_tiny_patch16_224,
    vit_small_patch16_224,
    vit_base_patch16_224,
    vit_large_patch16_224,
    get_2d_sincos_pos_embed,
)

from tests import DEVICE, TEST_BATCH_SIZE, TEST_IMAGE_SIZE, TEST_NUM_CLASSES


class TestOptimizedViT:
    """Test cases for OptimizedViT model"""
    
    @pytest.fixture
    def model_configs(self):
        """Standard model configurations for testing"""
        return {
            'tiny': {'dim': 192, 'depth': 12, 'heads': 3, 'mlp_dim': 768},
            'small': {'dim': 384, 'depth': 12, 'heads': 6, 'mlp_dim': 1536},
            'base': {'dim': 768, 'depth': 12, 'heads': 12, 'mlp_dim': 3072},
        }
    
    @pytest.fixture
    def sample_input(self):
        """Sample input tensor for testing"""
        return torch.randn(TEST_BATCH_SIZE, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)
    
    def test_model_creation(self, model_configs):
        """Test that models can be created with different configurations"""
        for config_name, config in model_configs.items():
            model = OptimizedViT(
                num_classes=TEST_NUM_CLASSES,
                **config
            )
            assert isinstance(model, OptimizedViT)
            assert model.num_classes == TEST_NUM_CLASSES
            assert model.dim == config['dim']
    
    def test_forward_pass(self, model_configs, sample_input):
        """Test forward pass produces correct output shapes"""
        for config_name, config in model_configs.items():
            model = OptimizedViT(
                num_classes=TEST_NUM_CLASSES,
                **config
            )
            model.eval()
            
            with torch.no_grad():
                output = model(sample_input)
            
            expected_shape = (TEST_BATCH_SIZE, TEST_NUM_CLASSES)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert torch.isfinite(output).all(), "Output contains infinite values"
    
    def test_feature_extraction(self, sample_input):
        """Test feature extraction functionality"""
        model = OptimizedViT(num_classes=TEST_NUM_CLASSES, dim=384, depth=6, heads=6, mlp_dim=1536)
        model.eval()
        
        with torch.no_grad():
            features = model.forward_features(sample_input)
        
        num_patches = (TEST_IMAGE_SIZE // 16) ** 2
        expected_shape = (TEST_BATCH_SIZE, num_patches, model.dim)
        assert features.shape == expected_shape
    
    def test_different_image_sizes(self):
        """Test model with different image sizes"""
        model = OptimizedViT(num_classes=TEST_NUM_CLASSES, dim=192, depth=6, heads=3, mlp_dim=768)
        model.eval()
        
        image_sizes = [224, 256, 320, 384]
        
        for img_size in image_sizes:
            if img_size % 16 == 0:  # Must be divisible by patch size
                x = torch.randn(1, 3, img_size, img_size)
                with torch.no_grad():
                    output = model(x)
                assert output.shape == (1, TEST_NUM_CLASSES)
    
    def test_parameter_count(self):
        """Test parameter counts match expected values"""
        configs = {
            'tiny': (vit_tiny_patch16_224, 5_500_000),
            'small': (vit_small_patch16_224, 22_000_000),
            'base': (vit_base_patch16_224, 86_000_000),
        }
        
        for name, (model_fn, expected_params) in configs.items():
            model = model_fn(num_classes=1000)
            actual_params = sum(p.numel() for p in model.parameters())
            
            # Allow 10% tolerance
            tolerance = expected_params * 0.1
            assert abs(actual_params - expected_params) < tolerance, \
                f"{name}: expected ~{expected_params:,}, got {actual_params:,}"
    
    def test_gradient_flow(self, sample_input):
        """Test that gradients flow properly through the model"""
        model = OptimizedViT(num_classes=TEST_NUM_CLASSES, dim=192, depth=6, heads=3, mlp_dim=768)
        model.train()
        
        # Forward pass
        output = model(sample_input)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                assert grad_norm > 0, f"Zero gradient for parameter: {name}"
        
        assert len(grad_norms) > 0, "No gradients found"
    
    def test_model_serialization(self):
        """Test model can be saved and loaded"""
        model = OptimizedViT(num_classes=TEST_NUM_CLASSES, dim=192, depth=6, heads=3, mlp_dim=768)
        
        # Save model state
        state_dict = model.state_dict()
        
        # Create new model and load state
        model2 = OptimizedViT(num_classes=TEST_NUM_CLASSES, dim=192, depth=6, heads=3, mlp_dim=768)
        model2.load_state_dict(state_dict)
        
        # Test that loaded model works
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        
        assert torch.allclose(out1, out2, atol=1e-6)


class TestTransformerBlock:
    """Test cases for TransformerBlock"""
    
    def test_transformer_block_creation(self):
        """Test TransformerBlock creation"""
        block = TransformerBlock(dim=384, num_heads=6, mlp_ratio=4.0)
        assert isinstance(block, TransformerBlock)
        assert isinstance(block.attn, Attention)
        assert isinstance(block.mlp, MLP)
    
    def test_transformer_block_forward(self):
        """Test TransformerBlock forward pass"""
        block = TransformerBlock(dim=384, num_heads=6, mlp_ratio=4.0)
        x = torch.randn(2, 196, 384)  # [batch, seq_len, dim]
        
        block.eval()
        with torch.no_grad():
            output = block(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestAttention:
    """Test cases for Attention module"""
    
    def test_attention_creation(self):
        """Test Attention module creation"""
        attn = Attention(dim=384, num_heads=6)
        assert isinstance(attn, Attention)
        assert attn.num_heads == 6
        assert attn.head_dim == 64  # 384 / 6
    
    def test_attention_forward(self):
        """Test Attention forward pass"""
        attn = Attention(dim=384, num_heads=6)
        x = torch.randn(2, 196, 384)
        
        attn.eval()
        with torch.no_grad():
            output = attn(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_attention_head_dimension_validation(self):
        """Test that attention validates head dimensions properly"""
        with pytest.raises(AssertionError):
            # 385 is not divisible by 6
            Attention(dim=385, num_heads=6)


class TestMLP:
    """Test cases for MLP module"""
    
    def test_mlp_creation(self):
        """Test MLP creation"""
        mlp = MLP(in_features=384, hidden_features=1536)
        assert isinstance(mlp, MLP)
        assert mlp.fc1.in_features == 384
        assert mlp.fc1.out_features == 1536
        assert mlp.fc2.in_features == 1536
        assert mlp.fc2.out_features == 384
    
    def test_mlp_forward(self):
        """Test MLP forward pass"""
        mlp = MLP(in_features=384, hidden_features=1536)
        x = torch.randn(2, 196, 384)
        
        mlp.eval()
        with torch.no_grad():
            output = mlp(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestDropPath:
    """Test cases for DropPath (Stochastic Depth)"""
    
    def test_drop_path_training(self):
        """Test DropPath in training mode"""
        drop_path = DropPath(drop_prob=0.1)
        drop_path.train()
        
        x = torch.randn(4, 196, 384)
        output = drop_path(x)
        
        assert output.shape == x.shape
        # In training mode with dropout, output should be different
        # (though this is probabilistic)
    
    def test_drop_path_evaluation(self):
        """Test DropPath in evaluation mode"""
        drop_path = DropPath(drop_prob=0.1)
        drop_path.eval()
        
        x = torch.randn(4, 196, 384)
        output = drop_path(x)
        
        # In eval mode, should be identity
        assert torch.equal(output, x)
    
    def test_drop_path_zero_prob(self):
        """Test DropPath with zero probability"""
        drop_path = DropPath(drop_prob=0.0)
        drop_path.train()
        
        x = torch.randn(4, 196, 384)
        output = drop_path(x)
        
        # With zero probability, should always be identity
        assert torch.equal(output, x)


class TestFactoryFunctions:
    """Test model factory functions"""
    
    def test_factory_functions_creation(self):
        """Test that all factory functions create valid models"""
        factory_functions = [
            vit_tiny_patch16_224,
            vit_small_patch16_224,
            vit_base_patch16_224,
            vit_large_patch16_224,
        ]
        
        for fn in factory_functions:
            model = fn(num_classes=100)
            assert isinstance(model, OptimizedViT)
            assert model.num_classes == 100
    
    def test_factory_functions_forward(self):
        """Test factory function models can perform forward pass"""
        x = torch.randn(1, 3, 224, 224)
        
        models = [
            vit_tiny_patch16_224(num_classes=10),
            vit_small_patch16_224(num_classes=10),
        ]  # Test smaller models to avoid memory issues
        
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 10)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_2d_sincos_pos_embed(self):
        """Test 2D sinusoidal position embedding generation"""
        embed_dim = 384
        grid_size = 14  # 14x14 = 196 patches for 224x224 image with 16x16 patches
        
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
        
        expected_shape = (grid_size * grid_size, embed_dim)
        assert pos_embed.shape == expected_shape
        assert not np.isnan(pos_embed).any()
        assert np.isfinite(pos_embed).all()
    
    def test_pos_embed_different_sizes(self):
        """Test position embeddings with different grid sizes"""
        embed_dim = 768
        grid_sizes = [7, 14, 16, 24]
        
        for grid_size in grid_sizes:
            pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
            expected_shape = (grid_size * grid_size, embed_dim)
            assert pos_embed.shape == expected_shape


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_invalid_patch_size(self):
        """Test model creation with invalid patch size"""
        with pytest.raises(AssertionError):
            # Image size not divisible by patch size
            OptimizedViT(image_size=225, patch_size=16)
    
    def test_single_sample_batch(self):
        """Test model with batch size of 1"""
        model = OptimizedViT(num_classes=10, dim=192, depth=6, heads=3, mlp_dim=768)
        x = torch.randn(1, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, 10)
    
    def test_large_batch(self):
        """Test model with larger batch size"""
        model = OptimizedViT(num_classes=10, dim=192, depth=6, heads=3, mlp_dim=768)
        batch_size = 16
        x = torch.randn(batch_size, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 10)
    
    def test_zero_classes(self):
        """Test model with zero classes (feature extractor)"""
        model = OptimizedViT(num_classes=0, dim=192, depth=6, heads=3, mlp_dim=768)
        x = torch.randn(2, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            features = model.forward_features(x)
            output = model(x)  # Should be identity when num_classes=0
        
        pooled_features = features.mean(dim=1)
        assert torch.allclose(output, pooled_features, atol=1e-6)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
