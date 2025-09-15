"""
Test package for Vision Transformer PyTorch
"""

import sys
import os

# Add the parent directory to the path so we can import the main package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test configuration
import pytest
import torch

# Set random seeds for reproducible tests
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Test device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test constants
TEST_BATCH_SIZE = 2
TEST_IMAGE_SIZE = 224
TEST_NUM_CLASSES = 10
TEST_CHANNELS = 3

print(f"Running tests on device: {DEVICE}")
