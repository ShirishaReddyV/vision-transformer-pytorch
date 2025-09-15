"""
Vision Transformer PyTorch Package
Advanced Vision Transformer implementation with state-of-the-art optimizations
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"
__description__ = "Advanced Vision Transformer implementation in PyTorch"

# Import main model classes for easy access
from .model import (
    OptimizedViT,
    TransformerBlock, 
    Attention,
    MLP,
    DropPath,
    vit_tiny_patch16_224,
    vit_small_patch16_224,
    vit_base_patch16_224,
    vit_large_patch16_224,
)

# Import utility functions
from .model import (
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed_from_grid,
)

# Package metadata
__all__ = [
    # Core model classes
    "OptimizedViT",
    "TransformerBlock",
    "Attention", 
    "MLP",
    "DropPath",
    
    # Model factory functions
    "vit_tiny_patch16_224",
    "vit_small_patch16_224", 
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    
    # Utility functions
    "get_2d_sincos_pos_embed",
    "get_2d_sincos_pos_embed_from_grid",
    "get_1d_sincos_pos_embed_from_grid",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
]

# Package configuration
import logging

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Check for required dependencies
try:
    import torch
    if not hasattr(torch, '__version__'):
        logger.warning("PyTorch version could not be determined")
    elif torch.__version__ < "1.12.0":
        logger.warning(f"PyTorch version {torch.__version__} detected. Recommended: >= 1.12.0")
except ImportError:
    logger.error("PyTorch is required but not installed. Please install with: pip install torch torchvision")

try:
    import torchvision
except ImportError:
    logger.warning("Torchvision not found. Some functionality may be limited.")

try:
    import numpy as np
except ImportError:
    logger.error("NumPy is required but not installed. Please install with: pip install numpy")

# Print welcome message
logger.info(f"Vision Transformer PyTorch v{__version__} loaded successfully")
logger.info("For documentation and examples, visit: https://github.com/yourusername/vision-transformer-pytorch")
