"""Core utilities for multi-modal few-shot learning."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic behavior for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for deterministic behavior
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. If None, auto-detect.
        
    Returns:
        PyTorch device object.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        config_path: Path to save configuration.
    """
    OmegaConf.save(config, config_path)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Dictionary with total and trainable parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def format_number(num: int) -> str:
    """Format large numbers with appropriate suffixes.
    
    Args:
        num: Number to format.
        
    Returns:
        Formatted string.
    """
    if num >= 1e9:
        return f"{num / 1e9:.1f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}K"
    else:
        return str(num)


class EarlyStopping:
    """Early stopping utility to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        restore_best_weights: Whether to restore best weights.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            model: Model to potentially restore weights for.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
        """
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
