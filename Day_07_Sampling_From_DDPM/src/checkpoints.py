"""
Checkpoint utilities for resolving and loading multiple DDPM checkpoints.
Handles EMA vs raw weights, epoch resolution, and model compatibility checking.
"""

import os
import re
import torch
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from .utils import load_checkpoint


def resolve_checkpoint_paths(ckpt_dir: Union[str, Path], 
                           ckpt_list: List[str],
                           verbose: bool = True) -> List[Path]:
    """
    Resolve checkpoint paths from directory and list of checkpoint names.
    
    Args:
        ckpt_dir: Directory containing checkpoints
        ckpt_list: List of checkpoint filenames or patterns
        verbose: Whether to print resolution info
        
    Returns:
        List of resolved checkpoint paths
    """
    ckpt_dir = Path(ckpt_dir)
    resolved_paths = []
    
    if verbose:
        print(f"Resolving checkpoints in: {ckpt_dir}")
    
    for ckpt_name in ckpt_list:
        ckpt_path = ckpt_dir / ckpt_name
        
        if ckpt_path.exists():
            resolved_paths.append(ckpt_path)
            if verbose:
                print(f"  ✓ Found: {ckpt_name}")
        else:
            if verbose:
                print(f"  ✗ Missing: {ckpt_name}")
    
    if not resolved_paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir} from list: {ckpt_list}")
    
    return resolved_paths


def extract_epoch_from_filename(filename: str) -> Optional[int]:
    """
    Extract epoch number from checkpoint filename.
    
    Args:
        filename: Checkpoint filename (e.g., "epoch_42.pt", "ema_epoch_100.pt")
        
    Returns:
        Epoch number if found, None otherwise
    """
    # Try various patterns
    patterns = [
        r'epoch[_-](\d+)',
        r'ep[_-](\d+)', 
        r'step[_-](\d+)',
        r'(\d+)(?:\.pt|\.pth|\.ckpt)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None


def sort_checkpoints_by_epoch(ckpt_paths: List[Path]) -> List[Path]:
    """Sort checkpoint paths by epoch number."""
    def get_sort_key(path: Path) -> Tuple[int, str]:
        epoch = extract_epoch_from_filename(path.name)
        if epoch is not None:
            return (epoch, path.name)
        else:
            # Put non-epoch checkpoints (like ema.pt) at the end
            return (float('inf'), path.name)
    
    return sorted(ckpt_paths, key=get_sort_key)


def load_model_checkpoint(ckpt_path: Path, model: torch.nn.Module, 
                         device: Optional[torch.device] = None,
                         use_ema: bool = True,
                         strict: bool = True) -> Dict[str, Any]:
    """
    Load model weights from checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load checkpoint on
        use_ema: Whether to prefer EMA weights if available
        strict: Whether to strictly match state dict keys
        
    Returns:
        Checkpoint metadata (epoch, step, etc.)
    """
    checkpoint = load_checkpoint(ckpt_path, device)
    
    # Extract model state dict - handle various checkpoint formats
    model_state = None
    
    if use_ema and 'model_ema' in checkpoint:
        model_state = checkpoint['model_ema']
        print(f"Loaded EMA weights from {ckpt_path.name}")
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
        print(f"Loaded model weights from {ckpt_path.name}")
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
        print(f"Loaded state_dict from {ckpt_path.name}")
    elif 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        print(f"Loaded model_state_dict from {ckpt_path.name}")
    else:
        # Assume the entire checkpoint is the model state
        model_state = checkpoint
        print(f"Loaded direct state from {ckpt_path.name}")
    
    # Load weights into model
    if model_state is not None:
        try:
            model.load_state_dict(model_state, strict=strict)
        except RuntimeError as e:
            if strict:
                raise RuntimeError(f"Failed to load checkpoint {ckpt_path}: {e}")
            else:
                print(f"Warning: Some keys mismatched when loading {ckpt_path}: {e}")
                model.load_state_dict(model_state, strict=False)
    
    # Extract metadata
    metadata = {
        'checkpoint_path': str(ckpt_path),
        'epoch': checkpoint.get('epoch', extract_epoch_from_filename(ckpt_path.name)),
        'step': checkpoint.get('step', None),
        'loss': checkpoint.get('loss', None),
        'ema_used': use_ema and 'model_ema' in checkpoint,
    }
    
    return metadata


class CheckpointManager:
    """Manager for handling multiple checkpoints and model loading."""
    
    def __init__(self, ckpt_dir: Union[str, Path], ckpt_list: List[str],
                 use_ema: bool = True, device: Optional[torch.device] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            ckpt_dir: Directory containing checkpoints
            ckpt_list: List of checkpoint filenames
            use_ema: Whether to prefer EMA weights
            device: Device to load checkpoints on
        """
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_list = ckpt_list
        self.use_ema = use_ema
        self.device = device
        
        # Resolve and sort checkpoint paths
        self.ckpt_paths = resolve_checkpoint_paths(ckpt_dir, ckpt_list)
        self.ckpt_paths = sort_checkpoints_by_epoch(self.ckpt_paths)
        
        print(f"CheckpointManager initialized with {len(self.ckpt_paths)} checkpoints")
        for i, path in enumerate(self.ckpt_paths):
            epoch = extract_epoch_from_filename(path.name)
            epoch_str = f"epoch {epoch}" if epoch is not None else "unknown epoch"
            print(f"  {i}: {path.name} ({epoch_str})")
    
    def load_checkpoint_into_model(self, model: torch.nn.Module, 
                                  ckpt_idx: int = -1) -> Dict[str, Any]:
        """
        Load specific checkpoint into model.
        
        Args:
            model: Model to load weights into
            ckpt_idx: Index of checkpoint to load (-1 for latest)
            
        Returns:
            Checkpoint metadata
        """
        if ckpt_idx < 0:
            ckpt_idx = len(self.ckpt_paths) + ckpt_idx
        
        if not (0 <= ckpt_idx < len(self.ckpt_paths)):
            raise ValueError(f"Checkpoint index {ckpt_idx} out of range [0, {len(self.ckpt_paths)})")
        
        ckpt_path = self.ckpt_paths[ckpt_idx]
        return load_model_checkpoint(ckpt_path, model, self.device, self.use_ema)
    
    def iterate_checkpoints(self, model: torch.nn.Module):
        """
        Generator that yields (checkpoint_metadata, model) for each checkpoint.
        
        Args:
            model: Model to load checkpoints into
            
        Yields:
            Tuple of (metadata, model) for each checkpoint
        """
        for i, ckpt_path in enumerate(self.ckpt_paths):
            metadata = load_model_checkpoint(ckpt_path, model, self.device, self.use_ema)
            metadata['checkpoint_index'] = i
            yield metadata, model
    
    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get information about all checkpoints without loading them."""
        info_list = []
        
        for i, ckpt_path in enumerate(self.ckpt_paths):
            info = {
                'index': i,
                'path': str(ckpt_path),
                'filename': ckpt_path.name,
                'epoch': extract_epoch_from_filename(ckpt_path.name),
                'size_mb': ckpt_path.stat().st_size / 1024 / 1024,
                'exists': ckpt_path.exists(),
            }
            info_list.append(info)
        
        return info_list


def verify_model_compatibility(ckpt_path: Path, model_config: Dict[str, Any]) -> bool:
    """
    Verify that a checkpoint is compatible with the given model configuration.
    
    Args:
        ckpt_path: Path to checkpoint
        model_config: Model configuration dict
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        checkpoint = load_checkpoint(ckpt_path)
        
        # Check if model config is stored in checkpoint
        if 'model_config' in checkpoint:
            ckpt_config = checkpoint['model_config']
            
            # Check key architectural parameters
            key_params = ['in_ch', 'base_ch', 'ch_mult', 'time_embed_dim']
            for param in key_params:
                if param in model_config and param in ckpt_config:
                    if model_config[param] != ckpt_config[param]:
                        print(f"Config mismatch for {param}: {model_config[param]} vs {ckpt_config[param]}")
                        return False
        
        return True
        
    except Exception as e:
        print(f"Error verifying compatibility: {e}")
        return False


def find_latest_checkpoint(ckpt_dir: Union[str, Path], pattern: str = "*.pt") -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        ckpt_dir: Directory to search
        pattern: Glob pattern for checkpoint files
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = list(ckpt_dir.glob(pattern))
    
    if not ckpt_files:
        return None
    
    # Sort by modification time
    latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    return latest_ckpt


def get_checkpoint_summary(ckpt_paths: List[Path]) -> str:
    """Generate a summary report of checkpoints."""
    if not ckpt_paths:
        return "No checkpoints found."
    
    summary = f"Checkpoint Summary ({len(ckpt_paths)} checkpoints):\n"
    summary += "=" * 50 + "\n"
    
    for i, path in enumerate(ckpt_paths):
        epoch = extract_epoch_from_filename(path.name)
        size_mb = path.stat().st_size / 1024 / 1024
        
        summary += f"{i+1:2d}. {path.name}\n"
        summary += f"    Epoch: {epoch if epoch is not None else 'Unknown'}\n"
        summary += f"    Size:  {size_mb:.1f} MB\n"
        summary += f"    Path:  {path}\n\n"
    
    return summary
