"""
Model architectures for Day 2: Denoising Autoencoder
"""

from .dae_conv import ConvDAE
from .dae_unet import UNetDAE

__all__ = ['ConvDAE', 'UNetDAE']


def create_model(
    model_name: str,
    in_ch: int,
    out_ch: int,
    **kwargs
):
    """Factory function to create model instances."""
    
    if model_name.lower() == "conv":
        return ConvDAE(in_ch=in_ch, out_ch=out_ch, **kwargs)
    elif model_name.lower() == "unet":
        return UNetDAE(in_ch=in_ch, out_ch=out_ch, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")