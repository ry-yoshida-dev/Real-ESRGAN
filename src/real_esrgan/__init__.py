"""Inference-only Real-ESRGAN package."""

from real_esrgan.model_factory import ModelFactory, ModelSpec
from real_esrgan.model_name import ModelName
from real_esrgan.upsampler import AlphaUpsampler, RealESRGANer

__all__ = [
    "AlphaUpsampler",
    "ModelFactory",
    "ModelName",
    "ModelSpec",
    "RealESRGANer",
]

__version__ = "0.3.0"
