"""Predefined Real-ESRGAN checkpoint identifiers."""

from __future__ import annotations

from enum import Enum


class ModelName(str, Enum):
    """Supported pretrained Real-ESRGAN model names."""

    REALESRGAN_X4PLUS = "RealESRGAN_x4plus"
    REALESRNET_X4PLUS = "RealESRNet_x4plus"
    REALESRGAN_X4PLUS_ANIME_6B = "RealESRGAN_x4plus_anime_6B"
    REALESRGAN_X2PLUS = "RealESRGAN_x2plus"
    REALESR_ANIME_VIDEOV3 = "realesr-animevideov3"
    REALESR_GENERAL_X4V3 = "realesr-general-x4v3"
    REALESR_GENERAL_WDN_X4V3 = "realesr-general-wdn-x4v3"
