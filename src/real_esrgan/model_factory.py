"""Factory helpers for building Real-ESRGAN inference models."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from torch import nn

from real_esrgan.archs.srvgg import ActivationType, SRVGGNetCompact
from real_esrgan.model_name import ModelName
from real_esrgan.upsampler import RealESRGANer


@dataclass(frozen=True)
class ModelSpec:
    """Static configuration for a pretrained Real-ESRGAN checkpoint."""

    model: nn.Module
    net_scale: int
    weight_urls: tuple[str, ...]


class ModelFactory:
    """Build pretrained Real-ESRGAN inference pipelines."""

    _RELEASE_BASE_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download"

    @classmethod
    def build_spec(cls, model_name: ModelName) -> ModelSpec:
        """Create the network and metadata for a model name.

        Parameters
        ----------
        model_name
            Pretrained checkpoint identifier.

        Returns
        -------
        ModelSpec
            Network instance and checkpoint metadata.
        """
        match model_name:
            case ModelName.REALESRGAN_X4PLUS:
                return ModelSpec(
                    model=RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=4,
                    ),
                    net_scale=4,
                    weight_urls=(f"{cls._RELEASE_BASE_URL}/v0.1.0/RealESRGAN_x4plus.pth",),
                )
            case ModelName.REALESRNET_X4PLUS:
                return ModelSpec(
                    model=RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=4,
                    ),
                    net_scale=4,
                    weight_urls=(f"{cls._RELEASE_BASE_URL}/v0.1.1/RealESRNet_x4plus.pth",),
                )
            case ModelName.REALESRGAN_X4PLUS_ANIME_6B:
                return ModelSpec(
                    model=RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=6,
                        num_grow_ch=32,
                        scale=4,
                    ),
                    net_scale=4,
                    weight_urls=(
                        f"{cls._RELEASE_BASE_URL}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
                    ),
                )
            case ModelName.REALESRGAN_X2PLUS:
                return ModelSpec(
                    model=RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=2,
                    ),
                    net_scale=2,
                    weight_urls=(f"{cls._RELEASE_BASE_URL}/v0.2.1/RealESRGAN_x2plus.pth",),
                )
            case ModelName.REALESR_ANIME_VIDEOV3:
                return ModelSpec(
                    model=SRVGGNetCompact(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_conv=16,
                        upscale=4,
                        act_type=ActivationType.PRELU,
                    ),
                    net_scale=4,
                    weight_urls=(
                        f"{cls._RELEASE_BASE_URL}/v0.2.5.0/realesr-animevideov3.pth",
                    ),
                )
            case ModelName.REALESR_GENERAL_X4V3:
                return ModelSpec(
                    model=SRVGGNetCompact(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_conv=32,
                        upscale=4,
                        act_type=ActivationType.PRELU,
                    ),
                    net_scale=4,
                    weight_urls=(
                        f"{cls._RELEASE_BASE_URL}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                        f"{cls._RELEASE_BASE_URL}/v0.2.5.0/realesr-general-x4v3.pth",
                    ),
                )
            case ModelName.REALESR_GENERAL_WDN_X4V3:
                return ModelSpec(
                    model=SRVGGNetCompact(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_conv=32,
                        upscale=4,
                        act_type=ActivationType.PRELU,
                    ),
                    net_scale=4,
                    weight_urls=(
                        f"{cls._RELEASE_BASE_URL}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                    ),
                )

    @classmethod
    def resolve_weight_path(
        cls,
        model_name: ModelName,
        weights_dir: str,
        model_path: str | None = None,
    ) -> str | list[str]:
        """Resolve a local checkpoint path, downloading weights when needed.

        Parameters
        ----------
        model_name
            Pretrained checkpoint identifier.
        weights_dir
            Directory used for downloaded checkpoints.
        model_path
            Optional explicit checkpoint path. When provided, no download occurs.

        Returns
        -------
        str | list[str]
            Local checkpoint path or DNI path pair.
        """
        if model_path is not None:
            return model_path
        spec = cls.build_spec(model_name)
        default_path = os.path.join(weights_dir, f"{model_name.value}.pth")
        if os.path.isfile(default_path):
            return default_path
        resolved_path = default_path
        for url in spec.weight_urls:
            resolved_path = load_file_from_url(
                url=url,
                model_dir=weights_dir,
                progress=True,
                file_name=None,
            )
        return resolved_path

    @classmethod
    def create_upsampler(
        cls,
        model_name: ModelName,
        *,
        model_path: str | None = None,
        weights_dir: str = "weights",
        denoise_strength: float = 0.5,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        is_half_precision: bool = True,
        device: torch.device | None = None,
        gpu_id: int | None = None,
    ) -> RealESRGANer:
        """Create a ready-to-run Real-ESRGAN upsampler.

        Parameters
        ----------
        model_name
            Pretrained checkpoint identifier.
        model_path
            Optional explicit checkpoint path.
        weights_dir
            Directory used for downloaded checkpoints.
        denoise_strength
            Denoise strength for ``realesr-general-x4v3`` via DNI.
        tile
            Tile size for tiled inference. ``0`` disables tiling.
        tile_pad
            Padding size for each tile.
        pre_pad
            Reflect padding applied before inference.
        is_half_precision
            Whether to run inference in half precision.
        device
            Optional explicit torch device.
        gpu_id
            Optional CUDA device index.

        Returns
        -------
        RealESRGANer
            Configured inference helper.
        """
        spec = cls.build_spec(model_name)
        resolved_path = cls.resolve_weight_path(
            model_name=model_name,
            weights_dir=weights_dir,
            model_path=model_path,
        )
        dni_weight: list[float] | None = None
        if model_name == ModelName.REALESR_GENERAL_X4V3 and denoise_strength != 1.0:
            if isinstance(resolved_path, str):
                wdn_model_path = resolved_path.replace(
                    ModelName.REALESR_GENERAL_X4V3.value,
                    ModelName.REALESR_GENERAL_WDN_X4V3.value,
                )
                resolved_path = [resolved_path, wdn_model_path]
                dni_weight = [denoise_strength, 1.0 - denoise_strength]
        return RealESRGANer(
            scale=spec.net_scale,
            model_path=resolved_path,
            dni_weight=dni_weight,
            model=spec.model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            is_half_precision=is_half_precision,
            device=device,
            gpu_id=gpu_id,
        )
