"""Compact VGG-style super-resolution architecture."""

from __future__ import annotations

from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F


class ActivationType(str, Enum):
    """Supported activation types for SRVGGNetCompact."""

    RELU = "relu"
    PRELU = "prelu"
    LEAKY_RELU = "leakyrelu"


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style network for efficient super-resolution."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: ActivationType | str = ActivationType.PRELU,
    ) -> None:
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = ActivationType(act_type)
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._build_activation(self.act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._build_activation(self.act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _build_activation(act_type: ActivationType, num_feat: int) -> nn.Module:
        match act_type:
            case ActivationType.RELU:
                return nn.ReLU(inplace=True)
            case ActivationType.PRELU:
                return nn.PReLU(num_parameters=num_feat)
            case ActivationType.LEAKY_RELU:
                return nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = tensor
        for layer in self.body:
            output = layer(output)
        output = self.upsampler(output)
        base = F.interpolate(tensor, scale_factor=self.upscale, mode="nearest")
        return output + base
