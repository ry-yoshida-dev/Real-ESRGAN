"""Real-ESRGAN inference upsampler."""

from __future__ import annotations

import math
import os
from enum import Enum

import cv2
import numpy as np
import torch
from basicsr.utils.download_util import load_file_from_url
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


class AlphaUpsampler(str, Enum):
    """Alpha-channel upsampling strategy."""

    REALESRGAN = "realesrgan"
    BICUBIC = "bicubic"


class RealESRGANer:
    """Upsample images with a pretrained Real-ESRGAN network."""

    def __init__(
        self,
        scale: int,
        model_path: str | list[str],
        model: nn.Module,
        dni_weight: list[float] | None = None,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        is_half_precision: bool = False,
        device: torch.device | None = None,
        gpu_id: int | None = None,
    ) -> None:
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale: int | None = None
        self.is_half_precision = is_half_precision
        self.device = self._resolve_device(device=device, gpu_id=gpu_id)
        loadnet = self._load_checkpoint(model_path=model_path, dni_weight=dni_weight)
        state_key = "params_ema" if "params_ema" in loadnet else "params"
        model.load_state_dict(loadnet[state_key], strict=True)
        model.eval()
        self.model = model.to(self.device)
        if self.is_half_precision:
            self.model = self.model.half()

    @staticmethod
    def _resolve_device(
        device: torch.device | None,
        gpu_id: int | None,
    ) -> torch.device:
        if device is not None:
            return device
        if gpu_id is not None:
            if torch.cuda.is_available():
                return torch.device(f"cuda:{gpu_id}")
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_checkpoint(
        self,
        model_path: str | list[str],
        dni_weight: list[float] | None,
    ) -> dict[str, object]:
        if isinstance(model_path, list):
            if dni_weight is None or len(model_path) != len(dni_weight):
                msg = "model_path and dni_weight must have the same length."
                raise ValueError(msg)
            return self._apply_dni(model_path[0], model_path[1], dni_weight)
        resolved_path = model_path
        if resolved_path.startswith("https://"):
            resolved_path = load_file_from_url(
                url=resolved_path,
                model_dir=os.path.join(_PACKAGE_ROOT, "weights"),
                progress=True,
                file_name=None,
            )
        checkpoint = torch.load(resolved_path, map_location=torch.device("cpu"))
        if not isinstance(checkpoint, dict):
            msg = "Checkpoint must be a state dictionary."
            raise TypeError(msg)
        return checkpoint

    @staticmethod
    def _apply_dni(
        net_a_path: str,
        net_b_path: str,
        dni_weight: list[float],
        key: str = "params",
        loc: str = "cpu",
    ) -> dict[str, object]:
        net_a = torch.load(net_a_path, map_location=torch.device(loc))
        net_b = torch.load(net_b_path, map_location=torch.device(loc))
        if not isinstance(net_a, dict) or not isinstance(net_b, dict):
            msg = "DNI checkpoints must be dictionaries."
            raise TypeError(msg)
        for weight_name, weight_a in net_a[key].items():
            net_a[key][weight_name] = (
                dni_weight[0] * weight_a + dni_weight[1] * net_b[key][weight_name]
            )
        return net_a

    def pre_process(self, image: NDArray[np.float32]) -> None:
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        self.img = tensor.unsqueeze(0).to(self.device)
        if self.is_half_precision:
            self.img = self.img.half()
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        self.mod_scale = None
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        self.mod_pad_h = 0
        self.mod_pad_w = 0
        if self.mod_scale is not None:
            _, _, height, width = self.img.size()
            if height % self.mod_scale != 0:
                self.mod_pad_h = self.mod_scale - height % self.mod_scale
            if width % self.mod_scale != 0:
                self.mod_pad_w = self.mod_scale - width % self.mod_scale
            self.img = F.pad(
                self.img,
                (0, self.mod_pad_w, 0, self.mod_pad_h),
                "reflect",
            )

    def process(self) -> None:
        self.output = self.model(self.img)

    def tile_process(self) -> None:
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        self.output = self.img.new_zeros((batch, channel, output_height, output_width))
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                offset_x = tile_x * self.tile_size
                offset_y = tile_y * self.tile_size
                input_start_x = offset_x
                input_end_x = min(offset_x + self.tile_size, width)
                input_start_y = offset_y
                input_end_y = min(offset_y + self.tile_size, height)
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = tile_y * tiles_x + tile_x + 1
                input_tile = self.img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]
                with torch.no_grad():
                    output_tile = self.model(input_tile)
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale
                self.output[
                    :,
                    :,
                    output_start_y:output_end_y,
                    output_start_x:output_end_x,
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]
                print(f"\tTile {tile_idx}/{tiles_x * tiles_y}")

    def post_process(self) -> torch.Tensor:
        if self.mod_scale is not None:
            _, _, height, width = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : height - self.mod_pad_h * self.scale,
                0 : width - self.mod_pad_w * self.scale,
            ]
        if self.pre_pad != 0:
            _, _, height, width = self.output.size()
            self.output = self.output[
                :,
                :,
                0 : height - self.pre_pad * self.scale,
                0 : width - self.pre_pad * self.scale,
            ]
        return self.output

    @torch.no_grad()
    def enhance(
        self,
        image: NDArray[np.uint8] | NDArray[np.uint16],
        outscale: float | None = None,
        alpha_upsampler: AlphaUpsampler | str = AlphaUpsampler.REALESRGAN,
    ) -> tuple[NDArray[np.uint8] | NDArray[np.uint16], str]:
        height_input, width_input = image.shape[0:2]
        image_float = image.astype(np.float32)
        if np.max(image_float) > 256:
            max_range = 65535
            print("\tInput is a 16-bit image")
        else:
            max_range = 255
        image_float = image_float / max_range
        alpha: NDArray[np.float32] | None = None
        if len(image_float.shape) == 2:
            image_mode = "L"
            image_rgb = cv2.cvtColor(image_float, cv2.COLOR_GRAY2RGB)
        elif image_float.shape[2] == 4:
            image_mode = "RGBA"
            alpha = image_float[:, :, 3]
            image_rgb = cv2.cvtColor(image_float[:, :, 0:3], cv2.COLOR_BGR2RGB)
            upsampler = AlphaUpsampler(alpha_upsampler)
            if upsampler == AlphaUpsampler.REALESRGAN:
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            image_mode = "RGB"
            image_rgb = cv2.cvtColor(image_float, cv2.COLOR_BGR2RGB)
        self.pre_process(image_rgb)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_image = self.post_process()
        output_image = output_image.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_image = np.transpose(output_image[[2, 1, 0], :, :], (1, 2, 0))
        if image_mode == "L":
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
        if image_mode == "RGBA" and alpha is not None:
            upsampler = AlphaUpsampler(alpha_upsampler)
            if upsampler == AlphaUpsampler.REALESRGAN:
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:
                alpha_height, alpha_width = alpha.shape[0:2]
                output_alpha = cv2.resize(
                    alpha,
                    (alpha_width * self.scale, alpha_height * self.scale),
                    interpolation=cv2.INTER_LINEAR,
                )
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)
            output_image[:, :, 3] = output_alpha
        if max_range == 65535:
            output = (output_image * 65535.0).round().astype(np.uint16)
        else:
            output = (output_image * 255.0).round().astype(np.uint8)
        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output,
                (int(width_input * outscale), int(height_input * outscale)),
                interpolation=cv2.INTER_LANCZOS4,
            )
        return output, image_mode
