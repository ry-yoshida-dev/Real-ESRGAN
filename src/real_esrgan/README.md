# real_esrgan

## Overview

Inference-only Python package extracted from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
It keeps Real-ESRGAN-specific inference code in this repository and depends on `basicsr` for shared
components such as `RRDBNet` and checkpoint download utilities.

## Components

| Component | Description |
| --- | --- |
| [`upsampler.py`](upsampler.py) | `RealESRGANer` inference helper for tiled and full-image upscaling. |
| [`model_factory.py`](model_factory.py) | Builds pretrained upsamplers from `ModelName`. |
| [`model_name.py`](model_name.py) | Enum of supported pretrained checkpoint names. |
| [`archs/srvgg.py`](archs/srvgg.py) | `SRVGGNetCompact`, the compact VGG-style architecture used by some Real-ESRGAN models. |

## Examples

```python
from real_esrgan import ModelFactory, ModelName

upsampler = ModelFactory.create_upsampler(
    ModelName.REALESRGAN_X4PLUS,
    weights_dir="weights",
    is_half_precision=True,
)
output, _ = upsampler.enhance(image_bgr)
```
