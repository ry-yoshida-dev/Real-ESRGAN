# real-esrgan

Inference-only Python package for [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).

## Installation

```bash
pip install .
```

## Usage

```python
from real_esrgan import ModelFactory, ModelName

upsampler = ModelFactory.create_upsampler(
    ModelName.REALESRGAN_X4PLUS,
    weights_dir="weights",
    is_half_precision=True,
)
output, _ = upsampler.enhance(image_bgr)
```

See [`src/real_esrgan/README.md`](src/real_esrgan/README.md) for package details.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
