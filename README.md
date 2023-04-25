# transformers

Collection easy to understand PyTorch transformer-based models.
The implementation is heavily commented and should be easy to follow.

## Installation

```bash
pip install git+https://github.com/willGuimont/transformers
```

## Implemented models

General:

- Transformer (Vaswani et al., 2017)
- Parallel Transformer (Dehghani et al., 2023)
- PerceiverIO (Jaegle et al., 2022)

Positional encoding:

- Sinusoidal positional encoding
- Learnable positional encoding
- Learnable Fourier positional encoding

Vision:

- VisionTransformer (Dosovitskiy et al., 2021)

NLP:

- Simple character-level Transformer language model

## Next steps

- Swin Transformer (Liu et al., 2021)
- DETR (Carion et al., 2020)
- PointBERT (Yu et al., 2022)
- Hyena Hierarchy: Towards Larger Convolutional Language Models (Poli et al., 2023)
- Thinking Like Transformers (Weiss et al., 2021)

## Cite this repository

```
@software{Guimont-Martin_transformer_flexible_and_2023,
    author = {Guimont-Martin, William},
    month = {2},
    title = {{transformer: flexible and easy to understand transformer models}},
    version = {0.1.0},
    year = {2023}
}
```
