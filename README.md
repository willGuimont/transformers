# transformers

Collection of easy to understand transformer-based models in PyTorch.
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
- Relative positional encoding
- Learnable positional encoding
- Learnable Fourier positional encoding (Li, 2021)

Vision:

- VisionTransformer (Dosovitskiy et al., 2021)

NLP:

- Simple character-level Transformer language model

## Next steps

- Rotary positional encoding https://arxiv.org/pdf/2104.09864.pdf
- Optimizing Deeper Transformers on Small Datasets (Xu et al., 2021)
- Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2016)
- Universal Transformers [Paper](https://arxiv.org/abs/2310.07096)
- Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles (Ryali et al., 2023)
- Swin Transformer (Liu et al., 2021)
- DINO: Emerging Properties in Self-Supervised Vision Transformers (Caron et al., 2021)
- FlashAttention (Dao et al., 2022)
- DETR (Carion et al., 2020)
- Unlimiformer: Long-Range Transformers with Unlimited Length Input (Bertsch et al., 2023)
- PointBERT (Yu et al., 2022)
- Hydra Attention: Efficient Attention with Many Heads (Bolya et al., 2022)
- Hyena Hierarchy: Towards Larger Convolutional Language Models (Poli et al., 2023)
- Thinking Like Transformers (Weiss et al., 2021)
- Long short-term memory (Schmidhuber, 1997)
- Rethinking Positional Encoding in Language Pre-training (Ke et al., 2021)

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
