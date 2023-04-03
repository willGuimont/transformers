import torch
import torch.nn as nn


def positional_encoding(n_tokens: int, d_model: int, device: torch.device = torch.device("cpu")):
    """
    Generate positional encoding.
    :param n_tokens: number of tokens
    :param d_model: dimension of model
    :param device: device to store positional encoding
    :return: positional encoding
    """
    # Generate position along sequence
    pos = torch.arange(n_tokens, dtype=torch.float, device=device).reshape(1, -1, 1)
    # Generate dimension along embedding
    dim = torch.arange(d_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    # Compute phase
    phase = pos / (10000 ** (dim / d_model))
    # Compute positional encoding as described in "Attention is all you need"
    pe = torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
    return pe


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self):
        """
        Absolute positional encoding.
        """
        super().__init__()

    def forward(self, x):
        """
        Generate positional encoding to input.
        :param x: input of shape (batch, n_tokens, d_model)
        :return: positional encoding of shape (1, n_tokens, d_model)
        """
        batch, n_tokens, d_model = x.shape
        pe = positional_encoding(n_tokens, d_model, device=x.device)
        return pe
