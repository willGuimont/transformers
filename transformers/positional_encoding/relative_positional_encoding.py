import numpy as np
import torch
from torch import nn


def relative_positional_encoding(n_tokens: int, embedding: nn.Embedding,
                                 device: torch.device = torch.device("cpu")):
    """
    Generate positional encoding.
    :param n_tokens: number of tokens
    :param embedding: embedding to use to embed relative position, should have at least `2 * n_tokens - 1` tokens
    :param device: device to store positional encoding
    :return: positional encoding
    """
    # Generate position along sequence
    pos = torch.arange(n_tokens, dtype=torch.float, device=device)
    # Compute relative position, this is an antisymmetric matrix
    offsets = pos[np.newaxis, :] - pos[:, np.newaxis]
    # Move offsets to positive values
    offsets += n_tokens - 1
    # Embed offsets
    pe = embedding(offsets.long())
    return pe


class RelativePositionalEncoding(nn.Module):
    def __init__(self, n_tokens: int, d_model: int):
        """
        Absolute positional encoding.
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.embeddings = nn.Embedding(2 * n_tokens - 1, d_model)

    def forward(self, x):
        """
        Generate positional encoding to input.
        :param x: input of shape (batch, n_tokens, d_model)
        :return: positional encoding of shape (1, n_tokens, d_model)
        """
        pe = relative_positional_encoding(self.n_tokens, self.embeddings, device=x.device)
        return pe
