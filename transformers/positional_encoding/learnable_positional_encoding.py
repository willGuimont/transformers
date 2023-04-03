import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n_tokens: int, d_model: int):
        """
        Learnable positional encoding.
        """
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, n_tokens, d_model))

    def forward(self, x):
        """
        Generate positional encoding to input.
        :param x: input of shape (batch, n_tokens, d_model)
        :return: positional encoding of shape (1, n_tokens, d_model)
        """
        return self.pe
