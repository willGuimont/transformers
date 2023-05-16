import numpy as np
import torch
import torch.nn as nn

from transformers.transformers import FeedForward


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, positional_groups: int, dimension_dim: int, fourier_dim: int, hidden_dim: int, pe_dim: int,
                 gamma: float):
        """
        Learnable Fourier Features from "Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding" (Li et al., 2021)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multidimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param positional_groups: positional groups (positions in different groups are independent)
        :param dimension_dim: each point has an M-dimensional positional values
        :param fourier_dim: depth of the Fourier feature dimension
        :param hidden_dim: hidden layer dimension
        :param pe_dim: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        # Parameters as described in the paper
        self.g = positional_groups
        self.m = dimension_dim
        self.f_dim = fourier_dim
        self.h_dim = hidden_dim
        self.d = pe_dim
        self.gamma = gamma

        # Projection matrix on learned lines (used in eq. 2)
        self.wr = nn.Linear(self.m, self.f_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.ff = FeedForward(self.f_dim, self.h_dim, self.d // self.g, dropout=0.0)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape (n, g, m) that represents n positions where each position is in the shape of (g, m),
                  where g is the positional group and each group has m-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        n, g, m = x.shape

        # Step 1. Compute Fourier features (eq. 2)
        projected = self.wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        fourier = 1 / np.sqrt(self.f_dim) * torch.cat([cosines, sines], dim=-1)

        # Step 2. Compute projected Fourier features (eq. 6)
        y = self.ff(fourier)

        # Step 3. Reshape to x's shape
        pe_x = y.reshape((n, self.d)).unsqueeze(0)

        return pe_x
