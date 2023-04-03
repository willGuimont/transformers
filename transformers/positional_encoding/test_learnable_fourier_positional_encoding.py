import unittest

import torch

from transformers.positional_encoding.learnable_fourier_positional_encoding import LearnableFourierPositionalEncoding


class TestLearnableFourierPositionalEncoding(unittest.TestCase):
    def test_fourier_pos_encoding(self):
        n_pos = 9
        positional_groups = 4
        pos_dim = 8
        fourier_dim = 384
        hidden_dim = 32
        pe_dim = 512
        enc = LearnableFourierPositionalEncoding(positional_groups, pos_dim, fourier_dim, hidden_dim, pe_dim, gamma=10)

        x = torch.randn((n_pos, positional_groups, pos_dim))
        pe = enc(x)

        self.assertEqual((1, n_pos, pe_dim), pe.shape)
