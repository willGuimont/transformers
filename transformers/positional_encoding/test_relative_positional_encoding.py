import unittest

import torch

from transformers.positional_encoding.relative_positional_encoding import relative_positional_encoding, \
    RelativePositionalEncoding


class TestRelativePositionalEncoding(unittest.TestCase):
    def test_positional_encoding(self):
        n_tokens = 10
        d_model = 128
        embedding = torch.nn.Embedding(2 * n_tokens - 1, d_model)
        device = torch.device("cpu")

        pe = relative_positional_encoding(n_tokens, embedding, device)

        self.assertEqual((n_tokens, n_tokens, d_model), pe.shape)

    def test_positional_encoding_class(self):
        batch = 10
        n_tokens = 10
        d_model = 128
        device = torch.device("cpu")
        x = torch.randn((batch, n_tokens, d_model), device=device)

        pe_module = RelativePositionalEncoding(n_tokens, d_model)

        pe = pe_module(x)

        self.assertEqual((n_tokens, n_tokens, d_model), pe.shape)
