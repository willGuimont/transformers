import unittest

import torch

from nnet.positional_encoding.absolute_positional_encoding import AbsolutePositionalEncoding, \
    absolute_positional_encoding


class TestAbsolutePositionalEncoding(unittest.TestCase):
    def test_positional_encoding(self):
        n_tokens = 10
        d_model = 128
        device = torch.device("cpu")

        pe = absolute_positional_encoding(n_tokens, d_model, device)

        self.assertEqual((1, n_tokens, d_model), pe.shape)

    def test_positional_encoding_class(self):
        batch = 10
        n_tokens = 10
        d_model = 128
        device = torch.device("cpu")
        x = torch.randn((batch, n_tokens, d_model), device=device)

        pe_module = AbsolutePositionalEncoding()

        pe = pe_module(x)

        self.assertEqual((1, n_tokens, d_model), pe.shape)
