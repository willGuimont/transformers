import unittest

from transformers.positional_encoding.learnable_positional_encoding import LearnablePositionalEncoding


class TestLearnablePositionalEncoding(unittest.TestCase):
    def test_pos_encoding(self):
        n_token = 10
        d_model = 128
        enc = LearnablePositionalEncoding(n_token, d_model)

        pe = enc(None)

        self.assertEqual((1, n_token, d_model), pe.shape)
