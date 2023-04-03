import unittest

import torch

from transformers.variants.parallel_transformer import ParallelTransformerEncoderLayer, ParallelTransformerEncoder, \
    ParallelSelfAttentionTransformerEncoder, ParallelTransformerDecoderLayer, ParallelTransformerDecoder, \
    ParallelTransformer


class TestTransformer(unittest.TestCase):
    def test_parallel_transformer_encoder_layer(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = ParallelTransformerEncoderLayer(d_model, 4, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, x, x, is_causal=True)

        self.assertEqual(x.shape, (batch, num_token, d_model))

    def test_parallel_transformer_encoder(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = ParallelTransformerEncoder(6, d_model, 4, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, x, x, is_causal=True)

        self.assertEqual(x.shape, (batch, num_token, d_model))

    def test_parallel_self_transformer_encoder(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = ParallelSelfAttentionTransformerEncoder(6, d_model, 4, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, is_causal=True)

        self.assertEqual(x.shape, (batch, num_token, d_model))

    def test_parallel_transformer_decoder_layer(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = ParallelTransformerDecoderLayer(d_model, 4, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, x)

        self.assertEqual(x.shape, (batch, num_token, d_model))

    def test_parallel_transformer_decoder(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = ParallelTransformerDecoder(3, d_model, 4, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, x)

        self.assertEqual(x.shape, (batch, num_token, d_model))

    def test_parallel_transformer(self):
        batch = 10
        num_token = 5
        d_model = 128
        out_size = 10

        transformer = ParallelTransformer(6, d_model, 4, out_size, 0.1)

        x = torch.randn((batch, num_token, d_model))
        target = torch.randn((batch, num_token, d_model))
        x = transformer(x, target)

        self.assertEqual(x.shape, (batch, num_token, out_size))
