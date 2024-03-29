import unittest

import torch

from nnet.transformers import FeedForward, TransformerEncoderLayer, TransformerEncoder, \
    SelfAttentionTransformerEncoder, TransformerDecoderLayer, TransformerDecoder, Transformer, AttentionHead, \
    MultiHeadAttention, RelativeTransformerEncoderLayer


class TestTransformer(unittest.TestCase):
    def test_feedforward(self):
        batch = 10
        num_token = 5
        d_model = 128

        ff = FeedForward(d_model, d_model * 4, d_model, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = ff(x)

        self.assertEqual((batch, num_token, d_model), x.shape)

    def test_transformer_encoder_layer(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = TransformerEncoderLayer(d_model, 4, 0.1, False)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, x, x, is_causal=True)

        self.assertEqual((batch, num_token, d_model), x.shape)

    def test_attention_head(self):
        batch = 10
        num_token = 5
        d_model = 128
        head_size = 6

        attn = AttentionHead(d_model, head_size, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = attn(x, x, x)

        self.assertEqual((batch, num_token, head_size), x.shape)

    def test_multihead_attention(self):
        batch = 10
        num_token = 5
        d_model = 128
        head_size = 6
        n_heads = 4

        attn = MultiHeadAttention(d_model, n_heads, head_size, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = attn(x, x, x)

        self.assertEqual((batch, num_token, d_model), x.shape)

    def test_relative_transformer_encoder_layer(self):
        batch = 10
        num_token = 5
        d_model = 128
        n_heads = 4

        trans = RelativeTransformerEncoderLayer(d_model, n_heads, 0.1, False, num_token)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, x, x)

        self.assertEqual((batch, num_token, d_model), x.shape)

    def test_transformer_encoder(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = TransformerEncoder(6, d_model, 4, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, x, x, is_causal=True)

        self.assertEqual((batch, num_token, d_model), x.shape)

    def test_self_transformer_encoder(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = SelfAttentionTransformerEncoder(6, d_model, 4, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, is_causal=True)

        self.assertEqual((batch, num_token, d_model), x.shape)

    def test_transformer_decoder_layer(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = TransformerDecoderLayer(d_model, 4, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, x)

        self.assertEqual((batch, num_token, d_model), x.shape)

    def test_transformer_decoder(self):
        batch = 10
        num_token = 5
        d_model = 128

        trans = TransformerDecoder(3, d_model, 4, 0.1)

        x = torch.randn((batch, num_token, d_model))
        x = trans(x, x)

        self.assertEqual((batch, num_token, d_model), x.shape)

    def test_transformer(self):
        batch = 10
        num_token = 5
        d_model = 128
        out_size = 10

        transformer = Transformer(6, 6, d_model, 4, out_size, 0.1)

        x = torch.randn((batch, num_token, d_model))
        target = torch.randn((batch, num_token, d_model))
        x = transformer(x, target)

        self.assertEqual((batch, num_token, out_size), x.shape)
