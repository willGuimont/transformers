from typing import Optional

import torch
import torch.nn as nn

from nnet.transformers import TransformerEncoderLayer, TransformerEncoder, SelfAttentionTransformerEncoder, \
    TransformerDecoderLayer, \
    TransformerDecoder, Transformer, generate_causal_mask


class NoBiasLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None):
        """
        LayerNorm without bias.
        See nn.LayerNorm for more details.
        """
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        # Delete bias and replace it with None
        del self.bias
        self.bias = None


class ParallelTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model: int, n_head: int, dropout: float, multihead_bias: bool = True):
        """
        Transformer encoder layer by "Scaling Vision Transformers to 22 Billion Parameters" from Dehghani et al.
        It uses
        - parallel layer:
            y' = LayerNorm(x)
            y = x + MLP(y') _ Attention(y')
        - QK Normalization
        - No biases on the QKV projection and LayerNorms

        :param d_model: dimension of model
        :param n_head: number of heads
        :param dropout: dropout rate
        :param multihead_bias: use bias on multi-head attention
        """
        # Use NoBiasLayerNorm instead of nn.LayerNorm and do not use bias on multi-head attention
        super().__init__(d_model, n_head, dropout, multihead_bias=multihead_bias, norm_layer=NoBiasLayerNorm)
        # Delete norm on values and replace it with None
        del self.norm1_v
        self.norm1_v = None
        self.attn_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mlp_bias = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, q, k, v, *, mask=None, is_causal=False):
        """
        Forward pass of parallel transformer encoder layer.
        :param q: queries
        :param k: keys
        :param v: values
        :param mask: attention mask. Mutually exclusive with is_causal.
        :param is_causal: is attention causal, prohibits attention to future tokens. Mutually exclusive with mask.
        :return: transformed queries
        """
        # Apply pre-norm
        q, k, v = self.norm1_q(q), self.norm1_k(k), v
        # Apply mlp
        mlp_out = self.ffn(q) + self.mlp_bias
        if mask is None and is_causal:
            mask = generate_causal_mask(q.size(1), q.device)
        # Apply multi-head attention
        attention_out = self.attention(
            q, k, v,
            attn_mask=mask,
            is_causal=is_causal)[0] + self.attn_bias
        # Add mlp output and attention output
        q = q + mlp_out + attention_out
        return q


class ParallelTransformerEncoder(TransformerEncoder):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, dropout: float, multihead_bias: bool = True):
        """
        Stack of parallel transformer encoder layers.
        Allows cross-attention.
        :param num_layers: number of transformer layers
        :param d_model: dimension of model
        :param n_heads: number of heads in each transformer layer
        :param dropout: dropout rate

        """
        # Use ParallelTransformerEncoderLayer instead of TransformerEncoderLayer
        # and do not use bias on multi-head attention
        super().__init__(num_layers, d_model, n_heads, dropout, multihead_bias=multihead_bias,
                         transformer_encoder_layer=ParallelTransformerEncoderLayer)


class ParallelSelfAttentionTransformerEncoder(SelfAttentionTransformerEncoder):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, dropout: float, multihead_bias: bool = True):
        """
        Self-attention parallel transformer encoder.
        Similar to ParallelTransformerEncoder but only use self-attention.
        :param num_layers: number of transformer layers
        :param d_model: dimension of model
        :param n_heads: number of heads in each transformer layer
        :param dropout: dropout rate
        :param multihead_bias: use bias on multi-head attention
        """
        # Use ParallelTransformerEncoderLayer instead of TransformerEncoderLayer
        # and do not use bias on multi-head attention
        super().__init__(num_layers, d_model, n_heads, dropout, multihead_bias=multihead_bias,
                         transformer_encoder_layer=ParallelTransformerEncoderLayer)


class ParallelTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model: int, n_head: int, dropout: float, multihead_bias: bool = True):
        """
        Parallel transformer decoder layer using multi-head attention.
        :param d_model: dimension of model
        :param n_head: number of heads
        :param dropout: dropout rate
        :param multihead_bias: use bias on multi-head attention
        """
        # Use NoBiasLayerNorm instead of nn.LayerNorm and do not use bias on multi-head attention
        super().__init__(d_model, n_head, dropout, multihead_bias=multihead_bias, norm_layer=NoBiasLayerNorm)
        self.attn_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mlp_bias = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, target, memory, *, target_mask=None, target_is_causal=False, memory_mask=None,
                memory_is_causal=False):
        """
        Forward pass of parallel decoder layer.
        :param target: input to decoder
        :param memory: tokens to attend to
        :param target_mask: target attention mask. Mutually exclusive with target_is_causal.
        :param target_is_causal: target is causal, prohibits attention to future tokens. Mutually exclusive with target_mask.
        :param memory_mask: memory attention mask. Mutually exclusive with memory_is_causal.
        :param memory_is_causal: memory is causal, prohibits attention to future tokens. Mutually exclusive with memory_mask.
        :return: transformed target
        """
        # Pre-norm
        target, memory = self.norm1_target(target), self.norm1_memory(memory)
        # Self-attention and residual connection
        target = target + \
                 self.self_attention(target, target, target, attn_mask=target_mask, is_causal=target_is_causal)[0]
        # Cross-attention and residual connection
        target = target + self.cross_attention(self.norm2(target), memory, memory, attn_mask=memory_mask,
                                               is_causal=memory_is_causal)[0]
        # Final feed forward and residual connection
        target = target + self.ffn(self.norm3(target))

        # Apply mlp
        mlp_out = self.ffn(target) + self.mlp_bias
        # Apply multi-head attention
        attention_out = self.cross_attention(self.norm2(target), memory, memory, attn_mask=memory_mask,
                                             is_causal=memory_is_causal)[0] + self.attn_bias
        # Add mlp output and attention output
        target = target + mlp_out + attention_out

        return target


class ParallelTransformerDecoder(TransformerDecoder):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, dropout: float, multihead_bias: bool = True):
        """
        Parallel transformer decoder.
        :param num_layers: number of transformer layers
        :param d_model: dimension of model
        :param n_heads: number of heads in each transformer layer
        :param dropout: dropout rate
        :param multihead_bias: use bias on multi-head attention
        """
        # Use ParallelTransformerDecoderLayer instead of TransformerDecoderLayer
        # and do not use bias on multi-head attention
        super().__init__(num_layers, d_model, n_heads, dropout, multihead_bias=multihead_bias,
                         transformer_decoder_layer=ParallelTransformerDecoderLayer)


class ParallelTransformer(Transformer):
    def __init__(self, num_layers_enc: int, num_layers_dec, d_model: int, n_heads: int, out_size: int, dropout: float,
                 multihead_bias: bool = True, positional_encoding: Optional[nn.Module] = None):
        """
        Transformer with parallel encoder and decoder.
        :param num_layers_enc: number of transformer layers in encoder
        :param num_layers_dec: number of transformer layers in decoder
        :param d_model: dimension of model
        :param n_heads: number of heads in each transformer layer
        :param out_size: output size
        :param dropout: dropout rate
        :param multihead_bias: use bias on multi-head attention
        :param positional_encoding: positional encoding type, defaults to AbsolutePositionalEncoding
        """
        # Use ParallelSelfAttentionTransformerEncoder instead of SelfAttentionTransformerEncoder
        # Use ParallelTransformerDecoder instead of TransformerDecoder
        # and do not use bias on multi-head attention
        super().__init__(num_layers_enc, num_layers_dec, d_model, n_heads, out_size, dropout, multihead_bias=multihead_bias,
                         self_attention_transformer_layer=ParallelSelfAttentionTransformerEncoder,
                         decoder_layer=ParallelTransformerDecoder, positional_encoding=positional_encoding)
