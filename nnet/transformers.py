from typing import Callable, Optional

import pipeline as pp
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet.positional_encoding.absolute_positional_encoding import AbsolutePositionalEncoding
from nnet.positional_encoding.relative_positional_encoding import RelativePositionalEncoding


class FeedForward(nn.Module):
    def __init__(self, in_channel: int, mid_channels: int, out_channels: int, dropout: float):
        """
        Simple feed forward network
        :param in_channel: input channel
        :param mid_channels: number of hidden channels
        :param out_channels: output channel
        :param dropout: dropout rate
        """
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_channel, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


def generate_causal_mask(size: int, device: torch.device):
    """
    Generate causal mask.
    :param size: size of mask
    :param device: device to put mask on
    :return: causal mask
    """
    return torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, multihead_bias: bool,
                 norm_layer: Callable = nn.LayerNorm):
        """
        Transformer encoder layer using multi-head attention.
        This implementation uses pre-norm instead of post-norm.
        :param d_model: dimension of model
        :param n_head: number of heads
        :param dropout: dropout rate
        :param multihead_bias: whether to use bias in multi-head attention
        :param norm_layer: normalization layer type, defaults to LayerNorm
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True, bias=multihead_bias)
        self.ffn = FeedForward(d_model, d_model * 4, d_model, dropout)
        self.norm1_q = norm_layer(d_model)
        self.norm1_k = norm_layer(d_model)
        self.norm1_v = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)

    def forward(self, q, k, v, *, mask=None, is_causal=False):
        """
        Forward pass of transformer encoder layer.
        :param q: queries
        :param k: keys
        :param v: values
        :param mask: attention mask. Mutually exclusive with is_causal.
        :param is_causal: is attention causal, prohibits attention to future tokens. Mutually exclusive with mask.
        :return: transformed queries
        """
        # Apply pre-norm
        q, k, v = self.norm1_q(q), self.norm1_k(k), self.norm1_v(v)
        # Multi-head attention and residual connection
        if mask is None and is_causal:
            mask = generate_causal_mask(q.size(1), q.device)
        q = q + self.attention(
            q, k, v,
            attn_mask=mask,
            is_causal=is_causal)[0]
        # Feed forward network and residual connection
        q = q + self.ffn(self.norm2(q))
        return q


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, head_size: int, dropout: float, attention_operation: Callable = pp.Identity()):
        """
        Single head attention that allows to apply operation to the attention matrix before masking and softmax.
        :param d_model: dimension of embedding
        :param head_size: size of head
        :param dropout: dropout rate
        :param attention_operation: operation to apply to attention matrix before masking and softmax
        """
        super().__init__()
        self.q_proj = nn.Linear(d_model, head_size, bias=False)
        self.k_proj = nn.Linear(d_model, head_size, bias=False)
        self.v_proj = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_operation = attention_operation

    def forward(self, q, k, v, attn_mask=None, is_causal=False):
        if is_causal is not False:
            raise ValueError("AttentionHead does not support is_causal")

        B, T, C = q.shape
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        attention = q.bmm(k.transpose(-2, -1))
        attention = self.attention_operation(attention)

        if attn_mask is not None:
            attention[:, attn_mask[:T, :T].logical_not()] = -torch.inf

        scale = q.size(-1) ** 0.5
        softmax = F.softmax(attention / scale, dim=-1)
        softmax = self.dropout(softmax)

        return softmax.bmm(v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, head_size: int, dropout: float,
                 attention_layer: Callable = AttentionHead, attention_layer_params: Optional[dict] = None):
        """
        Multi-head attention layer.
        :param d_model: dimension of model
        :param n_head: number of heads
        :param head_size: size of each head
        :param dropout: dropout rate
        :param attention_layer: attention layer type
        :param attention_layer_params: attention layer parameters
        """
        super().__init__()
        attention_layer_params = attention_layer_params or {}
        self.heads = nn.ModuleList(
            [attention_layer(d_model, head_size, dropout, **attention_layer_params) for _ in range(n_head)])
        self.fc = nn.Linear(n_head * head_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, is_causal=False):
        if is_causal is not False:
            raise ValueError("AttentionHead does not support is_causal")

        out = torch.cat([h(q, k, v, attn_mask) for h in self.heads], dim=-1)
        return self.dropout(self.fc(out))


class RelativeTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model: int, n_head: int, dropout: float, multihead_bias: bool, n_token: int,
                 norm_layer: Callable = nn.LayerNorm):
        super().__init__(d_model, n_head, dropout, multihead_bias, norm_layer)
        self.attention = MultiHeadAttention(d_model, n_head, d_model // n_head, dropout, AttentionHead,
                                            dict(attention_operation=RelativePositionalEncoding(n_token)))

    def forward(self, q, k, v, *, mask=None, is_causal=False):
        if is_causal is not False:
            raise ValueError('is_causal is not supported for RelativeTransformerEncoderLayer')
        return super().forward(q, k, v, mask=mask)


def _init_transformer_weights(module):
    """
    Initialize weights of linear and embedding layers.
    We use normal distribution with mean 0 and std 0.02 for linear layers with bias.
    This initialization help with optimizing the transformer.
    :param module: module to initialize
    :return: None
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, dropout: float,
                 multihead_bias: bool = True, transformer_encoder_layer: Callable = TransformerEncoderLayer,
                 transformer_encoder_layer_params: Optional[dict] = None):
        """
        Stack of transformer encoder layers.
        Allows cross-attention.
        :param num_layers: number of transformer layers
        :param d_model: dimension of model
        :param n_heads: number of heads in each transformer layer
        :param dropout: dropout rate
        :param multihead_bias: whether to use bias in multi-head attention
        :param transformer_encoder_layer: transformer encoder layer type, defaults to TransformerEncoderLayer
        :param transformer_encoder_layer_params: additional parameters to `transformer_encoder_layer`
        """
        super().__init__()
        transformer_encoder_layer_params = transformer_encoder_layer_params or {}
        self.layers = nn.ModuleList(
            [transformer_encoder_layer(d_model, n_heads, dropout, multihead_bias, **transformer_encoder_layer_params)
             for
             _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.apply(_init_transformer_weights)

    def forward(self, q, k, v, *, mask=None, is_causal=False):
        """
        Forward pass of transformer encoder.
        :param q: queries
        :param k: keys
        :param v: values
        :param mask: attention mask. Mutually exclusive with is_causal.
        :param is_causal: is attention causal, prohibits attention to future tokens. Mutually exclusive with mask.
        :return: transformed queries
        """
        # Apply each layer
        for layer in self.layers:
            # Cross-attention from q to (k, v)
            q = layer(q, k, v, mask=mask, is_causal=is_causal)
        # Final layer norm
        q = self.norm(q)
        return q


class SelfAttentionTransformerEncoder(TransformerEncoder):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, dropout: float,
                 multihead_bias: bool = True, transformer_encoder_layer: Callable = TransformerEncoderLayer,
                 transformer_encoder_layer_params: Optional[dict] = None):
        """
        Self-attention transformer encoder.
        Similar to TransformerEncoder but only use self-attention.
        :param num_layers: number of transformer layers
        :param d_model: dimension of model
        :param n_heads: number of heads in each transformer layer
        :param dropout: dropout rate
        :param multihead_bias: whether to use bias in multi-head attention
        :param transformer_encoder_layer: transformer encoder layer type, defaults to TransformerEncoderLayer
        :param transformer_encoder_layer_params: additional parameters to `transformer_encoder_layer`
        """
        super().__init__(num_layers, d_model, n_heads, dropout, multihead_bias, transformer_encoder_layer,
                         transformer_encoder_layer_params)

    def forward(self, x, *, mask=None, is_causal=False):
        """
        Forward pass of self-attention transformer encoder.
        :param x: tokens
        :param mask: attention mask. Mutually exclusive with is_causal.
        :param is_causal: is attention causal, prohibits attention to future tokens. Mutually exclusive with mask.
        :return: transformed tokens
        """
        # Apply each layer
        for layer in self.layers:
            # Self-attention
            x = layer(x, x, x, mask=mask, is_causal=is_causal)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, multihead_bias: bool = True,
                 norm_layer: Callable = nn.LayerNorm):
        """
        Transformer decoder layer using multi-head attention.
        :param d_model: dimension of model
        :param n_head: number of heads
        :param dropout: dropout rate
        :param multihead_bias: whether to use bias in multi-head attention
        :param norm_layer: normalization layer, defaults to nn.LayerNorm
        """
        super().__init__()
        # Self-attention block
        self.self_attention = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True, bias=multihead_bias)
        self.norm1_target = norm_layer(d_model)
        self.norm1_memory = norm_layer(d_model)

        # Cross-attention block
        self.norm2 = norm_layer(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_head, dropout, batch_first=True, bias=multihead_bias)

        #  Final feed forward block
        self.norm3 = norm_layer(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, d_model * 4, d_model, dropout)

    def forward(self, target, memory, *, target_mask=None, target_is_causal=False, memory_mask=None,
                memory_is_causal=False):
        """
        Forward pass of decoder layer.
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
        return target


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_heads: int, dropout: float,
                 multihead_bias: bool = True,
                 transformer_decoder_layer: Callable = TransformerDecoderLayer,
                 transformer_decoder_layer_params: Optional[dict] = None):
        """
        Transformer decoder.
        :param num_layers: number of transformer layers
        :param d_model: dimension of model
        :param n_heads: number of heads in each transformer layer
        :param dropout: dropout rate
        :param multihead_bias: whether to use bias in multi-head attention
        :param transformer_decoder_layer: transformer decoder layer type, defaults to TransformerDecoderLayer
        :param transformer_decoder_layer_params: additional parameters to `transformer_decoder_layer`
        """
        super().__init__()
        if transformer_decoder_layer_params is None:
            transformer_decoder_layer_params = {}
        self.layers = nn.ModuleList(
            [transformer_decoder_layer(d_model, n_heads, dropout, multihead_bias, **transformer_decoder_layer_params)
             for
             _ in
             range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        self.apply(_init_transformer_weights)

    def forward(self, target, memory, *, target_mask=None, target_is_causal=False, memory_mask=None,
                memory_is_causal=False):
        """
        Forward pass of transformer decoder.
        :param target: input to decoder
        :param memory: memory to attend to
        :param target_mask: target attention mask. Mutually exclusive with target_is_causal.
        :param target_is_causal: target is causal, prohibits attention to future tokens. Mutually exclusive with target_mask.
        :param memory_mask: memory attention mask. Mutually exclusive with memory_is_causal.
        :param memory_is_causal: memory is causal, prohibits attention to future tokens. Mutually exclusive with memory_mask.
        :return: transformed target
        """
        # Apply each layer
        for layer in self.layers:
            target = layer(target, memory, target_mask=target_mask, target_is_causal=target_is_causal,
                           memory_mask=memory_mask, memory_is_causal=memory_is_causal)
        # Final layer norm
        target = self.norm(target)
        return target


class Transformer(nn.Module):
    def __init__(self, num_layers_enc: int, num_layers_dec: int, d_model: int, n_heads: int, out_size: Optional[int], dropout: float,
                 multihead_bias: bool = True,
                 self_attention_transformer_layer: Callable = SelfAttentionTransformerEncoder,
                 self_attention_transformer_params: Optional[dict] = None,
                 decoder_layer: Callable = TransformerDecoder, decoder_params: Optional[dict] = None,
                 positional_encoding: Optional[nn.Module] = None):
        """
        Transformer model from "Attention is all you need" (https://arxiv.org/abs/1706.03762).
        :param num_layers_enc: number of transformer layers in encoder
        :param num_layers_dec: number of transformer layers in decoder
        :param d_model: dimension of model
        :param n_heads: number of heads in each transformer layer
        :param out_size: output size using final linear layer, None to not use final linear layer
        :param dropout: dropout rate
        :param multihead_bias: whether to use bias in multi-head attention
        :param self_attention_transformer_layer: self-attention transformer layer type, defaults to SelfAttentionTransformerEncoder
        :param self_attention_transformer_params: additional parameters to `self_attention_transformer_layer`
        :param decoder_layer: decoder layer type, defaults to TransformerDecoder
        :param decoder_params: additional parameters to `decoder_layer`
        :param positional_encoding: positional encoding type, defaults to AbsolutePositionalEncoding
        """
        super().__init__()
        if self_attention_transformer_params is None:
            self_attention_transformer_params = {}
        if decoder_params is None:
            decoder_params = {}
        self.transformer_encoder = self_attention_transformer_layer(num_layers_enc, d_model, n_heads, dropout,
                                                                    multihead_bias, **self_attention_transformer_params)
        self.transformer_decoder = decoder_layer(num_layers_dec, d_model, n_heads, dropout, multihead_bias,
                                                 **decoder_params)
        if out_size is not None:
            self.fc = nn.Linear(d_model, out_size)
        else:
            self.fc = pp.Identity()
        if positional_encoding is None:
            positional_encoding = AbsolutePositionalEncoding()
        self.positional_encoding = positional_encoding

    def forward(self, x, target, *, mask=None, is_causal=False, target_mask=None, target_is_causal=False,
                memory_mask=None, memory_is_causal=False):
        """
        Forward pass of transformer.
        :param x: input
        :param target: target values
        :param mask: attention mask of the encoder. Mutually exclusive with is_causal.
        :param is_causal: is encoder causal. Mutually exclusive with mask.
        :param target_mask: target attention mask. Mutually exclusive with target_is_causal.
        :param target_is_causal: target is causal, prohibits attention to future tokens. Mutually exclusive with target_mask.
        :param memory_mask: memory attention mask. Mutually exclusive with memory_is_causal.
        :param memory_is_causal: memory is causal, prohibits attention to future tokens. Mutually exclusive with memory_mask.
        :return: output of transformer (batch_size, seq_len, out_size)
        """
        _, seq_len, dim = x.shape
        # Add positional encoding to input
        pe = self.positional_encoding(x)
        x = x + pe

        # Encode input
        x = self.transformer_encoder(x, mask=mask, is_causal=is_causal)

        # Add positional encoding to target
        pe = self.positional_encoding(target)
        target = target + pe

        # Decode target, cross-attention to the encoder output
        x = self.transformer_decoder(target, x, target_mask=target_mask, target_is_causal=target_is_causal,
                                     memory_mask=memory_mask, memory_is_causal=memory_is_causal)

        # Final linear layer
        x = self.fc(x)

        return x
