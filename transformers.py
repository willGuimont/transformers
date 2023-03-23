import torch
import torch.nn as nn


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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        """
        Transformer encoder layer using multi-head attention.
        This implementation uses pre-norm instead of post-norm.
        :param d_model: dimension of model
        :param nhead: number of heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.ffn = FeedForward(d_model, d_model * 4, d_model, dropout)
        self.norm1_q = nn.LayerNorm(d_model)
        self.norm1_k = nn.LayerNorm(d_model)
        self.norm1_v = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

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
        q = q + self.attention(
            q, k, v,
            attn_mask=mask,
            is_causal=is_causal)[0]
        # Feed forward network and residual connection
        q = q + self.ffn(self.norm2(q))
        return q


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
    def __init__(self, num_layers: int, d_model: int, nheads: int, dropout: float):
        """
        Stack of Transformer encoder layers.
        Allows cross-attention.
        :param num_layers: number of transformer layers
        :param d_model: dimension of model
        :param nheads: number of heads in each transformer layer
        :param dropout: dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nheads, dropout) for _ in range(num_layers)])
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
    def __init__(self, num_layers: int, d_model: int, nheads: int, dropout: float):
        """
        Self-attention transformer encoder.
        Similar to TransformerEncoder but will only use self-attention.
        :param num_layers:
        :param d_model:
        :param nheads:
        :param dropout:
        """
        super().__init__(num_layers, d_model, nheads, dropout)

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
    def __init__(self, d_model: int, nhead: int, dropout: float):
        """
        Transformer decoder layer using multi-head attention.
        :param d_model: dimension of model
        :param nhead: number of heads
        :param dropout: dropout rate
        """
        super().__init__()
        # Self-attention block
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.norm1_target = nn.LayerNorm(d_model)
        self.norm1_memory = nn.LayerNorm(d_model)

        # Cross-attention block
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        #  Final feed forward block
        self.norm3 = nn.LayerNorm(d_model)
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
    def __init__(self, num_layers: int, d_model: int, nheads: int, dropout: float):
        """
        Transformer decoder.
        :param num_layers: number of transformer layers
        :param d_model: dimension of model
        :param nheads: number of heads in each transformer layer
        :param dropout: dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nheads, dropout) for _ in range(num_layers)])
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


def positional_encoding(n_tokens: int, d_model: int, device: torch.device = torch.device("cpu")):
    """
    Generate positional encoding.
    :param n_tokens: number of tokens
    :param d_model: dimension of model
    :param device: device to store positional encoding
    :return: positional encoding
    """
    # Generate position along sequence
    pos = torch.arange(n_tokens, dtype=torch.float, device=device).reshape(1, -1, 1)
    # Generate dimension along embedding
    dim = torch.arange(d_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    # Compute phase
    phase = pos / (10000 ** (dim / d_model))
    # Compute positional encoding as described in "Attention is all you need"
    pe = torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
    return pe


class Transformer(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nheads: int, out_size: int, dropout: float):
        """
        Transformer model.
        :param num_layers: number of transformer layers
        :param d_model: dimension of model
        :param nheads: number of heads in each transformer layer
        :param out_size: output size
        :param dropout: dropout rate
        """
        super().__init__()
        self.transformer_encoder = SelfAttentionTransformerEncoder(num_layers, d_model, nheads, dropout)
        self.transformer_decoder = TransformerDecoder(num_layers, d_model, nheads, dropout)
        self.fc = nn.Linear(d_model, out_size)

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
        x += positional_encoding(seq_len, dim, device=x.device)
        # Encode input
        x = self.transformer_encoder(x, mask=mask, is_causal=is_causal)

        # Add positional encoding to target
        target += positional_encoding(seq_len, dim, device=target.device)

        # Decode target, cross-attention to the encoder output
        x = self.transformer_decoder(target, x, target_mask=target_mask, target_is_causal=target_is_causal,
                                     memory_mask=memory_mask, memory_is_causal=memory_is_causal)

        # Final linear layer
        x = self.fc(x)

        return x
