import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, in_channel: int, mid_channels, out_channels, dropout: float):
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
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.ffn = FeedForward(d_model, d_model * 4, d_model, dropout)
        self.norm1_q = nn.LayerNorm(d_model)
        self.norm1_k = nn.LayerNorm(d_model)
        self.norm1_v = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, k, v, *, mask=None, is_causal=False):
        q = q + self.attention(
            self.norm1_q(q),
            self.norm1_k(k),
            self.norm1_v(v),
            attn_mask=mask,
            is_causal=is_causal)[0]
        q = q + self.ffn(self.norm2(q))
        return q


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nheads: int, out_size: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nheads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, out_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, q, k, v, *, mask=None, is_causal=False):
        for layer in self.layers:
            q = layer(q, k, v, mask=mask, is_causal=is_causal)
        q = self.norm(q)
        q = self.fc(q)
        return q


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.norm1_target = nn.LayerNorm(d_model)
        self.norm1_memory = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, d_model * 4, d_model, dropout)

    def forward(self, target, memory, *, target_mask=None, target_is_causal=False, memory_mask=None,
                memory_is_causal=False):
        target, memory = self.norm1_target(target), self.norm1_memory(memory)
        target = target + \
                 self.self_attention(target, target, target, attn_mask=target_mask, is_causal=target_is_causal)[0]
        try:
            target = target + self.cross_attention(self.norm2(target), memory, memory, attn_mask=memory_mask,
                                                   is_causal=memory_is_causal)[0]
        except:
            print('fuck')
        return target + self.ffn(self.norm3(target))


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nheads: int, out_size: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nheads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, out_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, target, memory, *, target_mask=None, target_is_causal=False, memory_mask=None,
                memory_is_causal=False):
        for layer in self.layers:
            target = layer(target, memory, target_mask=target_mask, target_is_causal=target_is_causal,
                           memory_mask=memory_mask, memory_is_causal=memory_is_causal)
        target = self.norm(target)
        target = self.fc(target)
        return target
