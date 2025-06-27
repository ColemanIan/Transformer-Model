# model/positional_encoding.py
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal PE that works with both (B, T, D) and (B, D, T) layouts.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 10_000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Pre-compute table:  (1, max_len, embed_dim) so it broadcasts over batch
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)          # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-math.log(10_000.0) / embed_dim))                       # (embed_dim//2,)

        pe = torch.zeros(max_len, embed_dim)                                          # (max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                                 # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts (B, T, D) **or** (B, D, T).  Returns a tensor in the same layout.
        """
        if x.dim() != 3:
            raise ValueError("Expected input of shape (B, T, D) or (B, D, T)")

        batch_first = x.size(1) != self.pe.size(2)          # True if layout is (B, T, D)

        if not batch_first:
            # x is (B, D, T) – swap to (B, T, D) to add PE
            x = x.transpose(1, 2)                           # (B, T, D)

        T = x.size(1)
        x = x + self.pe[:, :T, :].to(x.dtype)               # broadcast add
        x = self.dropout(x)

        if not batch_first:
            # restore original (B, D, T) layout
            x = x.transpose(1, 2)

        return x
