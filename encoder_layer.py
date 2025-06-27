import torch
import torch.nn as nn
from .multiheadattention import MultiheadAttention
from .positionwise_ff import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer consisting of a multi-head self-attention sublayer
    followed by a position-wise feed-forward sublayer, each with residual connections
    and layer normalization.

    Args:
        d_model (int): Dimensionality of input embeddings (model dimension).
        heads (int): Number of attention heads.
        d_fc_layer (int): Dimensionality of the feed-forward network's inner layer.
        dropout (float): Dropout probability applied after each sublayer.
    """
    def __init__(self, d_model: int, heads: int, d_fc_layer: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, heads, dropout)
        self.fc_layer = PositionwiseFeedForward(d_model, d_fc_layer, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Perform a forward pass of the encoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            key_padding_mask (Tensor or None): Optional mask to ignore padding tokens
                in attention, shape (batch_size, seq_length).

        Returns:
            Tensor: Output tensor of the same shape as input (batch_size, seq_length, d_model).
        """
        residual = x
        attn_out = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(residual + self.dropout(attn_out))

        residual = x
        ff_out = self.fc_layer(x)
        x = self.norm2(residual + self.dropout(ff_out))

        return x
