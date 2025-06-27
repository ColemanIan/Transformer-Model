import torch
import torch.nn as nn
import torch.nn.functional as F
from .multiheadattention import MultiheadAttention
from .positionwise_ff import PositionwiseFeedForward

class DecoderLayer(nn.Module):
  def __init__(self, embed_dim, heads, d_fc_layer, dropout=0.1):
    """
    Transformer encoder that maps token indices to continuous embeddings,
    adds positional information, and applies a stack of encoder layers.

    Args:
        vocab_size (int): Size of the input token vocabulary.
        embed_dim (int): Dimensionality of input embeddings and model (d_model).
        num_layers (int): Number of encoder layers to stack.
        heads (int): Number of attention heads in each multi-head attention block.
        d_fc_layer (int): Inner dimensionality of the position-wise feed-forward network.
        dropout (float): Dropout probability for all sub-layers.
        max_seq_len (int): Maximum sequence length for positional encoding.

    Forward Args:
        x (Tensor): Input tensor of token indices with shape (batch_size, seq_length).
        src_key_padding_mask (Tensor or None): Optional mask of shape (batch_size, seq_length)
            indicating which tokens should be ignored in attention (padding tokens).

    Returns:
        Tensor: Output tensor of shape (batch_size, seq_length, embed_dim) after processing
        through all encoder layers and final layer normalization.
    """
    super().__init__()
    self.self_attn = MultiheadAttention(embed_dim, heads, dropout)
    self.cross_attn = MultiheadAttention(embed_dim,heads,dropout)
    self.fc_layer = PositionwiseFeedForward(embed_dim=embed_dim, d_fc_layer=d_fc_layer, dropout=dropout)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.norm3 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, 
    x, 
    encoder_output, 
    self_attn_mask=None, 
    cross_attn_key_padding_mask=None, 
    self_attn_key_padding_mask=None) -> torch.Tensor:
    """

    """

    residual = x
    x = self.self_attn(x, x, x, attn_mask=self_attn_mask, key_padding_mask=self_attn_key_padding_mask)

    x = self.dropout(x)
    x = self.norm1(residual + x)

    residual = x
    x = self.cross_attn(x, encoder_output, encoder_output, key_padding_mask=cross_attn_key_padding_mask)
    x = self.dropout(x)
    x = self.norm2(residual + x)

    residual = x
    x = self.fc_layer(x)
    x = self.dropout(x)
    return self.norm3(residual + x)