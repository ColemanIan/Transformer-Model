import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding
from .encoder_layer       import EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        heads: int,
        d_fc_layer: int,
        dropout: float = 0.1,
        max_seq_len: int = 5000 #just needs to be high, really
    ):
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
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        ## max_seq_len is not the same as vocab_size
        self.pos_emb = PositionalEncoding(
            embed_dim=embed_dim,
            dropout=dropout,
            max_len=max_seq_len,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, heads, d_fc_layer, dropout=dropout)
                for l in range(num_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask=None
        )-> torch.Tensor:
        """
        Perform a forward pass of the encoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            key_padding_mask (Tensor or None): Optional mask to ignore padding tokens
                in attention, shape (batch_size, seq_length).

        Returns:
            Tensor: Output tensor of the same shape as input (batch_size, seq_length, d_model).
        """
        
        x = self.token_embeddings(x)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_key_padding_mask)
        return self.layernorm(x)