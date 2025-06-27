import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder_layer       import DecoderLayer
from .positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    """
    Transformer-TTS decoder (teacher-forcing mode).
    Expects mels_input shape (B, T_mel, n_mels).
    """
    def __init__(
        self,
        n_mels: int,
        embed_dim: int,
        num_layers: int,
        heads: int,
        d_fc_layer: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # project 80-bin mel frame → embed_dim
        self.input_proj = nn.Linear(n_mels, embed_dim)
        self.pos_emb    = PositionalEncoding(embed_dim, dropout=dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, heads, d_fc_layer, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------

    def forward(
        self,
        mels_input,                # (B, T_tgt, n_mels)
        mel_lens,                 # (B,)  lengths – *not* used here but handy if you add prenet dropout
        encoder_out,              # (B, T_src, D)
        tgt_attn_mask=None,       # (T_tgt, T_tgt) bool
        tgt_key_padding_mask=None,  # (B, T_tgt)  bool
        src_key_padding_mask=None,  # (B, T_src)  bool
    )-> torch.Tensor:
        """
        Teacher-forcing forward pass.
        """

        # 1) project + positional encoding
        x = self.input_proj(mels_input)        # (B, T_tgt, D)
        x = self.pos_emb(x)                    # add sin/cos PE

        # 2) stacked decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                self_attn_mask=tgt_attn_mask,
                self_attn_key_padding_mask=tgt_key_padding_mask,
                cross_attn_key_padding_mask=src_key_padding_mask,
            )

        return self.norm(x)                    # (B, T_tgt, D)
