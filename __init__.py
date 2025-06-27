# model/__init__.py
"""
High-level package interface for the Transformer-TTS project.

Having these re-exports means users can simply do:
>>> from model import Transformer, Encoder, Decoder
instead of importing from individual sub-modules.
"""

from .positional_encoding import PositionalEncoding
from .multiheadattention import MultiheadAttention
from .positionwise_ff import PositionwiseFeedForward
from .encoder_layer import EncoderLayer
from .encoder import Encoder
from .decoder_layer import DecoderLayer
from .decoder import Decoder
from .transformer import Transformer
# from .tokenizer import build_phoneme_tokenizer  # or Tokenizer, if that’s the object you expose

__all__ = [
    "PositionalEncoding",
    "MultiheadAttention",
    "PositionwiseFeedForward",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    "Transformer"
]





