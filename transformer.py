import torch
import torch.nn as nn
from typing import Callable, List, Dict
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        heads: int = 8,
        d_fc_layer: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, heads, d_fc_layer, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, heads, d_fc_layer, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.gate = nn.Linear(d_model, 1)  # (B, T, 1)

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device: torch.device = torch.device('cpu')
    ) -> "Transformer":
        """
        Load a Transformer from disk. Expects a checkpoint dict containing:
          - src_vocab_size, tgt_vocab_size, d_model, num_layers, heads,
            d_fc_layer, dropout
          - model_state_dict
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = cls(
            src_vocab_size=ckpt['src_vocab_size'],
            tgt_vocab_size=ckpt['tgt_vocab_size'],
            d_model=ckpt.get('d_model', 512),
            num_layers=ckpt.get('num_layers', 6),
            heads=ckpt.get('heads', 8),
            d_fc_layer=ckpt.get('d_fc_layer', 2048),
            dropout=ckpt.get('dropout', 0.1)
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
        return model

    # @classmethod
    # def load_inference(cls, bundle_path: str, device):
    # bundle = torch.load(bundle_path, map_location=device)

    # # rebuild model from config
    # cfg = bundle["config"]
    # model = cls(
    #     src_vocab_size=cfg["src_vocab_size"],
    #     tgt_vocab_size=cfg["tgt_vocab_size"],
    #     d_model=cfg["d_model"],
    #     num_layers=cfg["num_layers"],
    #     heads=cfg["heads"],
    #     d_fc_layer=cfg["d_fc_layer"],
    #     dropout=cfg["dropout"],
    # ).to(device)
    # model.load_state_dict(bundle["model_state_dict"])
    # model.eval()

    # # # rebuild vocab and tokenizer
    # # vocab = bundle["vocab"]
    # # id2token = {i:t for t,i in vocab.items()}
    # # def tokenizer_fn(text): …
    # # # or reuse load_vocab utility
    # #  = load_vocab()
    # # return model, tokenizer_fn, vocab, id2token, bundle["inference_args"]
    # return model, vocab, id2token, bundle["inference_args"]


    def generate_tgt_mask(
        self,
        mels_input: torch.Tensor,
        mel_lens: torch.Tensor
    ):
        """
        mels_input : (B, T, n_mels)
        mel_lens   : (B,)
        Returns:
            causal_mask         : (T, T)   bool
            key_padding_mask    : (B, T)   bool
        """
        T = mels_input.size(1)
        device = mels_input.device

        # mask padded frames
        key_padding_mask = torch.arange(T, device=device)[None, :] >= mel_lens[:, None]
        # causal mask
        causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), 1)

        return causal_mask, key_padding_mask

    def forward(
        self,
        src: torch.Tensor,
        mels_input: torch.Tensor,
        mel_lens: torch.Tensor
    ):
        src_key_padding_mask = (src == 0)  # assume 0 is pad in your text vocab

        enc_out = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        tgt_mask, tgt_key_padding_mask = self.generate_tgt_mask(mels_input, mel_lens)

        dec_out = self.decoder(
            mels_input,
            mel_lens,
            enc_out,
            tgt_attn_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        mel_logits = self.fc_out(dec_out)          # (B, T, vocab_size)
        gate_logits = self.gate(dec_out).squeeze(-1)  # (B, T)
        return mel_logits, gate_logits

    @torch.no_grad()
    def infer(
        self,
        text: str,
        model: nn.Module,
        tokenizer_fn: Callable[[str], List[int]],
        vocab: Dict[str, int],
        device: torch.device,
        max_len: int = 1000
    ):
        # 1. tokenize
        seq_ids = tokenizer_fn(text)
        src = torch.LongTensor(seq_ids)[None, :].to(device)

        # 2. init mel input
        n_mels = vocab.get('n_mels', 80)
        mel_input = torch.zeros(1, 1, n_mels, device=device)
        mel_lens = torch.ones(1, dtype=torch.long, device=device)

        output_mels = []
        for _ in range(max_len):
            mel_logits, gate_logits = model(src, mel_input, mel_lens)
            next_frame = mel_logits[:, -1, :].unsqueeze(1)
            output_mels.append(next_frame.cpu())

            mel_input = torch.cat([mel_input, next_frame], dim=1)
            mel_lens = torch.tensor([mel_input.size(1)], device=device)

            if torch.sigmoid(gate_logits[:, -1]) > 0.5:
                break

        mel_pred = torch.cat(output_mels, dim=1).squeeze(0)
        return mel_pred
