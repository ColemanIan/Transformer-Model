import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    Multi-head Attention module as described in "Attention Is All You Need".

    This module computes scaled dot-product attention over multiple heads,
    projects query, key, and value inputs into subspaces, applies attention,
    and then recombines and projects the result.

    Args:
        embed_dim (int): Total dimension of the input feature space (d_model).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to attention weights.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value, and final output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention for one timestep across all heads.

        Args:
            q (Tensor): Query tensor of shape (B, H, L_q, head_dim).
            k (Tensor): Key tensor of shape   (B, H, L_k, head_dim).
            v (Tensor): Value tensor of shape (B, H, L_k, head_dim).
            attn_mask (Tensor or None): Optional attention mask of shape (L_q, L_k).
            key_padding_mask (Tensor or None): Optional padding mask of shape (B, L_k).

        Returns:
            out (Tensor): Attention outputs of shape (B, H, L_q, head_dim).
            attn_weights (Tensor): Attention weights of shape (B, H, L_q, L_k).
        """
        B, H, L_q, D_h = q.shape
        L_k = k.size(-2)

        # Compute raw attention scores: (B, H, L_q, L_k)
        scores = (q @ k.transpose(-2, -1)) / (D_h ** 0.5)

        if attn_mask is not None:
            # attn_mask: (L_q, L_k) -> (1, 1, L_q, L_k)
            mask = attn_mask.bool()[None, None, :, :]
            scores = scores.masked_fill(mask, float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask: (B, L_k) -> (B, 1, 1, L_k)
            pad = key_padding_mask.bool()[:, None, None, :]
            scores = scores.masked_fill(pad, float('-inf'))

        # Convert scores to probabilities
        attn_weights = F.softmax(scores, dim=-1)  # zeros where scores were -inf
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        out = attn_weights @ v  # (B, H, L_q, head_dim)
        return out, attn_weights

    def _proj(
        self,
        linear: nn.Linear,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Project and reshape input tensor for multi-head attention.

        Args:
            linear (nn.Linear): Linear layer to apply (q_proj, k_proj, or v_proj).
            x (Tensor): Input tensor of shape (B, L, embed_dim).

        Returns:
            Tensor: Reshaped tensor of shape (B, H, L, head_dim).
        """
        B, L, _ = x.shape
        # Apply linear projection
        proj = linear(x)  # (B, L, embed_dim)
        # Split embedding into heads and head dimension
        proj = proj.view(B, L, self.num_heads, self.head_dim)
        # Move head dimension before sequence dimension
        proj = proj.transpose(1, 2)  # (B, H, L, head_dim)
        return proj

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.

        Args:
            query (Tensor): Query embeddings of shape (B, L_q, embed_dim).
            key (Tensor):   Key embeddings of shape   (B, L_k, embed_dim).
            value (Tensor): Value embeddings of shape (B, L_k, embed_dim).
            attn_mask (Tensor or None): Optional mask to apply to attention scores.
            key_padding_mask (Tensor or None): Mask padding tokens in the key.
            need_weights (bool): If True, also return attention weights.

        Returns:
            attn_output (Tensor): Output tensor of shape (B, L_q, embed_dim).
            attn_weights (Tensor, optional): Attention weights of shape (B, H, L_q, L_k).
        """
        B, L_q, _ = query.shape

        # Project inputs to multi-head representations
        q = self._proj(self.q_proj, query)
        k = self._proj(self.k_proj, key)
        v = self._proj(self.v_proj, value)

        # Compute scaled dot-product attention
        attn_out, attn_weights = self._scaled_dot_product(
            q, k, v, attn_mask, key_padding_mask
        )

        # Merge heads: (B, H, L_q, head_dim) -> (B, L_q, embed_dim)
        attn_merged = (
            attn_out
            .transpose(1, 2)               # (B, L_q, H, head_dim)
            .contiguous()
            .view(B, L_q, self.embed_dim)  # (B, L_q, embed_dim)
        )

        # Final linear projection
        out = self.o_proj(attn_merged)

        return (out, attn_weights) if need_weights else out
