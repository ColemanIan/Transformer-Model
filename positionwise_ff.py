import torch
import torch.nn as nn
import torch.nn.functional as f

class PositionwiseFeedForward(nn.Module):
    def __init__(self,
        embed_dim: int, 
        d_fc_layer: int, 
        dropout: float = 0.1
        ):
        """
        Feed forward network to be used for each head
        """
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, d_fc_layer)
        self.linear2 = nn.Linear(d_fc_layer, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.linear2(x)