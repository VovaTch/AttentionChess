import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import math
from positional_encodings import PositionalEncoding2D

from base.base_model import BaseModel


class AttChess(BaseModel):
    """Main model"""

    def __init__(self, hidden_dim=256, num_heads=8, num_encoder=15, num_decoder=2, dropout=0.1):
        super().__init__()

        self.transformer = nn.Transformer(hidden_dim, num_heads, num_encoder, num_decoder, dropout=0.0)
        self.move_embedding = nn.Parameter(torch.rand(200, hidden_dim))
        self.hidden_dim = hidden_dim

        self.positional_embedding = nn.Embedding(64, hidden_dim)

        self.backbone_embedding = nn.Embedding(36, hidden_dim)
        self.head_embedding = MLP(hidden_dim, hidden_dim, 6, 5, dropout=0.0)

    def forward(self, boards):
        """Input: chessboard embedding index input"""
        # embedding + pos embedding
        hidden_embedding = self.backbone_embedding(boards)
        flat_pos_embedding = self.positional_embedding.weight.unsqueeze(0).repeat(boards.size()[0], 1, 1)
        transformer_input = hidden_embedding.flatten(1, 2) + flat_pos_embedding

        # transformer encoder-decoder and outputs
        hs = self.transformer(transformer_input.permute(1, 0, 2),
                              self.move_embedding.unsqueeze(0).repeat((boards.size()[0], 1, 1)).permute(1, 0, 2))
        move_output = self.head_embedding(hs.permute(1, 0, 2))

        return move_output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = self.dropout(x) if i < self.num_layers - 1 else x
        return x
