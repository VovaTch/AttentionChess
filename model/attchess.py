import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from base.base_model import BaseModel


class AttChess(BaseModel):
    """Main model"""

    def __init__(self, hidden_dim=256, num_heads=8, num_encoder=15, num_decoder=2, dropout=0.1):
        super().__init__()

        self.transformer = nn.Transformer(hidden_dim, num_heads, num_encoder, num_decoder)
        self.move_embedding = nn.Parameter(torch.rand(200, hidden_dim))
        self.hidden_dim = hidden_dim

        self.backbone_embedding = MLP(5, hidden_dim, hidden_dim, 2, dropout=0)
        self.head_embedding = MLP(hidden_dim, hidden_dim, 6, 5, dropout=0)

    def forward(self, boards):
        """Input: tensor board representation of Nx5x8x8, output is a move tensor Nx6, for now ignore the resign flag"""
        flatten_boards = boards.flatten(2, 3)
        embedded_squares = self.backbone_embedding(flatten_boards.permute(0, 2, 1))
        hs = self.transformer(embedded_squares.permute(1, 0, 2),
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