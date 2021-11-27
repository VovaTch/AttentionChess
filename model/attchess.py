import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from base.base_model import BaseModel


class AttChess(BaseModel):
    """Main model"""

    def __init__(self, hidden_dim=256, num_heads=8, num_encoder=5, num_decoder=5, dropout=0.1):
        super().__init__()

        self.transformer = nn.Transformer(hidden_dim, num_heads, num_encoder, num_decoder)
        self.move_embedding = nn.Parameter(torch.rand(64, hidden_dim))
        self.hidden_dim = hidden_dim

        self.q_move_MLP = MLP(hidden_dim, hidden_dim, hidden_dim, 5, dropout=dropout)
        self.k_move_MLP = MLP(hidden_dim, hidden_dim, hidden_dim, 5, dropout=dropout)

        self.pos_embedding_stack = nn.Parameter(torch.rand(8, 8, hidden_dim - 2))

    def forward(self, boards, turn):
        """Input: tensor board representation, output is a matrix of move logits; illegal moves should be filtered."""

        turn_board = torch.zeros((boards.size()[0], 8, 8)).to(device=boards.device)
        for idx in range(boards.size()[0]):
            if turn[idx]:
                turn_board[idx, :, :] += 1

        pos = self.pos_embedding_stack.unsqueeze(0).repeat((boards.size()[0], 1, 1, 1))
        stacked_input = torch.cat((boards.unsqueeze(3), turn_board.unsqueeze(3), pos), 3)
        hs = self.transformer(stacked_input.flatten(1, 2), self.move_embedding.unsqueeze(0).repeat((boards.size()[0],
                                                                                                    1, 1)))

        q_hs_pass = self.q_move_MLP(hs)
        k_hs_pass = self.k_move_MLP(hs)

        output_logits = torch.matmul(q_hs_pass, k_hs_pass.permute(0, 2, 1)) / 64
        return output_logits


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
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = self.dropout(x) if i < self.num_layers - 1 else x
        return x