import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import math
from positional_encodings import PositionalEncoding2D

from base.base_model import BaseModel
from utils.util import word_to_move


class AttChess_old(BaseModel):
    """Main model"""

    def __init__(self, hidden_dim=256, num_heads=8, num_encoder=15, num_decoder=2, dropout=0.1):
        super().__init__()

        # Basic transformer
        self.transformer = nn.Transformer(hidden_dim, num_heads, num_encoder, num_decoder, dropout=dropout)
        self.move_embedding = nn.Parameter(torch.rand(200, hidden_dim))
        self.hidden_dim = hidden_dim

        # Chess board learned positional embedding
        self.positional_embedding = nn.Embedding(64, hidden_dim)

        # Chess backbone embedding
        self.backbone_embedding = nn.Embedding(36, hidden_dim)

        # Heads: legal classification, move coordinates, move quality
        self.legal_head_mlp = MLP(hidden_dim + 4, hidden_dim + 4, 1, 3, dropout=dropout)
        self.move_head_mlp = MLP(hidden_dim, hidden_dim, 4, 3, dropout=dropout)
        self.quality_head_mlp = MLP(hidden_dim + 4, hidden_dim + 4, 1, 5, dropout=dropout)

    def forward(self, boards):
        """Input: chessboard embedding index input"""
        # embedding + pos embedding
        hidden_embedding = self.backbone_embedding(boards)
        flat_pos_embedding = self.positional_embedding.weight.unsqueeze(0).repeat(boards.size()[0], 1, 1)
        transformer_input = hidden_embedding.flatten(1, 2) + flat_pos_embedding

        # transformer encoder-decoder and outputs
        hs = self.transformer(transformer_input.permute(1, 0, 2),
                              self.move_embedding.unsqueeze(0).repeat((boards.size()[0], 1, 1)).permute(1, 0, 2))

        # Heads # TODO: Continue
        move_pred = self.move_head_mlp(hs.permute((1, 0, 2)))
        move_pred = move_pred.sigmoid()
        hs_aug = torch.cat((hs.permute((1, 0, 2)), move_pred), 2)
        legal_pred = self.legal_head_mlp(hs_aug)
        quality_pred = self.quality_head_mlp(hs_aug)
        move_output = torch.cat((move_pred[:, :, 0:3], legal_pred, quality_pred, move_pred[:, :, 3].unsqueeze(2)), 2)

        return move_output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
            x = self.dropout(x) if i < self.num_layers - 1 else x
        return x


class AttChess(BaseModel):
    """Main model, new"""

    def __init__(self, hidden_dim=256, num_heads=8, num_encoder=10, num_decoder=5, dropout=0.1, query_word_len=256):
        super().__init__()

        # Basic constant
        self.hidden_dim = hidden_dim
        self.query_word_len = query_word_len

        # Chess board learned positional embedding, and backbone embedding
        self.positional_embedding = nn.Embedding(64, hidden_dim)
        self.backbone_embedding = nn.Embedding(36, hidden_dim)

        # transformer encoder and decoder
        self.chess_encoder = nn.TransformerEncoderLayer(batch_first=True, d_model=hidden_dim,
                                                        nhead=num_heads, dropout=dropout)
        self.chess_encoder_stack = nn.TransformerEncoder(self.chess_encoder, num_encoder)

        self.chess_decoder = nn.TransformerDecoderLayer(batch_first=True, d_model=hidden_dim,
                                                        nhead=num_heads, dropout=dropout)
        self.chess_decoder_stack = nn.TransformerDecoder(self.chess_decoder, num_decoder)

        # Move legality classification heads
        self.k_legal = MLP(hidden_dim, hidden_dim, hidden_dim, 3, dropout=dropout)
        self.q_legal = MLP(hidden_dim, hidden_dim, hidden_dim, 3, dropout=dropout)
        self.promotion_embedding = nn.Embedding(12, hidden_dim)  # Handling promotions, 4 possible pieces

        # Move classification embedding and MLP
        self.query_embedding = nn.Embedding(4865, hidden_dim, padding_idx=4864)
        self.move_quality_cls_head = MLP(hidden_dim, hidden_dim, 1, 5, dropout=dropout)

    def forward(self, boards: torch.Tensor):
        """Input: chessboard embedding index input"""

        batch_size = boards.size()[0]

        # Embedding + pos embedding
        hidden_embedding = self.backbone_embedding(boards)
        flat_pos_embedding = self.positional_embedding.weight.unsqueeze(0).repeat(boards.size()[0], 1, 1)
        transformer_input = hidden_embedding.flatten(1, 2) + flat_pos_embedding

        # Transformer encoder + classification head
        encoder_output = self.chess_encoder(transformer_input)
        enc_k_output = self.k_legal(encoder_output)
        encoder_output_aug = torch.cat((encoder_output,
                                        self.promotion_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)), 1)
        enc_q_output = self.q_legal(encoder_output_aug)
        legal_move_out = torch.matmul(enc_k_output, enc_q_output.permute((0, 2, 1)))
        legal_move_mask = legal_move_out > 0

        # Draw out the legal moves
        legal_move_idx = torch.nonzero(legal_move_mask)
        legal_move_word = torch.cat((legal_move_idx[:, 0].unsqueeze(1),
                                     legal_move_idx[:, 1].unsqueeze(1) +
                                     64 * legal_move_idx[:, 2].unsqueeze(1)), 1)
        query_words = torch.zeros((boards.size()[0], self.query_word_len)).to(boards.device) + 4864
        for batch_idx in range(boards.size()[0]):
            batch_legal_moves = legal_move_word[legal_move_word[:, 0] == batch_idx, 1]
            if batch_legal_moves.size()[0] < self.query_word_len - 1:
                query_words[batch_idx, 0: batch_legal_moves.size()[0]] = batch_legal_moves
            else:
                query_words[batch_idx, :-1] = batch_legal_moves[:self.query_word_len - 1]  # Cut off at 256. The max number of chess moves is 218.

        # Pass through decoder and classify
        queried_moves = self.query_embedding(query_words.long())
        decoder_output = self.chess_decoder(queried_moves, encoder_output)
        classification_scores = self.move_quality_cls_head(decoder_output)  # idx 255 is saved for resigning or draw

        return legal_move_out, classification_scores.squeeze(2)

    def post_process(self, legal_move_out, classification_scores):

        cls_score_batch = []
        end_game_flag = []
        legal_move_list = []

        for legal_move_ind, cls_end_score_ind in zip(legal_move_out, classification_scores):

            legal_move_mat = legal_move_ind > 0
            legal_move_idx = torch.nonzero(legal_move_mat)
            legal_move_word = legal_move_idx[:, 0] + 64 * legal_move_idx[:, 1]

            legal_move_list.append([])
            word_idx = 0
            # Collect all legal move in a 2d list
            for word_idx, word in enumerate(legal_move_word):
                move = word_to_move(word)
                legal_move_list[-1].append(move)
                if word_idx >= self.query_word_len - 1:
                    break

            end_game_flag.append(cls_end_score_ind[-1])

            # Handling classification scores. Has to be stored in a list because every one of them has a different size
            part_cls_score = cls_end_score_ind[:word_idx].clone()
            part_cls_score = torch.exp(part_cls_score) / torch.logsumexp(part_cls_score, 0)
            cls_score_batch.append(part_cls_score)

        return legal_move_list, cls_score_batch, end_game_flag

