import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import math
from typing import Optional
from positional_encodings import PositionalEncoding2D

from base.base_model import BaseModel
from utils.util import word_to_move, board_to_embedding_coord, move_to_coordinate
from model.chess_conv_attention import create_chess_kernel
from .chess_conv_attention import ChessEncoderLayer


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

    def __init__(self, hidden_dim=8, num_heads=8, num_encoder=10, num_decoder=5, dropout=0.1, query_word_len=256,
                 num_chess_conv_layers=2, p_embedding=True):
        super().__init__()

        self.relu = nn.ReLU()

        # Basic constant
        self.hidden_dim = hidden_dim
        self.query_word_len = query_word_len

        # Positional encoding
        self.positional_embedding = PositionalEncoding2D(self.hidden_dim)
        self.p_emb_flag = p_embedding

        # transformer encoder and decoder
        self.chess_encoder = ChessEncoderLayer(d_model=hidden_dim, heads=num_heads, dropout=dropout)
        self.chess_encoder_stack = nn.TransformerEncoder(self.chess_encoder, num_encoder)

        self.chess_decoder = nn.TransformerDecoderLayer(batch_first=True, d_model=hidden_dim,
                                                        nhead=num_heads, dropout=dropout)
        self.chess_decoder_stack = nn.TransformerDecoder(self.chess_decoder, num_decoder)

        self.move_quality_cls_head = MLP(hidden_dim, hidden_dim, 1, 5, dropout=dropout)

        # load board + move embeddings
        self.backbone_embedding = nn.Embedding(36, hidden_dim)
        # self.backbone_embedding.requires_grad_(requires_grad=False)
        # b_emb_weights = torch.load('model/board_embedding.pth', map_location=torch.device('cpu'))
        # self.backbone_embedding.load_state_dict(b_emb_weights)
        self.query_embedding = nn.Embedding(4865, hidden_dim, padding_idx=4864)
        # self.query_embedding.requires_grad_(requires_grad=False)
        # m_emb_weights = torch.load('model/move_embedding.pth', map_location=torch.device('cpu') )
        # self.query_embedding.load_state_dict(m_emb_weights)

        self.conv_weights = []
        self.num_conv = num_chess_conv_layers
        # Creates a convolution layer
        for idx in range(num_chess_conv_layers):
            self.conv_weights.append(create_chess_kernel(out_channels=8, in_channels=8))

    def board_forward(self, boards: list[chess.Board]):
        """Takes a list of boards and converts them to tensors, gets a list of python-chess boards."""

        if next(self.parameters()).is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        boards_torch = torch.zeros((len(boards), 8, 8)).to(device)
        legal_move_torch = torch.zeros((len(boards), 64, 76), requires_grad=False).to(device) - 1

        # Converting board to
        for board_idx, board in enumerate(boards):
            boards_torch[board_idx, :, :] = board_to_embedding_coord(board)
            for legal_move in board.legal_moves:
                move_coor = move_to_coordinate(legal_move)
                legal_move_torch[board_idx, move_coor[0], move_coor[1]] = 1

        return self.forward(boards_torch.int(), legal_move_torch)

    def forward(self, boards: torch.Tensor, legal_move_tensor=None):
        """Input: chessboard embedding index input"""

        batch_size = boards.size()[0]

        if next(self.parameters()).is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        # Embedding + pos embedding
        hidden_embedding = self.backbone_embedding(boards)
        for idx in range(self.num_conv):
            hidden_embedding = F.conv2d(hidden_embedding, weight=self.conv_weights[idx].to(device), padding=7)
            hidden_embedding = hidden_embedding if idx + 1 == self.num_conv else self.relu(hidden_embedding)

        pose_embedding = self.positional_embedding(hidden_embedding) * self.p_emb_flag
        transformer_input = hidden_embedding + pose_embedding

        # Transformer encoder + classification head
        boards_flattened = boards.flatten(1, 2)
        encoder_output_board = self.chess_encoder(transformer_input)
        encoder_output = encoder_output_board.flatten(1, 2)

        # If the moves are need to be learned
        if legal_move_tensor is None:
            legal_move_out = self.move_mlp(boards_flattened.float())
            legal_move_out = legal_move_out.view((batch_size, 64, 76))
        else:
            legal_move_out = legal_move_tensor
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

        legal_move_bool = legal_move_out > 0
        legal_move_idx_tot = legal_move_bool.nonzero()

        for batch_idx, cls_end_score_ind in enumerate(classification_scores):

            legal_move_idx = legal_move_idx_tot[legal_move_idx_tot[:, 0] == batch_idx]
            legal_move_idx = legal_move_idx[:, 1:]
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
            part_cls_score = cls_end_score_ind[:word_idx + 1].clone()
            part_cls_score = torch.softmax(part_cls_score, 0)
            cls_score_batch.append(part_cls_score)

        return legal_move_list, cls_score_batch, end_game_flag


class BoardEmbTrainNet(nn.Module):
    """A class to train the board embedding"""

    def __init__(self, hidden_size=64, emb_size=8):
        super(BoardEmbTrainNet, self).__init__()

        self.backbone_embedding = nn.Embedding(36, emb_size)
        self.intermid_layer_1 = nn.Linear(emb_size, hidden_size)
        self.intermid_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.piece_head = nn.Linear(hidden_size, 7)
        self.info_head = nn.Linear(hidden_size, 4)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.backbone_embedding(x)
        x = self.intermid_layer_1(x)
        x = self.relu(x)
        x = self.intermid_layer_2(x)
        x = self.relu(x)
        x_piece = self.piece_head(x)
        x_info = self.info_head(x)

        return x_piece, x_info


class MoveEmbTrainNet(nn.Module):
    """A class to train the move embedding"""

    def __init__(self, hidden_size=256, emb_size=8):
        super(MoveEmbTrainNet, self).__init__()

        self.query_embedding = nn.Embedding(4865, emb_size, padding_idx=4864)
        self.intermid_layer_1 = nn.Linear(emb_size, hidden_size)
        self.intermid_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.coor_head = nn.Linear(hidden_size, 4)
        self.promotion_head = nn.Linear(hidden_size, 5)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.query_embedding(x)
        x = self.intermid_layer_1(x)
        x = self.relu(x)
        x = self.intermid_layer_2(x)
        x = self.relu(x)
        x_coor = self.coor_head(x)
        x_prom = self.promotion_head(x)

        return x_coor, x_prom


class TransformerAuxDecoder(torch.nn.TransformerDecoder):
    """A sub class for outputting all aux outputs"""

    def __init__(self, decoder, num_layers, norm=None, aux_out_intervals=1):
        super(TransformerAuxDecoder, self).__init__(decoder, num_layers, norm=norm)
        self.aux_out_intervals = aux_out_intervals

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        output_aux = []

        for idx, mod in enumerate(self.layers):
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            if (idx + 1) % self.aux_out_intervals:
                output_aux.append(output.clone())

        if self.norm is not None:
            output = self.norm(output)

        return output, output_aux
