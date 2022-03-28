from turtle import forward
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
# from model.chess_conv_attention import hollow_chess_kernel
# from .chess_conv_attention import ChessEncoderLayer
# from ripple_linear_py import RippleLinear, RippleConv2d
# from .trigo_layers import TrigoEncoderLayer, TrigoDecoderLayer
from .backbone import ChessFeatureExpander


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1, ripple=False, activation='GELU'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.ripple = ripple
        if not ripple:
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(RippleLinear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
            self.layers[0] = nn.Linear(self.layers[0].input_size, self.layers[0].output_size)
            self.layers[-1] = nn.Linear(self.layers[-1].input_size, self.layers[-1].output_size)
        self.dropout = nn.Dropout(p=dropout)
        
        if activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x):
        for i, layer in enumerate(self.layers):         
            x = F.gelu(layer(x)) if i < self.num_layers - 1 and not self.ripple else layer(x)
            # Bypass
            if i == 0:
                x_bypass = x
            elif i == self.num_layers - 2:
                x += x_bypass
            x = self.dropout(x) if i < self.num_layers - 1 else x
        return x


class AttChess(BaseModel):
    """
    Main model, new again, now with a separate value score
    """

    def __init__(self, hidden_dim=8, num_heads=8, num_encoder=10, num_decoder=5, dropout=0.1, query_word_len=256,
                 num_chess_conv_layers=2, p_embedding=True, ripple_net=False, aux_outputs=False):
        super().__init__()

        self.relu = nn.ReLU()

        # Basic constant
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim * 4
        self.query_word_len = query_word_len
        self.aux_outputs_flag = aux_outputs

        # Positional encoding
        self.positional_embedding = PositionalEncoding2D(self.hidden_dim_2)
        self.p_emb_flag = p_embedding

        # transformer encoder and decoder
        # self.chess_encoder = ChessEncoderLayer(d_model=hidden_dim, heads=num_heads, dropout=dropout)
        self.chess_encoder = nn.TransformerEncoderLayer(batch_first=True, d_model=self.hidden_dim_2,
                                                        nhead=num_heads, dropout=dropout, norm_first=True)
        self.chess_encoder_stack = nn.TransformerEncoder(self.chess_encoder, num_encoder)
        self.chess_decoder = nn.TransformerDecoderLayer(batch_first=True, d_model=self.hidden_dim_2,
                                                        nhead=num_heads, dropout=dropout, norm_first=True)
        self.chess_decoder_stack = TransformerAuxDecoder(self.chess_decoder, num_decoder, aux_out_intervals=num_decoder)
        self.move_quality_cls_head = MLP(self.hidden_dim_2, self.hidden_dim_2, 1, 3, dropout=dropout, ripple=ripple_net)
        # self.chess_encoder_value = ChessEncoderLayer(d_model=hidden_dim, heads=num_heads, dropout=dropout)
        # self.chess_encoder_value = TrigoEncoderLayer(batch_first=True, d_model=self.hidden_dim_2,
        #                                                nhead=num_heads, dropout=dropout)
        # self.chess_encoder_value_stack = nn.TransformerEncoder(self.chess_encoder_value, num_encoder)

        # load board + move embeddings
        self.backbone_embedding = nn.Embedding(36, hidden_dim)
        self.backbone_embedding.requires_grad_(requires_grad=False)
        b_emb_weights = torch.load('model/board_embedding.pth', map_location=torch.device('cpu'))
        self.backbone_embedding.load_state_dict(b_emb_weights)
        self.query_embedding = nn.Embedding(4865, self.hidden_dim_2, padding_idx=4864)
        self.query_embedding.requires_grad_(requires_grad=False)
        m_emb_weights = torch.load('model/move_embedding.pth', map_location=torch.device('cpu') )
        self.query_embedding.load_state_dict(m_emb_weights)

        self.num_conv = num_chess_conv_layers
        self.end_conv = list()
        # Creates a convolution layer
        # for _ in range(num_chess_conv_layers):
        #     self.end_conv.append(nn.Conv2d(self.hidden_dim_2, self.hidden_dim_2, (15, 15), padding=7))
        #     hollow_chess_kernel(self.end_conv[-1].weight)
            
        self.conv_end_stack = nn.ModuleList(self.end_conv)
        
        self.bypass_parameter_encoder = nn.Parameter(torch.zeros(1))
        self.enc_dec_MLP = MLP(self.hidden_dim_2, self.hidden_dim_2 * 8, self.hidden_dim_2, 3, dropout=dropout, ripple=ripple_net)
        # self.bypass_parameter_decoder = nn.Parameter(torch.zeros(1))
        # self.encoder_info_parameter = nn.Parameter(torch.zeros(1))
          
        # self.end_head = MLP(self.hidden_dim_2 * 512, self.hidden_dim, 1, 3, ripple=ripple_net, activation='tanh')
        self.end_head = EndHead(query_word_len, self.hidden_dim_2)
        #self.batch_norm = nn.BatchNorm2d(self.hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()

        # And another try for a backbone, this is a multi-layer convolution that produces features.
        # self.backbone = ChessFeatureExpander(hidden_dim) TODO: Conv backbone

        # ANOTHER TRY FOR THE CHESS CONV
        # self.conv_backbone_1 = nn.Conv2d(hidden_dim, self.hidden_dim_2 * 2, kernel_size=(15, 15), padding=7)
        # hollow_chess_kernel(self.conv_backbone_1.weight)
        # self.conv_backbone_2 = nn.Conv2d(self.hidden_dim_2 * 2, self.hidden_dim_2 * 2, kernel_size=(15, 15), padding=7)
        # hollow_chess_kernel(self.conv_backbone_1.weight)
        # self.conv_backbone_3 = nn.Conv2d(self.hidden_dim_2 * 2, self.hidden_dim_2 * 4, kernel_size=(3, 3), padding=1)
        # self.conv_backbone_4 = nn.Conv2d(self.hidden_dim_2 * 4, self.hidden_dim_2 * 8, kernel_size=(3, 3), padding=1)
        
        # self.backbone_conv = nn.Sequential(
        #     self.conv_backbone_1,
        #     nn.GELU(),
        #     nn.LayerNorm([self.hidden_dim_2 * 2, 8, 8]),
        #     self.conv_backbone_2,
        #     nn.GELU(),
        #     nn.LayerNorm([self.hidden_dim_2 * 2, 8, 8]),
        #     self.conv_backbone_3,
        #     nn.GELU(),
        #     nn.LayerNorm([self.hidden_dim_2 * 4, 8, 8]),
        #     self.conv_backbone_4,
        #     nn.GELU(),
        #     nn.LayerNorm([self.hidden_dim_2 * 8, 8, 8])
        # )
        

    def forward(self, boards: list[chess.Board]):
        """
        Takes a list of boards and converts them to tensors, gets a list of python-chess boards.
        """

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

        return self.raw_forward(boards_torch.int(), legal_move_torch)

    def raw_forward(self, boards: torch.Tensor, legal_move_tensor=None):
        """
        Input: chessboard embedding index input
        """

        batch_size = boards.size()[0]

        if next(self.parameters()).is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        # Embedding + pos embedding
        hidden_embedding = self.backbone_embedding(boards.int())

        pose_embedding = self.positional_embedding(hidden_embedding) * self.p_emb_flag
        conv_input = torch.cat((hidden_embedding, pose_embedding, hidden_embedding, pose_embedding), dim=1)
        # transformer_input = self.backbone(conv_input) # Size BS x HS x W x H TODO: Try again later if necessary
        transformer_input = conv_input
        transformer_input = transformer_input.flatten(2, 3).permute((0, 2, 1)) # Size BS x C x HS

        # Transformer encoder + classification head
        boards_flattened = boards.flatten(1, 2)
        # encoder_output_board = transformer_input.flatten(1, 2)
        head_eo = self.chess_encoder_stack(transformer_input)
        # board_value = self.end_head(head_eo).squeeze(-1)

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
            if batch_legal_moves.size()[0] < self.query_word_len:
                query_words[batch_idx, 0: batch_legal_moves.size()[0]] = batch_legal_moves
            else:
                query_words[batch_idx, :] = batch_legal_moves[:self.query_word_len]  # Cut off at 256. The max number of chess moves is 218.

        # Pass through decoder and classify
        query_input = self.enc_dec_MLP(head_eo)
        queried_moves = self.query_embedding(query_words.long())
        value_decoder_input = torch.cat((transformer_input.view(batch_size, -1, self.hidden_dim_2), query_input), 1)
        decoder_output, decoder_output_aux = self.chess_decoder_stack(queried_moves, value_decoder_input)
        board_value = self.end_head(decoder_output)
        classification_scores = self.move_quality_cls_head(decoder_output)
        
        if self.aux_outputs_flag:
            aux_outputs = {f'loss_quality_{idx}': self.move_quality_cls_head(dec_out_aux).squeeze() 
                           for idx, dec_out_aux in enumerate(decoder_output_aux)}
            aux_value = {f'loss_board_value_{idx}': self.end_head(dec_out_aux).squeeze() 
                         for idx, dec_out_aux in enumerate(decoder_output_aux)}
            aux_outputs.update(aux_value)
            return legal_move_out, classification_scores.squeeze(2), board_value, aux_outputs
        else:
            return legal_move_out, classification_scores.squeeze(2), board_value
    
    def post_process(self, legal_move_out, classification_scores, board_value, num_pruned_moves=None):
        # TODO: filter low probability moves to prevent the bot from doing nonsense.

        cls_score_batch = []
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
                if word_idx >= self.query_word_len:
                    break

            # Handling classification scores. Has to be stored in a list because every one of them has a different size
            part_cls_score = cls_end_score_ind[:word_idx + 1].clone()
            part_cls_score = torch.softmax(part_cls_score, 0)
            cls_score_batch.append(part_cls_score)

        # Consider only the 5 top moves in the policy
        if num_pruned_moves is not None:
            
            cls_score_pruned = list()
            legal_move_list_pruned = list()
            
            for batch_idx, (cls_score, legal_move_sublist) in enumerate(zip(cls_score_batch, legal_move_list)):
                cls_score_sorted, sort_idx = torch.sort(cls_score, descending=True)
                if cls_score_sorted.shape[-1] < num_pruned_moves:
                    cls_score_pruned_ind = cls_score_sorted
                    legal_moves_pruned_ind = [legal_move_sublist[idx] for idx in sort_idx]
                else:
                    cls_score_pruned_ind = cls_score_sorted[:num_pruned_moves]
                    idx_pruned = sort_idx[:num_pruned_moves]
                    legal_moves_pruned_ind = [legal_move_sublist[idx] for idx in idx_pruned]
                    
                cls_score_pruned_ind /= torch.sum(cls_score_pruned_ind)
                cls_score_pruned.append(cls_score_pruned_ind)
                legal_move_list_pruned.append(legal_moves_pruned_ind)
                
            legal_move_list = legal_move_list_pruned
            cls_score_batch = cls_score_pruned

        return legal_move_list, cls_score_batch, board_value * 100 / 3

class EndHead(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, ripple=False) -> None:
        super().__init__()
        
        if not ripple:
            self.lin_1 = nn.Linear(hidden_dim, 1)
            self.activation = nn.ReLU()
            self.lin_2 = nn.Linear(input_dim, input_dim)
            self.lin_3 = nn.Linear(input_dim, 1)
            
        else:
            self.lin_1 = RippleLinear(hidden_dim, 1)
            self.activation = nn.ReLU()
            self.lin_2 = RippleLinear(input_dim, input_dim)
            self.lin_3 = RippleLinear(input_dim, 1)
            
    def forward(self, x):
        x = self.lin_1(x)
        x = self.activation(x)
        x = x.squeeze()
        x = self.lin_2(x)
        x = self.activation(x)
        x = self.lin_3(x)
        return x.squeeze()
        


class BoardEmbTrainNet(nn.Module):
    """A class to train the board embedding"""

    def __init__(self, hidden_size=64, emb_size=8):
        super(BoardEmbTrainNet, self).__init__()

        self.backbone_embedding = nn.Embedding(36, emb_size)
        self.piece_head = nn.Linear(emb_size, 7)
        self.info_head = nn.Linear(emb_size, 4)
        self.relu = nn.GELU()

    def forward(self, x):

        x = self.backbone_embedding(x)
        x_piece = self.piece_head(x)
        x_info = self.info_head(x)

        return x_piece, x_info


class MoveEmbTrainNet(nn.Module):
    """A class to train the move embedding"""

    def __init__(self, hidden_size=256, emb_size=8):
        super(MoveEmbTrainNet, self).__init__()

        self.query_embedding = nn.Embedding(4865, emb_size, padding_idx=4864)
        self.coor_head = nn.Linear(emb_size, 4)
        self.promotion_head = nn.Linear(emb_size, 5)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.query_embedding(x)
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
        """Pass the inputs (and mask) through the decoder layer in turn.

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
