import copy

import chess.pgn
import torch
from torch.utils.data import Dataset
from base import BaseDataLoader
import numpy as np

from utils.util import move_to_tensor, move_to_coordinate, board_to_embedding_coord
from .game_roll import GameRoller


class BoardEmbeddingLoader(BaseDataLoader):
    """Dataloader for board square embedding training, 36 possible words."""
    def __init__(self, batch_size, shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset = BoardEmbeddingDataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BoardEmbeddingDataset(Dataset):
    """Dataloader for board square embedding training, 36 possible words."""
    def __init__(self):
        super(BoardEmbeddingDataset, self).__init__()

    def __getitem__(self, idx):
        """From the index, outputs 2 vectors, a probability vector of pieces, and a probability vector of properties."""

        # Piece probabilities
        target_prob_vector = torch.zeros(7)
        white_prob = 0  # 1 for white, 0 for black
        ep_flag = 0
        castling_right = 1  # 1 for there is right, 0 for there isn't
        turn = 1  # 1 for white, 0 for black

        if idx in [7, 25]:
            target_prob_vector[1] = 1  # white pawn
            white_prob = 1
        elif idx in [8, 26]:
            target_prob_vector[2] = 1  # white knight
            white_prob = 1
        elif idx in [9, 27]:
            target_prob_vector[3] = 1  # white bishop
            white_prob = 1
        elif idx in [10, 28]:
            target_prob_vector[4] = 1  # white rook
            white_prob = 1
        elif idx in [11, 29]:
            target_prob_vector[5] = 1  # white queen
            white_prob = 1
        elif idx in [12, 30]:
            target_prob_vector[6] = 1  # white king
            white_prob = 1
            castling_right = 0
        elif idx in [14, 32]:
            target_prob_vector[6] = 1  # white king
            white_prob = 1
        elif idx in [5, 23]:
            target_prob_vector[1] = 1  # black pawn
        elif idx in [4, 22]:
            target_prob_vector[2] = 1  # black knight
        elif idx in [3, 21]:
            target_prob_vector[3] = 1  # black bishop
        elif idx in [2, 20]:
            target_prob_vector[4] = 1  # black rook
        elif idx in [1, 19]:
            target_prob_vector[5] = 1  # black queen
        elif idx in [0, 18]:
            target_prob_vector[6] = 1  # black king
            castling_right = 0
        elif idx in [15, 33]:
            target_prob_vector[6] = 1  # black king
        elif idx in [16, 17, 34, 35]:
            target_prob_vector[0] = 1
            ep_flag = 1
        else:
            target_prob_vector[0] = 1

        if idx >= 18:
            turn = 0

        flags_vector = torch.tensor([white_prob, ep_flag, castling_right, turn])
        return target_prob_vector, flags_vector

    def __len__(self):
        return 38


class MoveEmbeddingLoader(BaseDataLoader):
    """Dataloader for move embedding training, 4864 possible words. """
    def __init__(self, batch_size, shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset = MoveEmbeddingDataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MoveEmbeddingDataset(Dataset):
    """Dataset for moves, 4864 moves, you get word in and get out move coordinates and possible promotion
    classification vector."""
    def __init__(self):
        super(MoveEmbeddingDataset, self).__init__()

    def __getitem__(self, word):

        # Initialize promotion probability vector: 0: no promotion, 1: queen, 2: rook, 3: bishop, 4: knight
        promotion_prob = torch.zeros(5)

        # Decompose to individual move coordinates
        coordinates_from_to = (int(word % 64), word // 64)
        coordinates_from = (int(coordinates_from_to[0] % 8), coordinates_from_to[0] // 8)  # 0 is a,b,c... 1 is numbers

        # If not promoting
        if coordinates_from_to[1] < 64:
            coordinates_to = (int(coordinates_from_to[1] % 8), coordinates_from_to[1] // 8)
            promotion_prob[0] = 1

        # If promoting
        else:
            if 64 <= coordinates_from_to[1] < 67:
                coor_shift = 65 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[1] = 1
            elif 67 <= coordinates_from_to[1] < 70:
                coor_shift = 68 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[2] = 1
            elif 70 <= coordinates_from_to[1] < 73:
                coor_shift = 71 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[3] = 1
            else:
                coor_shift = 74 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[4] = 1

            coordinates_to = (int(coordinates_from[0] - coor_shift), int(coor_up_down))

        coor_tensor = torch.tensor([coordinates_from[0], coordinates_from[1], coordinates_to[0], coordinates_to[1]]) / 7

        return coor_tensor, promotion_prob

    def __len__(self):
        return 4864


class RuleAttentionChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, collate_fn, data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn',
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset_path = data_dir
        self.dataset = RuleChessDataset(data_dir, query_word_len=query_word_len)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)


class RuleChessDataset(Dataset):

    def __init__(self, dataset_path, query_word_len=256):
        super(RuleChessDataset, self).__init__()
        self.pgn = open(dataset_path, encoding="utf-8")
        self.game = None

        self.query_word_len = query_word_len
        self.follow_idx = 0
        self.game_length = 0
        self.board_collection = None
        self.move_quality_batch = None

    def __getitem__(self, _):

        if self.follow_idx == 0:
            self.load_game()
            self.game_length = len(self.board_collection)

        sampled_board = copy.deepcopy(self.board_collection[self.follow_idx])
        sampled_quality_batch = self.move_quality_batch[self.follow_idx, :].clone()

        self.follow_idx += 1
        if self.follow_idx == self.game_length:
            self.follow_idx = 0

        return sampled_board, sampled_quality_batch

    def load_game(self):

        while True:
            game = chess.pgn.read_game(self.pgn)
            if 'Termination' in game.headers and game.headers['Termination'] == 'Normal':
                break

        board = game.board()
        result = game.headers['Result']
        if result == '1-0':
            base_eval = 1
        elif result == '0-1':
            base_eval = -1
        else:
            base_eval = 0

        self.board_collection = [board]
        self.move_quality_batch = torch.zeros((0, self.query_word_len))

        score_factor = base_eval

        for idx, move in enumerate(game.mainline_moves()):

            # Fill the legal move matrix
            legal_move_mat = torch.zeros((1, 64, 76))
            quality_vector = torch.zeros((1, self.query_word_len))
            quality_vector[0, -1] = 0
            for idx_move, legal_move in enumerate(board.legal_moves):

                move_legal_coor = move_to_coordinate(legal_move)
                legal_move_mat[0, move_legal_coor[0], move_legal_coor[1]] = 1
                quality_vector[0, idx_move] = 1e-10

            legal_move_mat = legal_move_mat == 1
            legal_move_idx = torch.nonzero(legal_move_mat)
            legal_move_word = torch.cat((legal_move_idx[:, 0].unsqueeze(1),
                                         legal_move_idx[:, 1].unsqueeze(1) +
                                         64 * legal_move_idx[:, 2].unsqueeze(1)), 1)
            move_per_coor = move_to_coordinate(move)
            move_per_word = move_per_coor[0] + 64 * move_per_coor[1]

            # Find the correct move
            matching_idx = torch.nonzero(legal_move_word[:, 1] == move_per_word)
            if score_factor == 1:
                quality_vector[0, matching_idx] = 1
            elif score_factor == -1:
                quality_vector[0, matching_idx] = -1

            quality_vector[0, :-1] = quality_vector[0, :-1] / (torch.sum(quality_vector[0, :-1]) + 1e-6)

            # self.move_collection = torch.cat((self.move_collection, move_tensor.unsqueeze(0)), 0)
            board_new = copy.deepcopy(board)
            board_new.push(move)
            board.push(move)
            self.board_collection.append(board_new)

            # concat also legals + quality
            self.move_quality_batch = torch.cat((self.move_quality_batch, quality_vector), 0)

        # Fill the legal move matrix
        quality_vector = torch.zeros((1, self.query_word_len))
        for idx_move, legal_move in enumerate(board.legal_moves):

            quality_vector[0, idx_move] = 1e-10

        if result == '1-0' or result == '0-1':
            quality_vector[0, self.query_word_len - 1] = 1
        else:
            quality_vector[0, self.query_word_len - 1] = 0

        quality_vector[0, :-1] = quality_vector[0, :-1] / (torch.sum(quality_vector[0, :-1]) + 1e-6)

        # concat also legals + quality
        self.move_quality_batch = torch.cat((self.move_quality_batch, quality_vector), 0)

    def get_item_new(self, _):
        pass

    def __len__(self):
        return int(1e6)


def collate_fn(batch):
    """Required collate function because the boards are a unique class"""
    chess_boards = [batch[idx][0] for idx in range(len(batch))]
    quality_vectors = torch.zeros((len(batch), batch[0][1].size()[0]))
    for idx in range(len(batch)):
        quality_vectors[idx, :] = batch[idx][1]
    return chess_boards, quality_vectors


