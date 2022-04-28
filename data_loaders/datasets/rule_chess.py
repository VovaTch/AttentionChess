import copy

import torch
from torch.utils.data import Dataset
import chess
import chess.pgn
import logging
import numpy as np

from model.score_functions import ScoreScaling
from utils.util import move_to_coordinate

class RuleChessDataset(Dataset):

    # Set to ignore the errors, such that it won't flood the log
    logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

    def __init__(self, dataset_path, query_word_len=256, base_multiplier=0.95):
        super(RuleChessDataset, self).__init__()
        self.pgn = open(dataset_path, encoding="utf-8")
        self.game = None

        self.query_word_len = query_word_len
        self.follow_idx = 0
        self.game_length = 0
        self.board_collection = None
        self.move_quality_batch = None
        self.board_value_batch = None
        self.selected_move_idx = None
        self.base_multiplier = base_multiplier

    def __getitem__(self, _):

        if self.follow_idx == 0:
            while self.game_length == 0:
                self.load_game()
                self.game_length = len(self.board_collection)

        sampled_board = copy.deepcopy(self.board_collection[self.follow_idx])
        sampled_quality_batch = self.move_quality_batch[self.follow_idx, :].clone()
        sampled_board_value_batch = self.board_value_batch[self.follow_idx].clone()
        sampled_move_idx = self.selected_move_idx[self.follow_idx].clone()

        self.follow_idx += 1
        if self.follow_idx == self.game_length:
            self.follow_idx = 0
            self.game_length = 0

        return sampled_board, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx

    def load_game(self):

        while True:
            game = chess.pgn.read_game(self.pgn)
            move_counter = 0
            for move in enumerate(game.mainline_moves()):
                move_counter += 1
            last_move = 1 if move_counter % 2 == 1 else -1
            
            if 'Termination' in game.headers and game.headers['Termination'] == 'Normal' and move_counter > 0:
                break

        board = game.board()
        result = game.headers['Result']
        if result == '1-0':
            base_eval = 1
        elif result == '0-1':
            base_eval = -1
        else:
            base_eval = 0

        self.board_collection = []
        self.move_quality_batch = torch.zeros((0, self.query_word_len))
        self.board_value_batch = torch.zeros(0)
        board_value_list = []
        move_idx_list = []

        score_factor = copy.copy(base_eval)
        
        # Create the score function
        score_function = ScoreScaling(moves_to_end=move_counter, score_max=5)
            

        for idx, move in enumerate(game.mainline_moves()):

            # Fill the legal move matrix
            legal_move_mat = torch.zeros((1, 64, 76))
            quality_vector_logit = torch.zeros((1, self.query_word_len)) - torch.inf
            for idx_move, legal_move in enumerate(board.legal_moves):

                move_legal_coor = move_to_coordinate(legal_move)
                legal_move_mat[0, move_legal_coor[0], move_legal_coor[1]] = 1
                quality_vector_logit[0, idx_move] = 0

            legal_move_mat = legal_move_mat == 1
            legal_move_idx = torch.nonzero(legal_move_mat)
            legal_move_word = torch.cat((legal_move_idx[:, 0].unsqueeze(1),
                                         legal_move_idx[:, 1].unsqueeze(1) +
                                         64 * legal_move_idx[:, 2].unsqueeze(1)), 1)
            move_per_coor = move_to_coordinate(move)
            move_per_word = move_per_coor[0] + 64 * move_per_coor[1]

            # Find the correct move
            board_value = 0
            matching_idx = torch.nonzero(legal_move_word[:, 1] == move_per_word).squeeze().item()
            if score_factor > 0:
                quality_vector_logit[0, matching_idx] = abs(score_function(idx)) * 10
            elif score_factor < 0:
                quality_vector_logit[0, matching_idx] = 0# -abs(score_function(idx)) * 10
                
            board_value = np.tanh(base_eval * abs(score_function(idx)))

            quality_vector = quality_vector_logit.softmax(dim=1)

            # self.move_collection = torch.cat((self.move_collection, move_tensor.unsqueeze(0)), 0)
            board_recorded = copy.deepcopy(board)
            board.push(move)

            # Collect all the data
            # if score_factor * base_eval == 1:
            board_value_list.append(board_value)
            move_idx_list.append(matching_idx)
            self.board_collection.append(board_recorded)
            self.move_quality_batch = torch.cat((self.move_quality_batch, quality_vector), 0)
            
                
            score_factor *= -1

        self.board_value_batch = torch.tensor(board_value_list)
        print(move_idx_list)
        self.selected_move_idx = torch.tensor(move_idx_list)
        # print(self.selected_move_idx)

    def __len__(self):
        return int(1e6)