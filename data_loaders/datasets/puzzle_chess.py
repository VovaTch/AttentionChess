import copy

import chess
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from model.score_functions import ScoreScaling
from utils.util import move_to_coordinate

class PuzzleChessDataset(Dataset):

    def __init__(self, dataset_path, query_word_len=256):
        super(PuzzleChessDataset, self).__init__()
        self.puzzle_path = dataset_path
        self.puzzle_df = pd.read_csv(dataset_path)
            
        self.game = None

        self.query_word_len = query_word_len
        self.follow_idx = 0
        self.game_length = 0
        self.board_collection = None
        self.move_quality_batch = None
        self.board_value_batch = None
        self.selected_move_idx = None

    def __getitem__(self, item):

        board_collection, move_quality_batch, board_value_batch, selected_move_idx = self.load_puzzle(item)
        return board_collection, move_quality_batch, board_value_batch, selected_move_idx
        
    
    def load_puzzle(self, item):
        
        board = chess.Board(fen=self.puzzle_df[item, 1])
        moves_str = self.puzzle_df[item, 2]
        moves_list = moves_str.split()
        moves_uci = [chess.Move(move_ind) for move_ind in moves_list]
        
        # Create the score function
        score_function = ScoreScaling(moves_to_end=len(moves_uci), score_max=3)
        starting_move = board.turn
        
        # Resetting the variables
        board_collection = []
        move_quality_batch = torch.zeros((0, self.query_word_len))
        board_value_list = []
        move_idx_list = []
        
        # Loop over puzzle moves
        for move_idx, move in enumerate(moves_uci):

            # Find the GT move index
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
            matching_idx = torch.nonzero(legal_move_word[:, 1] == move_per_word)
            if starting_move:
                quality_vector_logit[0, matching_idx] = score_function(move_idx) * 10
            else:
                quality_vector_logit[0, matching_idx] = -score_function(move_idx) * 10
                
            board_value = np.tanh(score_function(move_idx)) if starting_move else -np.tanh(score_function(move_idx))

            quality_vector = quality_vector_logit.softmax(dim=1)

            # self.move_collection = torch.cat((self.move_collection, move_tensor.unsqueeze(0)), 0)
            board_recorded = copy.deepcopy(board)
            board.push(move)
            
            # Collect all the data
            board_value_list.append(board_value)
            move_idx_list.append(matching_idx)
            board_collection.append(board_recorded)
            move_quality_batch = torch.cat((move_quality_batch, quality_vector), 0)
            
        board_value_batch = torch.tensor(board_value_list)
        selected_move_idx = torch.tensor(move_idx_list)
        
        return board_collection, move_quality_batch, board_value_batch, selected_move_idx

    def __len__(self):
        return int(self.puzzle_df)