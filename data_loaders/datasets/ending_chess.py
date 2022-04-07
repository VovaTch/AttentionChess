import copy

import torch
from torch.utils.data import Dataset
import chess
import chess.pgn

class EndingChessDataset(Dataset):

    def __init__(self, dataset_path, query_word_len=256, base_multiplier=0.95):
        super(EndingChessDataset, self).__init__()
        self.pgn = open(dataset_path, encoding="utf-8")
        self.game = None

        self.query_word_len = query_word_len
        self.game_length = 0
        self.board_collection = None
        self.move_quality_batch = None
        self.board_value_batch = None
        self.selected_move_idx = None
        self.base_multiplier = base_multiplier

    def __getitem__(self, _):

        while self.game_length == 0:
            self.load_game()
            self.game_length = len(self.board_collection)

        sampled_board = copy.deepcopy(self.board_collection[0])
        sampled_quality_batch = self.move_quality_batch[0].clone()
        sampled_board_value_batch = self.board_value_batch[0].clone()
        sampled_move_idx = self.selected_move_idx[0].clone()

        return sampled_board, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx

    def load_game(self):

        while True:
            board_collection_ind = list()
            game = chess.pgn.read_game(self.pgn)
            board_game = chess.Board()
            board_collection_ind.append(copy.deepcopy(board_game))
            move_counter = 0
            for move in game.mainline_moves():
                move_counter += 1
                board_game.push(move)
                board_collection_ind.append(copy.deepcopy(board_game))
            
            if 'Termination' in game.headers and game.headers['Termination'] == 'Normal' and move_counter > 0:
                
                result = game.headers['Result']
                
                if board_game.is_checkmate and result == '1-0': # White wins
                    base_eval = 1
                    break
                elif board_game.is_checkmate and result == '0-1': # Black wins
                    base_eval = -1
                    break
                elif result == '0-0':
                    base_eval = 0
                    break
                
        self.board_collection = [board_game]
        self.move_quality_batch = torch.zeros((1, self.query_word_len))
        self.board_value_batch = torch.zeros(1)
        board_value_list = [base_eval]
        move_idx_list = [-1]

        self.board_value_batch = torch.tensor(board_value_list)
        self.selected_move_idx = torch.tensor(move_idx_list)

    def __len__(self):
        return int(1e6)
