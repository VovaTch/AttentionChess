import chess.pgn
import torch
from torch.utils.data import Dataset
from base import BaseDataLoader
import numpy as np

from utils.util import board_to_tensor, board_to_tensor_full, move_to_tensor, legal_move_mask
from .game_roll import GameRoller


class S1AttentionChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn',
                 shuffle=True, validation_split=0.1, num_workers=1, training=True):

        self.dataset_path = data_dir
        self.dataset = S1ChessDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class S1ChessDataset(Dataset):

    def __init__(self, dataset_path):
        super(S1ChessDataset, self).__init__()
        self.pgn = open(dataset_path, encoding="utf-8")

    def __getitem__(self, idx):

        while True:
            game = chess.pgn.read_game(self.pgn)
            if game.headers['Termination'] == 'Normal':
                break

        board = game.board()
        result = game.headers['Result']
        if result == '1-0':
            base_eval = 1
        elif result == '0-1':
            base_eval = -1
        else:
            base_eval = 0

        board_collection = board_to_tensor_full(board).unsqueeze(0)
        move_collection = torch.zeros((0, 6))

        for idx, move in enumerate(game.mainline_moves()):

            move_tensor = move_to_tensor(move)
            move_tensor[3] = 10
            move_tensor[0: 2] += 0.5
            if board_collection[-1, 1, 0, 0] == 1:
                move_tensor[4] = base_eval
            else:
                move_tensor[4] = -base_eval

            move_collection = torch.cat((move_collection, move_tensor.unsqueeze(0)), 0)
            board.push(move)
            board_new = board_to_tensor_full(board).unsqueeze(0)
            board_collection = torch.cat((board_collection, board_new), 0)

        move_last = torch.tensor([0, 0, 0, 10, 0, np.abs(base_eval) + 0.5]) # Last move; resigning is qualified as checkmate here.
        move_collection = torch.cat((move_collection, move_last.unsqueeze(0)), 0)

        return board_collection, move_collection

    def __len__(self):
        return int(2e3)


class S2AttentionChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, adversarial_model, shuffle=True, validation_split=0.0, num_workers=1,
                 move_limit=250, training=True, device='cuda'):

        self.dataset = S2ChessDataset(move_limit=move_limit, adversarial_model=adversarial_model, device=device)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class S2ChessDataset(Dataset):
    """Self playing dataset, shortest game reward"""

    def __init__(self, move_limit, adversarial_model, game_roller, device='cuda'):

        self.board = chess.Board()
        super(S2ChessDataset, self).__init__()
        self.move_limit = move_limit
        self.adversarial_model = adversarial_model
        self.game_roller = game_roller
        self.game_roller.move_limit = move_limit

    def __getitem__(self, _):

        turn = []
        turn_ind = True

        with torch.no_grad():
            result = self.game_roller.roll_game(self.board)
            move_mat_buffer = self.game_roller.get_selected_move_buffer()
            board_buffer = self.game_roller.get_board_buffer()

            if result['result'] == 1:
                cost_multiplier = 1
            elif result['result'] == -1:
                cost_multiplier = -1
            else:
                cost_multiplier = 0

            game_length = board_buffer.size()[0]

            for idx in range(game_length):
                turn.append(turn_ind)
                turn_ind = not turn_ind
                move_mat_buffer[idx][move_mat_buffer[idx] == 1] *= cost_multiplier
                cost_multiplier *= -1

        self.game_roller.reset_buffers()
        return board_buffer, turn, move_mat_buffer

    def __len__(self):
        return int(1)


class RuleAttentionChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn',
                 shuffle=True, validation_split=0.1, num_workers=1, training=True):

        self.dataset_path = data_dir
        self.dataset = RuleChessDataset(data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class RuleChessDataset(Dataset):

    def __init__(self, dataset_path):
        super(RuleChessDataset, self).__init__()
        self.pgn = open(dataset_path, encoding="utf-8")

    def __getitem__(self, idx):

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

        board_collection = board_to_tensor_full(board).unsqueeze(0)
        move_collection = torch.zeros((0, 6))
        legal_move_batch = torch.zeros((0, 200, 6))

        for idx, move in enumerate(game.mainline_moves()):

            move_tensor = move_to_tensor(move)
            move_tensor[3] = 10
            move_tensor[0: 2] += 0.5
            if board_collection[-1, 1, 0, 0] == 1:
                move_tensor[4] = base_eval
            else:
                move_tensor[4] = -base_eval

            legal_move_collection = torch.zeros((0, 6))
            for legal_move in board.legal_moves:

                legal_move_tensor = move_to_tensor(legal_move)
                legal_move_tensor[3] = 10
                legal_move_tensor[0: 3] += 0.5
                legal_move_tensor[5] += 0.5
                if board_collection[-1, 1, 0, 0] == 1:
                    legal_move_tensor[4] = 0
                else:
                    legal_move_tensor[4] = 0
                legal_move_collection = torch.cat((legal_move_collection, legal_move_tensor.unsqueeze(0)), 0)

            while legal_move_collection.size()[0] < 200:
                illegal_move = torch.zeros((1, 6))
                illegal_move[0, 3] = -100
                legal_move_collection = torch.cat((legal_move_collection, illegal_move), 0)

            legal_move_batch = torch.cat((legal_move_batch, legal_move_collection.unsqueeze(0)), 0)

            move_collection = torch.cat((move_collection, move_tensor.unsqueeze(0)), 0)
            board.push(move)
            board_new = board_to_tensor_full(board).unsqueeze(0)
            board_collection = torch.cat((board_collection, board_new), 0)

        if result == '1-0' or result == '0-1':
            move_last = torch.tensor([0, 0, 0, 10, 0, 1.5]).unsqueeze(0)
        else:
            move_last = torch.tensor([0, 0, 0, 10, 0, -0.5]).unsqueeze(0)
        move_last_after = torch.tensor([0, 0, 0, -100, 0, 0]).unsqueeze(0).repeat((1, 199, 1)) # Last move; resigning is qualified as checkmate here.
        move_last_after = torch.cat((move_last.unsqueeze(0), move_last_after), 1)
        legal_move_batch = torch.cat((legal_move_batch, move_last_after), 0)

        # trying to figure out how to get rid of the data leaks
        del board_new, illegal_move, legal_move_collection, move_collection, move_last, move_last_after, move_tensor, \
            legal_move_tensor

        return board_collection, legal_move_batch

    def __len__(self):
        return int(5e3)

