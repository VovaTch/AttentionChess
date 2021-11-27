import chess.pgn
import torch
from torch.utils.data import Dataset
from base import BaseDataLoader
import numpy as np

from utils.util import board_to_tensor, legal_move_mask
from .game_roll import GameRoller


# TODO: see if the -inf on all but the played moves is needed

class S1AttentionChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, data_dir='liches_data/lichess_db_standard_rated_2016-09.pgn',
                 shuffle=True, validation_split=0.0, num_workers=1, training=True):

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

        board_collection = board_to_tensor(board).unsqueeze(0)
        reward_collection = torch.zeros([0, 64, 64])
        turn_collection = [True]

        for idx, move in enumerate(game.mainline_moves()):

            ind_reward = torch.zeros([1, 64, 64]) - np.inf
            ind_reward[0, move.from_square, move.to_square] = base_eval
            legal_move_mat = legal_move_mask(board)
            ind_reward += legal_move_mat
            reward_collection = torch.cat((reward_collection, ind_reward), 0)

            base_eval = -base_eval

            board.push(move)
            board_next = board_to_tensor(board).unsqueeze(0)
            board_collection = torch.cat((board_collection, board_next), 0)
            turn_collection.append(not turn_collection[-1])

        return board_collection[:-1], turn_collection[:-1], reward_collection

    def __len__(self):
        return int(1e6)


class S2AttentionChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, adversarial_model, game_roller, shuffle=True, validation_split=0.0, num_workers=1,
                 move_limit=250, training=True, device='cuda'):

        self.dataset = S2ChessDataset(move_limit=move_limit, adversarial_model=adversarial_model, device=device,
                                      game_roller=game_roller)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class S2ChessDataset(Dataset):
    """Self playing dataset, shortest game reward"""

    def __init__(self, move_limit, adversarial_model, game_roller, device='cuda'):

        self.board = chess.Board()
        super(S2ChessDataset, self).__init__()
        self.move_limit = move_limit
        self.adversarial_model = adversarial_model
        self.game_roller = game_roller

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

        return board_buffer, turn, move_mat_buffer

    def __len__(self):
        return int(1)

