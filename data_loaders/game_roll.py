import torch
import chess
import numpy as np
import copy

from model.attchess import AttChess
from utils.util import board_to_tensor, legal_move_mask


class GameRoller:
    """A class for the model to play against itself. It plays a game and outputs the moves + result"""

    def __init__(self, model: AttChess, device='cuda', move_limit=100):
        """Loads the model"""
        self.model = copy.deepcopy(model)
        self.device = device
        self.move_limit = move_limit

        self.board_buffer = torch.zeros((0, 8, 8)).to(self.device)
        self.move_mat_buffer = torch.zeros((0, 64, 64)).to(self.device)
        self.selected_move_buffer = torch.zeros((0, 64, 64)).to(self.device)

    def roll_game(self, board: chess.Board):
        """Plays the entire game"""
        # torch.multiprocessing.set_start_method('spawn')
        board_torch = board_to_tensor(board).unsqueeze(0)
        board_torch = board_torch.to(self.device)
        turn = [board.turn]

        while True:

            self.board_buffer = torch.cat((self.board_buffer, board_torch), 0)

            # Run the network to determine the next move, filter illegal moves and normalize
            move_mat = self.model(board_torch, turn)
            legal_move_matrix = legal_move_mask(board).to(self.device).unsqueeze(0)
            move_mat += legal_move_matrix
            move_mat = torch.exp(move_mat) / torch.sum(torch.exp(move_mat))

            self.move_mat_buffer = torch.cat((self.move_mat_buffer, move_mat), 0)

            # Sample the move
            sampled_move_torch_flatten = torch.multinomial(move_mat[0].float().flatten(), 1)
            sampled_move_torch = (sampled_move_torch_flatten % 64, torch.div(sampled_move_torch_flatten,
                                                                             64, rounding_mode='floor'))
            sampled_move = chess.Move(sampled_move_torch[1], sampled_move_torch[0])
            sampled_move_matrix = torch.zeros((1, 64, 64)).to(self.device)
            sampled_move_matrix[0, sampled_move_torch[1], sampled_move_torch[0]] = 1
            sampled_move_matrix[sampled_move_matrix != 1] = -np.inf

            self.selected_move_buffer = torch.cat((self.selected_move_buffer, sampled_move_matrix), 0)

            if sampled_move.promotion:
                print('Promotion, deal with it')  # TODO: Deal with promotions

            # Plan the sampled move
            board.push(sampled_move)
            board_torch = board_to_tensor(board).to(self.device).unsqueeze(0)

            # Conditions for termination
            num_moves = self.move_mat_buffer.size()[0]
            if num_moves > self.move_limit:  # Exceeds the number of moves allowed
                result = {'result': 0, 'moves': num_moves}
                break
            if board.is_checkmate():  # Someone wins by checkmate
                if turn:
                    result = {'result': 1, 'moves': num_moves}
                else:
                    result = {'result': -1, 'moves': num_moves}
                break
            if board.is_insufficient_material() or board.is_fivefold_repetition() \
                    or board.is_seventyfive_moves() or board.is_stalemate():  # Drawing conditions
                result = {'result': 0, 'moves': num_moves}
                break
            turn[-1] = not turn[-1]

        return result

    def reset_buffers(self):

        self.board_buffer = torch.zeros((0, 8, 8)).to(self.device)
        self.move_mat_buffer = torch.zeros((0, 64, 64)).to(self.device)
        self.selected_move_buffer = torch.zeros((0, 64, 64)).to(self.device)

    def get_board_buffer(self):
        return self.board_buffer

    def get_move_mat_buffer(self):
        return self.move_mat_buffer

    def get_selected_move_buffer(self):
        return self.selected_move_buffer






