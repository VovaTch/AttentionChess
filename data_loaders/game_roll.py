import torch
import chess
import numpy as np
import copy

from model.attchess import AttChess
from utils.util import board_to_tensor, legal_move_mask


class GameRoller:
    """A class for the model to play against itself. It plays a game and outputs the moves + result"""

    @torch.no_grad()
    def __init__(self, model_good: AttChess, model_evil: AttChess, device='cuda', move_limit=150):
        """Loads the model"""
        self.model_good = copy.deepcopy(model_good)
        self.model_evil = copy.deepcopy(model_evil)
        self.device = device
        self.move_limit = move_limit

        self.board_buffer = []
        self.move_buffer = []
        self.move_vec_buffer = []
        self.reward_vec_buffer = []
        self.result = {}  # a dict that describes what happened and with how many moves

    @torch.no_grad()
    def roll_game(self, init_board: chess.Board, num_of_branches=1):
        """Plays the entire game"""
        # torch.multiprocessing.set_start_method('spawn') TODO: CODE THE FUNCTION, MAYBE APPLY MULTIPLE GAMES SIMULTANEOUSLY

        # Initialization, limiting game length to the constant
        move_count = init_board.fullmove_number
        turn = init_board.turn
        is_good = True

        # Add board to buffer
        self.board_buffer.append([copy.deepcopy(init_board) for idx in range(num_of_branches)])
        board_idx_buffer = [[idx for idx in range(num_of_branches)]]

        while True:

            if is_good:
                outputs_legal, outputs_class_vec = self.model_good.board_forward(self.board_buffer[-1])
                legal_move_list, cls_vec, endgame_flag = self.model_good.post_process(outputs_legal, outputs_class_vec)
            else:
                outputs_legal, outputs_class_vec = self.model_evil.board_forward(self.board_buffer[-1])
                legal_move_list, cls_vec, endgame_flag = self.model_evil.post_process(outputs_legal, outputs_class_vec)

            # New board idx buffer
            new_board_idx = copy.copy(board_idx_buffer[-1])
            board_idx_buffer.append(new_board_idx)

            # Collect individual boards and moves. If one game ends, reroll reward to the beginning.
            model_sub_buffer = []
            move_sub_buffer = []
            move_vec_sub_buffer = []
            reward_vec_sub_buffer = []

            # Simpler one
            init_move_list = [move for move in init_board.legal_moves]
            final_probability_vec = torch.zeros((len(init_move_list))).to(self.device) + 1e-10

            for idx, (legal_move_ind, cls_vec_ind) in enumerate(zip(legal_move_list, cls_vec)):

                # Sample moves
                cat = torch.distributions.Categorical(cls_vec_ind)
                sample_idx = cat.sample()
                sample = legal_move_ind[sample_idx]
                move_sub_buffer.append(sample)

                # Move to buffer
                copied_board = copy.deepcopy(self.board_buffer[-1][idx])
                copied_board.push(sample)
                model_sub_buffer.append(copied_board)
                move_vec_sub_buffer.append(copied_board.legal_moves)
                reward_vec_sub_buffer.append(torch.zeros(len(copied_board.legal_moves)).to(self.device))

                # End game conditions
                if model_sub_buffer[-1].is_checkmate():
                    print(f'Victory to white: {self.board_buffer[-1][idx].turn}')  # TODO: TEMP
                    model_sub_buffer.pop()
                    init_move_idx = board_idx_buffer[-1][idx]  # index of initial move
                    board_idx_buffer[-1].pop()


                if model_sub_buffer[-1].is_stalemate():
                    print('Stalemate')
                    model_sub_buffer.pop()
                    board_idx_buffer[-1].pop()
                if model_sub_buffer[-1].is_insufficient_material():
                    print('Insufficient Material draw')
                    model_sub_buffer.pop()
                    board_idx_buffer[-1].pop()
                if model_sub_buffer[-1].can_claim_threefold_repetition():
                    print('Repetition')
                    model_sub_buffer.pop()
                    board_idx_buffer[-1].pop()

            self.board_buffer.append(model_sub_buffer)
            self.move_buffer.append(move_sub_buffer)

            print(self.board_buffer[-1][0].fullmove_number)

            # Too many moves break
            if self.board_buffer[-1][0].fullmove_number > self.move_limit:
                # TODO: There may be more
                break

        pass

    @torch.no_grad()
    def reset_buffers(self):

        self.board_buffer = []
        self.move_buffer = []

    @torch.no_grad()
    def get_board_buffer(self):
        return self.board_buffer

    @torch.no_grad()
    def move_word_buffer(self):
        return self.move_buffer






