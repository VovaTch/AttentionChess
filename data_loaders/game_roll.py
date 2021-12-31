import torch
import chess
import numpy as np
import copy
import random

from model.attchess import AttChess
from utils.util import board_to_tensor, legal_move_mask


class GameRoller:
    """A class for the model to play against itself. It plays a game and outputs the moves + result"""

    @torch.no_grad()
    def __init__(self, model_good: AttChess, model_evil: AttChess, device='cuda', move_limit=300, argmax_start=300):
        """Loads the model"""
        self.model_good = copy.deepcopy(model_good)
        self.model_evil = copy.deepcopy(model_evil)
        self.device = device
        self.move_limit = move_limit
        self.argmax_start = argmax_start

        self.board_buffer = []
        self.move_buffer = []
        self.move_vec_buffer = []
        self.reward_vec_buffer = []
        self.result = {}  # a dict that describes what happened and with how many moves

    @torch.no_grad()
    def roll_game(self, init_board: chess.Board, num_of_branches=1, expansion_constant=0.0):
        """Plays the entire game"""

        # Score function  TODO: this is hard coded, change
        score = ScoreWinFast(100)

        # Initializing board list
        board_init = [copy.deepcopy(init_board) for idx in range(num_of_branches)]
        board_list = board_init
        current_nodes = [BoardNode(board_ind, None, score, self.device) for board_ind in board_init]
        init_node = current_nodes[0]
        init_flag = True # Use this flag for initial sampling, such that we can collapse everything to a single list+ tensor
        
        while True:
            
            board_list_new = []
            new_nodes = []
            
            used_model = self.model_good if board_list[0].turn else self.model_evil  
            raw_legals, raw_outputs = used_model.board_forward(board_list)
            legal_moves, quality_vectors, _ = used_model.post_process(raw_legals, raw_outputs)
            
            for current_node, legal_move, quality_vec in zip(current_nodes, legal_moves, quality_vectors):
                
                # Check if we reached the endgame
                move_num = current_node.moves_performed
                if move_num + 1 >= self.argmax_start:
                    sample_move = False
                else:
                    sample_move = True
                    
                # Create node; append to new list only if the endgame wasn't reached
                if init_flag:
                    
                    new_node = init_node.sample_move(legal_move, quality_vec, sample_move=sample_move)
                    if not new_node.endgame_flag:
                        new_nodes.append(new_node)
                        board_list_new.append(new_node.board)
                    else:
                        new_node.propagate_score()
                        
                else: 
                    while True:
                        new_node = current_node.sample_move(legal_move, quality_vec, sample_move=sample_move)
                        new_node.score_function.last_move_idx = move_num + 2  # The +2 hack is to prevent division by zero
                        
                        if not new_node.endgame_flag and move_num <= self.move_limit:  # TODO: once debugged get rid of the end condition
                            new_nodes.append(new_node)
                            board_list_new.append(new_node.board)
                        else:
                            print('Game ended')
                            new_node.propagate_score()
                        
                        # Create random expansion
                        random_num = random.uniform(0, 1)
                        if random_num > expansion_constant:
                            break
                        else:
                            print('New branch')
                
                    
            # After the initial roll, the next rolls are from the tree.
            init_flag = False
                
            current_nodes = new_nodes
            board_list = board_list_new
            
            # Check if all nodes reached endgame state
            if len(current_nodes) == 0 or move_num > self.move_limit:
                break
            
        self.board_buffer, self.reward_vec_buffer = init_node.flatten_tree()  # TODO: perform additive operation

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


class BoardNode:
    """Create a board node for the MCTS.
    Include forward rolls for moves and backward rolls for quality vector updates. Score is based on length of game and the board"""

    def __init__(self, board: chess.Board, parent, score_function, device='cuda', moves_performed=0):

        self.device = device

        self.score_function = score_function
        self.num_legal_moves = 0
        self.legal_move_list = []
        for move in board.legal_moves:
            self.num_legal_moves += 1
            self.legal_move_list.append(move)

        self.quality_vector_logit = torch.zeros(
            self.num_legal_moves).to(device)

        self.parent = parent  # Board node if not first move, None if it is
        self.board = board
        self.children = []
        self.endgame_flag, self.result = self._is_game_end()
        self.moves_performed = moves_performed

    def perform_move(self, move):
        """Get a move and create a child node"""
        board_copy = copy.deepcopy(self.board)
        board_copy.push(move)
        new_child = BoardNode(board_copy, self, self.score_function, device=self.device, moves_performed=self.moves_performed)
        new_child.moves_performed += 1
        print(f'MOves performed: {new_child.moves_performed}')
        self.children.append(new_child)
        return new_child

    def sample_move(self, legal_move_list, quality_vector, sample_move=True):
        """Sample a move given outputs from an engine and create a child node"""
        
        if sample_move:
            cat = torch.distributions.Categorical(quality_vector)
            sample_idx = cat.sample()
        else:
            sample_idx = torch.argmax(quality_vector)
        sample = legal_move_list[sample_idx]
        new_child = self.perform_move(sample)

        return new_child

    def propagate_score(self):
        """Once the game has ended, propogate down the score"""

        # If there is no parent, we reached the bottom of the barrel
        if self.parent is None:
            return

        turn_variable = 1 if self.board.turn is False else -1
        self.parent.score_function = self.score_function
        current_score = self.score_function(self.moves_performed, self.board)
        last_move = self.board.peek()
        self.parent.result = self.result
        quality_idx = [idx for idx, move in enumerate(self.parent.legal_move_list) if move == last_move]
        self.parent.quality_vector_logit[quality_idx[0]] += current_score * self.result * turn_variable  # TODO: Check if works. The 1 replaced the result
        self.parent.propagate_score()  # Recursively apply the score propogation

    def flatten_tree(self):
        """Call on tree root to collapse all boards into a single vector, with the corresponding quality vectors. 
        This is the most essential method, it prepares the data for learning."""

        # If there is no child, i.e. the game ended, return an empty stack. Otherwise, start collecting
        
        move_tensor = torch.zeros((0, 256)).to(self.device)

        if len(self.children) == 0:
            print(f'Result: {self._is_game_end()}')
            return [], move_tensor
        else:
            # board_list = [child.board for child in self.children if len(child.children) != 0]
            board_list = [self.board]

        turn_variable = 1 if self.board.turn is False else -1
        
        quality_vector_append = torch.zeros((1, 256)).to(self.device) - torch.inf
        quality_vector_append[0, :self.quality_vector_logit.size()[0]] = self.quality_vector_logit
        quality_vector_append[0, -1] = turn_variable * self.result
        move_tensor = torch.cat((move_tensor, quality_vector_append))

        # Extract everything from the children nodes recursively
        for child in self.children:
            ind_board, ind_move_tensor = child.flatten_tree()
            if ind_move_tensor.dim() == 1:
                ind_move_tensor = ind_move_tensor.unsqueeze(0)
            board_list.extend(ind_board)
            move_tensor = torch.cat((move_tensor, ind_move_tensor), 0)

        # Return the collapsed tree
        return board_list, move_tensor

    def _is_game_end(self):
        """Checks if the game ends."""
        if self.board.is_checkmate():
            return True, -1 * self.board.turn + 1 * (not self.board.turn)
        elif self.board.is_stalemate() or self.board.is_repetition() or \
                self.board.is_seventyfive_moves() or self.board.is_insufficient_material():
            return True, 0
        return False, 0


class ScoreWinFast:

    def __init__(self, moves_to_end, score_max=100):
        self.last_move_idx = moves_to_end
        self.score_max = score_max

    def __call__(self, move_idx, board: chess.Board):
        score = self.score_max / (self.last_move_idx - move_idx)
        return score
