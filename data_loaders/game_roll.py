import torch
import chess
import numpy as np
import copy
import random

from model.attchess import AttChess
from model.score_functions import ScoreWinFast
from utils.util import board_to_tensor, legal_move_mask


class GameRoller:
    """A class for the model to play against itself. It plays a game and outputs the moves + result"""

    @torch.no_grad()
    def __init__(self, model_good: AttChess, model_evil: AttChess, device='cuda', move_limit=300, argmax_start=300, 
                 discard_draws=False):
        """Loads the model"""
        self.model_good = copy.deepcopy(model_good)
        self.model_evil = copy.deepcopy(model_evil)
        self.device = device
        self.move_limit = move_limit
        self.argmax_start = argmax_start
        self.discard_draws = discard_draws

        self.board_buffer = []
        self.move_buffer = []
        self.move_vec_buffer = []
        self.reward_vec_buffer = []
        self.result = {}  # a dict that describes what happened and with how many moves

    @torch.no_grad()
    def roll_game(self, init_board: chess.Board, num_of_branches=1, expansion_constant=0.0, exploration_prob=1.0):
        """Plays the entire game"""

        # Keep track of branches, wins, draws, and losses
        results_dict = {'branches': 0, 'white wins': 0, 'black wins': 0, 'draws': 0}

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
                    exp_prob = 0.0
                else:
                    exp_prob = exploration_prob
                    
                # Create node; append to new list only if the endgame wasn't reached
                if init_flag:
                    
                    new_node = init_node.sample_move(legal_move, quality_vec, exploration_prob=exp_prob)
                    if not new_node.endgame_flag:
                        new_nodes.append(new_node)
                        board_list_new.append(new_node.board)
                    else:
                        new_node.propagate_score()
                        
                else: 
                    while True:
                        new_node = current_node.sample_move(legal_move, quality_vec, exploration_prob=exp_prob)
                        new_node.score_function.last_move_idx = (move_num + 2) // 2  # The +2 hack is to prevent division by zero
                        
                        if not new_node.endgame_flag and move_num <= self.move_limit: 
                            new_nodes.append(new_node)
                            board_list_new.append(new_node.board)
                        else:
                            # Log the scores
                            score_out = new_node._is_game_end()
                            results_dict['branches'] += 1
                            if score_out[1] == 1:
                                results_dict['white wins'] += 1
                            elif score_out[1] == -1:
                                results_dict['black wins'] += 1
                            elif score_out[0] is True and score_out[1] == 0:
                                results_dict['draws'] += 1
                            
                            new_node.propagate_score()
                        
                        # Create random expansion
                        random_num = random.uniform(0, 1)
                        if random_num > expansion_constant:
                            break
                
                    
            # After the initial roll, the next rolls are from the tree.
            init_flag = False
                
            current_nodes = new_nodes
            board_list = board_list_new
            
            # Check if all nodes reached endgame state
            if len(current_nodes) == 0 or move_num > self.move_limit:
                break
            
        print(f'Results of all the branches: {results_dict}')
        self.board_buffer, self.reward_vec_buffer = init_node.flatten_tree(discard_draws=self.discard_draws)  # TODO: perform additive operation

    @torch.no_grad()
    def reset_buffers(self):

        self.board_buffer = []
        self.reward_vec_buffer = []
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
        self.value_score = 0

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
        self.children.append(new_child)
        return new_child

    def sample_move(self, legal_move_list, quality_vector, exploration_prob=1.0):
        """Sample a move given outputs from an engine and create a child node"""
        
        rand_num = random.uniform(0, 1)
        
        if rand_num <= exploration_prob:
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
            if self.result != 0:
                self.value_score = -100
            else:
                self.value_score = 0
            return

        turn_variable = 1 if self.board.turn is False else -1
        self.parent.score_function = self.score_function
        current_score = self.score_function(self.moves_performed, self.board)
        last_move = self.board.peek()
        self.parent.result = self.result
        result_marker = self.result if self.result != 0 else 0 # was -turn_variable
        quality_idx = [idx for idx, move in enumerate(self.parent.legal_move_list) if move == last_move]
        self.parent.quality_vector_logit[quality_idx[0]] += current_score * result_marker * turn_variable 
        
        # Position value score; take the value and divide by the number of children.
        self.parent.value_score += current_score * result_marker * turn_variable / len(self.parent.children)
        
        self.parent.propagate_score()  # Recursively apply the score propogation

    def flatten_tree(self, discard_draws=False):
        """Call on tree root to collapse all boards into a single vector, with the corresponding quality vectors. 
        This is the most essential method, it prepares the data for learning."""

        # If there is no child, i.e. the game ended, return an empty stack. Otherwise, start collecting
        
        move_tensor = torch.zeros((0, 256)).to(self.device)

        if len(self.children) == 0:
            return [], move_tensor
        else:
            # board_list = [child.board for child in self.children if len(child.children) != 0]
            board_list = [self.board]

        turn_variable = 1 if self.board.turn is False else -1
        
        if discard_draws or self.result != 0:
            quality_vector_append = torch.zeros((1, 256)).to(self.device) - torch.inf
            quality_vector_append[0, :self.quality_vector_logit.size()[0]] = self.quality_vector_logit
            quality_vector_append[0, -1] = self.value_score
            move_tensor = torch.cat((move_tensor, quality_vector_append))
        else:
            board_list.pop()
            

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

class InferenceBoardNode(BoardNode):
    """
    A sister class for inference; perform MCTS when the score is pre-determined
    """
    
    def __init__(self, board: chess.Board, parent, score_function, quality_prob_vec, value_score, 
                 device='cuda', moves_performed=0):
        super().__init__(board, parent, score_function, device=device, moves_performed=moves_performed)
        self.quality_vector = quality_prob_vec
        self.value_score = value_score
        self.num_of_visits = 0
        self.back_moves = 1
        
        # Need to do this hack because can't check directly if the board is an initial position or not.
        try:
            self.last_move = board.peek()
        except:
            self.last_move = None
        
    def propagate_score(self):
        """
        Propagate score for inference; instead of going for the mid, always select min-max TODO: Do the min-max
        """

        # If there is no parent, we reached the bottom of the barrel
        if self.parent is None:
            return

        self.parent.score_function = self.score_function

        # Position value score; take the value and divide by the number of children.
        self.parent.value_score += (self.score * self.back_moves / (self.back_moves + 1)  - self.parent.value_score) \
            / len(self.parent.children)
        
        self.parent.propagate_score()  # Recursively apply the score propogation
        
        
class InferenceMoveSearcher:
    """
    MCTS move searcher; expand tree according to policy, find the best move according to value.
    """
    def __init__(self, engine: AttChess):
        self.leaf_nodes = []
        self.engine = copy.deepcopy(engine)
    
    def return_sampled_node(self, node_query: InferenceBoardNode):
        
        new_node = node_query.sample_move(legal_move_list=node_query.board.legal_moves, 
                                    quality_vector=node_query.quality_vector)
        
        node_check = [node for node in node_query.children if node.last_move is new_node.last_move]
        
        # Check if the newly created node exists in the children list already. If not, create a new child node
        if len(node_check) == 1:
            current_node = new_node
            new_flag = True
        else:
            current_node = node_check[0]
            node_query.children.pop()
            new_flag = False
        current_node.num_of_visits += 1
        
        return current_node, new_flag
    
    @torch.no_grad()
    def run_engine(self, current_node):

        legal_move_out, quality_out, value_out = self.engine.board_forward([current_node.board])
        legal_move_list, quality_vec, value_pred = self.engine.post_process(legal_move_out, quality_out, value_out)
        current_node.legal_move_list = legal_move_list[0]
        current_node.quality_vector = quality_vec[0]
        current_node.value_score = value_pred[0]
    
    def __call__(self, init_node: InferenceBoardNode, number_of_moves):
        
        for idx in range(number_of_moves):
            
            current_node, new_flag = self.return_sampled_node(init_node)
            
            # Activate the net
            self.run_engine(current_node)

            # Search further down
            while not new_flag:
                current_node, new_flag = self.return_sampled_node(current_node)
                self.run_engine(current_node)
            
            self.leaf_nodes.append(current_node)
            
        for leaf_node in self.leaf_nodes:
            leaf_node.propagate_score()
            
            # TODO: Continue the selection algorithm