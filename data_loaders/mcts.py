from jmespath import search
import torch
import chess
import copy
import numpy as np
import math

from model.score_functions import ScoreWinFast
from model.attchess import AttChess



class Node:
    
    def __init__(self, board: chess.Board, prior_prob: float) -> None:
        """
        Initiates the node for the MCTS
        """
        
        self.prior_prob = prior_prob
        self.turn = board.turn
        self.half_move = None  # Used to compute the cost function
        
        self.children = {}
        self.visit_count = 0
        self.value_candidates = [] 
        self.value_sum = 0.0
        self.board = board
        
    def expanded(self):
        """
        Return a boolian to represent if it's an expanded node or not
        """
        return len(self.children) > 0
    
    def select_action(self, temperature = 0):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action
    
    def select_child(self):
        """
        Select the child with the highest UCB score
        """
        ucb_score_dict = ucb_scores(self, self.children)
        max_score_move = max(ucb_score_dict, key=ucb_score_dict.get)
        
        best_child = self.children[max_score_move]
        
        return max_score_move, best_child
    
    def value_avg(self):
        """
        Return the average of the children's value
        """
        
        return self.value_sum / self.visit_count
    
    def value_max(self):
        """
        Return the maximum value of the children
        """
        
        if len(self.value_candidates) == 0:
            return None

        if self.turn:
            return max(self.value_candidates)
        else:
            return min(self.value_candidates)
        
    def expand(self, legal_move_list, cls_prob_vec):
        """
        Expand the node to include all possible moves.
        """
        
        for move, prob in zip(legal_move_list, cls_prob_vec):
            new_board = copy.deepcopy(self.board)
            
            if type(move) is str:
                new_board.push_san(move)
            else:
                new_board.push(move)
                
            self.children[self.board.san(move)] = Node(prior_prob=prob, board=new_board)
            
    def __repr__(self):
        """
        Debug: display pretty info
        """
        
        prior = "{0:.2f}".format(self.prior_prob)
        return "{} Prior: {} Count: {} Value: {}".format(self.board.move_stack, float(prior), int(self.visit_count), np.tanh(float(self.value_avg())))
        
        
      
def ucb_scores(parent, children: dict[str, Node]):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_scores = {move: child.prior_prob * math.sqrt(parent.visit_count) / (child.visit_count + 1) for move, child in children.items()}
    value_scores = {}
    for move, child in children.items():
        if child.visit_count > 0 and child.value_avg() is not None:
            value_scores[move] = -np.tanh(child.value_avg()) if child.board.turn else np.tanh(child.value_avg())
        else:
            value_scores[move] = 0
    
    collector_scores = {move: value_scores[move] + prior_scores[move] for move, _ in children.items()}       
     
    return collector_scores
        
        
class MCTS:
    """
    A class to run MCTS search; compared to simple AlphaZero, need to configure this to run simultaneous 
    board evaluations to levarage CUDA capabilities.
    """
    
    def __init__(self, model_good: AttChess, model_evil: AttChess, args):
        self.model_good = copy.deepcopy(model_good)
        self.model_evil = copy.deepcopy(model_evil)
        self.args = args
        self.model_good_flag = True
        
        self.board_list = []
        self.move_count_vec = torch.zeros((0, 256))
        self.board_value_vec = torch.zeros(0)
        
    @torch.no_grad()
    def run_engine(self, boards: list[chess.Board]):
        legal_moves_pred, cls_logit_pred, value_raw_pred = self.model_good(boards) if self.model_good_flag else self.model_evil(boards)
        legal_moves_list, cls_prob_list, _ = self.model_good.post_process(legal_moves_pred, cls_logit_pred, value_raw_pred)
        return legal_moves_list, cls_prob_list, value_raw_pred
        
    def get_endgame_value(self, board: chess.Board):
        """
        Get final value if the game ended, else get None
        """
        game_end_flag, result = self._is_game_end(board)
        if game_end_flag:
            return result * 100
        else:
            return None
    
    def _is_game_end(self, board: chess.Board):
        """Checks if the game ends."""
        if board.is_checkmate():
            return True, -1 * board.turn + 1 * (not board.turn)
        elif board.is_stalemate() or board.is_repetition() or \
                board.is_seventyfive_moves() or board.is_insufficient_material():
            return True, 0
        return False, 0
        
    def run(self, board: chess.Board, verbose=False):
        
        self.model_good_flag = True
        root = Node(board, 0.0)
        
        # Expand the root node
        legal_move_list, cls_prob_list, _ = self.run_engine([board])
        root.expand(legal_move_list[0], cls_prob_list[0])
        
        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]
            
            # Select move to make
            while node.expanded():
                move, node = node.select_child()
                search_path.append(node)
                
            parent = search_path[-2]
            board = parent.board
            next_board = copy.deepcopy(board)
            next_board.push_san(move)
            
            value = self.get_endgame_value(next_board)
            
            if value is None:
                # Expand if game not ended
                legal_move_list, cls_prob_list, value = self.run_engine([next_board])
                node.expand(legal_move_list[0], cls_prob_list[0])
            else:
                node.half_move = 1 # Used to compute the value function
                
            self.backpropagate(search_path, value)
            if verbose:
                for node in search_path:
                    print(node)
                
        return root
    
    
    def collect_nodes(self, node):
        pass
    
    
    def run_multi(self, boards: list[chess.Board], verbose=False):
        pass # TODO: MAKE THE MULTI-SIM HAPPEN, THIS WILL MAKE EVERYTHING EASIER
    
        

    def backpropagate(self, search_path, value):
        
        half_move_accumilated = 1
        
        for node in reversed(search_path):
            node.value_sum += value
            node.value_candidates.append(value)
            node.visit_count += 1
            value *= 0.95