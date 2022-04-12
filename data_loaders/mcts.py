from jmespath import search
import torch
import torch.nn.functional as F
import chess
import copy
import numpy as np
import math

from model.score_functions import ScoreWinFast
from model.attchess import AttChess



class Node:
    
    def __init__(self, board: chess.Board, prior_prob: float, device='cpu') -> None:
        """
        Initiates the node for the MCTS
        """
        
        self.device = device
        self.prior_prob = prior_prob
        self.turn = board.turn
        self.half_move = None  # Used to compute the cost function
        
        self.children = {}
        self.visit_count = 0
        self.value_candidates = {}
        self.value_sum = 0.0
        self.board = board
        
    def expanded(self):
        """
        Return a boolian to represent if it's an expanded node or not
        """
        return len(self.children) > 0
    
    def select_action(self, temperature = 0, print_action_count=False):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = torch.tensor([child.visit_count for child in self.children.values()]).to(self.device)
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[torch.argmax(visit_counts)]
        elif temperature == float("inf"):
            
            logits = torch.tensor([0.0 for _ in actions]).to(self.device)
            cat_dist = torch.distributions.Categorical(logits=logits)
            action = actions[cat_dist.sample()]
                    
        else:
            
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / torch.sum(visit_count_distribution)
            cat_dist = torch.distributions.Categorical(probs=visit_count_distribution)
            action = actions[cat_dist.sample()]
            
        if print_action_count:
            action_value = {act: round(float(move.value_avg()), 5) for act, move in zip(actions, self.children.values())}
            print(f'Action values: {action_value}')
            action_dict = {act: int(vis_c) for act, vis_c in zip(actions, visit_counts)}
            print(f'Action list: {action_dict}')

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

        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    
    def value_max(self):
        """
        Return the maximum value of the children
        """
        
        non_none_candidate = {move: value for move, value in self.value_candidates.items() if self.value_candidates[move] is not None}
        
        if len(non_none_candidate) == 0:
            return None

        if self.turn:
            return non_none_candidate[max(non_none_candidate, key=non_none_candidate.get)]
        else:
            return non_none_candidate[min(non_none_candidate, key=non_none_candidate.get)]
        
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
                
            self.children[self.board.san(move)] = Node(prior_prob=prob, board=new_board, device=self.device)
            # if self.board.turn:
            #     self.value_candidates[self.board.san(move)] = - torch.inf
            # else:
            #     self.value_candidates[self.board.san(move)] = torch.inf
            
            self.value_candidates[self.board.san(move)] = None
            
    def __repr__(self):
        """
        Debug: display pretty info
        """
        
        prior = "{0:.2f}".format(self.prior_prob)
        return "{} Prior: {} Count: {} Value: {}".format(self.board.move_stack, float(prior), int(self.visit_count), np.tanh(float(self.value_avg())))
        
        
      
def ucb_scores(parent, children: dict):
    """
    The score for an action that would transition between the parent and child.
    """
    c_puct = 1
    
    prior_scores = {move: child.prior_prob * math.sqrt(parent.visit_count) / (child.visit_count + 1) for move, child in children.items()}
    value_scores = {}
    for move, child in children.items():
        if child.visit_count > 0 and child.value_avg() is not None:
            # value_scores[move] = -torch.tanh(torch.tensor(child.value_avg())) \
            #     if child.board.turn else torch.tanh(torch.tensor(child.value_avg()))
            value_scores[move] = -torch.tensor(child.value_avg()) \
                if child.board.turn else torch.tensor(child.value_avg())
    
        else:
            value_scores[move] = 0
    
    collector_scores = {move: value_scores[move] + c_puct * prior_scores[move] for move, _ in children.items()}
    # print(f'Turn: {child.board.turn}, Value scores: {value_scores}\n')
    # print(f'Turn: {child.board.turn}, Prior scores: {prior_scores}\n')
    
    # print(f'UCB adjustment: {math.sqrt(parent.visit_count) / (child.visit_count + 1)}')

    return collector_scores
        
        
class MCTS:
    """
    A class to run MCTS search; compared to simple AlphaZero, need to configure this to run simultaneous 
    board evaluations to levarage CUDA capabilities.
    """
    
    def __init__(self, model_good: AttChess, model_evil: AttChess, num_sims, device='cpu'):
        self.model_good = model_good
        self.model_evil = model_evil
        self.num_sims = num_sims
        self.model_good_flag = True
        self.device = device
        
        self.model_good.to(self.device)
        self.model_evil.to(self.device)
        
        self.board_list = []
        self.move_count_vec = torch.zeros((0, 256)).to(self.device)
        self.board_value_vec = torch.zeros(0).to(self.device)
        
    @torch.no_grad()
    def run_engine(self, boards: list):
        legal_moves_pred, cls_logit_pred, value_raw_pred = self.model_good(boards) if self.model_good_flag else self.model_evil(boards)
        legal_moves_list, cls_prob_list, _ = self.model_good.post_process(legal_moves_pred, cls_logit_pred, value_raw_pred)
        return legal_moves_list, cls_prob_list, value_raw_pred
        
    def get_endgame_value(self, board: chess.Board):
        """
        Get final value if the game ended, else get None
        """
        game_end_flag, result = self._is_game_end(board)
        if game_end_flag:
            return result * 5.0
        else:
            return None
    
    def _is_game_end(self, board: chess.Board):
        """Checks if the game ends."""
        if board.is_checkmate():
            result_const = -1 if board.turn else 1
            return True, result_const
        elif board.is_stalemate() or board.is_repetition() or \
                board.is_seventyfive_moves() or board.is_insufficient_material():
            return True, 0
        return False, 0
        
    def run(self, board: chess.Board, verbose=False):
        
        self.model_good_flag = True
        root = Node(board, 0.0, device=self.device)
        
        # Expand the root node
        legal_move_list, cls_prob_list, _ = self.run_engine([board])
        root.expand(legal_move_list[0], cls_prob_list[0])
        
        for _ in range(self.num_sims):
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
                
            if type(value) == float:
                self.backpropagate(search_path, value)
            else:
                self.backpropagate(search_path, value[0])
                
            if verbose:
                for node in search_path:
                    print(node)
                
        return root
    
    
    def collect_nodes_for_training(self, node: Node, min_counts = 5):
        """Consider all nodes that have X or more visits for future training of self play."""
        
        board_collection = [node.board]
        cls_vec_collection = torch.zeros((1, 256)).to(self.device)
        board_value_collection = torch.tensor([node.value_avg()]).to(self.device)
        
        for idx, (_, child) in enumerate(node.children.items()):
            
            cls_vec_collection[0, idx] += child.visit_count
            if child.visit_count >= min_counts:
                board_add, cls_vec_add, board_value_add = self.collect_nodes_for_training(child, min_counts=min_counts)
                
                # Recursivelly add the nodes with the correct count number
                board_collection.extend(board_add)
                cls_vec_collection = torch.cat((cls_vec_collection, cls_vec_add), dim=0)
                board_value_collection = torch.cat((board_value_collection, board_value_add), dim=0)
               
        cls_vec_collection = F.normalize(cls_vec_collection, p=1, dim=1)
        return board_collection, cls_vec_collection, torch.tanh(board_value_collection)      
    
    
    def run_multi(self, boards: list, verbose=False, print_enchors=True):
        
        self.model_good_flag = True
        roots = [Node(board, 0.0, self.device) for board in boards]
        root_boards = [node.board for node in roots]
        
        # Expand every root node
        legal_move_list, cls_prob_list, _ = self.run_engine(root_boards)
        for idx, root in enumerate(roots):
            root.expand(legal_move_list[idx], cls_prob_list[idx])
            
        # Create win/loss/draw counters for printing
        white_win_count = 0
        black_win_count = 0
        draw_count = 0
            
        # Run sim for every board
        for _ in range(self.num_sims):
            node_edge_list = [None for _ in roots] # Need to do this, otherwise the roots will be overridden by the leaf nodes
            search_path_list = [[node] for node in roots]
            
            # Select a move to make per each node
            value_list = [None for _ in roots]
            board_slice_list = []
            
            # Expand tree nodes and input values until every value in the list is filled
            for idx, node in enumerate(roots):
                while node.expanded():
                    move, node = node.select_child()
                    node_edge_list[idx] = node
                    search_path_list[idx].append(node)

                # necessary to not break the loop when the game ended in one of the branches
                if len(search_path_list[idx]) >= 2:
                    
                    parent = search_path_list[idx][-2]
                    board = parent.board 
                    
                    next_board = copy.deepcopy(board)
                    next_board.push_san(move)
                    
                else:
                    
                    next_board = search_path_list[idx][-1].board
                
                value = self.get_endgame_value(next_board)
                value_list[idx] = value
                
                if value == 5:
                    white_win_count += 1
                elif value == -5:
                    black_win_count += 1
                elif value == 0:
                    draw_count += 1
                
                if value is None:
                    board_slice_list.append(next_board)
                else:
                    node.half_move = 1 # Used to compute the value function in old version
                 
                    
            # Forward all boards through the net
            if len(board_slice_list) > 0:
                legal_move_list, cls_prob_list, value = self.run_engine(board_slice_list)
            
            # Expand every node that didn't reach the end
            node_selection_idx = 0
            for idx, node in enumerate(node_edge_list):
                if value_list[idx] is None:
                    node.expand(legal_move_list[node_selection_idx], cls_prob_list[node_selection_idx])
                    value_list[idx] = value[node_selection_idx]
                    node_selection_idx += 1
                    
            for idx, search_path in enumerate(search_path_list):
                self.backpropagate(search_path, value_list[idx])
                
                if verbose:
                    for node in search_path:
                        print(node)
                        
        if print_enchors:
            print(f'Out of {self.num_sims} simulations, {len(roots)} roots, {white_win_count} white wins, {black_win_count} black wins, {draw_count} draws.')
                        
        return roots
                
            

    def backpropagate(self, search_path, value, value_multiplier=0.95):
        
        half_move_accumilated = 1
        
        for node_idx, node in reversed(list(enumerate(search_path))):
            node.value_sum += value
            
            if node_idx != 0:
                prior_board = copy.deepcopy(node.board)
                prior_board.pop()
                last_move = prior_board.san(node.board.peek())
                
                if search_path[node_idx - 1].value_candidates[last_move] is None:
                    search_path[node_idx - 1].value_candidates[last_move] = value * value_multiplier
                
                elif not node.board.turn and search_path[node_idx - 1].value_candidates[last_move] > value * value_multiplier:   
                    search_path[node_idx - 1].value_candidates[last_move] = value * value_multiplier
                    
                elif node.board.turn and search_path[node_idx - 1].value_candidates[last_move] < value * value_multiplier:    
                    search_path[node_idx - 1].value_candidates[last_move] = value * value_multiplier
            
            node.visit_count += 1
            value *= value_multiplier
            
