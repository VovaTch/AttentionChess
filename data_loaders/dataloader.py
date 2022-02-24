import copy

import chess.pgn
import chess
import torch
from torch.utils.data import Dataset
from base import BaseDataLoader
import numpy as np
from model.attchess import AttChess
from colorama import Fore

from utils.util import move_to_coordinate, is_game_end
from model.score_functions import ScoreScaling
from .mcts import MCTS

class BoardEmbeddingLoader(BaseDataLoader):
    """
    Dataloader for board square embedding training, 36 possible words.
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset = BoardEmbeddingDataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BoardEmbeddingDataset(Dataset):
    """
    Dataloader for board square embedding training, 36 possible words.
    """
    def __init__(self):
        super(BoardEmbeddingDataset, self).__init__()

    def __getitem__(self, idx):
        """
        From the index, outputs 2 vectors, a probability vector of pieces, and a probability vector of properties.
        """

        # Piece probabilities
        target_prob_vector = torch.zeros(7)
        white_prob = 0  # 1 for white, 0 for black
        ep_flag = 0  # En passant 1 if available
        castling_right = 1  # 1 for there is right, 0 for there isn't
        turn = 1  # 1 for white, 0 for black

        if idx in [7, 25]:
            target_prob_vector[1] = 1  # white pawn
            white_prob = 1
        elif idx in [8, 26]:
            target_prob_vector[2] = 1  # white knight
            white_prob = 1
        elif idx in [9, 27]:
            target_prob_vector[3] = 1  # white bishop
            white_prob = 1
        elif idx in [10, 28]:
            target_prob_vector[4] = 1  # white rook
            white_prob = 1
        elif idx in [11, 29]:
            target_prob_vector[5] = 1  # white queen
            white_prob = 1
        elif idx in [12, 30]:
            target_prob_vector[6] = 1  # white king
            white_prob = 1
            castling_right = 0
        elif idx in [14, 32]:
            target_prob_vector[6] = 1  # white king
            white_prob = 1
        elif idx in [5, 23]:
            target_prob_vector[1] = 1  # black pawn
        elif idx in [4, 22]:
            target_prob_vector[2] = 1  # black knight
        elif idx in [3, 21]:
            target_prob_vector[3] = 1  # black bishop
        elif idx in [2, 20]:
            target_prob_vector[4] = 1  # black rook
        elif idx in [1, 19]:
            target_prob_vector[5] = 1  # black queen
        elif idx in [0, 18]:
            target_prob_vector[6] = 1  # black king
            castling_right = 0
        elif idx in [15, 33]:
            target_prob_vector[6] = 1  # black king
        elif idx in [16, 17, 34, 35]:
            target_prob_vector[0] = 1
            ep_flag = 1
        else:
            target_prob_vector[0] = 1

        if idx >= 18:
            turn = 0

        flags_vector = torch.tensor([white_prob, ep_flag, castling_right, turn])
        return idx, target_prob_vector, flags_vector

    def __len__(self):
        return 36


class MoveEmbeddingLoader(BaseDataLoader):
    """
    Dataloader for move embedding training, 4864 possible words. 
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset = MoveEmbeddingDataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MoveEmbeddingDataset(Dataset):
    """
    Dataset for moves, 4864 moves, you get word in and get out move coordinates and possible promotion
    classification vector.
    """
    def __init__(self):
        super(MoveEmbeddingDataset, self).__init__()

    def __getitem__(self, word):

        # Initialize promotion probability vector: 0: no promotion, 1: queen, 2: rook, 3: bishop, 4: knight
        promotion_prob = torch.zeros(5)

        # Decompose to individual move coordinates
        coordinates_from_to = (int(word % 64), word // 64)
        coordinates_from = (int(coordinates_from_to[0] % 8), coordinates_from_to[0] // 8)  # 0 is a,b,c... 1 is numbers

        # If not promoting
        if coordinates_from_to[1] < 64:
            coordinates_to = (int(coordinates_from_to[1] % 8), coordinates_from_to[1] // 8)
            promotion_prob[0] = 1

        # If promoting
        else:
            if 64 <= coordinates_from_to[1] < 67:
                coor_shift = 65 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[1] = 1
            elif 67 <= coordinates_from_to[1] < 70:
                coor_shift = 68 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[2] = 1
            elif 70 <= coordinates_from_to[1] < 73:
                coor_shift = 71 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[3] = 1
            else:
                coor_shift = 74 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[4] = 1

            coordinates_to = (int(coordinates_from[0] - coor_shift), int(coor_up_down))

        coor_tensor = torch.tensor([coordinates_from[0], coordinates_from[1], coordinates_to[0], coordinates_to[1]]) / 7

        return word, coor_tensor, promotion_prob

    def __len__(self):
        return 4864


class RuleAttentionChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, collate_fn, data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn',
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset_path = data_dir
        self.dataset = RuleChessDataset(data_dir, query_word_len=query_word_len)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)


class RuleChessDataset(Dataset):

    def __init__(self, dataset_path, query_word_len=256):
        super(RuleChessDataset, self).__init__()
        self.pgn = open(dataset_path, encoding="utf-8")
        self.game = None

        self.query_word_len = query_word_len
        self.follow_idx = 0
        self.game_length = 0
        self.board_collection = None
        self.move_quality_batch = None
        self.board_value_batch = None
        self.selected_move_idx = None

    def __getitem__(self, _):

        if self.follow_idx == 0:
            while self.game_length == 0:
                self.load_game()
                self.game_length = len(self.board_collection)

        sampled_board = copy.deepcopy(self.board_collection[self.follow_idx])
        sampled_quality_batch = self.move_quality_batch[self.follow_idx, :].clone()
        sampled_board_value_batch = self.board_value_batch[self.follow_idx].clone()
        sampled_move_idx = self.selected_move_idx[self.follow_idx].clone()

        self.follow_idx += 1
        if self.follow_idx == self.game_length:
            self.follow_idx = 0
            self.game_length = 0

        return sampled_board, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx

    def load_game(self):

        while True:
            game = chess.pgn.read_game(self.pgn)
            move_counter = 0
            for move in enumerate(game.mainline_moves()):
                move_counter += 1
            last_move = 1 if move_counter % 2 == 1 else -1
            
            if 'Termination' in game.headers and game.headers['Termination'] == 'Normal' and move_counter > 0:
                break

        board = game.board()
        result = game.headers['Result']
        if result == '1-0':
            base_eval = 1
        elif result == '0-1':
            base_eval = -1
        else:
            base_eval = 0

        self.board_collection = []
        self.move_quality_batch = torch.zeros((0, self.query_word_len))
        self.board_value_batch = torch.zeros(0)
        board_value_list = []
        move_idx_list = []

        score_factor = base_eval
        
        # Create the score function
        score_function = ScoreScaling(moves_to_end=move_counter, score_max=3)
            

        for idx, move in enumerate(game.mainline_moves()):

            # Fill the legal move matrix
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
            opposite_win_add = 1 if base_eval != last_move else 0 # A fix for when the the player resigns right after doing his move
            board_value = 0
            matching_idx = torch.nonzero(legal_move_word[:, 1] == move_per_word)
            if score_factor == 1:
                quality_vector_logit[0, matching_idx] = score_function(idx - opposite_win_add) * 10
            elif score_factor == -1:
                quality_vector_logit[0, matching_idx] = -score_function(idx - opposite_win_add) * 10
                
            board_value = np.tanh(score_function(idx - opposite_win_add) * base_eval)

            quality_vector = quality_vector_logit.softmax(dim=1)

            # self.move_collection = torch.cat((self.move_collection, move_tensor.unsqueeze(0)), 0)
            board_recorded = copy.deepcopy(board)
            board.push(move)

            # Collect all the data
            # if score_factor * base_eval == 1:
            board_value_list.append(board_value)
            move_idx_list.append(matching_idx)
            self.board_collection.append(board_recorded)
            self.move_quality_batch = torch.cat((self.move_quality_batch, quality_vector), 0)
            
                
            score_factor *= -1

        self.board_value_batch = torch.tensor(board_value_list)
        self.selected_move_idx = torch.tensor(move_idx_list)

    def __len__(self):
        return int(1e6)


class GuidedSelfPlayLoader(BaseDataLoader):
    """
    Data loader for self playing games with moves from database.
    """
    def __init__(self, batch_size, collate_fn,
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256, 
                 num_of_sims=100, epochs_per_game=1, min_counts=10, data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn',
                 device='cpu'):

        self.dataset = GuidedSelfPlayDataset(query_word_len=query_word_len, num_of_sims=num_of_sims, data_dir=data_dir,
                                             epochs_per_game=epochs_per_game, min_counts=min_counts, simultaneous_mcts=batch_size)
        self.device = device
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
    def set_mcts(self, mcts: MCTS):
        
        self.dataset.mcts = mcts
        


class GuidedSelfPlayDataset(Dataset):
    
    def __init__(self, query_word_len=256, num_of_sims=100, epochs_per_game=1, min_counts=10, 
                 data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn', simultaneous_mcts=32):
        super().__init__()
        
        # Load game collection file
        self.pgn = open(data_dir, encoding="utf-8")
        
        # Initiate engines, will later assert that they aren't empty.
        self.good_engine = None
        self.evil_engine = None
        
        # Initiate variables from outside
        self.query_word_len = query_word_len
        self.num_of_sims = num_of_sims
        self.epochs_per_game = epochs_per_game
        self.min_counts = min_counts
        self.simultaneous_mcts = simultaneous_mcts
        
        # Initiate variables from the inside
        self.follow_idx = 0
        self.game_length = 0
        self.board_collection = None
        self.move_quality_batch = None
        self.board_value_batch = None
        self.selected_move_idx = None
        
        # Initialize MCTS:
        self.mcts = None
        
    def __getitem__(self, _):
        
        # Assert engines are inputed
        assert self.mcts is not None, 'Must load an MCTS object into the dataloader'

        if self.follow_idx == 0:
            while self.game_length == 0:
                self.load_game()
                self.game_length = len(self.board_collection)

        sampled_board = copy.deepcopy(self.board_collection[self.follow_idx])
        sampled_quality_batch = self.move_quality_batch[self.follow_idx, :].clone()
        sampled_board_value_batch = self.board_value_batch[self.follow_idx].clone()
        sampled_move_idx = self.selected_move_idx[self.follow_idx].clone()

        self.follow_idx += 1
        if self.follow_idx == self.game_length:
            self.follow_idx = 0
            self.game_length = 0

        return sampled_board, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx 
    
    def load_game(self):
        """
        Game loading function;
        """

        while True:
            game = chess.pgn.read_game(self.pgn)
            move_counter = 0
            for move in enumerate(game.mainline_moves()):
                move_counter += 1
            last_move = 1 if move_counter % 2 == 1 else -1
            
            if 'Termination' in game.headers and game.headers['Termination'] == 'Normal' and move_counter > 0:
                break

        board = game.board()

        self.board_collection = []
        self.move_quality_batch = torch.zeros((0, self.query_word_len)).to(self.mcts.device)
        self.board_value_batch = torch.zeros(0).to(self.mcts.device)
        move_idx_list = []
            
        board_list_to_mcts = []

        for move_num, move in enumerate(game.mainline_moves()):
            
            # Collect all boards until either we get to the batch size or finish the game
            current_board = copy.deepcopy(board)
            board_list_to_mcts.append(current_board)
            
            # Collect data when the bath is collected
            if (move_num + 1) % self.simultaneous_mcts == 0 or move_num == move_counter - 1:
                
                current_node_list_expanded = self.mcts.run_multi(board_list_to_mcts)
                
                # Collect all individual data from the nodes
                for current_node_expanded in current_node_list_expanded:
                    
                    boards_added, cls_vec_added, value_added = self.mcts.collect_nodes_for_training(current_node_expanded, 
                                                                                            min_counts=self.min_counts)
                    
                    
                    # Collect data for the class variables
                    self.board_collection.extend(boards_added)
                    self.move_quality_batch = torch.cat((self.move_quality_batch, cls_vec_added), dim=0).to(self.mcts.device)
                    self.board_value_batch = torch.cat((self.board_value_batch, value_added), dim=0).to(self.mcts.device)
                    move_idx_list.extend([-torch.inf for _ in range(len(boards_added))]) # Necessary for all of this to work; TODO: make the loss don't count it
            
                # Reset the board list
                board_list_to_mcts = []
            
            # print(f'Pushed move: {board.san(move)}, move number: {move_num // 2  + 1}')
            board.push(move)
            
        self.selected_move_idx = torch.tensor(move_idx_list).to(self.mcts.device)
        
        result = game.headers['Result']
        if result == '1-0':
            result_string = 'white wins'
        elif result == '0-1':
            result_string = 'black wins'
        else:
            result_string = 'draw'
        print(f'Game result: {result_string}')
        
        # Repeat the data as long as needed
        self.board_collection *= self.epochs_per_game
        self.move_quality_batch = self.move_quality_batch.repeat((self.epochs_per_game, 1))
        self.board_value_batch = self.board_value_batch.repeat(self.epochs_per_game)
        self.selected_move_idx = self.selected_move_idx.repeat(self.epochs_per_game)
        

    def __len__(self):
        return int(1e5)


class FullSelfPlayLoader(BaseDataLoader):
    """
    Data loader for self playing games with moves from database.
    """
    def __init__(self, batch_size, collate_fn,
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256, 
                 num_of_sims=100, epochs_per_game=1, min_counts=10, device='cpu', win_multiplier=1):

        self.dataset = FullSelfPlayDataset(query_word_len=query_word_len, num_of_sims=num_of_sims, 
                                             epochs_per_game=epochs_per_game, min_counts=min_counts, simultaneous_mcts=batch_size,
                                             win_multipler=win_multiplier)
        self.device = device
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
    def set_mcts_game(self, mcts: MCTS):
        
        self.dataset.mcts_game = mcts
        
    def set_mcts_learn(self, mcts: MCTS):
        
        self.dataset.mcts = mcts
        
    def set_mcts(self, mcts: MCTS):
        
        self.set_mcts_game(mcts)
        self.set_mcts_learn(mcts)
        
        


class FullSelfPlayDataset(Dataset):
    
    def __init__(self, query_word_len=256, num_of_sims=100, epochs_per_game=1, min_counts=10, simultaneous_mcts=32, move_limit=300, 
                 win_multipler=2):
        super().__init__()
        
        # Initiate engines, will later assert that they aren't empty.
        self.good_engine = None
        self.evil_engine = None
        
        # Initiate variables from outside
        self.query_word_len = query_word_len
        self.num_of_sims = num_of_sims
        self.epochs_per_game = epochs_per_game
        self.min_counts = min_counts
        self.simultaneous_mcts = simultaneous_mcts
        self.move_limit = move_limit
        self.win_multiplier = win_multipler
        
        # Initiate variables from the inside
        self.follow_idx = 0
        self.game_length = 0
        self.board_collection = None
        self.move_quality_batch = None
        self.board_value_batch = None
        self.selected_move_idx = None
        
        # Initialize MCTS:
        self.mcts = None
        self.mcts_game = None
        
    def __getitem__(self, _):
        
        # Assert engines are inputed
        assert self.mcts is not None, 'Must load an MCTS object into the dataloader'

        if self.follow_idx == 0:
            while self.game_length == 0:
                self.load_game()
                self.game_length = len(self.board_collection)

        sampled_board = copy.deepcopy(self.board_collection[self.follow_idx])
        sampled_quality_batch = self.move_quality_batch[self.follow_idx, :].clone()
        sampled_board_value_batch = self.board_value_batch[self.follow_idx].clone()
        sampled_move_idx = self.selected_move_idx[self.follow_idx].clone()

        self.follow_idx += 1
        if self.follow_idx == self.game_length:
            self.follow_idx = 0
            self.game_length = 0

        return sampled_board, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx 
    
    def load_game(self):
        """
        Game running function;
        """

        board = chess.Board()
        game_board_list = []
        
        for move_idx in range(self.move_limit):
            
            move_counter = move_idx + 1
            
            # Append all board states
            game_board_list.append(copy.deepcopy(board))
            
            # Find the best move from a short search
            sample_node = self.mcts_game.run(board)
            sample = sample_node.select_action(temperature=0.3)
            
            # Append the move to the board
            board.push_san(sample)
            print(f'[FullSelfPlay] Pushed move: ' + Fore.YELLOW + f'{sample}' + 
                  Fore.RESET + f',   \t move: {move_idx // 2 + 1}')
            
            # Check for game end
            ending_flag, result = is_game_end(board)
            if ending_flag:
                
                if result == 1:
                    result_string = 'white wins'
                    multiplier = self.win_multiplier
                elif result == -1:
                    result_string = 'black wins'
                    multiplier = self.win_multiplier
                else:
                    result_string = 'draw'
                    multiplier = 1
                    
                print(Fore.RED + f'[FullSelfPlay] Game result: {result_string}' + Fore.RESET)
                
                break
            

        self.board_collection = []
        self.move_quality_batch = torch.zeros((0, self.query_word_len)).to(self.mcts.device)
        self.board_value_batch = torch.zeros(0).to(self.mcts.device)
        move_idx_list = []
            
        board_list_to_mcts = []

        for board_num, board in enumerate(game_board_list):
            
            # Collect all boards until either we get to the batch size or finish the game
            current_board = copy.deepcopy(board)
            board_list_to_mcts.append(current_board)
            
            # Collect data when the bath is collected
            if (board_num + 1) % self.simultaneous_mcts == 0 or board_num == move_counter - 1:
                
                current_node_list_expanded = self.mcts.run_multi(board_list_to_mcts)
                
                # Collect all individual data from the nodes
                for current_node_expanded in current_node_list_expanded:
                    
                    boards_added, cls_vec_added, value_added = self.mcts.collect_nodes_for_training(current_node_expanded, 
                                                                                            min_counts=self.min_counts)
                    
                    
                    # Collect data for the class variables
                    self.board_collection.extend(boards_added)
                    self.move_quality_batch = torch.cat((self.move_quality_batch, cls_vec_added), dim=0).to(self.mcts.device)
                    self.board_value_batch = torch.cat((self.board_value_batch, value_added), dim=0).to(self.mcts.device)
                    move_idx_list.extend([-torch.inf for _ in range(len(boards_added))]) # Necessary for all of this to work; TODO: make the loss don't count it
            
                # Reset the board list
                board_list_to_mcts = []
            
        self.selected_move_idx = torch.tensor(move_idx_list).to(self.mcts.device)
        
        # Repeat the data as long as needed
        self.board_collection *= self.epochs_per_game * multiplier
        self.move_quality_batch = self.move_quality_batch.repeat((self.epochs_per_game * multiplier, 1))
        self.board_value_batch = self.board_value_batch.repeat(self.epochs_per_game * multiplier)
        self.selected_move_idx = self.selected_move_idx.repeat(self.epochs_per_game * multiplier)

    def __len__(self):
        return int(1e5)


def collate_fn(batch):
    """
    Required collate function because the boards are a unique class
    """
    chess_boards = [batch[idx][0] for idx in range(len(batch))]
    quality_vectors = torch.zeros((len(batch), batch[0][1].size()[0]))
    board_values = torch.zeros(len(batch))
    move_idx = torch.zeros(len(batch))
    for idx in range(len(batch)):
        quality_vectors[idx, :] = batch[idx][1]
        board_values[idx] = batch[idx][2]
        move_idx[idx] = batch[idx][3]
    return chess_boards, quality_vectors, board_values, move_idx
