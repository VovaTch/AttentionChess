import copy

import chess
import torch
from torch.utils.data import Dataset
from colorama import Fore

from utils.util import is_game_end

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
        multiplier = 1
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
            
            # print(board_num)
            
            # Collect all boards until either we get to the batch size or finish the game
            current_board = copy.deepcopy(board)
            board_list_to_mcts.append(current_board)
            
            # Collect data when the bath is collected
            if (board_num + 1) % self.simultaneous_mcts == 0 or board_num == move_counter - 1:
                
                len(board_list_to_mcts)
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
        
        print('exploding before repeating')
        
        # Repeat the data as long as needed
        self.board_collection *= self.epochs_per_game * multiplier
        self.move_quality_batch = self.move_quality_batch.repeat((self.epochs_per_game * multiplier, 1))
        self.board_value_batch = self.board_value_batch.repeat(self.epochs_per_game * multiplier)
        self.selected_move_idx = self.selected_move_idx.repeat(self.epochs_per_game * multiplier)
        
        print('exploding after repeating')

    def __len__(self):
        return int(1e5)