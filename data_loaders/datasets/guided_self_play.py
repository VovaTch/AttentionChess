import copy

import chess
import torch
from torch.utils.data import Dataset

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
