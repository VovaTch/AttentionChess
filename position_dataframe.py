import argparse
import copy

import chess
import chess.pgn
import csv
import numpy as np
import torch

from model.score_functions import ScoreScaling
from utils.util import move_to_coordinate

# TODO: Work on it
QUERY_WORD_LEN = 256

def load_game(pgn_handle, args):

    while True:
        game = chess.pgn.read_game(pgn_handle)
        if game is None:
            return None, None, None, None
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

    board_collection = []
    board_value_list = []
    move_idx_list = []

    score_factor = copy.copy(base_eval)
    
    # Create the score function
    score_function = ScoreScaling(moves_to_end=move_counter, score_max=5)
        

    for idx, move in enumerate(game.mainline_moves()):

        # Fill the legal move matrix
        legal_move_mat = torch.zeros((1, 64, 76))
        for idx_move, legal_move in enumerate(board.legal_moves):

            move_legal_coor = move_to_coordinate(legal_move)
            legal_move_mat[0, move_legal_coor[0], move_legal_coor[1]] = 1

        legal_move_mat = legal_move_mat == 1
        legal_move_idx = torch.nonzero(legal_move_mat)
        legal_move_word = torch.cat((legal_move_idx[:, 0].unsqueeze(1),
                                        legal_move_idx[:, 1].unsqueeze(1) +
                                        64 * legal_move_idx[:, 2].unsqueeze(1)), 1)
        move_per_coor = move_to_coordinate(move)
        move_per_word = move_per_coor[0] + 64 * move_per_coor[1]

        # Find the correct move
        board_value = 0
        matching_idx = torch.nonzero(legal_move_word[:, 1] == move_per_word)
            
        board_value = base_eval * np.tanh(score_function(idx))

        # self.move_collection = torch.cat((self.move_collection, move_tensor.unsqueeze(0)), 0)
        board_recorded = copy.deepcopy(board)
        board.push(move)

        # Collect all the data
        # if score_factor * base_eval == 1:
        board_value_list.append(board_value)
        move_idx_list.append(matching_idx)
        board_collection.append(board_recorded)
        
        score_factor *= -1
        
    return board_collection, board_value_list, move_idx_list, base_eval

def main(args):
    
    pgn_database = open(args.path, encoding="utf-8")
    pgn_dict = {}
        
    game_count = 0
    file_count = 0
    
    # Organize all data into a massive dict. Save later as a json/csv
    while game_count < args.game_limit:
        
        board_collection, board_value_list, move_idx_list, base_eval = load_game(pgn_database, args)
        if board_collection is None:
            break
        
        for (board_ind, board_value, move_idx) in \
               zip(board_collection, board_value_list, move_idx_list):
            
            board_fen = board_ind.fen()
            turn_number = 1 if board_ind.turn else -1
            
            # Create new entry if not in dict
            if board_fen not in pgn_dict:
                move_vec = np.zeros((QUERY_WORD_LEN))
                
                # If it's the winning side turn
                if base_eval * turn_number == 1:
                    move_vec[move_idx] += 1
                pgn_dict[board_fen] = [1, board_value, move_vec]
                
            # Update entry if in dict
            else:
                pgn_dict[board_fen][0] += 1
                pgn_dict[board_fen][1] += (board_value - pgn_dict[board_fen][1]) / pgn_dict[board_fen][0]
                if base_eval * turn_number == 1:
                    pgn_dict[board_fen][2][move_idx] += 1
                    
        game_count += 1
        
        # Keep track of the games processed
        if game_count % 100 == 0:
            print(f'Game processed: {game_count}', end='\r')
            
        if game_count % 50000 == 0 or args.game_limit == game_count:
            
            file_count += 1
            
            # Write a table PGN with headers based on a dict; should be faster access?
            with open(f'lichess_data/lichess_data_raw_{file_count}.csv', mode='w') as csv_file:
                fieldnames = ['Game fen', 'Board value']
                fieldnames.extend(f'index {idx}' for idx in range(QUERY_WORD_LEN))
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for key in pgn_dict:
                    written_dict = {}
                    written_dict['Game fen'] = key
                    written_dict['Board value'] = pgn_dict[key][1]
                    written_dict.update({f'index {idx}': int(pgn_dict[key][2][idx]) for idx in range(QUERY_WORD_LEN)})
                    writer.writerow(written_dict)
                    
                pgn_dict = {}
                    
                    
        
    print(f'Number of games is {game_count}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to save a database as csv file with non-repeating positions.')
    parser.add_argument('-p', '--path', type=str, default = 'lichess_data/lichess_db_standard_rated_2016-09.pgn', 
                        help='The path of the game dataset.')
    parser.add_argument('-g', '--game_limit', type=int, default=2e5, 
                        help='Number of games processed. Infinite - all the games in the file.')
    parser.add_argument('-m', '--base_multiplier', type=float, default=0.95,
                        help='Multiplier for decaying reward for long games')
    args = parser.parse_args()
    main(args) 