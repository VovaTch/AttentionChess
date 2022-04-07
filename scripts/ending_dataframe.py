import argparse
import copy

import chess
import chess.pgn
import csv
import numpy as np
import torch

from model.score_functions import ScoreScaling
from utils import move_to_coordinate, board_to_embedding_coord

# TODO: Work on it
QUERY_WORD_LEN = 256

def load_game(pgn_handle, args):

    while True:
        game = chess.pgn.read_game(pgn_handle)
        if game is None:
            return None, None, None, None
        move_counter = 0
        board_running = chess.Board()
        for move in game.mainline_moves():
            move_counter += 1
            board_running.push(move)
        
        if 'Termination' in game.headers and game.headers['Termination'] == 'Normal' and move_counter > 0:
            result = game.headers['Result']
            if result == '1-0' and board_running.is_checkmate():
                value = 1
                break
            elif result == '0-1' and board_running.is_checkmate():
                value = -1
                break
            elif result == '0-0':
                value = 0
                break
        
    return [board_running], [value], [-1], value / 5

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
                pgn_dict[board_fen] = [1, board_value, move_vec]
                
            # Update entry if in dict
            else:
                pgn_dict[board_fen][0] += 1
                pgn_dict[board_fen][1] += (board_value - pgn_dict[board_fen][1]) / pgn_dict[board_fen][0]
                    
        game_count += 1
        
        # Keep track of the games processed
        if game_count % 100 == 0:
            print(f'Game processed: {game_count}', end='\r')
            
        if args.game_limit == game_count:
            
            file_count += 1
            
            # Write a table PGN with headers based on a dict; should be faster access?
            with open(f'lichess_data/endgame_data_raw_{file_count}.csv', mode='w') as csv_file:
                fieldnames = ['Game board', 'Board value', 'Legal move matrix']
                fieldnames.extend(f'index {idx}' for idx in range(QUERY_WORD_LEN))
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for key in pgn_dict:
                    
                    try:
                        check_board = chess.Board(key)
                        board_torch = board_to_embedding_coord(check_board).int()
                        board_list = board_torch.tolist()
                        legal_move_torch = torch.zeros((64, 76), requires_grad=False) - 1
                        for legal_move in check_board.legal_moves:
                            move_coor = move_to_coordinate(legal_move)
                            legal_move_torch[move_coor[0], move_coor[1]] = 1
                    except:
                        print(f'Fen is illegal')
                        continue
                        
                    
                    written_dict = {}
                    written_dict['Game board'] = board_list
                    written_dict['Board value'] = pgn_dict[key][1]
                    written_dict['Legal move matrix'] = legal_move_torch.int().tolist()
                    
                    written_dict.update({f'index {idx}': int(pgn_dict[key][2][idx]) for idx in range(QUERY_WORD_LEN)})
                    writer.writerow(written_dict)
                    
                pgn_dict = {}
                    
                    
        
    print(f'Number of games is {game_count}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to save an ending database as csv file.')
    parser.add_argument('-p', '--path', type=str, default = 'lichess_data/lichess_db_standard_rated_2016-09.pgn', 
                        help='The path of the game dataset.')
    parser.add_argument('-g', '--game_limit', type=int, default=200000, 
                        help='Number of games processed. Infinite - all the games in the file.')
    parser.add_argument('-m', '--base_multiplier', type=float, default=0.95,
                        help='Multiplier for decaying reward for long games')
    args = parser.parse_args()
    main(args) 