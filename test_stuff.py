import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy

from utils.util import board_to_tensor, legal_move_mask, move_to_tensor, \
    board_to_embedding_coord, move_to_coordinate
from utils.matcher import match_moves
from model.attchess import AttChess, BoardEmbTrainNet, MoveEmbTrainNet
from data_loaders.game_roll import GameRoller
from data_loaders.dataloader import RuleAttentionChessLoader, collate_fn, BoardEmbeddingLoader, MoveEmbeddingLoader, SelfPlayChessLoader
from model import loss


def main():

    board = chess.Board()
    board_embed = board_to_embedding_coord(board)
    loader = RuleAttentionChessLoader(batch_size=32, collate_fn=collate_fn)
    attchess = AttChess(hidden_dim=32, num_heads=8, num_encoder=6, num_decoder=6, query_word_len=256, 
                        num_chess_conv_layers=0, p_embedding=False)
    attchess = attchess.eval().to('cuda')
    
    # Load trained model
    checkpoint = torch.load('model_best_init.pth')
    attchess.load_state_dict(checkpoint['state_dict'])
    
    move = chess.Move.from_uci('e2e4')
    coordinates = move_to_coordinate(move)

    game_roller = GameRoller(copy.deepcopy(attchess), copy.deepcopy(attchess), device='cuda')
    self_play_loader = SelfPlayChessLoader(batch_size=32, game_roller=game_roller, collate_fn=collate_fn)
    
    for idx, (board_list, quality_tensor) in enumerate(self_play_loader):
        quality_tensor[:, :-1] = quality_tensor[:, :-1].softmax(dim=1)
        print(quality_tensor)
        # if idx == 2:
        #     break    

    game_roller = GameRoller(model_good=attchess, model_evil=attchess, move_limit=300)
    game_roller.roll_game(board, num_of_branches=1  , expansion_constant=0.02)

    print(1111)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  # Necessary for this to work; maybe it will run out of memory like that
    main()
