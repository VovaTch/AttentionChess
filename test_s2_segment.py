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
from model.attchess import AttChess
from data_loaders.mcts import MCTS, Node
from data_loaders.game_roll import GameRoller, BoardNode, ScoreWinFast, InferenceBoardNode, InferenceMoveSearcher
from data_loaders.dataloader import RuleAttentionChessLoader, collate_fn, BoardEmbeddingLoader, MoveEmbeddingLoader, SelfPlayChessLoader
from model import loss


def main():

    board = chess.Board()
    attchess = AttChess(conv_hidden=32, num_heads=1, feature_multiplier=8, num_encoder=6, num_decoder=4, dropout=0.1, query_word_len=256)
    attchess = attchess.eval().to('cuda')
    
    # Load trained model
    checkpoint = torch.load('model_best_init.pth')
    attchess.load_state_dict(checkpoint['state_dict'])
    score = ScoreWinFast(moves_to_end=4 // 2)
    
    # Run an episode
    args = {}
    args['num_simulations'] = 10
    mcts = MCTS(model_good=attchess, model_evil=attchess, args=args)
    root = mcts.run(board)
    
    

    print(1111)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  # Necessary for this to work; maybe it will run out of memory like that
    main()
