import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
from model.chess_conv_attention import attention

from utils.util import board_to_tensor, legal_move_mask, move_to_tensor, \
    board_to_embedding_coord, move_to_coordinate
from utils.matcher import match_moves
from model.attchess import AttChess, BoardEmbTrainNet, MoveEmbTrainNet
from data_loaders.game_roll import GameRoller
from data_loaders.dataloader import RuleAttentionChessLoader, collate_fn, BoardEmbeddingLoader, MoveEmbeddingLoader, SelfPlayChessLoader
from model import loss


def main():

    board = chess.Board()
    board2 = chess.Board()
    board_embed = board_to_embedding_coord(board)
    loader = RuleAttentionChessLoader(batch_size=4, collate_fn=collate_fn, shuffle=False)
    attchess = AttChess(hidden_dim=32, num_heads=8, num_encoder=6, num_decoder=6, query_word_len=256, 
                        num_chess_conv_layers=1, p_embedding=False)
    attchess = attchess.eval().to('cpu')
    
    # Load trained model
    # checkpoint = torch.load('model_best_init.pth', map_location=torch.device('cpu'))
    # attchess.load_state_dict(checkpoint['state_dict'])
    
    for idx, (board, quality_vec, board_val) in enumerate(loader):
        outputs = attchess.board_forward(board)
        print(board_val)
    

    

    print(1111)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')  # Necessary for this to work; maybe it will run out of memory like that
    main()
