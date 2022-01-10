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
from data_loaders.game_roll import GameRoller, BoardNode, ScoreWinFast, InferenceBoardNode, InferenceMoveSearcher
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
    score = ScoreWinFast(moves_to_end=4 // 2)
    
    # Make a quick checkmate
    nodes = list()
    nodes.append(InferenceBoardNode(board, None, score))
    nodes.append(nodes[-1].perform_move(chess.Move.from_uci('f2f3')))
    nodes.append(nodes[-1].perform_move(chess.Move.from_uci('e7e5')))
    nodes.append(nodes[-1].perform_move(chess.Move.from_uci('g2g4')))
    nodes.append(nodes[-1].perform_move(chess.Move.from_uci('d8h4')))
    nodes[-1].propagate_score()
    outputs = nodes[0].flatten_tree()
    
    

    print(1111)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  # Necessary for this to work; maybe it will run out of memory like that
    main()
