import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.util import board_to_tensor, legal_move_mask, move_to_tensor, \
    board_to_embedding_coord, move_to_coordinate
from utils.matcher import match_moves
from model.attchess import AttChess
from data_loaders.game_roll import GameRoller
from data_loaders.dataloader import RuleAttentionChessLoader, collate_fn
from model import loss


def main():

    board = chess.Board()
    board_embed = board_to_embedding_coord(board)
    loader = RuleAttentionChessLoader(batch_size=4, collate_fn=collate_fn)
    attchess = AttChess()
    attchess = attchess.eval().to('cuda')
    move = chess.Move.from_uci('e2e4')
    coordinates = move_to_coordinate(move)

    attchess.board_forward([board, board, board])

    game_roller = GameRoller(model_good=attchess, model_evil=attchess)
    game_roller.roll_game(board, num_of_branches=3)

    print(1111)


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')  # TODO: This is necessary for the game generating code to work
    main()
