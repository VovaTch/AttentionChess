import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.util import board_to_tensor, board_to_tensor_full, legal_move_mask, move_to_tensor, push_torch
from model.attchess import AttChess
from data_loaders.game_roll import GameRoller
from data_loaders.dataloader import S1AttentionChessLoader, S2AttentionChessLoader
from model import loss


def main():

    board = chess.Board()
    move_tensor = move_to_tensor(chess.Move.from_uci('e2e4'))
    move_tensor[3] = 1
    board_full_tensor = board_to_tensor_full(board)
    board_full_tensor_e4 = push_torch(board_full_tensor, move_tensor)
    move_tensor = move_to_tensor(chess.Move.from_uci('e7e6'))
    board_full_tensor_e4_e6 = push_torch(board_full_tensor_e4, move_tensor)

    print(1111)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  # TODO: This is necessary for the game generating code to work
    main()
