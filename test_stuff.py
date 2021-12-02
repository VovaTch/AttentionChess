import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.util import board_to_tensor, board_to_tensor_full, legal_move_mask, move_to_tensor, push_torch
from utils.matcher import match_moves
from model.attchess import AttChess
from data_loaders.game_roll import GameRoller
from data_loaders.dataloader import S1AttentionChessLoader, S2AttentionChessLoader, RuleAttentionChessLoader
from model import loss


def main():

    board = chess.Board()
    board_torch = board_to_tensor_full(board)
    loader = RuleAttentionChessLoader(batch_size=1)
    attchess = AttChess()
    attchess = attchess.eval().to('cuda')

    for idx, (boards, moves) in enumerate(loader):
        print(moves)
        outputs = attchess(boards.squeeze(0).to('cuda'))
        break

    match_idx = match_moves(outputs, moves.squeeze(0).to('cuda'))
    print(1111)





if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')  # TODO: This is necessary for the game generating code to work
    main()
