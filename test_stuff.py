import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.util import board_to_tensor, legal_move_mask
from model.attchess import AttChess
from data_loaders.game_roll import GameRoller
from data_loaders.dataloader import S1AttentionChessLoader, S2AttentionChessLoader
from model import loss


def main():

    board = chess.Board()
    turn = board.turn
    turn_list = [turn]
    print(f'board: \n{board}')
    torch_board = board_to_tensor(board)

    attchess = AttChess()
    attchess = attchess.to('cuda')
    attchess.eval()

    model_parameters = filter(lambda p: p.requires_grad, attchess.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of parameters: {params}')

    game_roller = GameRoller(attchess, device='cuda')
    dataloader = S2AttentionChessLoader(batch_size=1, game_roller=game_roller, adversarial_model=attchess)

    with torch.no_grad():
        for batch_idx, (board, turn, score) in enumerate(dataloader):
            board = board.to('cuda').squeeze()
            score = score.to('cuda').squeeze()
            output_moves = attchess(board, turn)
            loss.des_boost_l1(board, turn, predicted_logits=output_moves, played_logits=score)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  # TODO: This is necessary for the game generating code to work
    main()
