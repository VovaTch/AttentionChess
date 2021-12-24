import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.util import board_to_tensor, legal_move_mask, move_to_tensor, \
    board_to_embedding_coord, move_to_coordinate
from utils.matcher import match_moves
from model.attchess import AttChess, BoardEmbTrainNet, MoveEmbTrainNet
from data_loaders.game_roll import GameRoller
from data_loaders.dataloader import RuleAttentionChessLoader, collate_fn, BoardEmbeddingLoader, MoveEmbeddingLoader
from model import loss


def main():

    board = chess.Board()
    board_embed = board_to_embedding_coord(board)
    loader = RuleAttentionChessLoader(batch_size=4, collate_fn=collate_fn)
    attchess = AttChess()
    attchess = attchess.eval().to('cuda')
    boardnet = BoardEmbTrainNet()
    boardnet = boardnet.eval().to('cuda')
    movenet = MoveEmbTrainNet()
    movenet = movenet.eval().to('cuda')
    move = chess.Move.from_uci('e2e4')
    coordinates = move_to_coordinate(move)

    board_data_loader = BoardEmbeddingLoader(batch_size=8)
    move_data_loader = MoveEmbeddingLoader(batch_size=8)

    for batch_idx, (board_state, piece_prob, properties) in enumerate(board_data_loader):
        board_state, piece_prob, properties = board_state.to('cuda'), piece_prob.to('cuda'), properties.to('cuda')
        print(batch_idx)
        print(board_state)
        print(piece_prob)
        print(properties)
        output_piece, output_prop = boardnet(board_state)
        break

    for batch_idx, (move_word, move_coord, promotion_prob) in enumerate(move_data_loader):
        move_word, move_coord, promotion_prob = move_word.to('cuda'), move_coord.to('cuda'), promotion_prob.to('cuda')
        print(batch_idx)
        print(move_word)
        print(move_coord * 7)
        print(promotion_prob)
        output_coor, output_prom = movenet(move_word)
        break

    # game_roller = GameRoller(model_good=attchess, model_evil=attchess)
    # game_roller.roll_game(board, num_of_branches=3)

    print(1111)


if __name__ == '__main__':
    main()
