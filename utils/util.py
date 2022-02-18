import chess
import torch
import pandas as pd
import json
import numpy as np

from pathlib import Path
from itertools import repeat
from collections import OrderedDict


# TODO: 1. convert states to word embeddings.


# Convert a chess board to a reduced alpha-zero-like representation. 
# 6 white, 6 black, 1 turn, 1 white castling, 1 black castling, 1 en-passant flag
def board_to_bitboard(board: chess.Board):
    x = torch.zeros((64,16), dtype=torch.float)
    for pos in range(64):
        piece = board.piece_type_at(pos)
        if piece:
            color = int(bool(board.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
            col = int(pos % 8)
            row = int(pos / 8)
            x[row * 8 + col, piece + color*6 - 1] = 1 
            
    # turn
    if board.turn:
        x[:, 12] = 1
            
    # Check for castling rights
    if board.has_castling_rights(chess.WHITE):
        x[:, 13] = 1
    if board.has_castling_rights(chess.BLACK):
        x[:, 14] = 1
        
    # check for en-passant
    if board.ep_square:
        x[board.ep_square, 15] = 1
        
    x = x.reshape(8, 8, 16)
    return x

# Converts a chess board to pytorch tensor
def board_to_tensor(board):
    # Python chess uses flattened representation of the board
    x = torch.zeros(64, dtype=torch.float)
    for pos in range(64):
        piece = board.piece_type_at(pos)
        if piece:
            color = int(bool(board.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
            col = int(pos % 8)
            row = int(pos / 8)
            x[row * 8 + col] = -piece if color else piece
    x = x.reshape(8, 8)
    return x


# Convert board to a grid of embedding coordinates
def board_to_embedding_coord(board: chess.Board):

    # Python chess uses flattened representation of the board
    x = torch.zeros(64, dtype=torch.float)
    for pos in range(64):
        piece = board.piece_type_at(pos)
        if piece:
            color = int(bool(board.occupied_co[chess.BLACK] & chess.BB_SQUARES[pos]))
            col = int(pos % 8)
            row = int(pos / 8)
            x[row * 8 + col] = -piece if color else piece
    x = x.reshape(8, 8)
    x += 6

    if board.ep_square:
        coordinates = (np.floor(board.ep_square / 8), board.ep_square % 8)
        if x[int(coordinates[0]), int(coordinates[1])] == 5:
            x[int(coordinates[0]), int(coordinates[1])] = 16
        else:
            x[int(coordinates[0]), int(coordinates[1])] = 17

    # Check for castling rights
    if board.has_castling_rights(chess.WHITE):
        x[0, 4] = 14
    if board.has_castling_rights(chess.BLACK):
        x[7, 4] = 15

    x += (not board.turn) * 18
    x = x.int()
    return x


# Convert the move to a format for attchess learning, also used to advance the attchess board format
def move_to_tensor(move: chess.Move):
    from_square = move.from_square
    to_square = move.to_square

    # Handle promotions
    if move.promotion is not None:
        promotion_symbol = move.promotion
        if promotion_symbol is chess.QUEEN:
            promotion = 5
        elif promotion_symbol is chess.ROOK:
            promotion = 4
        elif promotion_symbol is chess.BISHOP:
            promotion = 3
        else:
            promotion = 2
    else:
        promotion = 0

    move_legality_legal = 0
    move_quality = 0
    resign_flag = 0  # 1 for resign, -1 for draw

    move_torch = torch.Tensor([from_square/64, to_square/64, promotion, move_legality_legal,
                               move_quality, resign_flag]).float()

    return move_torch


def word_to_move(word):

    coor_col = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    promotion_char = ['n', 'b', 'r', 'q']

    # Decompose to individual move coordinates
    coordinates_from_to = (int(word % 64), 
                           torch.div(word, 64, rounding_mode='floor').int())
    coordinates_from = (int(coordinates_from_to[0] % 8), coordinates_from_to[0] // 8)  # 0 is a,b,c... 1 is numbers

    coor_char_from = coor_col[coordinates_from[0]] + str(int(coordinates_from[1] + 1))

    # If not promoting
    if coordinates_from_to[1] < 64:
        coordinates_to = (int(coordinates_from_to[1] % 8), 
                          torch.div(coordinates_from_to[1], 8, rounding_mode='floor').int())
        coor_char_to = coor_col[coordinates_to[0]] + str(int(coordinates_to[1] + 1))

    # If promoting
    else:
        if 64 <= coordinates_from_to[1] < 67:
            coor_shift = 65 - coordinates_from_to[1]
            coor_up_down = 0 if coordinates_from[1] == 1 else 7
            coor_char_to = coor_col[coordinates_from[0] + coor_shift] + str(coor_up_down + 1) + promotion_char[3]
        elif 67 <= coordinates_from_to[1] < 70:
            coor_shift = 68 - coordinates_from_to[1]
            coor_up_down = 0 if coordinates_from[1] == 1 else 7
            coor_char_to = coor_col[coordinates_from[0] + coor_shift] + str(coor_up_down + 1) + promotion_char[2]
        elif 70 <= coordinates_from_to[1] < 73:
            coor_shift = 71 - coordinates_from_to[1]
            coor_up_down = 0 if coordinates_from[1] == 1 else 7
            coor_char_to = coor_col[coordinates_from[0] + coor_shift] + str(coor_up_down + 1) + promotion_char[1]
        else:
            coor_shift = 74 - coordinates_from_to[1]
            coor_up_down = 0 if coordinates_from[1] == 1 else 7
            coor_char_to = coor_col[coordinates_from[0] + coor_shift] + str(coor_up_down + 1) + promotion_char[0]

    move = chess.Move.from_uci(coor_char_from + coor_char_to)

    return move


def move_to_coordinate(move: chess.Move):
    from_square = move.from_square
    to_square = move.to_square

    from_square_coor = (int(from_square % 8), int(from_square / 8))  # 0 for a,b,c, 1 for numbers
    to_square_coor = (int(to_square % 8), int(to_square / 8))

    # Handle promotions
    if move.promotion is not None:
        direction = from_square_coor[0] - to_square_coor[0]  # 1 for left, -1 for right
        promotion_symbol = move.promotion
        if promotion_symbol is chess.QUEEN:
            to_square = 65 + direction
        elif promotion_symbol is chess.ROOK:
            to_square = 68 + direction
        elif promotion_symbol is chess.BISHOP:
            to_square = 71 + direction
        else:
            to_square = 74 + direction

    return torch.tensor([int(from_square), int(to_square)])


def legal_move_mask(board: chess.Board):
    """Given a board, produce a masked logits matrix that considers only legal moves"""
    filter_mask = torch.zeros((64, 64)) - np.inf
    legal_moves = board.legal_moves
    for legal_move in legal_moves:
        from_square = legal_move.from_square
        to_square = legal_move.to_square
        filter_mask[from_square, to_square] = 0


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """''' wrapper function for endless data loader. '''"""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
