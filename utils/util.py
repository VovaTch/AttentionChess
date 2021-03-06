import chess
import torch
import pandas as pd
import json
import numpy as np

from pathlib import Path
from itertools import repeat
from collections import OrderedDict


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
    

def is_game_end(board: chess.Board):
    """Checks if the board is an endgame state board."""
    if board.is_checkmate():
        return True, -1 * board.turn + 1 * (not board.turn)
    elif board.is_stalemate() or board.is_repetition() or \
            board.is_seventyfive_moves() or board.is_insufficient_material():
        return True, 0
    return False, 0
