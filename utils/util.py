import chess
import torch
import pandas as pd
import json
import numpy as np

from pathlib import Path
from itertools import repeat
from collections import OrderedDict


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


# Conerts a chess board to tensor with all the data needed for attchess.
def board_to_tensor_full(board: chess.Board):

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
    x = x.unsqueeze(0)

    x_turn = torch.zeros((1, 8, 8), dtype=torch.float) + board.turn
    x_c_grid, x_r_grid = torch.meshgrid(torch.tensor(range(8)), torch.tensor(range(8)), indexing='xy')
    x_c_grid, x_r_grid = x_c_grid.unsqueeze(0), x_r_grid.unsqueeze(0)
    x_special = torch.zeros((1, 8, 8), dtype=torch.float)

    # Check for en-passant
    if board.ep_square:
        coordinates = (np.floor(board.ep_square / 8), board.ep_square % 8)
        x_special[0, int(coordinates[0]), int(coordinates[1])] = 1

    # Check for castling rights
    if board.has_castling_rights(chess.WHITE):
        x_special[0, 0, 4] = 1
    if board.has_castling_rights(chess.BLACK):
        x_special[0, 7, 4] = 1

    # Concatenate all matrices
    x_total = torch.cat((x, x_turn, x_r_grid, x_c_grid, x_special), 0)

    return x_total


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

    move_legality = 0
    move_quality = 0

    move_torch = torch.Tensor([from_square, to_square, promotion, move_legality, move_quality]).float()

    return move_torch


# push move on full tensor board, assuming the move is legal
def push_torch(board_full_torch, move_torch):
    board_full_torch_after = board_full_torch.clone()

    # Check if move is legal; assume it is because the net was taught to make only legal moves
    if move_torch[3] <= 0:
        return board_full_torch_after

    move_coor_from = (int(np.floor(move_torch[0] / 8)), int(np.floor(move_torch[0]) % 8))
    move_coor_to = (int(np.floor(move_torch[1] / 8)), int(np.floor(move_torch[1]) % 8))

    castling_flag = False

    # White's turn
    if board_full_torch_after[1, 0, 0] == 1:

        board_full_torch_after[0, move_coor_to[0], move_coor_to[1]] = \
            board_full_torch_after[0, move_coor_from[0], move_coor_from[1]]
        board_full_torch_after[0, move_coor_from[0], move_coor_from[1]] = 0
        # castling kingside
        if move_coor_from == (4, 0) and move_coor_to == (6, 0):
            board_full_torch_after[0, 5, 0] = 4
            castling_flag = True
        # castling queenside
        elif move_coor_from == (4, 0) and move_coor_to == (2, 0):
            board_full_torch_after[0, 3, 0] = 4
            castling_flag = True
        # promotion
        if np.floor(move_torch[2]) != 0:
            board_full_torch_after[0, move_coor_to[0], move_coor_to[1]] = move_torch[2]
        # turn
        board_full_torch_after[1, :, :] = 0
        # castling
        if castling_flag:
            board_full_torch_after[0, 4, 0] = 0
        # En passant
        board_full_torch_after[4, :, 2:6] = 0
        if move_coor_from[0] == 1 and move_coor_to[0] == 3 and \
                board_full_torch_after[0, move_coor_to[0], move_coor_to[1]] == 1:
            board_full_torch_after[4, move_coor_to[1], 2] = 1

    # Black's turn
    else:
        board_full_torch_after[0, move_coor_to[0], move_coor_to[1]] = \
            board_full_torch_after[0, move_coor_from[0], move_coor_from[1]]
        board_full_torch_after[0, move_coor_from[0], move_coor_from[1]] = 0
        # castling kingside
        if move_coor_from == (4, 7) and move_coor_to == (6, 7):
            board_full_torch_after[0, 5, 7] = 4
            castling_flag = True
        # castling queenside
        elif move_coor_from == (4, 7) and move_coor_to == (2, 7):
            board_full_torch_after[0, 3, 7] = 4
            castling_flag = True
        # promotion
        if np.floor(move_torch[2]) != 0:
            board_full_torch_after[0, move_coor_to[0], move_coor_to[1]] = -move_torch[2]
        # turn
        board_full_torch_after[1, :, :] = 1
        # castling
        if castling_flag:
            board_full_torch_after[0, 4, 7] = 0
        # En passant
        board_full_torch_after[4, :, 2:6] = 0
        if move_coor_from[0] == 6 and move_coor_to[0] == 4 and \
                board_full_torch_after[0, move_coor_to[0], move_coor_to[1]] == 1:
            board_full_torch_after[4, move_coor_to[1], 5] = 1

    return board_full_torch_after


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
    ''' wrapper function for endless data loader. '''
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
