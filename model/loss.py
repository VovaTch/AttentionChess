import torch.nn
import torch.nn.functional as F
import numpy as np


def des_boost_l1(board, turns, predicted_logits, played_logits):
    """Policy number 1: go for the quickest win, encouraging aggressive play and sacrifices"""
    num_of_moves = len(turns)
    pred_targets = predicted_logits[played_logits != -np.inf]
    played_targets = played_logits[played_logits != -np.inf]

    for idx in range(num_of_moves):
        played_targets[idx] *= 10 / (num_of_moves - idx)

    loss = torch.nn.MSELoss()(pred_targets, played_targets)
    return loss


def stable_l1(board, turns, predicted_logits, played_logits):
    """Policy number 2: go for the win, no matter what. Stockfish/AlphaZero style"""
    pred_targets = predicted_logits[played_logits != -np.inf]
    played_targets = played_logits[played_logits != -np.inf]
    loss = torch.nn.MSELoss()(pred_targets, played_targets)
    return loss


def greedy_l1(board: torch.Tensor, turns, predicted_logits, played_logits):
    """Policy number 3: go for the greatest material advantage, refrains from sacrifices except for
    short-term material gain"""
    num_of_moves = len(turns)
    board_copy = torch.clone(board)

    # Turn markers of pieces into values, white
    board_copy[board_copy == 2] = 2.9
    board_copy[board_copy == 3] = 3.1
    board_copy[board_copy == 4] = 5
    board_copy[board_copy == 5] = 9
    board_copy[board_copy == 6] = 0

    # Turn markers of pieces into values, black
    board_copy[board_copy == -2] = -2.9
    board_copy[board_copy == -3] = -3.1
    board_copy[board_copy == -4] = -5
    board_copy[board_copy == -5] = -9
    board_copy[board_copy == -6] = 0

    pred_targets = predicted_logits[played_logits != -np.inf]
    played_targets = played_logits[played_logits != -np.inf]

    turn_invertor = 1
    for idx in range(num_of_moves):
        material_factor = torch.sum(board_copy[idx]) / 40
        played_targets[idx] += material_factor * turn_invertor
        turn_invertor *= -1

    loss = torch.nn.MSELoss()(pred_targets, played_targets)

    return loss


def rule_teaching_loss(boards: torch.Tensor, moves: torch.Tensor):
    pass
