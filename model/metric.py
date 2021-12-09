import torch
import numpy as np

from utils.matcher import match_moves
from model.loss import move_lineup
from model.loss import rule_teaching_loss as rtl


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


@torch.no_grad()
def smile_pass(output, target):
    """Return an empty metric; next time I'll try to figure out how to incorporate winning percentage and ELO"""
    return 1


@torch.no_grad()
def rule_teaching_loss(pred_moves: torch.Tensor, target_moves: torch.Tensor, **kwargs):
    """Loss specifically for teaching the net how to play chess and when to resign."""
    return rtl(pred_moves, target_moves)



@torch.no_grad()
def cardinality_loss(pred_moves, target_moves):
    """From DETR, cardinality error is a great representive for the detection of the correct bounding boxes."""
    num_of_moves = target_moves.size()[0]
    pred_flat = pred_moves.flatten(0, 1)
    target_flat = target_moves.flatten(0, 1)
    num_targets = torch.sum(target_flat[:, 3] == 10)
    num_preds = torch.sum(pred_flat[:, 3] > 0)
    cardinality_error = np.abs(int(num_targets) - int(num_preds)) / num_of_moves
    return cardinality_error


@torch.no_grad()
def a_rule_precision(pred_moves, target_moves):
    """Average precision for the rules"""
    pred_flat = pred_moves.flatten(0, 1).unsqueeze(0)
    target_flat = target_moves.flatten(0, 1).unsqueeze(0)

    tp_count = 0
    fp_count = 0

    matched_idx = match_moves(pred_flat, target_flat)
    for idx_pair in matched_idx[0]:
        if pred_flat[0, int(idx_pair[0]), 3] > 0 and target_flat[0, int(idx_pair[1]), 3] == 10:
            tp_count += 1
        elif pred_flat[0, int(idx_pair[0]), 3] > 0 and target_flat[0, int(idx_pair[1]), 3] == -100:
            fp_count += 1

    return tp_count / (tp_count + fp_count + 1e-5)

@torch.no_grad()
def a_rule_recall(pred_moves, target_moves):
    """Average precision for the rules"""
    pred_flat = pred_moves.flatten(0, 1).unsqueeze(0)
    target_flat = target_moves.flatten(0, 1).unsqueeze(0)

    tp_count = 0

    matched_idx = match_moves(pred_flat, target_flat)
    for idx_pair in matched_idx[0]:
        if pred_flat[0, int(idx_pair[0]), 3] > 0 and target_flat[0, int(idx_pair[1]), 3] == 10:
            tp_count += 1

    total_det_count = torch.sum(target_flat[0, :, 3] == 10)

    return tp_count / (total_det_count + 1e-5)

