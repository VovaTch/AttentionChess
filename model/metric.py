import torch
import numpy as np

from utils.matcher import match_moves
from model.loss import move_lineup


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

    if 'eos_loss' in kwargs:
        eos_loss = kwargs['eos_loss']
    else:
        eos_loss = 0.25

    if 'moves_coef' in kwargs:
        moves_coef = kwargs['moves_coef']
    else:
        moves_coef = 5

    matching_indices = match_moves(pred_moves, target_moves)
    lined_up_preds, lined_up_targets = move_lineup(pred_moves, target_moves, matching_indices)
    lined_up_preds_legals = lined_up_preds[:, 3]
    lined_up_targets_legals = lined_up_targets[:, 3]

    # Both of this line for the kinda messy legal move flag
    lined_up_targets_legals[lined_up_targets_legals == 10] = 1
    lined_up_targets_legals[lined_up_targets_legals == -100] = 0

    # BCE with logits loss, more numerically stable than classic BCE #TODO: CHECK HOW I DID IT IN ADJDETR
    weight_vector = torch.ones(lined_up_targets_legals.size()[0]).to(pred_moves.device)
    weight_vector[lined_up_targets_legals == 1] = eos_loss
    bceloss = torch.nn.BCEWithLogitsLoss(weight=weight_vector)
    loss_bce = bceloss(lined_up_preds_legals, lined_up_targets_legals)

    # matching losses on moves; ignores the move quality in this loss
    l1_loss = torch.nn.L1Loss()
    lined_up_pred_moves = lined_up_preds[:, [0, 1, 2, 5]]
    lined_up_pred_moves = lined_up_pred_moves[lined_up_targets_legals == 1]
    lined_up_target_moves = lined_up_targets[:, [0, 1, 2, 5]]
    lined_up_target_moves = lined_up_target_moves[lined_up_targets_legals == 1]
    loss = l1_loss(lined_up_pred_moves, lined_up_target_moves)

    # Total loss
    loss_tot = loss + moves_coef * loss_bce
    return loss_tot


@torch.no_grad()
def cardinality_loss(pred_moves, target_moves):
    """From DETR, cardinality error is a great representive for the detection of the correct bounding boxes."""
    num_of_moves = target_moves.size()[0]
    pred_flat = pred_moves.flatten(0, 1)
    target_flat = target_moves.flatten(0, 1)
    num_targets = torch.sum(target_flat[:, 3] == 10)
    num_preds = torch.sum(pred_flat[:, 3] >= 0)
    cardinality_error = np.abs(int(num_targets) - int(num_preds)) / num_of_moves
    return cardinality_error
