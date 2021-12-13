import torch
import numpy as np

from utils.matcher import match_moves
from model.loss import move_lineup
from model.loss import Criterion

# Losses directly from the loss function


@torch.no_grad()
def mse_score_loss(pred_moves, target_moves, criterion: Criterion):
    """Goes to the criterion for the loss"""
    matching_indices = match_moves(pred_moves, target_moves)
    lined_up_preds, lined_up_targets = move_lineup(pred_moves, target_moves, matching_indices)
    loss = criterion.mse_score_loss(lined_up_preds, lined_up_targets)
    return loss['loss_score']


@torch.no_grad()
def label_loss(pred_moves, target_moves, criterion: Criterion):
    """Goes to the criterion for the loss"""
    matching_indices = match_moves(pred_moves, target_moves)
    lined_up_preds, lined_up_targets = move_lineup(pred_moves, target_moves, matching_indices)
    loss = criterion.label_loss(lined_up_preds, lined_up_targets)
    return loss['loss_labels']


@torch.no_grad()
def move_loss(pred_moves, target_moves, criterion: Criterion):
    """Goes to the criterion for the loss"""
    matching_indices = match_moves(pred_moves, target_moves)
    lined_up_preds, lined_up_targets = move_lineup(pred_moves, target_moves, matching_indices)
    loss = criterion.move_loss(lined_up_preds, lined_up_targets)
    return loss['loss_move']


@torch.no_grad()
def cardinality_loss(pred_moves, target_moves, criterion: Criterion):
    """Goes to the criterion for the loss"""
    matching_indices = match_moves(pred_moves, target_moves)
    lined_up_preds, lined_up_targets = move_lineup(pred_moves, target_moves, matching_indices)
    loss = criterion.cardinality_loss(lined_up_preds, lined_up_targets)
    return loss['loss_cardinality']


@torch.no_grad()
def cardinality_loss_direction(pred_moves, target_moves, criterion: Criterion):
    """Goes to the criterion for the loss"""
    matching_indices = match_moves(pred_moves, target_moves)
    lined_up_preds, lined_up_targets = move_lineup(pred_moves, target_moves, matching_indices)
    loss = criterion.cardinality_loss_direction(lined_up_preds, lined_up_targets)
    return loss['loss_cardinality_direction']


@torch.no_grad()
def smile_pass(output, target):
    """Return an empty metric; next time I'll try to figure out how to incorporate winning percentage and ELO"""
    return 1


@torch.no_grad()
def a_rule_precision(pred_moves, target_moves, *args):
    """Average precision for the rules"""

    matching_indices = match_moves(pred_moves, target_moves)
    lined_up_preds, lined_up_targets = move_lineup(pred_moves, target_moves, matching_indices)

    tp_count = 0
    fp_count = 0

    for pred_ind, target_ind in zip(lined_up_preds, lined_up_targets):
        if pred_ind[3] > 0 and target_ind[3] == 0:
            tp_count += 1
        elif pred_ind[3] > 0 and target_ind[3] == 1:
            fp_count += 1

    return tp_count / (tp_count + fp_count + 1e-5)


@torch.no_grad()
def a_rule_recall(pred_moves, target_moves, *args):
    """Average precision for the rules"""

    matching_indices = match_moves(pred_moves, target_moves)
    lined_up_preds, lined_up_targets = move_lineup(pred_moves, target_moves, matching_indices)

    tp_count = 0
    fn_count = 0

    for pred_ind, target_ind in zip(lined_up_preds, lined_up_targets):
        if pred_ind[3] > 0 and target_ind[3] == 0:
            tp_count += 1
        elif pred_ind[3] <= 0 and target_ind[3] == 0:
            fn_count += 1

    return tp_count / (tp_count + fn_count + 1e-5)

