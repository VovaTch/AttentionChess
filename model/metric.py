import torch
import numpy as np

from utils.matcher import match_moves
from model.loss import move_lineup
from model.loss import Criterion

# Losses directly from the loss function


@torch.no_grad()
def mse_score_loss(pred_legal_mat: torch.Tensor, pred_quality_vec: torch.Tensor,
                   target_legal_mat: torch.Tensor, target_quality_vec: torch.Tensor, criterion: Criterion):
    """Goes to the criterion for the loss"""
    loss = criterion.mse_score_loss(pred_legal_mat, pred_quality_vec, target_legal_mat, target_quality_vec)
    return loss['loss_score']


@torch.no_grad()
def label_loss(pred_legal_mat: torch.Tensor, pred_quality_vec: torch.Tensor,
               target_legal_mat: torch.Tensor, target_quality_vec: torch.Tensor, criterion: Criterion):
    """Goes to the criterion for the loss"""
    loss = criterion.label_loss(pred_legal_mat, pred_quality_vec, target_legal_mat, target_quality_vec)
    return loss['loss_labels']


@torch.no_grad()
def cardinality_loss(pred_legal_mat: torch.Tensor, pred_quality_vec: torch.Tensor,
                     target_legal_mat: torch.Tensor, target_quality_vec: torch.Tensor, criterion: Criterion):
    """Goes to the criterion for the loss"""
    loss = criterion.cardinality_loss(pred_legal_mat, pred_quality_vec, target_legal_mat, target_quality_vec)
    return loss['loss_cardinality']


@torch.no_grad()
def cardinality_loss_direction(pred_legal_mat: torch.Tensor, pred_quality_vec: torch.Tensor,
                               target_legal_mat: torch.Tensor, target_quality_vec: torch.Tensor, criterion: Criterion):
    """Goes to the criterion for the loss"""
    loss = criterion.cardinality_loss_direction(pred_legal_mat, pred_quality_vec, target_legal_mat, target_quality_vec)
    return loss['loss_cardinality_direction']


@torch.no_grad()
def smile_pass(output, target):
    """Return an empty metric; next time I'll try to figure out how to incorporate winning percentage and ELO"""
    return 1


@torch.no_grad()
def a_rule_precision(pred_legal_mat: torch.Tensor, pred_quality_vec: torch.Tensor,
                     target_legal_mat: torch.Tensor, target_quality_vec: torch.Tensor, *args):
    """Average precision for the rules"""

    flatten_pred_legal_mat = pred_legal_mat.flatten()
    flatten_target_legal_mat = target_legal_mat.flatten()

    tp_count = 0
    fp_count = 0

    for pred_ind, target_ind in zip(flatten_pred_legal_mat, flatten_target_legal_mat):
        if pred_ind > 0 and target_ind == 1:
            tp_count += 1
        elif pred_ind > 0 and target_ind == 0:
            fp_count += 1

    return tp_count / (tp_count + fp_count + 1e-5)


@torch.no_grad()
def a_rule_recall(pred_legal_mat: torch.Tensor, pred_quality_vec: torch.Tensor,
                  target_legal_mat: torch.Tensor, target_quality_vec: torch.Tensor, *args):
    """Average precision for the rules"""

    flatten_pred_legal_mat = pred_legal_mat.flatten()
    flatten_target_legal_mat = target_legal_mat.flatten()

    tp_count = 0
    fn_count = 0

    for pred_ind, target_ind in zip(flatten_pred_legal_mat, flatten_target_legal_mat):
        if pred_ind > 0 and target_ind == 1:
            tp_count += 1
        elif pred_ind <= 0 and target_ind == 1:
            fn_count += 1

    return tp_count / (tp_count + fn_count + 1e-5)

