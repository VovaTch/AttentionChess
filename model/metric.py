import torch
import numpy as np

from utils.matcher import match_moves
from model.loss import move_lineup
from model.loss import Criterion

# Losses directly from the loss function


@torch.no_grad()
def quality_loss(pred_quality_vec: torch.Tensor, pred_value: torch.Tensor, 
                 target_quality_vec: torch.Tensor, target_value: torch.Tensor, 
                 matching_idx: torch.Tensor, criterion: Criterion):
    """Goes to the criterion for the loss"""
    loss = criterion.quality_loss(pred_quality_vec, pred_value, target_quality_vec, target_value, matching_idx)
    return loss['loss_quality']

@torch.no_grad()
def board_value_loss(pred_quality_vec: torch.Tensor, pred_value: torch.Tensor, 
                 target_quality_vec: torch.Tensor, target_value: torch.Tensor, 
                 matching_idx: torch.Tensor, criterion: Criterion):
    """Goes to the criterion for the loss"""
    loss = criterion.board_value_loss(pred_quality_vec, pred_value, target_quality_vec, target_value, matching_idx)
    return loss['loss_board_value']


@torch.no_grad()
def smile_pass(output, target):
    """Return an empty metric; next time I'll try to figure out how to incorporate winning percentage and ELO"""
    return 1

