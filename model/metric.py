import torch
import numpy as np

from utils.matcher import match_moves
from model.loss import move_lineup
from model.loss import Criterion

# Losses directly from the loss function


@torch.no_grad()
def mse_score_loss(pred_quality_vec: torch.Tensor, target_quality_vec: torch.Tensor, criterion: Criterion):
    """Goes to the criterion for the loss"""
    loss = criterion.mse_score_loss(pred_quality_vec, target_quality_vec)
    return loss['loss_score']


@torch.no_grad()
def smile_pass(output, target):
    """Return an empty metric; next time I'll try to figure out how to incorporate winning percentage and ELO"""
    return 1

