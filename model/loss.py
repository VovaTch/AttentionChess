import torch.nn
import torch.nn.functional as F
import numpy as np

from utils.matcher import match_moves


def des_boost_l1(board, turns, predicted_logits, played_logits, **kwargs):
    """Policy number 1: go for the quickest win, encouraging aggressive play and sacrifices"""
    num_of_moves = len(turns)
    pred_targets = predicted_logits[played_logits != -np.inf]
    played_targets = played_logits[played_logits != -np.inf]

    for idx in range(num_of_moves):
        played_targets[idx] *= 10 / (num_of_moves - idx)

    loss = torch.nn.MSELoss()(pred_targets, played_targets)
    return loss


def stable_l1(board, turns, predicted_logits, played_logits, **kwargs):
    """Policy number 2: go for the win, no matter what. Stockfish/AlphaZero style"""
    pred_targets = predicted_logits[played_logits != -np.inf]
    played_targets = played_logits[played_logits != -np.inf]
    loss = torch.nn.MSELoss()(pred_targets, played_targets)
    return loss


def greedy_l1(board: torch.Tensor, turns, predicted_logits, played_logits, **kwargs):
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


def move_lineup(pred_moves: torch.Tensor, target_moves: torch.Tensor, matching_indices):

    lined_up_targets = torch.zeros((0, 6)).to(pred_moves.device)
    lined_up_preds = pred_moves.flatten(0, 1)
    for img_idx, match_idx in enumerate(matching_indices):
        lined_up_targets_img = torch.Tensor([0, 0, 0, -100, 0, 0]).repeat((pred_moves.size()[1], 1)).\
            to(pred_moves.device)
        for ind_match in match_idx:
            lined_up_targets_img[ind_match[0], :] = target_moves[img_idx, ind_match[1], :]
        lined_up_targets = torch.cat((lined_up_targets, lined_up_targets_img), 0)

    return lined_up_preds, lined_up_targets


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


# TODO: a more ordered criteria that encompasses everything
class Criterion:

    def __init__(self, eos_coef = 0.25, moves_coef=5, type_of_score='stable'):
        self.eos_coes = eos_coef
        self.moves_coef = moves_coef

        assert type_of_score in ['stable', 'greedy', 'aggressive'], 'type_of_score must be stable, ' \
                                                                    'greedy, or aggressive'





