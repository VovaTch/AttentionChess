import torch.nn
import torch.nn.functional as F
import numpy as np

from utils.matcher import match_moves


def test_loss(pred_moves: torch.Tensor, target_moves: torch.Tensor):
    l1_loss = torch.nn.L1Loss()
    return l1_loss(pred_moves, target_moves)


# TODO: a more ordered criteria that encompasses everything. Fix moves tensor back to 6
class Criterion(torch.nn.Module):

    def __init__(self, losses, eos_coef=0.25):
        super().__init__()
        self.eos_coef = eos_coef
        self.losses = losses
        self.batch_size = 0
        self.query_size = 0

    def mse_score_loss(self, lined_up_preds: torch.Tensor, lined_up_targets: torch.Tensor):
        """Score for the move strength. The move strength should be drawn from outside, or another class"""

        # Collecting all general probability vector loss of everything
        loss_ce_gen = 0
        for idx_batch in range(self.batch_size):
            prob_vec_pred = lined_up_preds[self.query_size * idx_batch: self.query_size * (1 + idx_batch), :]
            prob_vec_target = lined_up_targets[self.query_size * idx_batch: self.query_size * (1 + idx_batch), :]
            legal_lined_up_preds = prob_vec_pred[prob_vec_target[:, 3] == 0]
            legal_lined_up_targets = prob_vec_target[prob_vec_target[:, 3] == 0]
            prob_target_filtered = (legal_lined_up_targets[:, 4] + 1e-6) / \
                                   torch.sum(legal_lined_up_targets[:, 4] + 1e-6)
            loss_ce_gen += cross_entropy_gen(legal_lined_up_preds[:, 4], prob_target_filtered) / self.batch_size

        loss = {'loss_score': loss_ce_gen}
        return loss

    def label_loss(self, lined_up_preds: torch.Tensor, lined_up_targets: torch.Tensor):
        """Loss function for the matching of moves. Uses l1 loss for matching the moves.
        Later, the prediction need to be multiplied by 64"""

        lined_up_preds_legals = lined_up_preds[:, 3]
        # lined_up_preds_legals[:, 1] = 0
        lined_up_targets_legals = lined_up_targets[:, 3]

        # Both of this line for the kinda messy legal move flag
        # lined_up_targets_legals[lined_up_targets_legals == 10] = 0
        # lined_up_targets_legals[lined_up_targets_legals == -100] = 1
        lined_up_targets_legals = lined_up_targets_legals.to(dtype=torch.int64)

        # Focal loss loss, more numerically stable than classic BCE #TODO: CHECK HOW I DID IT IN ADJDETR
        # weight_vector = torch.tensor([1, self.eos_coef]).to(lined_up_preds.device)
        # Floss = FocalLoss(weight=weight_vector, reduction='mean')
        # loss_ce = Floss(lined_up_preds_legals, lined_up_targets_legals)
        weight_vector = torch.ones(lined_up_preds.size()[0]).to(lined_up_preds.device)
        weight_vector[lined_up_targets_legals == 1] = self.eos_coef
        crit = torch.nn.BCEWithLogitsLoss(weight=weight_vector)
        loss_ce = crit(-lined_up_preds_legals.float(), lined_up_targets_legals.float())

        loss = {'loss_labels': loss_ce}
        return loss

    def move_loss(self, lined_up_preds: torch.Tensor, lined_up_targets: torch.Tensor):
        """Loss function for the legality of moves. Uses focal loss with gamma=2 as a default."""

        lined_up_targets_legals = lined_up_targets[:, 3]

        # Both of this line for the kinda messy legal move flag
        lined_up_targets_legals[lined_up_targets_legals == 10] = 0
        lined_up_targets_legals[lined_up_targets_legals == -100] = 1
        lined_up_targets_legals = lined_up_targets_legals.to(dtype=torch.int64)

        # matching losses on moves; ignores the move quality in this loss
        mse_loss = torch.nn.MSELoss()
        lined_up_pred_moves = lined_up_preds[:, [0, 1, 2, 5]]
        lined_up_pred_moves = lined_up_pred_moves[lined_up_targets_legals == 0]
        lined_up_target_moves = lined_up_targets[:, [0, 1, 2, 5]]
        lined_up_target_moves = lined_up_target_moves[lined_up_targets_legals == 0]
        loss_move = mse_loss(lined_up_pred_moves, lined_up_target_moves)

        loss = {'loss_move': loss_move}

        return loss

    @torch.no_grad()
    def cardinality_loss(self, pred_moves: torch.Tensor, target_moves: torch.Tensor):
        """From DETR, cardinality error is a great representive for the detection of the correct bounding boxes."""

        num_targets = torch.sum(target_moves[:, 3] == 0)
        num_preds = torch.sum(pred_moves[:, 3] > 0)
        cardinality_error = np.abs(int(num_targets) - int(num_preds)) / self.batch_size

        loss = {'loss_cardinality': cardinality_error}
        return loss

    @torch.no_grad()
    def cardinality_loss_direction(self, pred_moves: torch.Tensor, target_moves: torch.Tensor):
        """From DETR, cardinality error is a great representive for the detection of the correct bounding boxes."""

        num_targets = torch.sum(target_moves[:, 3] == 0)
        num_preds = torch.sum(pred_moves[:, 3] > 0)
        cardinality_error = -(int(num_targets) - int(num_preds)) / self.batch_size

        loss = {'loss_cardinality_direction': cardinality_error}
        return loss

    def get_loss(self, loss, lined_up_preds, lined_up_targets, matching_indices):
        loss_map = {
            'labels': self.label_loss,
            'cardinality': self.cardinality_loss,
            'cardinality_direction': self.cardinality_loss_direction,
            'move': self.move_loss,
            'mse_score': self.mse_score_loss
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](lined_up_preds, lined_up_targets)

    def forward(self, pred_moves: torch.Tensor, target_moves: torch.Tensor):

        self.batch_size = pred_moves.size()[0]
        self.query_size = pred_moves.size()[1]

        matching_indices = match_moves(pred_moves, target_moves)
        lined_up_preds, lined_up_targets = move_lineup(pred_moves, target_moves, matching_indices)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, lined_up_preds, lined_up_targets, matching_indices))

        return losses


def move_lineup(pred_moves: torch.Tensor, target_moves: torch.Tensor, matching_indices):
    """A utility function for loss computation; lines up predicted moves with matched targets"""
    lined_up_targets = torch.zeros((0, 6)).to(pred_moves.device)
    lined_up_preds = pred_moves.flatten(0, 1)
    for img_idx, match_idx in enumerate(matching_indices):
        lined_up_targets_img = torch.Tensor([0, 0, 0, 1, 0, 0]).repeat((pred_moves.size()[1], 1)).\
            to(pred_moves.device)
        for ind_match in match_idx:
            lined_up_targets_img[ind_match[0], :] = target_moves[img_idx, ind_match[1], :]
        lined_up_targets = torch.cat((lined_up_targets, lined_up_targets_img), 0)

    return lined_up_preds, lined_up_targets


class FocalLoss(torch.nn.Module):
    """A class for implementation of focal loss; zero the gradients in correct classifications,
    enhance on incorrect ones."""

    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        torch.nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


def cross_entropy_gen(input, target):
    """Implementing a general cross entropy function, accepting logits"""
    return torch.mean(-torch.sum(target * (input - torch.logsumexp(input, 0)), 0))

