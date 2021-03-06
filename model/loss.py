import torch.nn
import torch.nn.functional as F

def test_loss(pred_moves: torch.Tensor, target_moves: torch.Tensor):
    l1_loss = torch.nn.L1Loss()
    return l1_loss(pred_moves, target_moves)


# TODO: a more ordered criteria that encompasses everything. Fix moves tensor back to 6
class Criterion(torch.nn.Module):

    def __init__(self, losses, eos_coef=0.25, f_loss_gamma=2, mse_weight=0.01):
        super().__init__()
        self.eos_coef = eos_coef
        self.f_loss_gamma = f_loss_gamma
        self.losses = losses
        self.batch_size = 0
        self.query_size = 0
        self.mse_weight = mse_weight

    def board_value_loss(self, pred_quality_vec: torch.Tensor, pred_value: torch.Tensor, 
                         target_quality_vec: torch.Tensor, target_value: torch.Tensor, matching_idx: torch.Tensor):
        """Score for the board value; doing this MSE style"""
        
        mse_loss_handle = torch.nn.MSELoss()
        
        loss_mse = mse_loss_handle(torch.tanh(pred_value), target_value)
        
        loss = {'loss_board_value': loss_mse}
        return loss

    def quality_loss(self, pred_quality_vec: torch.Tensor, pred_value: torch.Tensor, 
                     target_quality_vec: torch.Tensor, target_value: torch.Tensor, matching_idx: torch.Tensor):
        """Score for the move strength. The move strength should be drawn from outside, or another class. -inf for considering all the vector, 
        -1 for 0 error STILL NEED TO IMPLEMENT, NOT SURE IT MATTERS"""

        loss_ce = torch.nn.CrossEntropyLoss()
        move_consider_index = torch.argmax(target_quality_vec, dim=1) == matching_idx
        if torch.sum(matching_idx) < 0:  # TODO: Need to check if it works
            loss_ce_gen = loss_ce(pred_quality_vec, target_quality_vec)
        else:
            loss_ce_gen = loss_ce(pred_quality_vec[move_consider_index], torch.argmax(target_quality_vec, dim=1)[move_consider_index])
            #loss_ce_gen = loss_ce(pred_quality_vec[move_consider_index], target_quality_vec[move_consider_index])
            if torch.isnan(loss_ce_gen):
                loss_ce_gen = 0

        # loss_ce_gen = cross_entropy_gen(pred_quality_vec, target_quality_vec, weights=weight_mat)

        loss = {'loss_quality': loss_ce_gen}
        return loss

    def get_loss(self, loss, pred_quality_vec, pred_value, target_quality_vec, target_value, matching_idx):
        loss_map = {
            'loss_quality': self.quality_loss,
            'loss_board_value': self.board_value_loss
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](pred_quality_vec, pred_value, target_quality_vec, target_value, matching_idx)

    def forward(self, pred_quality_vec: torch.Tensor, pred_value: torch.Tensor, 
                target_quality_vec: torch.Tensor, target_value: torch.Tensor, matching_idx: torch.Tensor, output_aux=None):

        self.batch_size = target_quality_vec.size()[0]
        self.query_size = target_quality_vec.size()[1]

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, pred_quality_vec, pred_value, target_quality_vec, 
                                        target_value, matching_idx))
            
        if output_aux is not None:
            loss_aux = {}
            for key, pred_value in output_aux.items():
                loss = key[:-2]
                loss_aux = self.get_loss(loss, pred_value, pred_value, target_quality_vec, 
                                              target_value, matching_idx)
                losses[key] = loss_aux[loss]

        return losses


def move_lineup(pred_legal_mat: torch.Tensor, pred_quality_vec: torch.Tensor,
                target_legal_mat: torch.Tensor):
    """A utility function for loss computation; lines up predicted moves with matched targets"""

    lined_up_all_preds = torch.zeros((0, pred_quality_vec.size()[1])).to(pred_quality_vec.device)
    match_find = torch.zeros((0, pred_quality_vec.size()[1])).to(pred_quality_vec.device)

    for batch_idx in range(pred_legal_mat.size()[0]):

        pred_q_lined = torch.zeros((pred_quality_vec.size()[1])).to(pred_quality_vec.device) - 1e6
        match_find_batch = torch.zeros((pred_quality_vec.size()[1]), dtype=torch.bool).to(pred_quality_vec.device)

        t_legal_move_mat = target_legal_mat[batch_idx, :, :] == 1
        t_legal_move_idx = torch.nonzero(t_legal_move_mat)
        t_legal_move_word = t_legal_move_idx[:, 0] + 64 * t_legal_move_idx[:, 1]

        p_legal_move_mat = pred_legal_mat[batch_idx, :, :] > 0
        p_legal_move_idx = torch.nonzero(p_legal_move_mat)
        p_legal_move_word = p_legal_move_idx[:, 0] + 64 * p_legal_move_idx[:, 1]

        for t_idx, t_legal in enumerate(t_legal_move_word):
            if t_legal in p_legal_move_word:
                p_finder = p_legal_move_word == t_legal
                p_idx = torch.nonzero(p_finder)
                if p_idx < pred_quality_vec.size()[1] - 1:
                    pred_q_lined[t_idx] = pred_quality_vec[batch_idx, p_idx]
                    match_find_batch[t_idx] = True

        pred_q_lined[pred_quality_vec.size()[1] - 1] = pred_quality_vec[batch_idx, pred_quality_vec.size()[1] - 1]
        lined_up_all_preds = torch.cat((lined_up_all_preds, pred_q_lined.unsqueeze(0)), 0)
        match_find = torch.cat((match_find, match_find_batch.unsqueeze(0)), 0)

    match_find[:, -1] = 1
    return lined_up_all_preds, match_find


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


def cross_entropy_gen(input, target, weights=None):
    """Implementing a general cross entropy function, accepting logits"""
    log_sum_vector = torch.logsumexp(input, 1)
    log_sum_vector_large = log_sum_vector.unsqueeze(0).repeat((input.size()[1], 1))
    target_normal = torch.sum(target, 1)
    target_normal_large = target_normal.unsqueeze(0).repeat((target.size()[1]), 1) + 1e-10
    targets_normalized = torch.div(target, target_normal_large.permute((1, 0)))
    if weights is None:
        return torch.mean(-torch.sum(targets_normalized * (input - log_sum_vector_large.permute((1, 0))), 1))
    else:
        return torch.mean(-torch.sum(weights * targets_normalized * (input - log_sum_vector_large.permute((1, 0))), 1))

