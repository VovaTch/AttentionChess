import torch
import numpy as np

from scipy.optimize import linear_sum_assignment


# TODO: Use linear sum assignment from scipi
@torch.no_grad()
def match_moves(output_moves: torch.Tensor, target_moves: torch.Tensor):

    pair_idx_list = []

    for idx_board in range(output_moves.size()[0]):

        target_moves_legal = target_moves[idx_board, target_moves[idx_board, :, 3] == 0, :]

        # Create a large coordinate block
        target_coor = target_moves_legal[:, :2]
        target_block = target_coor.flatten().unsqueeze(0).repeat((output_moves.size()[1], 1))

        # Create a large output block
        output_coor = output_moves[idx_board, :, :2]
        output_block = output_coor.repeat((1, int(target_block.size()[1] / 2)))

        # Subtraction block
        sub_block = target_block - output_block

        norm_matrix = torch.zeros((output_coor.size()[0], 0)).to(output_moves.device)
        # compute norm
        for idx_d in range(0, sub_block.size()[1], 2):
            norm_column = torch.norm(sub_block[:, idx_d: idx_d + 2], dim=1)
            norm_matrix = torch.cat((norm_matrix, norm_column.unsqueeze(1)), 1)

        output_matches = linear_sum_assignment(norm_matrix.detach().cpu())
        output_matches = np.array(output_matches)
        output_matches_torch = torch.Tensor(output_matches).permute(1, 0).int().to(norm_matrix.device)

        pair_idx_list.append(output_matches_torch)

    return pair_idx_list



