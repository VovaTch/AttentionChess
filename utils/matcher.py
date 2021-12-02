import torch
import numpy as np


def match_moves(output_moves: torch.Tensor, target_moves: torch.Tensor):

    pair_idx_list = []

    for idx_board in range(output_moves.size()[0]):

        target_moves_legal = target_moves[idx_board, target_moves[idx_board, :, 3] == 10, :]
        board_pair_idx_list = [idx_target for idx_target in torch.where(target_moves[idx_board, :, 3] == 10)]

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

        output_matches = []
        # Compute closest matches to targets
        for idx_c in range(norm_matrix.size()[1]):
            indices_match_flat = torch.argmin(norm_matrix)
            indices_match = [int(torch.floor(indices_match_flat / norm_matrix.size()[1])),
                             int(indices_match_flat % norm_matrix.size()[1])]
            norm_matrix[indices_match[0], :] = torch.inf
            norm_matrix[:, indices_match[1]] = torch.inf
            output_matches.append([int(board_pair_idx_list[0][indices_match[1]]), indices_match[0]])

        output_matches_torch = torch.tensor(output_matches)

        pair_idx_list.append(output_matches_torch)

    return pair_idx_list



