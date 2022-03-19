import torch
from torch.utils.data import Dataset

class BoardEmbeddingDataset(Dataset):
    """
    Dataloader for board square embedding training, 36 possible words.
    """
    def __init__(self):
        super(BoardEmbeddingDataset, self).__init__()

    def __getitem__(self, idx):
        """
        From the index, outputs 2 vectors, a probability vector of pieces, and a probability vector of properties.
        """

        # Piece probabilities
        target_prob_vector = torch.zeros(7)
        white_prob = 0  # 1 for white, 0 for black
        ep_flag = 0  # En passant 1 if available
        castling_right = 1  # 1 for there is right, 0 for there isn't
        turn = 1  # 1 for white, 0 for black

        if idx in [7, 25]:
            target_prob_vector[1] = 1  # white pawn
            white_prob = 1
        elif idx in [8, 26]:
            target_prob_vector[2] = 1  # white knight
            white_prob = 1
        elif idx in [9, 27]:
            target_prob_vector[3] = 1  # white bishop
            white_prob = 1
        elif idx in [10, 28]:
            target_prob_vector[4] = 1  # white rook
            white_prob = 1
        elif idx in [11, 29]:
            target_prob_vector[5] = 1  # white queen
            white_prob = 1
        elif idx in [12, 30]:
            target_prob_vector[6] = 1  # white king
            white_prob = 1
            castling_right = 0
        elif idx in [14, 32]:
            target_prob_vector[6] = 1  # white king
            white_prob = 1
        elif idx in [5, 23]:
            target_prob_vector[1] = 1  # black pawn
        elif idx in [4, 22]:
            target_prob_vector[2] = 1  # black knight
        elif idx in [3, 21]:
            target_prob_vector[3] = 1  # black bishop
        elif idx in [2, 20]:
            target_prob_vector[4] = 1  # black rook
        elif idx in [1, 19]:
            target_prob_vector[5] = 1  # black queen
        elif idx in [0, 18]:
            target_prob_vector[6] = 1  # black king
            castling_right = 0
        elif idx in [15, 33]:
            target_prob_vector[6] = 1  # black king
        elif idx in [16, 17, 34, 35]:
            target_prob_vector[0] = 1
            ep_flag = 1
        else:
            target_prob_vector[0] = 1

        if idx >= 18:
            turn = 0

        flags_vector = torch.tensor([white_prob, ep_flag, castling_right, turn])
        return idx, target_prob_vector, flags_vector

    def __len__(self):
        return 36
