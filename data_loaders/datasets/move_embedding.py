import torch
from torch.utils.data import Dataset

class MoveEmbeddingDataset(Dataset):
    """
    Dataset for moves, 4864 moves, you get word in and get out move coordinates and possible promotion
    classification vector.
    """
    def __init__(self):
        super(MoveEmbeddingDataset, self).__init__()

    def __getitem__(self, word):

        # Initialize promotion probability vector: 0: no promotion, 1: queen, 2: rook, 3: bishop, 4: knight
        promotion_prob = torch.zeros(5)

        # Decompose to individual move coordinates
        coordinates_from_to = (int(word % 64), word // 64)
        coordinates_from = (int(coordinates_from_to[0] % 8), coordinates_from_to[0] // 8)  # 0 is a,b,c... 1 is numbers

        # If not promoting
        if coordinates_from_to[1] < 64:
            coordinates_to = (int(coordinates_from_to[1] % 8), coordinates_from_to[1] // 8)
            promotion_prob[0] = 1

        # If promoting
        else:
            if 64 <= coordinates_from_to[1] < 67:
                coor_shift = 65 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[1] = 1
            elif 67 <= coordinates_from_to[1] < 70:
                coor_shift = 68 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[2] = 1
            elif 70 <= coordinates_from_to[1] < 73:
                coor_shift = 71 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[3] = 1
            else:
                coor_shift = 74 - coordinates_from_to[1]
                coor_up_down = 0 if coordinates_from[1] == 1 else 7
                promotion_prob[4] = 1

            coordinates_to = (int(coordinates_from[0] - coor_shift), int(coor_up_down))

        coor_tensor = torch.tensor([coordinates_from[0], coordinates_from[1], coordinates_to[0], coordinates_to[1]]) / 7

        return word, coor_tensor, promotion_prob

    def __len__(self):
        return 4864