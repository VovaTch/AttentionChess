import torch
from base import BaseDataLoader

from .mcts import MCTS
from .datasets import BoardEmbeddingDataset, MoveEmbeddingDataset, RuleChessDataset,\
    GuidedSelfPlayDataset, FullSelfPlayDataset, LichessDatabaseChessDataset, RandomSelfPlayDataset, EndingChessDataset, \
        PreEndingChessDataset

class BoardEmbeddingLoader(BaseDataLoader):
    """
    Dataloader for board square embedding training, 36 possible words.
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset = BoardEmbeddingDataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MoveEmbeddingLoader(BaseDataLoader):
    """
    Dataloader for move embedding training, 4864 possible words. 
    """
    def __init__(self, batch_size, shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset = MoveEmbeddingDataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class RuleAttentionChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, collate_fn, data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn',
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256, base_multiplier=1.0):

        self.dataset_path = data_dir
        self.dataset = RuleChessDataset(data_dir, query_word_len=query_word_len, base_multiplier=base_multiplier)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)

    
class PuzzleAttentionChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, collate_fn, data_dir='lichess_data/lichess_db_puzzle.csv',
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset_path = data_dir
        self.dataset = RuleChessDataset(data_dir, query_word_len=query_word_len)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)


class GuidedSelfPlayLoader(BaseDataLoader):
    """
    Data loader for self playing games with moves from database.
    """
    def __init__(self, batch_size, collate_fn,
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256, 
                 num_of_sims=100, epochs_per_game=1, min_counts=10, data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn',
                 device='cpu'):

        self.dataset = GuidedSelfPlayDataset(query_word_len=query_word_len, num_of_sims=num_of_sims, data_dir=data_dir,
                                             epochs_per_game=epochs_per_game, min_counts=min_counts, simultaneous_mcts=batch_size)
        self.device = device
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
    def set_mcts(self, mcts: MCTS):
        
        self.dataset.mcts = mcts
        

class FullSelfPlayLoader(BaseDataLoader):
    """
    Data loader for self playing games with moves from database.
    """
    def __init__(self, batch_size, collate_fn,
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256, 
                 num_of_sims=100, epochs_per_game=1, min_counts=10, device='cpu', win_multiplier=1):

        self.dataset = FullSelfPlayDataset(query_word_len=query_word_len, num_of_sims=num_of_sims, 
                                             epochs_per_game=epochs_per_game, min_counts=min_counts, simultaneous_mcts=batch_size,
                                             win_multipler=win_multiplier)
        self.device = device
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
    def set_mcts_game(self, mcts: MCTS):
        
        self.dataset.mcts_game = mcts
        
    def set_mcts_learn(self, mcts: MCTS):
        
        self.dataset.mcts = mcts
        
    def set_mcts(self, mcts: MCTS):
        
        self.set_mcts_game(mcts)
        self.set_mcts_learn(mcts)
        
    def set_white_engine(self, engine):
        
        self.dataset.white_engine = engine
        
    def set_black_engine(self, engine):
        
        self.dataset.black_engine = engine
        
        
class PreEndingChessLoader(BaseDataLoader):
    """
    Data loader for self playing games with moves from database.
    """
    def __init__(self, batch_size, collate_fn, data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn',
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256, 
                 num_of_sims=100, epochs_per_game=1, min_counts=10, device='cpu', win_multiplier=1, boards_to_end=6, *args, **kwargs):

        self.dataset = PreEndingChessDataset(dataset_path=data_dir, query_word_len=query_word_len, num_of_sims=num_of_sims, 
                                             epochs_per_game=epochs_per_game, min_counts=min_counts, simultaneous_mcts=batch_size, 
                                             boards_to_end=boards_to_end)
        self.device = device
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
    def set_mcts_game(self, mcts: MCTS):
        
        self.dataset.mcts_game = mcts
        
    def set_mcts_learn(self, mcts: MCTS):
        
        self.dataset.mcts = mcts
        
    def set_mcts(self, mcts: MCTS):
        
        self.set_mcts_game(mcts)
        self.set_mcts_learn(mcts)


class RandomSelfPlayLoader(BaseDataLoader):
    """
    Data loader to expand tree from random chess positions.
    """
    def __init__(self, batch_size, collate_fn,
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256, 
                 num_of_sims=100, epochs_per_game=1, min_counts=10, device='cpu', boards_per_sample=128):

        self.dataset = RandomSelfPlayDataset(query_word_len=query_word_len, num_of_sims=num_of_sims, 
                                             epochs_per_game=epochs_per_game, min_counts=min_counts, simultaneous_mcts=batch_size,
                                             boards_per_sample=boards_per_sample)
        self.device = device
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)
        
    def set_mcts(self, mcts: MCTS):
        
        self.dataset.mcts = mcts


class LichessDatabaseChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, collate_fn, data_dir='lichess_data/lichess_data_raw.csv',
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256):

        self.dataset_path = data_dir
        self.dataset = LichessDatabaseChessDataset(data_dir, query_word_len=query_word_len)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)


class EndingChessLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, collate_fn, data_dir='lichess_data/lichess_db_standard_rated_2016-09.pgn',
                 shuffle=True, validation_split=0.1, num_workers=1, training=True, query_word_len=256, base_multiplier=0.95):

        self.dataset_path = data_dir
        self.dataset = EndingChessDataset(data_dir, query_word_len=query_word_len, base_multiplier=base_multiplier)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=collate_fn)


def collate_fn(batch):
    """
    Required collate function because the boards are a unique class
    """
    chess_boards = [batch[idx][0] for idx in range(len(batch))]
    quality_vectors = torch.zeros((len(batch), batch[0][1].size()[0]))
    board_values = torch.zeros(len(batch))
    move_idx = torch.zeros(len(batch))
    for idx in range(len(batch)):
        quality_vectors[idx, :] = batch[idx][1]
        board_values[idx] = batch[idx][2]
        move_idx[idx] = batch[idx][3]
    return chess_boards, quality_vectors, board_values, move_idx