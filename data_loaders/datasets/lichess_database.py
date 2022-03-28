import csv
import collections
import itertools

import chess
import torch
from torch.utils.data import Dataset

class LichessDatabaseChessDataset(Dataset):

    def __init__(self, dataset_path, query_word_len=256):
        super(LichessDatabaseChessDataset, self).__init__()
        self.game = None

        self.query_word_len = query_word_len
        self.follow_idx = 0
        self.game_length = 0
        self.board_collection = None
        self.move_quality_batch = None
        self.board_value_batch = None
        self.selected_move_idx = None
        
        # Load the csv file
        self.csv_file = open(dataset_path, 'r')
        self.csv_reader = csv.reader(self.csv_file)
        self.length_of_dataset = 0
        for _ in open(dataset_path):
            self.length_of_dataset += 1
        self.headers = next(self.csv_reader)

    def __getitem__(self, _):
        
        sampled_row_raw = next(self.csv_reader)
        sampled_row = {header: data for (header, data) in zip(self.headers, sampled_row_raw)}
        
        sampled_board = chess.Board(sampled_row['Game fen'])
        sampled_board_value_batch = float(sampled_row['Board value'])
        sampled_quality_batch = torch.tensor([float(sampled_row[idx]) for idx in list(sampled_row.keys())[2:]])
        
        # Fix if the size doesn't fit
        if sampled_quality_batch.size()[0] < self.query_word_len:
            zeros_tensor = torch.zeros(self.query_word_len)
            zeros_tensor[:sampled_quality_batch.size()[0]] = sampled_quality_batch
            sampled_quality_batch = zeros_tensor
        
        sampled_move_idx = -1
        
        if torch.sum(sampled_quality_batch) == 0:
            legal_move_counter = [1 for _ in sampled_board.legal_moves]
            legal_move_counter = sum(legal_move_counter)
            sampled_quality_batch[:legal_move_counter] += 1e-10
            
        sampled_quality_batch /= torch.sum(sampled_quality_batch)

        return sampled_board, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx

    def __len__(self):
        return int(self.length_of_dataset)
    
    def __del__(self):
        self.csv_file.close()
        
    def reset_reader(self):
        self.csv_file.seek(0)
        self.csv_reader = csv.reader(self.csv_file)
        next(self.csv_reader)
        
    @staticmethod
    def consume(iterator, n):
        "Advance the iterator n-steps ahead. If n is none, consume entirely."
        # Use functions that consume iterators at C speed.
        if n is None or n == -1:
            # feed the entire iterator into a zero-length deque
            collections.deque(iterator, maxlen=0)
        else:
            # advance to the empty slice starting at position n
            next(itertools.islice(iterator, n, n), None)