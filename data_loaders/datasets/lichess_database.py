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
        next(self.csv_reader)
             
    def __getitem__(self, _):
        
        sampled_row_raw = next(self.csv_reader)
        sampled_row = {header: data for (header, data) in zip(self.headers, sampled_row_raw)}
        sampled_board = sampled_row['Game board']
        sampled_board_legal_moves = sampled_row['Legal move matrix']
        sampled_board, sampled_board_legal_moves = self._string2list_int(sampled_board), self._string2list_int(sampled_board_legal_moves)
        sampled_board_value_batch = float(sampled_row['Board value'])
        sampled_quality_batch = torch.tensor([float(sampled_row[idx]) for idx in list(sampled_row.keys())[3:]])
        
        # Fix if the size doesn't fit
        if sampled_quality_batch.size()[0] < self.query_word_len:
            zeros_tensor = torch.zeros(self.query_word_len)
            zeros_tensor[:sampled_quality_batch.size()[0]] = sampled_quality_batch
            sampled_quality_batch = zeros_tensor
        
        sampled_move_idx = -1
        
        if torch.sum(sampled_quality_batch) == 0:
            legal_move_counter = torch.sum(torch.tensor(sampled_board_legal_moves)) + 64 * 76
            legal_move_counter = legal_move_counter.item()
            sampled_quality_batch[:legal_move_counter] += 1e-10
            
        sampled_quality_batch /= torch.sum(sampled_quality_batch)

        return sampled_board, sampled_board_legal_moves, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx

    def __len__(self):
        return int(self.length_of_dataset) # A workaround to hopefully not get the training freeze
    
    def __del__(self):
        self.csv_file.close()
        
    def reset_reader(self):
        self.csv_file.seek(0)
        self.csv_reader = csv.reader(self.csv_file)
        next(self.csv_reader)
    
    @staticmethod
    def _string2list_int(string: str):
        
        list_1 = string.split('], [')
        list_1[0] = list_1[0][2:]
        list_1[-1] = list_1[-1][:-2]
        list_2 = list()
        for list_ind in list_1:
            list_2.append(list_ind.split(', '))
            list_2[-1] = list(map(int, list_2[-1]))
        return list_2
        
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