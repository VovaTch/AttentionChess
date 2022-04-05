import torch
from ffcv import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze
from ffcv.fields import NDArrayField, FloatField, IntField
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder
import numpy as np

from .datasets import RuleChessDataset, LichessDatabaseChessDataset
from utils.util import board_to_embedding_coord, move_to_coordinate


class RuleChessNumpy:
    """
    FFCV Dataset class wrapper for the lichess database dataset. TODO: Apply to all games if not too large of a file.
    """
    
    def __init__(self) -> None:
        self.dataset = RuleChessDataset('lichess_data/lichess_db_standard_rated_2016-09.pgn')
        
    def __getitem__(self, idx):
        sampled_board, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx = self.dataset.__getitem__(idx)
        sampled_board_coor = board_to_embedding_coord(sampled_board)
        
        # Output the legal moves, necessary for the net to work.
        legal_move_torch = torch.zeros((64, 76), requires_grad=False) - 1
        for legal_move in sampled_board.legal_moves:
            move_coor = move_to_coordinate(legal_move)
            legal_move_torch[move_coor[0], move_coor[1]] = 1
        
        return (np.array(sampled_board_coor).astype('uint8'), np.array(sampled_quality_batch).astype('float32'), 
                float(sampled_board_value_batch), int(sampled_move_idx), np.array(legal_move_torch).astype('int8'))
    
    def __len__(self):
        return self.dataset.__len__()
    

class LichessDatabaseNumpy:
    """
    FFCV Dataset class wrapper for the lichess database dataset. TODO: Apply to all games if not too large of a file.
    """
    
    def __init__(self, path='lichess_data/lichess_data_raw_2.csv') -> None:
        self.dataset = LichessDatabaseChessDataset(path)
        
    def __getitem__(self, idx):
        sampled_board, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx = self.dataset.__getitem__(idx)
        sampled_board_coor = board_to_embedding_coord(sampled_board)
        
        # Output the legal moves, necessary for the net to work.
        legal_move_torch = torch.zeros((64, 76), requires_grad=False) - 1
        for legal_move in sampled_board.legal_moves:
            move_coor = move_to_coordinate(legal_move)
            legal_move_torch[move_coor[0], move_coor[1]] = 1
        
        return (np.array(sampled_board_coor).astype('uint8'), np.array(sampled_quality_batch).astype('float32'), 
                float(sampled_board_value_batch), int(sampled_move_idx), np.array(legal_move_torch).astype('int8'))
    
    def __len__(self):
        return self.dataset.__len__()
    
    
class EndgameDatabaseNumpy:
    """
    FFCV Dataset class wrapper for the lichess database dataset. TODO: Apply to all games if not too large of a file.
    """
    
    def __init__(self, path='lichess_data/endgame_data_raw.csv') -> None:
        self.dataset = LichessDatabaseChessDataset(path)
        
    def __getitem__(self, idx):
        sampled_board, sampled_quality_batch, sampled_board_value_batch, sampled_move_idx = self.dataset.__getitem__(idx)
        sampled_board_coor = board_to_embedding_coord(sampled_board)
        
        # Output the legal moves, necessary for the net to work.
        legal_move_torch = torch.zeros((64, 76), requires_grad=False) - 1
        for legal_move in sampled_board.legal_moves:
            move_coor = move_to_coordinate(legal_move)
            legal_move_torch[move_coor[0], move_coor[1]] = 1
        
        return (np.array(sampled_board_coor).astype('uint8'), np.array(sampled_quality_batch).astype('float32'), 
                float(sampled_board_value_batch), int(sampled_move_idx), np.array(legal_move_torch).astype('int8'))
    
    def __len__(self):
        return self.dataset.__len__()
    
    
def create_dataset_ffcv(path='lichess_data/ffcv_rule_data.beton', dataset_type='rule'):

    dataset = RuleChessNumpy() if dataset_type == 'rule' else LichessDatabaseNumpy()

    if dataset_type == 'rule':
        dataset = RuleChessNumpy() 
    elif dataset_type == 'lichess':
        dataset = LichessDatabaseNumpy()
    elif dataset_type == 'ending':
        dataset = LichessDatabaseNumpy(path='lichess_data/endgame_data_raw_1.csv')

    writer = DatasetWriter(path, {
        'board': NDArrayField(shape=(8, 8), dtype=np.dtype("uint8")),
        'quality': NDArrayField(shape=(256,), dtype=np.dtype("float32")),
        'value': FloatField(),
        'index': IntField(),
        'legal_moves': NDArrayField(shape=(64, 76), dtype=np.dtype("int8"))
    }, num_workers=4)
    writer.from_indexed_dataset(dataset, chunksize=128)


def get_ffcv_loader(batch_size, num_workers, device, shuffle=True, path='lichess_data/ffcv_ending_data.beton', *args):
    order = OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL
    loader = Loader(path, batch_size=batch_size, num_workers=num_workers, order=order, pipelines={
        'board': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
        'quality': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
        'value': [FloatDecoder(), ToTensor(), ToDevice(device), Squeeze()],
        'index': [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()],
        'legal_moves': [NDArrayDecoder(), ToTensor(), ToDevice(device)]
    }, batches_ahead=2)
    return loader