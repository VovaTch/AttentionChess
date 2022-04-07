# The package containing all the datasets classes I use

from .board_embedding import BoardEmbeddingDataset
from .full_self_play import FullSelfPlayDataset
from .guided_self_play import GuidedSelfPlayDataset
from .lichess_database import LichessDatabaseChessDataset
from .move_embedding import MoveEmbeddingDataset
from .puzzle_chess import PuzzleChessDataset
from .rule_chess import RuleChessDataset
from .random_self_play import RandomSelfPlayDataset
from .ending_chess import EndingChessDataset