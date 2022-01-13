import chess
import chess.pgn
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
from model.chess_conv_attention import attention

from utils.util import board_to_tensor, legal_move_mask, move_to_tensor, \
    board_to_embedding_coord, move_to_coordinate
from utils.matcher import match_moves
from model.attchess import AttChess, BoardEmbTrainNet, MoveEmbTrainNet
from model.score_functions import ScoreWinFast
from data_loaders.game_roll import GameRoller, InferenceBoardNode, InferenceMoveSearcher
from data_loaders.dataloader import RuleAttentionChessLoader, collate_fn, BoardEmbeddingLoader, MoveEmbeddingLoader, SelfPlayChessLoader
from model import loss

# fix random seeds for reproducibility
SEED = 69
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main():

    device = 'cpu'

    board = chess.Board()
    board.push_san('e4')
    board2 = chess.Board()
    board_embed = board_to_embedding_coord(board)
    loader = RuleAttentionChessLoader(batch_size=4, collate_fn=collate_fn, shuffle=False)
    attchess = AttChess(hidden_dim=32, num_heads=8, num_encoder=4, num_decoder=4, query_word_len=256, 
                        num_chess_conv_layers=0, p_embedding=True)
    attchess = attchess.eval().to(device)
    
    # Load trained model
    checkpoint = torch.load('model_best_init.pth', map_location=torch.device('cpu'))
    attchess.load_state_dict(checkpoint['state_dict'])
    
    # Run the net
    with torch.no_grad():
        legal_moves, quality_pred, value_pred = attchess([board])
        legal_move_list, quality_vec, value = attchess.post_process(legal_moves, quality_pred, value_pred)
    
    # Load move selector
    score_function = ScoreWinFast(100)
    init_node = InferenceBoardNode(board, None, score_function, quality_vec[0], value[0], device=device)
    move_searcher = InferenceMoveSearcher(engine=attchess)
    
    # Perform search
    move_selected = move_searcher(init_node, 2000)
    print(init_node)
    print(move_selected)
    

    

    print(1111)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')  # Necessary for this to work; maybe it will run out of memory like that
    main()
