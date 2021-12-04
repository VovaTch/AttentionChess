import argparse
import collections
import torch
import chess

from model.attchess import AttChess
from parse_config import ConfigParser
from utils.util import board_to_tensor_full


def main(config):

    logger = config.get_logger('Chess game')

    # Load network
    engine = AttChess()
    checkpoint = torch.load(config.resume)
    engine.load_state_dict(checkpoint['model'])
    engine = engine.to(config.device)
    logger.info('Engine loaded')

    # Load chess board, choose side
    board = chess.Board()
    while True:
        side = input('Please choose side (white or black): ')
        if side == 'white' or side == 'black':
            break

    if side == 'white':
        player_turn = True
    else:
        player_turn = False

    # Main playing loop
    while True:
        logger.info(board)

        # Player move; doesn't matter if black or white
        if player_turn:
            while True:

                move_candidate = input(f'Please play a move as {side}:')
                if move_candidate in board.legal_moves:
                    break

            board.push_san(move_candidate)

        # Computer move
        else:
            # Run the network and get a move sample
            board_tensor = board_to_tensor_full(board).to(config.device)
            outputs_raw = engine(board_tensor.unsqueeze(0))
            outputs_raw = outputs_raw.squeeze(0)
            outputs = outputs_raw[:, outputs_raw[:, 3] >= 0]
            outputs_probs = outputs[:, 4]
            sm = torch.nn.Softmax(dim=1)
            outputs_probs = sm(outputs_probs)
            cat = torch.distributions.Categorical(outputs_probs)
            sample = cat.sample()





if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config_s1.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='cuda', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
