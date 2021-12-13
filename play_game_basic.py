import argparse
import collections
import torch
import chess

import model.attchess as module_arch
from model.attchess import AttChess
from parse_config import ConfigParser
from utils.util import board_to_embedding_coord, torch_to_move
from utils.util import prepare_device


@torch.no_grad()
def main(config):
    logger = config.get_logger('Chess game')
    device, _ = prepare_device(config['n_gpu'])

    # Load network
    engine = config.init_obj('arch', module_arch)
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    engine.load_state_dict(checkpoint['state_dict'])
    engine = engine.to(device).eval()
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

                move_candidate_str = input(f'Please play a move as {side}:')
                try:
                    move_candidate_uci = board.parse_san(move_candidate_str).uci()
                except:
                    continue
                move_candidate = chess.Move.from_uci(move_candidate_uci)
                if move_candidate in board.legal_moves:
                    break

            board.push_san(move_candidate_str)
            player_turn = not player_turn

        # Computer move
        else:
            # Run the network and get a move sample
            board_tensor = board_to_embedding_coord(board).to(device)
            outputs_raw = engine(board_tensor.unsqueeze(0))
            outputs_raw = outputs_raw.squeeze(0)
            outputs = outputs_raw[outputs_raw[:, 3] >= 0]
            outputs_probs = outputs[:, 5]
            outputs_probs = outputs_probs.softmax(dim=0)
            cat = torch.distributions.Categorical(outputs_probs)

            # Force legal move
            while True:

                sample_idx = cat.sample()
                sample = outputs[sample_idx, :]

                # perform the move
                try:
                    sample_move = torch_to_move(sample)
                except:
                    continue
                print(sample_move)
                if sample_move in board.legal_moves:
                    board.push(sample_move)
                    break
            player_turn = not player_turn


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config_s1.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='model_best.pth', type=str,
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
