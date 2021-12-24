import argparse

import torch

from model.attchess import BoardEmbTrainNet


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of board embedding script.')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device of the net.')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs.')
    arg = parser.parse_args()
    main(arg)
