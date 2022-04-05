import argparse

from data_loaders.ffcv_dataloader import create_dataset_ffcv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['rule', 'lichess', 'ending'], default='rule',
                        help='Type of FFCV dataset to generate')
    args = parser.parse_args()
    create_dataset_ffcv(dataset_type=args.dataset_type, path='lichess_data/ffcv_ending_data.beton')