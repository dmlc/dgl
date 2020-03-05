import time

from utils import setup, load_data

def main(args):
    setup(args)
    train_set, val_set, test_set = load_data(args['num_processes'])

if __name__ == '__main__':
    from argparse import ArgumentParser

    from configure import reaction_center_config

    parser = ArgumentParser(description='Reaction Center Identification')
    parser.add_argument('-r', '--result-path', type=str, default='results',
                        help='Path to training results')
    parser.add_argument('-np', '--num-processes', type=int,
                        help='Number of processes for preprocessing the dataset')
    args = parser.parse_args().__dict__
    args.update(reaction_center_config)

    main(args)
