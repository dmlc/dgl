from dgllife.data import USPTORank, WLNRankDataset

from configure import reaction_center_config, candidate_ranking_config
from utils import setup, prepare_reaction_center

def main(args, path_to_candidate_bonds):
    if args['train_path'] is None:
        train_set = USPTORank(subset='train', candidate_bond_path=path_to_candidate_bonds['train'])
    else:
        train_set = WLNRankDataset(raw_file_path=args['train_path'],
                                   candidate_bond_path=path_to_candidate_bonds['train'],
                                   mol_graph_path='train_rank_graphs.bin')
    train_set.ignore_large()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Candidate Ranking')
    parser.add_argument('--result-path', type=str, default='candidate_results',
                        help='Path to save modeling results')
    parser.add_argument('--train-path', type=str, default=None,
                        help='Path to a new training set. '
                             'If None, we will use the default training set in USPTO.')
    parser.add_argument('--val-path', type=str, default=None,
                        help='Path to a new validation set. '
                             'If None, we will use the default validation set in USPTO.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Path to a new test set.'
                             'If None, we will use the default test set in USPTO.')
    parser.add_argument('-cmp', '--center-model-path', type=str, default=None,
                        help='Path to a pre-trained model for reaction center prediction. '
                             'By default we use the official pre-trained model. If not None, '
                             'the model should follow the hyperparameters specified in '
                             'reaction_center_config.')
    parser.add_argument('-np', '--num-processes', type=int, default=64,
                        help='Number of processes to use for data pre-processing')
    args = parser.parse_args().__dict__
    args.update(candidate_ranking_config)
    args = setup(args)

    path_to_candidate_bonds = prepare_reaction_center(args, reaction_center_config)
    main(args, path_to_candidate_bonds)
