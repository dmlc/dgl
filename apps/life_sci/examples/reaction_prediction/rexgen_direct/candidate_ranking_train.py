import torch

from dgllife.data import USPTORank, WLNRankDataset
from dgllife.model import WLNReactionRanking

from configure import reaction_center_config, candidate_ranking_config
from utils import prepare_reaction_center, mkdir_p, set_seed

def main(args, path_to_candidate_bonds):
    if args['train_path'] is None:
        train_set = USPTORank(subset='train', candidate_bond_path=path_to_candidate_bonds['train'],
                              num_processes=args['num_processes'])
    else:
        train_set = WLNRankDataset(path_to_save_results=args['result_path'] + '/train',
                                   raw_file_path=args['train_path'],
                                   candidate_bond_path=path_to_candidate_bonds['train'],
                                   mol_graph_path='train_rank_graphs.bin',
                                   num_processes=args['num_processes'])
    train_set.ignore_large()
    if args['val_path'] is None:
        val_set = USPTORank(subset='val', candidate_bond_path=path_to_candidate_bonds['val'],
                            num_processes=args['num_processes'])
    else:
        val_set = WLNRankDataset(path_to_save_results=args['result_path'] + '/val',
                                 raw_file_path=args['val_path'],
                                 candidate_bond_path=path_to_candidate_bonds['val'],
                                 mol_graph_path='val_rank_graphs.bin',
                                 train_mode=False,
                                 num_processes=args['num_processes'])

    model = WLNReactionRanking(
        node_in_feats=args['node_in_feats'],
        edge_in_feats=args['edge_in_feats'],
        node_hidden_feats=args['hidden_size'],
        num_encode_gnn_layers=args['num_encode_gnn_layers']).to(args['device'])

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
    parser.add_argument('-cmp', '--center-model-path', type=str, default=None,
                        help='Path to a pre-trained model for reaction center prediction. '
                             'By default we use the official pre-trained model. If not None, '
                             'the model should follow the hyperparameters specified in '
                             'reaction_center_config.')
    parser.add_argument('-rcb', '--reaction-center-batch-size', type=int, default=200,
                        help='Batch size to use for preparing candidate bonds from a trained '
                             'model on reaction center prediction')
    parser.add_argument('-np', '--num-processes', type=int, default=32,
                        help='Number of processes to use for data pre-processing')
    args = parser.parse_args().__dict__
    args.update(candidate_ranking_config)
    mkdir_p(args['result_path'])
    set_seed()
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    path_to_candidate_bonds = prepare_reaction_center(args, reaction_center_config)
    main(args, path_to_candidate_bonds)
