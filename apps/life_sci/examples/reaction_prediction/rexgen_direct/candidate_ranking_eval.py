import torch

from dgllife.data import USPTORank, WLNRankDataset
from dgllife.model import WLNReactionRanking
from torch.utils.data import DataLoader

from configure import candidate_ranking_config, reaction_center_config
from utils import mkdir_p, prepare_reaction_center, collate_rank_eval, candidate_ranking_eval

def main(args, path_to_candidate_bonds):
    if args['test_path'] is None:
        test_set = USPTORank(subset='test', candidate_bond_path=path_to_candidate_bonds['test'],
                             num_processes=args['num_processes'])
    else:
        test_set = WLNRankDataset(path_to_save_results=args['result_path'] + '/test',
                                  raw_file_path=args['test_path'],
                                  candidate_bond_path=path_to_candidate_bonds['test'],
                                  train_mode=False,
                                  num_processes=args['num_processes'])

    test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_rank_eval,
                             shuffle=False, num_workers=args['num_workers'])

    # Todo: load from a pre-trained model
    model = WLNReactionRanking(
        node_in_feats=args['node_in_feats'],
        edge_in_feats=args['edge_in_feats'],
        node_hidden_feats=args['hidden_size'],
        num_encode_gnn_layers=args['num_encode_gnn_layers']).to(args['device'])

    prediction_summary = candidate_ranking_eval(args, model, test_loader)
    with open(args['result_path'], 'w') as f:
        f.write(prediction_summary)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Candidate Ranking')
    parser.add_argument('--result-path', type=str, default='candidate_results',
                        help='Path to save modeling results')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Path to a new test set. '
                             'If None, we will use the default test set in USPTO.')
    parser.add_argument('-cmp', '--center-model-path', type=str, default=None,
                        help='Path to a pre-trained model for reaction center prediction. '
                             'By default we use the official pre-trained model. If not None, '
                             'the model should follow the hyperparameters specified in '
                             'reaction_center_config.')
    parser.add_argument('-rcb', '--reaction-center-batch-size', type=int, default=200,
                        help='Batch size to use for preparing candidate bonds from a trained '
                             'model on reaction center prediction')
    parser.add_argument('-np', '--num-processes', type=int, default=8,
                        help='Number of processes to use for data pre-processing')
    parser.add_argument('-nw', '--num-workers', type=int, default=32,
                        help='Number of workers to use for data loading in PyTorch data loader')
    args = parser.parse_args().__dict__
    args.update(candidate_ranking_config)
    mkdir_p(args['result_path'])
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    path_to_candidate_bonds = prepare_reaction_center(args, reaction_center_config)
    main(args, path_to_candidate_bonds)
