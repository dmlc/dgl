import torch

from dgllife.data import USPTOCenter, WLNCenterDataset
from dgllife.model import WLNReactionCenter, load_pretrained
from torch.utils.data import DataLoader

from utils import reaction_center_final_eval, set_seed, collate

def main(args):
    set_seed()
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')
    # Set current device
    torch.cuda.set_device(args['device'])

    if args['test_path'] is None:
        test_set = USPTOCenter('test', num_processes=args['num_processes'])
    else:
        test_set = WLNCenterDataset(raw_file_path=args['test_path'],
                                    mol_graph_path='test.bin',
                                    num_processes=args['num_processes'])
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             collate_fn=collate, shuffle=False)

    if args['pre_trained']:
        model = load_pretrained('wln_center_uspto')
    else:
        model = WLNReactionCenter(node_in_feats=args['node_in_feats'],
                                  edge_in_feats=args['edge_in_feats'],
                                  node_pair_in_feats=args['node_pair_in_feats'],
                                  node_out_feats=args['node_out_feats'],
                                  n_layers=args['n_layers'],
                                  n_tasks=args['n_tasks'])
        model.load_state_dict(torch.load(args['result_path'] + '/model.pkl',
                                         map_location='cpu')['model_state_dict'])
    model = model.to(args['device'])

    print('Evaluation on the test set.')
    test_result = reaction_center_final_eval(args, model, test_loader, args['easy'])
    print(test_result)
    with open(args['result_path'] + '/results.txt', 'w') as f:
        f.write(test_result)

if __name__ == '__main__':
    from argparse import ArgumentParser

    from configure import reaction_center_config

    parser = ArgumentParser(description='Reaction Center Identification -- Evaluation')
    parser.add_argument('--result-path', type=str, default='center_results',
                        help='Path where we saved model training results')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Path to a new test set.'
                             'If None, we will use the default test set in USPTO.')
    parser.add_argument('--easy', action='store_true', default=False,
                        help='Whether to exclude reactants not contributing heavy atoms to the '
                             'product in top-k atom pair selection, which will make the '
                             'task easier.')
    parser.add_argument('-p', '--pre-trained', action='store_true', default=False,
                        help='If true, we will directly evaluate a '
                             'pretrained model on the test set.')
    parser.add_argument('-np', '--num-processes', type=int, default=32,
                        help='Number of processes to use for data pre-processing')
    args = parser.parse_args().__dict__
    args.update(reaction_center_config)

    assert args['max_k'] >= max(args['top_ks']), \
        'Expect max_k to be no smaller than the possible options ' \
        'of top_ks, got {:d} and {:d}'.format(args['max_k'], max(args['top_ks']))
    main(args)
