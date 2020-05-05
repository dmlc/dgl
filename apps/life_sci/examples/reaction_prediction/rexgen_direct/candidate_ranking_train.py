import time
import torch

from dgllife.data import USPTORank, WLNRankDataset
from dgllife.model import WLNReactionRanking
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from configure import reaction_center_config, candidate_ranking_config
from utils import prepare_reaction_center, mkdir_p, set_seed, collate_rank_train, \
    collate_rank_eval, candidate_ranking_eval

def main(args, path_to_candidate_bonds):
    if args['train_path'] is None:
        train_set = USPTORank(subset='train', candidate_bond_path=path_to_candidate_bonds['train'],
                              num_processes=args['num_processes'])
    else:
        train_set = WLNRankDataset(path_to_save_results=args['result_path'] + '/train',
                                   raw_file_path=args['train_path'],
                                   candidate_bond_path=path_to_candidate_bonds['train'],
                                   num_processes=args['num_processes'])
    train_set.ignore_large()
    if args['val_path'] is None:
        val_set = USPTORank(subset='val', candidate_bond_path=path_to_candidate_bonds['val'],
                            num_processes=args['num_processes'])
    else:
        val_set = WLNRankDataset(path_to_save_results=args['result_path'] + '/val',
                                 raw_file_path=args['val_path'],
                                 candidate_bond_path=path_to_candidate_bonds['val'],
                                 train_mode=False,
                                 num_processes=args['num_processes'])

    train_loader = DataLoader(train_set, batch_size=1, collate_fn=collate_rank_train,
                              shuffle=True, num_workers=args['num_workers'])
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_rank_eval,
                            shuffle=False, num_workers=args['num_workers'])

    model = WLNReactionRanking(
        node_in_feats=args['node_in_feats'],
        edge_in_feats=args['edge_in_feats'],
        node_hidden_feats=args['hidden_size'],
        num_encode_gnn_layers=args['num_encode_gnn_layers']).to(args['device'])
    criterion = BCEWithLogitsLoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=args['lr'])
    from utils import Optimizer
    optimizer = Optimizer(model, args['lr'], optimizer, max_grad_norm=args['max_norm'],
                          num_accum_times=args['batch_size'])

    total_samples = 0
    acc_sum = 0
    grad_norm_sum = 0
    dur = []

    for epoch in range(args['num_epochs']):
        t0 = time.time()
        model.train()
        for batch_id, batch_data in enumerate(train_loader):
            reactant_graph, product_graphs, combo_scores, labels = batch_data

            # No valid candidate products have been predicted
            if reactant_graph is None:
                continue

            combo_scores, labels = combo_scores.to(args['device']), labels.to(args['device'])
            reactant_node_feats = reactant_graph.ndata.pop('hv').to(args['device'])
            reactant_edge_feats = reactant_graph.edata.pop('he').to(args['device'])
            product_node_feats = product_graphs.ndata.pop('hv').to(args['device'])
            product_edge_feats = product_graphs.edata.pop('he').to(args['device'])

            pred = model(reactant_graph=reactant_graph,
                         reactant_node_feats=reactant_node_feats,
                         reactant_edge_feats=reactant_edge_feats,
                         product_graphs=product_graphs,
                         product_node_feats=product_node_feats,
                         product_edge_feats=product_edge_feats,
                         candidate_scores=combo_scores)
            # Check if the ground truth candidate has the highest score
            acc_sum += float(pred.max(dim=0)[1].detach().cpu().data.item() == 0)
            loss = criterion(pred, labels)
            total_samples += 1
            grad_norm_sum += optimizer.backward_and_step(loss)

            if total_samples % args['print_every'] == 0:
                progress = 'Epoch {:d}/{:d}, iter {:d}/{:d} | time {:.4f} |' \
                           'accuracy {:.4f} | grad norm {:.4f}'.format(
                    epoch + 1, args['num_epochs'], (batch_id + 1) // args['print_every'],
                    len(train_loader) // args['print_every'],
                    (sum(dur) + time.time() - t0) / total_samples * args['print_every'],
                    acc_sum / args['print_every'],
                    grad_norm_sum / args['print_every'])
                print(progress)
                acc_sum = 0
                grad_norm_sum = 0

            if total_samples % args['decay_every'] == 0:
                old_lr = optimizer.lr
                optimizer.decay_lr(args['lr_decay_factor'])
                new_lr = optimizer.lr
                print('Learning rate decayed from {:.4f} to {:.4f}'.format(old_lr, new_lr))
                torch.save({'model_state_dict': model.state_dict()},
                           args['result_path'] + '/model.pkl')

        optimizer._reset()
        dur.append(time.time() - t0)
        prediction_summary = candidate_ranking_eval(args, model, val_loader)
        prediction_summary = 'Epoch {:d}/{:d}\n'.format(epoch + 1, args['num_epochs']) + \
                             prediction_summary
        print(prediction_summary)
        with open(args['result_path'] + '/val_eval.txt', 'a') as f:
            f.write(prediction_summary)

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
    parser.add_argument('-np', '--num-processes', type=int, default=8,
                        help='Number of processes to use for data pre-processing')
    parser.add_argument('-nw', '--num-workers', type=int, default=100,
                        help='Number of workers to use for data loading in PyTorch data loader')
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
