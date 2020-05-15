import time
import torch

from dgllife.data import USPTORank, WLNRankDataset
from dgllife.model import WLNReactionRanking
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from configure import reaction_center_config, candidate_ranking_config
from utils import prepare_reaction_center, mkdir_p, set_seed, collate_rank_train, \
    collate_rank_eval, candidate_ranking_eval

def main(args, path_to_candidate_bonds):
    if args['train_path'] is None:
        train_set = USPTORank(
            subset='train', candidate_bond_path=path_to_candidate_bonds['train'],
            max_num_change_combos_per_reaction=args['max_num_change_combos_per_reaction_train'],
            num_processes=args['num_processes'])
    else:
        train_set = WLNRankDataset(
            raw_file_path=args['train_path'],
            candidate_bond_path=path_to_candidate_bonds['train'], mode='train',
            max_num_change_combos_per_reaction=args['max_num_change_combos_per_reaction_train'],
            num_processes=args['num_processes'])
    train_set.ignore_large()
    if args['val_path'] is None:
        val_set = USPTORank(
            subset='val', candidate_bond_path=path_to_candidate_bonds['val'],
            max_num_change_combos_per_reaction=args['max_num_change_combos_per_reaction_eval'],
            num_processes=args['num_processes'])
    else:
        val_set = WLNRankDataset(
            raw_file_path=args['val_path'],
            candidate_bond_path=path_to_candidate_bonds['val'], mode='val',
            max_num_change_combos_per_reaction=args['max_num_change_combos_per_reaction_eval'],
            num_processes=args['num_processes'])

    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate_rank_train,
                              shuffle=True, num_workers=args['num_workers'])
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            collate_fn=collate_rank_eval,
                            shuffle=False, num_workers=args['num_workers'])

    model = WLNReactionRanking(
        node_in_feats=args['node_in_feats'],
        edge_in_feats=args['edge_in_feats'],
        node_hidden_feats=args['hidden_size'],
        num_encode_gnn_layers=args['num_encode_gnn_layers']).to(args['device'])
    criterion = CrossEntropyLoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=args['lr'])
    from utils import Optimizer
    optimizer = Optimizer(model, args['lr'], optimizer, max_grad_norm=args['max_norm'])

    acc_sum = 0
    grad_norm_sum = 0
    dur = []
    total_samples = 0
    for epoch in range(args['num_epochs']):
        t0 = time.time()
        model.train()
        for batch_id, batch_data in enumerate(train_loader):
            batch_reactant_graphs, batch_product_graphs, \
            batch_combo_scores, batch_labels, batch_num_candidate_products = batch_data

            batch_combo_scores = batch_combo_scores.to(args['device'])
            batch_labels = batch_labels.to(args['device'])
            reactant_node_feats = batch_reactant_graphs.ndata.pop('hv').to(args['device'])
            reactant_edge_feats = batch_reactant_graphs.edata.pop('he').to(args['device'])
            product_node_feats = batch_product_graphs.ndata.pop('hv').to(args['device'])
            product_edge_feats = batch_product_graphs.edata.pop('he').to(args['device'])

            pred = model(reactant_graph=batch_reactant_graphs,
                         reactant_node_feats=reactant_node_feats,
                         reactant_edge_feats=reactant_edge_feats,
                         product_graphs=batch_product_graphs,
                         product_node_feats=product_node_feats,
                         product_edge_feats=product_edge_feats,
                         candidate_scores=batch_combo_scores,
                         batch_num_candidate_products=batch_num_candidate_products)

            # Check if the ground truth candidate has the highest score
            batch_loss = 0
            product_graph_start = 0
            for i in range(len(batch_num_candidate_products)):
                product_graph_end = product_graph_start + batch_num_candidate_products[i]
                reaction_pred = pred[product_graph_start:product_graph_end, :]
                acc_sum += float(reaction_pred.max(dim=0)[1].detach().cpu().data.item() == 0)
                batch_loss += criterion(reaction_pred.reshape(1, -1), batch_labels[i, :])
                product_graph_start = product_graph_end

            grad_norm_sum += optimizer.backward_and_step(batch_loss)
            total_samples += args['batch_size']
            if total_samples % args['print_every'] == 0:
                progress = 'Epoch {:d}/{:d}, iter {:d}/{:d} | time {:.4f} | ' \
                           'accuracy {:.4f} | grad norm {:.4f}'.format(
                    epoch + 1, args['num_epochs'],
                    (batch_id + 1) * args['batch_size'] // args['print_every'],
                    len(train_set) // args['print_every'],
                    (sum(dur) + time.time() - t0) / total_samples * args['print_every'],
                    acc_sum / args['print_every'],
                    grad_norm_sum / args['print_every'])
                print(progress)
                acc_sum = 0
                grad_norm_sum = 0

            if total_samples % args['decay_every'] == 0:
                dur.append(time.time() - t0)
                old_lr = optimizer.lr
                optimizer.decay_lr(args['lr_decay_factor'])
                new_lr = optimizer.lr
                print('Learning rate decayed from {:.4f} to {:.4f}'.format(old_lr, new_lr))
                torch.save({'model_state_dict': model.state_dict()},
                           args['result_path'] + '/model_{:d}.pkl'.format(total_samples))
                prediction_summary = 'total samples {:d}, (epoch {:d}/{:d}, iter {:d}/{:d})\n'.format(
                    total_samples, epoch + 1, args['num_epochs'],
                    (batch_id + 1) * args['batch_size'] // args['print_every'],
                    len(train_set) // args['print_every']) + candidate_ranking_eval(args, model, val_loader)
                print(prediction_summary)
                with open(args['result_path'] + '/val_eval.txt', 'a') as f:
                    f.write(prediction_summary)
                t0 = time.time()
                model.train()

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
