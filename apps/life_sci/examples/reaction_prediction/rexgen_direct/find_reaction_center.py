import numpy as np
import time
import torch

from dgllife.data import USPTO, WLNReactionDataset
from dgllife.model import WLNReactionCenter, load_pretrained
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils import setup, collate, reaction_center_prediction, \
    rough_eval_on_a_loader, reaction_center_final_eval

def main(args):
    setup(args)
    if args['train_path'] is None:
        train_set = USPTO('train')
    else:
        train_set = WLNReactionDataset(raw_file_path=args['train_path'],
                                       mol_graph_path='train.bin')
    if args['val_path'] is None:
        val_set = USPTO('val')
    else:
        val_set = WLNReactionDataset(raw_file_path=args['val_path'],
                                     mol_graph_path='val.bin')
    if args['test_path'] is None:
        test_set = USPTO('test')
    else:
        test_set = WLNReactionDataset(raw_file_path=args['test_path'],
                                      mol_graph_path='test.bin')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            collate_fn=collate, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             collate_fn=collate, shuffle=False)

    if args['pre_trained']:
        model = load_pretrained('wln_center_uspto').to(args['device'])
        args['num_epochs'] = 0
    else:
        model = WLNReactionCenter(node_in_feats=args['node_in_feats'],
                                  edge_in_feats=args['edge_in_feats'],
                                  node_pair_in_feats=args['node_pair_in_feats'],
                                  node_out_feats=args['node_out_feats'],
                                  n_layers=args['n_layers'],
                                  n_tasks=args['n_tasks']).to(args['device'])

        criterion = BCEWithLogitsLoss(reduction='sum')
        optimizer = Adam(model.parameters(), lr=args['lr'])
        scheduler = StepLR(optimizer, step_size=args['decay_every'], gamma=args['lr_decay_factor'])

        total_iter = 0
        grad_norm_sum = 0
        loss_sum = 0
        dur = []

    for epoch in range(args['num_epochs']):
        t0 = time.time()
        for batch_id, batch_data in enumerate(train_loader):
            total_iter += 1

            batch_reactions, batch_graph_edits, batch_mols, batch_mol_graphs, \
            batch_complete_graphs, batch_atom_pair_labels = batch_data
            labels = batch_atom_pair_labels.to(args['device'])
            pred, biased_pred = reaction_center_prediction(
                args['device'], model, batch_mol_graphs, batch_complete_graphs)
            loss = criterion(pred, labels) / len(batch_reactions)
            loss_sum += loss.cpu().detach().data.item()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), args['max_norm'])
            grad_norm_sum += grad_norm
            optimizer.step()
            scheduler.step()

            if total_iter % args['print_every'] == 0:
                progress = 'Epoch {:d}/{:d}, iter {:d}/{:d} | time/minibatch {:.4f} | ' \
                           'loss {:.4f} | grad norm {:.4f}'.format(
                    epoch + 1, args['num_epochs'], batch_id + 1, len(train_loader),
                    (np.sum(dur) + time.time() - t0) / total_iter, loss_sum / args['print_every'],
                    grad_norm_sum / args['print_every'])
                grad_norm_sum = 0
                loss_sum = 0
                print(progress)

            if total_iter % args['decay_every'] == 0:
                torch.save(model.state_dict(), args['result_path'] + '/model.pkl')

        dur.append(time.time() - t0)
        print('Epoch {:d}/{:d}, validation '.format(epoch + 1, args['num_epochs']) + \
              rough_eval_on_a_loader(args, model, val_loader))

    del train_loader
    del val_loader
    del train_set
    del val_set
    print('Evaluation on the test set.')
    test_result = reaction_center_final_eval(args, model, test_loader, args['easy'])
    print(test_result)
    with open(args['result_path'] + '/results.txt', 'w') as f:
        f.write(test_result)

if __name__ == '__main__':
    from argparse import ArgumentParser

    from configure import reaction_center_config

    parser = ArgumentParser(description='Reaction Center Identification')
    parser.add_argument('--result-path', type=str, default='center_results',
                        help='Path to training results')
    parser.add_argument('--train-path', type=str, default=None,
                        help='Path to a new training set. '
                             'If None, we will use the default training set in USPTO.')
    parser.add_argument('--val-path', type=str, default=None,
                        help='Path to a new validation set. '
                             'If None, we will use the default validation set in USPTO.')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Path to a new test set.'
                             'If None, we will use the default test set in USPTO.')
    parser.add_argument('-p', '--pre-trained', action='store_true', default=False,
                        help='If true, we will directly evaluate a '
                             'pretrained model on the test set.')
    parser.add_argument('--easy', action='store_true', default=False,
                        help='Whether to exclude reactants not contributing atoms to the '
                             'product in top-k atom pair selection, which will make the '
                             'task easier.')
    args = parser.parse_args().__dict__
    args.update(reaction_center_config)

    main(args)
