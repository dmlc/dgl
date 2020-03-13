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

from utils import setup, collate

def perform_prediction(device, model, mol_graphs, complete_graphs):
    node_feats = mol_graphs.ndata.pop('hv').to(device)
    edge_feats = mol_graphs.edata.pop('he').to(device)
    node_pair_feats = complete_graphs.edata.pop('feats').to(device)

    return model(mol_graphs, complete_graphs, node_feats, edge_feats, node_pair_feats)

def eval(complete_graphs, preds, graph_edits, mols, num_correct, max_k):
    # 0 for losing the bond
    # 1, 2, 3, 1.5 separately for forming a single, double, triple or aromatic bond.
    bond_change_to_id = {0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
    id_to_bond_change = {v: k for k, v in bond_change_to_id.items()}
    num_change_types = len(bond_change_to_id)

    batch_size = complete_graphs.batch_size
    start = 0
    for i in range(batch_size):
        num_nodes = complete_graphs.batch_num_nodes[i]
        end = start + complete_graphs.batch_num_edges[i]
        preds_i = preds[start:end, :].flatten()
        mol_i = mols[i]
        candidate_bonds = []
        topk_values, topk_indices = torch.topk(preds_i, max_k)
        for j in range(max_k):
            preds_i_j = topk_indices[j]
            # A bond change can be either losing the bond or forming a
            # single, double, triple or aromatic bond
            change_id = preds_i_j % num_change_types
            change_type = id_to_bond_change[change_id]
            pair_id = preds_i_j // num_change_types
            atom1 = pair_id // num_nodes
            atom2 = pair_id % num_nodes
            if atom1 >= atom2:
                continue
            bond = mol_i.GetBondBetweenAtoms(atom1, atom2)
            # Filter out existing bonds in the reactants
            if (bond is None) or (bond.GetBondTypeAsDouble() != change_type):
                candidate_bonds.append((int(atom1), int(atom2), float(change_type)))

        gold_bonds = []
        gold_edits = graph_edits[i]
        for edit in gold_edits.split(';'):
            atom1, atom2, change_type = edit.split('-')
            atom1, atom2 = int(atom1), int(atom2)
            gold_bonds.append((min(atom1, atom2), max(atom1, atom2), float(change_type)))

        for k in num_correct.keys():
            if set(gold_bonds) <= set(candidate_bonds[:k]):
                num_correct[k] += 1
        start = end

def eval_on_a_loader(args, model, data_loader):
    model.eval()
    num_correct = {k: 0 for k in args['top_ks']}
    for batch_id, batch_data in enumerate(data_loader):
        batch_reactions, batch_graph_edits, batch_mols, batch_mol_graphs, \
        batch_complete_graphs, batch_atom_pair_labels = batch_data
        with torch.no_grad():
            pred, biased_pred = perform_prediction(
                args['device'], model, batch_mol_graphs, batch_complete_graphs)
        eval(batch_complete_graphs, biased_pred,  batch_atom_pair_labels,
             batch_graph_edits, batch_mols, num_correct, args['max_k'])

    msg = '|'
    for k, correct_count in num_correct.items():
        msg += ' acc@{:d} {:.4f} |'.format(k, correct_count / len(data_loader.dataset))

    return msg

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
            pred, biased_pred = perform_prediction(args['device'], model,
                                                   batch_mol_graphs, batch_complete_graphs)
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
              eval_on_a_loader(args, model, val_loader))

    del train_loader
    del val_loader
    del train_set
    del val_set
    print('Evaluation on the test set.')
    test_result = eval_on_a_loader(args, model, test_loader)
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
    args = parser.parse_args().__dict__
    args.update(reaction_center_config)

    main(args)
