import numpy as np
import time
import torch

from dgllife.data import USPTO
from dgllife.model import WLNReactionCenter
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

def eval(complete_graphs, preds, labels, num_correct):
    batch_size = complete_graphs.batch_size
    start = 0
    for i in range(batch_size):
        end = start + complete_graphs.batch_num_edges[i]
        preds_i = preds[start:end, :].flatten()
        labels_i = labels[start:end, :].flatten()
        for k in num_correct.keys():
            topk_values, topk_indices = torch.topk(preds_i, k)
            is_correct = labels_i[topk_indices].sum() == labels_i.sum().float().cpu().data.item()
            num_correct[k].append(is_correct)
        start = end

def eval_on_a_loader(args, model, data_loader):
    model.eval()
    num_correct = {k: [] for k in args['top_ks']}
    for batch_id, batch_data in enumerate(data_loader):
        batch_reactions, batch_graph_edits, batch_mols, batch_mol_graphs, \
        batch_complete_graphs, batch_atom_pair_labels = batch_data
        with torch.no_grad():
            pred, biased_pred = perform_prediction(
                args['device'], model, batch_mol_graphs, batch_complete_graphs)
        eval(batch_complete_graphs, biased_pred, batch_atom_pair_labels, num_correct)

    msg = '|'
    for k, correct_count in num_correct.items():
        msg += ' acc@{:d} {:.4f} |'.format(k, np.mean(correct_count))

    return msg

def main(args):
    setup(args)
    train_set = USPTO('train')
    val_set = USPTO('val')
    test_set = USPTO('test')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                              collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             collate_fn=collate)

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
        model.train()
        for batch_id, batch_data in enumerate(train_loader):
            total_iter += 1
            if total_iter >= 3:
                t0 = time.time()

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

            if total_iter >= 3:
                dur.append(time.time() - t0)

            if total_iter % args['print_every'] == 0:
                progress = 'Epoch {:d}/{:d}, iter {:d}/{:d} | time/minibatch {:.4f} | ' \
                           'loss {:.4f} | grad norm {:.4f}'.format(
                    epoch + 1, args['num_epochs'], batch_id + 1, len(train_loader),
                    np.mean(dur), loss_sum / args['print_every'],
                    grad_norm_sum / args['print_every'])
                grad_norm_sum = 0
                loss_sum = 0
                print(progress)

            if total_iter % args['decay_every'] == 0:
                torch.save(model.state_dict(), args['result_path'] + '/model.pkl')

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
    parser.add_argument('-r', '--result-path', type=str, default='center_results',
                        help='Path to training results')
    args = parser.parse_args().__dict__
    args.update(reaction_center_config)

    main(args)
