import numpy as np
import time
import torch

from copy import deepcopy
from dgllife.data import USPTOCenter, WLNCenterDataset
from dgllife.model import WLNReactionCenter, load_pretrained
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import collate, reaction_center_prediction, reaction_center_rough_eval_on_a_loader, \
    mkdir_p, set_seed, synchronize, get_center_subset

def load_dataset(args):
    if args['train_path'] is None:
        train_set = USPTOCenter('train', num_processes=args['num_processes'])
    else:
        train_set = WLNCenterDataset(raw_file_path=args['train_path'],
                                     mol_graph_path='train.bin',
                                     num_processes=args['num_processes'])
    if args['val_path'] is None:
        val_set = USPTOCenter('val', num_processes=args['num_processes'])
    else:
        val_set = WLNCenterDataset(raw_file_path=args['val_path'],
                                   mol_graph_path='val.bin',
                                   num_processes=args['num_processes'])

    return train_set, val_set

def main(rank, dev_id, args, train_set, val_set):
    set_seed()
    # Remove the line below will result in problems for multiprocess
    torch.set_num_threads(1)
    if dev_id == -1:
        args['device'] = torch.device('cpu')
    else:
        args['device'] = torch.device('cuda:{}'.format(dev_id))
    # Set current device
    torch.cuda.set_device(args['device'])

    """
    train_set, val_set = load_dataset(args)
    get_center_subset(train_set, rank, args['num_devices'])
    """
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              collate_fn=collate, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            collate_fn=collate, shuffle=False)

    model = WLNReactionCenter(node_in_feats=args['node_in_feats'],
                              edge_in_feats=args['edge_in_feats'],
                              node_pair_in_feats=args['node_pair_in_feats'],
                              node_out_feats=args['node_out_feats'],
                              n_layers=args['n_layers'],
                              n_tasks=args['n_tasks']).to(args['device'])

    criterion = BCEWithLogitsLoss(reduction='sum')
    optimizer = Adam(model.parameters(), lr=args['lr'])
    if args['num_devices'] <= 1:
        from utils import Optimizer
        optimizer = Optimizer(model, args['lr'], optimizer)
    else:
        from utils import MultiProcessOptimizer
        optimizer = MultiProcessOptimizer(args['num_devices'], model, args['lr'], optimizer)

    total_iter = 0
    grad_norm_sum = 0
    loss_sum = 0
    dur = []

    for epoch in range(args['num_epochs']):
        if rank == 0:
            t0 = time.time()
        for batch_id, batch_data in enumerate(train_loader):
            total_iter += 1

            batch_reactions, batch_graph_edits, batch_mol_graphs, \
            batch_complete_graphs, batch_atom_pair_labels = batch_data
            labels = batch_atom_pair_labels.to(args['device'])
            pred, biased_pred = reaction_center_prediction(
                args['device'], model, batch_mol_graphs, batch_complete_graphs)
            loss = criterion(pred, labels) / len(batch_reactions)
            loss_sum += loss.cpu().detach().data.item()
            grad_norm = optimizer.backward_and_step(loss)
            grad_norm_sum += grad_norm
            if total_iter % args['decay_every']:
                optimizer.decay_lr(args['lr_decay_factor'])

            if total_iter % args['print_every'] == 0 and rank == 0:
                progress = 'Epoch {:d}/{:d}, iter {:d}/{:d} | time/minibatch {:.4f} | ' \
                           'loss {:.4f} | grad norm {:.4f}'.format(
                    epoch + 1, args['num_epochs'], batch_id + 1, len(train_loader),
                    (np.sum(dur) + time.time() - t0) / total_iter, loss_sum / args['print_every'],
                    grad_norm_sum / args['print_every'])
                grad_norm_sum = 0
                loss_sum = 0
                print(progress)

            if total_iter % args['decay_every'] == 0 and rank == 0:
                torch.save({'model_state_dict': model.state_dict()},
                           args['result_path'] + '/model.pkl')

        if rank == 0:
            dur.append(time.time() - t0)
            print('Epoch {:d}/{:d}, validation '.format(epoch + 1, args['num_epochs']) + \
                  reaction_center_rough_eval_on_a_loader(args, model, val_loader))
        synchronize(args['num_devices'])

def run(rank, dev_id, args, train_set, val_set):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=args['master_ip'], master_port=args['master_port'])
    torch.distributed.init_process_group(backend="nccl",
                                         init_method=dist_init_method,
                                         world_size=args['num_devices'],
                                         rank=dev_id)
    assert torch.distributed.get_rank() == dev_id
    main(rank, dev_id, args, train_set, val_set)

if __name__ == '__main__':
    from argparse import ArgumentParser

    from configure import reaction_center_config

    parser = ArgumentParser(description='Reaction Center Identification -- Training')
    parser.add_argument('--gpus', default='0', type=str,
                        help='To use multi-gpu training, '
                             'pass multiple gpu ids with --gpus id1,id2,...')
    parser.add_argument('--result-path', type=str, default='center_results',
                        help='Path to save modeling results')
    parser.add_argument('--train-path', type=str, default=None,
                        help='Path to a new training set. '
                             'If None, we will use the default training set in USPTO.')
    parser.add_argument('--val-path', type=str, default=None,
                        help='Path to a new validation set. '
                             'If None, we will use the default validation set in USPTO.')
    parser.add_argument('-np', '--num-processes', type=int, default=32,
                        help='Number of processes to use for data pre-processing')
    parser.add_argument('--master-ip', type=str, default='127.0.0.1',
                        help='master ip address')
    parser.add_argument('--master-port', type=str, default='12345',
                        help='master port')
    args = parser.parse_args().__dict__
    args.update(reaction_center_config)

    assert args['max_k'] >= max(args['top_ks']), \
        'Expect max_k to be no smaller than the possible options ' \
        'of top_ks, got {:d} and {:d}'.format(args['max_k'], max(args['top_ks']))
    mkdir_p(args['result_path'])

    devices = list(map(int, args['gpus'].split(',')))
    args['num_devices'] = len(devices)
    train_set, val_set = load_dataset(args)

    if len(devices) == 1:
        device_id = devices[0] if torch.cuda.is_available() else -1
        main(0, device_id, args, train_set, val_set)
    else:
        mp = torch.multiprocessing.get_context('spawn')
        procs = []
        for id, device_id in enumerate(devices):
            print('Preparing for gpu {:d}/{:d}'.format(id + 1, args['num_devices']))
            train_subset = deepcopy(train_set)
            val_susbet = deepcopy(val_set)
            get_center_subset(train_subset, id, args['num_devices'])
            get_center_subset(val_susbet, id, args['num_devices'])
            procs.append(mp.Process(target=run, args=(
                id, device_id, args, train_subset, val_susbet), daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()
