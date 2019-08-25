"""
Learning Deep Generative Models of Graphs
Paper: https://arxiv.org/pdf/1803.03324.pdf
"""
import datetime
import time
import torch
import torch.distributed as dist
from dgl import model_zoo
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import MoleculeDataset, Printer, set_random_seed, synchronize, launch_a_process

def evaluate(epoch, model, data_loader, printer):
    model.eval()
    batch_size = data_loader.batch_size
    total_log_prob = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            log_prob = model(actions=data, compute_log_prob=True).detach()
            total_log_prob -= log_prob
            if printer is not None:
                prob = log_prob.detach().exp()
                printer.update(epoch + 1, - log_prob / batch_size, prob / batch_size)
    return total_log_prob / len(data_loader)

def main(rank, args):
    """
    Parameters
    ----------
    rank : int
        Subprocess id
    args : dict
        Configuration
    """
    if rank == 0:
        t1 = time.time()

    set_random_seed(args['seed'])
    # Remove the line below will result in problems for multiprocess
    torch.set_num_threads(1)

    # Setup dataset and data loader
    dataset = MoleculeDataset(args['dataset'], args['order'], ['train', 'val'],
                              subset_id=rank, n_subsets=args['num_processes'])

    # Note that currently the batch size for the loaders should only be 1.
    train_loader = DataLoader(dataset.train_set, batch_size=args['batch_size'],
                              shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(dataset.val_set, batch_size=args['batch_size'],
                            shuffle=True, collate_fn=dataset.collate)

    if rank == 0:
        try:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(args['log_dir'])
        except ImportError:
            print('If you want to use tensorboard, install tensorboardX with pip.')
            writer = None
        train_printer = Printer(args['nepochs'], len(dataset.train_set), args['batch_size'], writer)
        val_printer = Printer(args['nepochs'], len(dataset.val_set), args['batch_size'])
    else:
        val_printer = None

    # Initialize model
    model = model_zoo.chem.DGMG(atom_types=dataset.atom_types,
                                bond_types=dataset.bond_types,
                                node_hidden_size=args['node_hidden_size'],
                                num_prop_rounds=args['num_propagation_rounds'],
                                dropout=args['dropout'])

    if args['num_processes'] == 1:
        from utils import Optimizer
        optimizer = Optimizer(args['lr'], Adam(model.parameters(), lr=args['lr']))
    else:
        from utils import MultiProcessOptimizer
        optimizer = MultiProcessOptimizer(args['num_processes'], args['lr'],
                                          Adam(model.parameters(), lr=args['lr']))

    if rank == 0:
        t2 = time.time()
    best_val_prob = 0

    # Training
    for epoch in range(args['nepochs']):
        model.train()
        if rank == 0:
            print('Training')

        for i, data in enumerate(train_loader):
            log_prob = model(actions=data, compute_log_prob=True)
            prob = log_prob.detach().exp()

            loss_averaged = - log_prob
            prob_averaged = prob
            optimizer.backward_and_step(loss_averaged)
            if rank == 0:
                train_printer.update(epoch + 1, loss_averaged.item(), prob_averaged.item())

        synchronize(args['num_processes'])

        # Validation
        val_log_prob = evaluate(epoch, model, val_loader, val_printer)
        if args['num_processes'] > 1:
            dist.all_reduce(val_log_prob, op=dist.ReduceOp.SUM)
        val_log_prob /= args['num_processes']
        # Strictly speaking, the computation of probability here is different from what is
        # performed on the training set as we first take an average of log likelihood and then
        # take the exponentiation. By Jensen's inequality, the resulting value is then a
        # lower bound of the real probabilities.
        val_prob = (- val_log_prob).exp().item()
        val_log_prob = val_log_prob.item()
        if val_prob >= best_val_prob:
            if rank == 0:
                torch.save({'model_state_dict': model.state_dict()}, args['checkpoint_dir'])
                print('Old val prob {:.10f} | new val prob {:.10f} | model saved'.format(best_val_prob, val_prob))
            best_val_prob = val_prob
        elif epoch >= args['warmup_epochs']:
            optimizer.decay_lr()

        if rank == 0:
            print('Validation')
            if writer is not None:
                writer.add_scalar('validation_log_prob', val_log_prob, epoch)
                writer.add_scalar('validation_prob', val_prob, epoch)
                writer.add_scalar('lr', optimizer.lr, epoch)
            print('Validation log prob {:.4f} | prob {:.10f}'.format(val_log_prob, val_prob))

        synchronize(args['num_processes'])

    if rank == 0:
        t3 = time.time()
        print('It took {} to setup.'.format(datetime.timedelta(seconds=t2 - t1)))
        print('It took {} to finish training.'.format(datetime.timedelta(seconds=t3 - t2)))
        print('--------------------------------------------------------------------------')
        print('On average, an epoch takes {}.'.format(datetime.timedelta(
            seconds=(t3 - t2) / args['nepochs'])))

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser(description='Training DGMG for molecule generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # configure
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-w', '--warmup-epochs', type=int, default=10,
                        help='Number of epochs where no lr decay is performed.')

    # dataset and setting
    parser.add_argument('-d', '--dataset',
                        help='dataset to use')
    parser.add_argument('-o', '--order', choices=['random', 'canonical'],
                        help='order to generate graphs')
    parser.add_argument('-tf', '--train-file', type=str, default=None,
                        help='Path to a file with one SMILES a line for training data. '
                             'This is only necessary if you want to use a new dataset.')
    parser.add_argument('-vf', '--val-file', type=str, default=None,
                        help='Path to a file with one SMILES a line for validation data. '
                             'This is only necessary if you want to use a new dataset.')

    # log
    parser.add_argument('-l', '--log-dir', default='./training_results',
                        help='folder to save info like experiment configuration')

    # multi-process
    parser.add_argument('-np', '--num-processes', type=int, default=32,
                        help='number of processes to use')
    parser.add_argument('-mi', '--master-ip', type=str, default='127.0.0.1')
    parser.add_argument('-mp', '--master-port', type=str, default='12345')

    args = parser.parse_args()
    args = setup(args, train=True)

    if args['num_processes'] == 1:
        main(0, args)
    else:
        mp = torch.multiprocessing.get_context('spawn')
        procs = []
        for rank in range(args['num_processes']):
            procs.append(mp.Process(target=launch_a_process, args=(rank, args, main), daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()
