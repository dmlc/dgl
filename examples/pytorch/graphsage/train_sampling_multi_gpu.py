import os
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.multiprocessing as mp
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
from torch.nn.parallel import DistributedDataParallel
import tqdm

from model import SAGE
from load_graph import load_reddit, inductive_split, load_ogb

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id):
    """
    Extracts features and labels for a subset of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(dev_id)
    batch_labels = labels[seeds].to(dev_id)
    return batch_inputs, batch_labels

#### Entry point

def run(proc_id, n_gpus, args, devices, data):
    # Start up distributed training, if enabled.
    device = th.device(devices[proc_id])
    if n_gpus > 0:
        th.cuda.set_device(device)
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)

    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, val_nfeat, test_nfeat, \
    train_labels, val_labels, test_labels, train_nid, val_nid, test_nid = data

    if args.data_device == 'gpu':
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)
    elif args.data_device == 'uva':
        train_nfeat = dgl.contrib.UnifiedTensor(train_nfeat, device=device)
        train_labels = dgl.contrib.UnifiedTensor(train_labels, device=device)

    in_feats = train_nfeat.shape[1]

    if args.graph_device == 'gpu':
        train_nid = train_nid.to(device)
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        args.num_workers = 0
    elif args.graph_device == 'uva':
        train_nid = train_nid.to(device)
        train_g.pin_memory_()
        args.num_workers = 0

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        use_ddp=n_gpus > 1,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            if proc_id == 0:
                tic_step = time.time()

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if proc_id == 0:
                iter_tput.append(len(seeds) * n_gpus / (time.time() - tic_step))
            if step % args.log_every == 0 and proc_id == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), th.cuda.max_memory_allocated() / 1000000))

        if n_gpus > 1:
            th.distributed.barrier()

        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:
                if n_gpus == 1:
                    eval_acc = evaluate(
                        model, val_g, val_nfeat, val_labels, val_nid, devices[0])
                    test_acc = evaluate(
                        model, test_g, test_nfeat, test_labels, test_nid, devices[0])
                else:
                    eval_acc = evaluate(
                        model.module, val_g, val_nfeat, val_labels, val_nid, devices[0])
                    test_acc = evaluate(
                        model.module, test_g, test_nfeat, test_labels, test_nid, devices[0])
                print('Eval Acc {:.4f}'.format(eval_acc))
                print('Test Acc: {:.4f}'.format(test_acc))

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--graph-device', choices=('cpu', 'gpu', 'uva'), default='cpu',
                           help="Device to perform the sampling. "
                                "Must have 0 workers for 'gpu' and 'uva'")
    argparser.add_argument('--data-device', choices=('cpu', 'gpu', 'uva'), default='gpu',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "Use 'cpu' to keep the features on host memory and "
                                "'uva' to enable UnifiedTensor (GPU zero-copy access on "
                                "pinned host memory).")
    args = argparser.parse_args()

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-papers100M':
        g, n_classes = load_ogb('ogbn-papers100M')
        g = dgl.add_reverse_edges(g)
        # convert labels to integer
        g.ndata['labels'] = th.as_tensor(g.ndata['labels'], dtype=th.int64)
        g.ndata.pop('year')
    else:
        raise Exception('unknown dataset')

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_labels = test_g.ndata.pop('labels')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
        train_labels = val_labels = test_labels = g.ndata.pop('labels')

    test_nid = test_g.ndata.pop('test_mask',
        ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])).nonzero().squeeze()
    train_nid = train_g.ndata.pop('train_mask').nonzero().squeeze()
    val_nid = val_g.ndata.pop('val_mask').nonzero().squeeze()

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    # this to avoid competition overhead on machines with many cores.
    # Change it to a proper number on your machine, especially for multi-GPU training.
    os.environ['OMP_NUM_THREADS'] = str(mp.cpu_count() // 2 // n_gpus)
    if n_gpus > 1:
        # Copy the graph to shared memory explicitly before pinning.
        # In other cases, we can just rely on fork's copy-on-write.
        # TODO: the original train_g is not freed.
        if args.graph_device == 'uva':
            train_g = train_g.shared_memory('train_g')
        if args.data_device == 'uva':
            train_nfeat = train_nfeat.share_memory_()
            train_labels = train_labels.share_memory_()

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, val_nfeat, test_nfeat, \
           train_labels, val_labels, test_labels, train_nid, val_nid, test_nid

    if devices[0] == -1:
        assert args.graph_device == 'cpu', \
               f"Must have GPUs to enable {args.graph_device} sampling."
        assert args.data_device == 'cpu', \
               f"Must have GPUs to enable {args.data_device} feature storage."
        run(0, 0, args, ['cpu'], data)
    elif n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
