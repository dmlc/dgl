import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import argparse
from dgl.data import register_data_args, load_data
from model import GraphSAGEBatchMultiGPU,GraphSAGEBatch
from node_classification import batch_evaluate, evaluate

def train(rank, world_size, args, graph, devices, data):
    in_feats, num_classes, train_idx, valid_idx, test_idx = data
    device = torch.device(devices[rank])
    if world_size > 0:
        torch.cuda.set_device(device)
    else:
        world_size = 1
        
    if world_size > 1:
        dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)
        model = GraphSAGEBatchMultiGPU(in_feats, args.n_hidden, num_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        model = model.module
    else:
        model = GraphSAGEBatch(in_feats, args.n_hidden, num_classes, args.n_layers, F.relu, args.dropout, args.aggregator_type).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # move ids to GPU
    train_idx = train_idx.to(device)
    valid_idx = valid_idx.to(device)
    
    # For training, each process/GPU will get a subset of the
    # train_idx/valid_idx, and generate mini-batches indepednetly. This allows
    # the only communication neccessary in training to be the all-reduce for
    # the gradients performed by the DDP wrapper (created above).
    sampler = dgl.dataloading.NeighborSampler(
            [15, 10, 5], prefetch_node_feats=['feat'], prefetch_labels=['label'])
    train_dataloader = dgl.dataloading.DataLoader(
            graph, train_idx, sampler,
            device=device, batch_size=args.batch_size, shuffle=True, drop_last=False,
            num_workers=0, use_ddp=world_size > 1, use_uva=True)
    valid_dataloader = dgl.dataloading.DataLoader(
            graph, valid_idx, sampler, device=device, batch_size=args.batch_size, shuffle=True,
            drop_last=False, num_workers=0, use_ddp=world_size > 1,
            use_uva=True)

    dur = []

    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()
        total_loss = torch.tensor(0.0).to(device)
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label'][:, 0]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
        tt = time.time()
        dur.append(tt - t0)
        acc = batch_evaluate(model, graph, valid_dataloader) / world_size
        if world_size > 1: dist.reduce(acc, 0)
        if rank == 0:
            mem = torch.cuda.max_memory_allocated() / 1000000
            print("Epoch {:05d} | Avg. Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "GPU Mem(MB) {:.2f}".format(epoch, np.mean(dur), total_loss.item() / (it+1),
                                              acc.item(), mem))
        if world_size > 1: dist.barrier()

    acc = evaluate(model, graph, device, test_idx, 2**16) # 2**16: batch_size for evaluation
    if rank==0:
        print("Test Accuracy on Master {:.4f}".format(acc.item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=str, default='-1',
                        help="GPU, can be a list of gpus for multi-gpu training,"
                                " e.g., 0,1,2,3; -1 for CPU")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--pure-gpu', action='store_true',
                    help='Perform both sampling and training on GPU.')
    args = parser.parse_args()
    
    dataset = DglNodePropPredDataset('ogbn-products')
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)
    g, labels = dataset[0]
    g.ndata['label'] = labels
    features = g.ndata['feat']
    g.create_formats_()     # must be called before mp.spawn().
    split_idx = dataset.get_idx_split()
    n_classes = dataset.num_classes
    n_edges = g.number_of_edges()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    data = features.shape[1], n_classes, train_idx, valid_idx, test_idx

    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
    (n_edges, n_classes,
     len(train_idx), len(valid_idx), len(test_idx)))
     
    # Tested with mp.spawn and fork.  Both worked and got 4s per epoch with 4 GPUs
    # and 3.86s per epoch with 8 GPUs on p2.8x, compared to 5.2s from official examples.
    import torch.multiprocessing as mp
    if devices[0] == -1:
        train(0, 0, args, g, ['cpu'], data)
    elif n_gpus == 1:
        train(0, 1, args, g, devices, data)
    else:
        mp.spawn(train, args=(n_gpus, args, g, devices, data), nprocs=n_gpus)
