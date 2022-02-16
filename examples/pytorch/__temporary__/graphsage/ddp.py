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

USE_WRAPPER = False
MODE = 'uva'           # 'cuda' or 'uva'

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def train(rank, world_size, graph, num_classes, split_idx):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)

    model = SAGE(graph.ndata['feat'].shape[1], 256, num_classes).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    if MODE == 'uva':
        train_idx = train_idx.to('cuda')
    use_prefetch_thread = (MODE == 'cuda')
    pin_prefetcher = (MODE == 'cuda')
    num_workers = 0 if (MODE == 'uva') else 4

    sampler = dgl.dataloading.NeighborSampler(
            [5, 5, 5], prefetch_node_feats=['feat'], prefetch_labels=['label'])
    dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            train_idx,
            sampler,
            device='cuda',
            batch_size=1000,
            shuffle=True,
            drop_last=False,
            pin_prefetcher=pin_prefetcher,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            use_ddp=True,
            use_prefetch_thread=use_prefetch_thread,
            use_uva=(MODE == 'uva'))

    durations = []
    for _ in range(10):
        t0 = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label'][:, 0]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 20 == 0:
                acc = MF.accuracy(y_hat, y)
                mem = torch.cuda.max_memory_allocated() / 1000000
                print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
        tt = time.time()
        if rank == 0:
            print(tt - t0)
            durations.append(tt - t0)
    if rank == 0:
        print(np.mean(durations[4:]), np.std(durations[4:]))

if __name__ == '__main__':
    dataset = DglNodePropPredDataset('ogbn-products')
    graph, labels = dataset[0]
    graph.ndata['label'] = labels
    graph.create_formats_()
    split_idx = dataset.get_idx_split()
    num_classes = dataset.num_classes
    n_procs = 4

    if MODE == 'uva':
        # put the graph into shared memory
        new_graph = graph.shared_memory('shm')
        new_graph.ndata['feat'] = graph.ndata['feat']
        new_graph.ndata['label'] = graph.ndata['label']
        new_graph.ndata['feat'].share_memory_()
        new_graph.ndata['label'].share_memory_()
        new_graph.create_formats_()
        graph = new_graph

    # Tested with mp.spawn and fork.  Both worked and got 4s per epoch with 4 GPUs
    # and 3.86s per epoch with 8 GPUs on p2.8x, compared to 5.2s from official examples.
    import torch.multiprocessing as mp
    mp.spawn(train, args=(n_procs, graph, num_classes, split_idx), nprocs=n_procs)
