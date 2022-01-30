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
    if USE_WRAPPER:
        import dglnew
        graph = dglnew.graph.wrapper.DGLGraphStorage(graph)

    sampler = dgl.dataloading.NeighborSampler(
            [5, 5, 5], output_device='cpu', prefetch_node_feats=['feat'],
            prefetch_labels=['label'])
    dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            train_idx,
            sampler,
            device='cuda',
            batch_size=1000,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            use_ddp=True,
            use_prefetch_thread=True)       # TBD: could probably remove this argument

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

    # Tested with mp.spawn and fork.  Both worked and got 4s per epoch with 4 GPUs
    # and 3.86s per epoch with 8 GPUs on p2.8x, compared to 5.2s from official examples.
    #import torch.multiprocessing as mp
    #mp.spawn(train, args=(n_procs, graph, num_classes, split_idx), nprocs=n_procs)
    import dgl.multiprocessing as mp
    procs = []
    for i in range(n_procs):
        p = mp.Process(target=train, args=(i, n_procs, graph, num_classes, split_idx))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
