import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.utils import pin_memory_inplace, unpin_memory_inplace, gather_pinned_tensor_rows
import time
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm


def shared_tensor(*shape, device, q):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        y = torch.zeros(
            *shape, device=device)
        for i in range(1, world_size):
            q.put(y)
    else:
        y = q.get()
    dist.barrier()
    pin_memory_inplace(y)
    return y


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size,
                  buffer_device, q):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.NodeDataLoader(
                g, torch.arange(g.num_nodes(), device=device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0, use_ddp=True, use_uva=True)

        feat = g.ndata['feat']
        for l, layer in enumerate(self.layers):
            y = shared_tensor(
                    g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                    device=buffer_device, q=q)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader) \
                    if dist.get_rank() == 0 else dataloader:
                x = gather_pinned_tensor_rows(feat, input_nodes)
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes] = h.to(buffer_device)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            if l > 0:
                unpin_memory_inplace(feat)
            feat = y
        return y


def train(rank, world_size, graph, num_classes, split_idx, q):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)

    model = SAGE(graph.ndata['feat'].shape[1], 256, num_classes).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

    # move ids to GPU
    train_idx = train_idx.to('cuda')
    valid_idx = valid_idx.to('cuda')
    test_idx = test_idx.to('cuda')

    sampler = dgl.dataloading.NeighborSampler(
            [15, 10, 5], prefetch_node_feats=['feat'], prefetch_labels=['label'])
    train_dataloader = dgl.dataloading.DataLoader(
            graph, train_idx, sampler,
            device='cuda', batch_size=1024, shuffle=True, drop_last=False,
            num_workers=0, use_ddp=True, use_uva=True)
    valid_dataloader = dgl.dataloading.NodeDataLoader(
            graph, valid_idx, sampler, device='cuda', batch_size=1024, shuffle=True,
            drop_last=False, num_workers=0, use_ddp=True,
            use_uva=True)

    durations = []
    for _ in range(10):
        model.train()
        t0 = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label'][:, 0]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 20 == 0 and rank == 0:
                acc = MF.accuracy(torch.argmax(y_hat, dim=1), y)
                mem = torch.cuda.max_memory_allocated() / 1000000
                print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
        tt = time.time()

        if rank == 0:
            print(tt - t0)
        durations.append(tt - t0)

        model.eval()
        ys = []
        y_hats = []
        for it, (input_nodes, output_nodes, blocks) in enumerate(valid_dataloader):
            with torch.no_grad():
                x = blocks[0].srcdata['feat']
                ys.append(blocks[-1].dstdata['label'])
                y_hats.append(model.module(blocks, x))
        acc = MF.accuracy(torch.argmax(torch.cat(y_hats), dim=1), torch.cat(ys)) / world_size
        dist.reduce(acc, 0)
        if rank == 0:
            print('Validation acc:', acc.item())
        dist.barrier()

    if rank == 0:
        print(np.mean(durations[4:]), np.std(durations[4:]))
    model.eval()
    with torch.no_grad():
        # since we do 1-layer at a time, use a very large batch size
        pred = model.module.inference(graph, device='cuda', batch_size=2**16,
                                      buffer_device='cpu', q=q)
        if rank == 0:
            acc = MF.accuracy(torch.argmax(pred[test_idx], dim=1), graph.ndata['label'][test_idx])
            print('Test acc:', acc.item())

if __name__ == '__main__':
    dataset = DglNodePropPredDataset('ogbn-products')
    graph, labels = dataset[0]
    graph.ndata['label'] = labels
    graph.create_formats_()     # must be called before mp.spawn().
    split_idx = dataset.get_idx_split()
    num_classes = dataset.num_classes
    # use all available GPUs
    n_procs = torch.cuda.device_count()

    # Tested with mp.spawn and fork.  Both worked and got 4s per epoch with 4 GPUs
    # and 3.86s per epoch with 8 GPUs on p2.8x, compared to 5.2s from official examples.
    import torch.multiprocessing as mp
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    mp.spawn(train, args=(n_procs, graph, num_classes, split_idx, q), nprocs=n_procs)
