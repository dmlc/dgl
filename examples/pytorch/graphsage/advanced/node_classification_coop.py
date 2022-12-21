import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.distributed as thd
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.contrib.dist_sampling import DistConv, DistGraph, DistSampler, uniform_partition, reorder_graph_wrapper
import argparse
import os

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, replicated=False):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(DistConv(dglnn.SAGEConv(in_size, hid_size, 'mean'), False))
        self.layers.append(DistConv(dglnn.SAGEConv(hid_size, hid_size, 'mean'), not replicated))
        self.layers.append(DistConv(dglnn.SAGEConv(hid_size, out_size, 'mean'), not replicated))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

def producer(g, fanouts, idx, batch_size, device):
    sampler = DistSampler(g, NeighborSampler, fanouts, ['feat'], ['label'])
    it = 0
    outputs = [None, None]
    perm = torch.randperm(idx.shape[0], device=device)
    num_items = torch.tensor(idx.shape[0], device=g.device)
    thd.all_reduce(num_items, thd.ReduceOp.MAX, g.comm)
    for i in range(0, num_items, batch_size):
        seeds = idx[perm[i: i + batch_size]] if i < idx.shape[0] else torch.tensor([], dtype=idx.dtype)
        out = sampler.sample(g.g, seeds.to(device))
        wait = out[-1][0].slice_features(out[-1][0])
        out[-1][-1].slice_labels(out[-1][-1])
        outputs[it % 2] = out + (wait,)
        it += 1
        if it > 1:
            out = outputs[it % 2]
            out[-1]()
            yield out[:-1]
    it += 1
    out = outputs[it % 2]
    out[-1]()
    yield out[:-1]

def evaluate(model, g, fanouts, idx, batch_size, device):
    model.eval()
    ys = []
    y_hats = []
    for input_nodes, output_nodes, blocks in producer(g, fanouts, idx, batch_size, device):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'])
            y_hats.append(model(blocks, x))
    cnt = torch.tensor(sum(y.shape[0] for y in ys), dtype=torch.int64, device=device)
    acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys)).to(device) * cnt
    thd.all_reduce(acc, thd.ReduceOp.SUM, g.comm)
    thd.all_reduce(cnt, thd.ReduceOp.SUM, g.comm)
    return acc / cnt

def train(local_rank, local_size, group_rank, world_size, g, parts, dataset, args):
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    global_rank = group_rank * local_size + local_rank
    thd.init_process_group('nccl', 'env://', world_size=world_size, rank=global_rank)

    g = DistGraph(g, parts, args.replication, True)

    train_idx = torch.nonzero(g.dstdata['train_mask'], as_tuple=True)[0] + g.l_offset
    val_idx = torch.nonzero(g.dstdata['val_mask'], as_tuple=True)[0] + g.l_offset
    test_idx = torch.nonzero(~(g.dstdata['train_mask'] | g.dstdata['val_mask']), as_tuple=True)[0] + g.l_offset

    # model training
    if global_rank == 0:
        print('Training...')
    # create GraphSAGE model
    in_size = g.dstdata['feat'].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size, args.replication == 1).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        cnt = 0
        for input_nodes, output_nodes, blocks in producer(g, [10, 10, 10], train_idx, args.batch_size, device):
            x = blocks[0].srcdata.pop('feat')
            y = blocks[-1].dstdata.pop('label')
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.shape[0] if y.shape[0] > 0 else 0
            cnt += y.shape[0]
        loss = torch.tensor(total_loss, dtype=torch.int64, device=device)
        cnt = torch.tensor(cnt, dtype=torch.int64, device=device)
        thd.all_reduce(loss, thd.ReduceOp.SUM, g.comm)
        thd.all_reduce(cnt, thd.ReduceOp.SUM, g.comm)
        
        acc = evaluate(model, g, [10, 10, 10], val_idx, args.batch_size, device)
        if global_rank == 0:
            print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                  .format(epoch, (loss / cnt).item(), acc.item()))
    
    if global_rank == 0:
        print('Testing...')
    acc = evaluate(model, g, [-1, -1, -1], test_idx, 100000, device)
    if global_rank == 0:
        print("Test Accuracy {:.4f}".format(acc.item()))

    thd.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replication', type=int, default=0, help="how many gpus should cooperate, \
            default is all")
    parser.add_argument('--batch-size', type=int, default=1024) # per rank
    args = parser.parse_args()
    assert torch.cuda.is_available(), "cuda is required"

    local_size = torch.cuda.device_count()
    group_rank = int(os.environ["GROUP_RANK"])
    num_groups = int(os.environ["WORLD_SIZE"])
    world_size = local_size * num_groups
    if args.replication <= 0:
        args.replication = world_size

    # load and preprocess dataset
    print('Loading data')
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]

    torch.manual_seed(0)
    parts = uniform_partition(g, world_size)

    g = reorder_graph_wrapper(g, parts)
    g.create_formats_()

    torch.multiprocessing.spawn(train, args=(local_size, group_rank, world_size, g, [len(part) for part in parts], dataset, args), nprocs=local_size)