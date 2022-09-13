import argparse

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader
from dgl.contrib.cugraph.nn.conv.relgraphconv_ops import RgcnConv

# debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        # two-layer RGCN
        self.conv1 = RgcnConv(h_dim, h_dim, num_rels,
                              regularizer='basis', num_bases=num_bases, self_loop=False)
        self.conv2 = RgcnConv(h_dim, out_dim, num_rels,
                              regularizer='basis', num_bases=num_bases, self_loop=False)

    def forward(self, g):
        x = self.emb(g[0].srcdata[dgl.NID])
        h = F.relu(self.conv1(g[0], x, g[0].edata[dgl.ETYPE], norm=g[0].edata['norm']))
        h = self.conv2(g[1], h, g[1].edata[dgl.ETYPE], norm=g[1].edata['norm'])
        return h

def evaluate(model, label, dataloader, inv_target):
    model.eval()
    eval_logits = []
    eval_seeds = []
    with th.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            output_nodes = inv_target[output_nodes]
            for block in blocks:
                block.edata['norm'] = dgl.norm_by_dst(block).unsqueeze(1)
            logits = model(blocks)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(output_nodes.cpu().detach())
    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
    return accuracy(eval_logits.argmax(dim=1), labels[eval_seeds].cpu()).item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for entity classification')
    parser.add_argument("--dataset", type=str, default="aifb",
                        help="Dataset name ('aifb', 'mutag', 'bgs', 'am'), default to 'aifb'.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose data information.")
    args = parser.parse_args()

    # load and preprocess dataset
    if args.dataset == 'aifb':
        data = AIFBDataset()
    elif args.dataset == 'mutag':
        data = MUTAGDataset()
    elif args.dataset == 'bgs':
        data = BGSDataset()
    elif args.dataset == 'am':
        data = AMDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    hg = data[0]
    hg = hg.to(device)

    if args.verbose:
        print(f'# nodes: {hg.num_nodes()}')
        print(f'# destination nodes: {hg.num_dst_nodes()}')
        print(f'# source nodes: {hg.num_src_nodes()}')
        print(f'# edges: {hg.num_edges()}')
        print(f'# Node types: {len(hg.ntypes)}')
        print(f'# Canonical edge types: {len(hg.etypes)}')
        print(f'# Unique edge type names: {len(set(hg.etypes))}')

    num_rels = len(hg.canonical_etypes)
    category = data.predict_category
    labels = hg.nodes[category].data.pop('labels')
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    # calculate normalization weight for each edge, and find target category and node id
    for cetype in hg.canonical_etypes:
        hg.edges[cetype].data['norm'] = dgl.norm_by_dst(hg, cetype).unsqueeze(1)
    category_id = hg.ntypes.index(category)
    g = dgl.to_homogeneous(hg, edata=['norm'])
    node_ids = th.arange(g.num_nodes()).to(device)
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]  # target node index

    # rename the field as they can be changed by DataLoader
    # cugraph-ops requires node/edge type to be int32
    g.ndata['ntype'] = g.ndata.pop(dgl.NTYPE).type(th.int32)
    g.ndata['type_id'] = g.ndata.pop(dgl.NID)
    g.edata[dgl.ETYPE] = g.edata[dgl.ETYPE].type(th.int32)

    # find the mapping (inv_target) from global node IDs to type-specific node IDs
    inv_target = th.empty((g.num_nodes(),), dtype=th.int64).to(device)
    inv_target[target_idx] = th.arange(0, target_idx.shape[0], dtype=inv_target.dtype).to(device)

    # construct sampler and dataloader
    fanouts = [4, 4]
    sampler = MultiLayerNeighborSampler(fanouts)
    train_loader = DataLoader(g, target_idx[train_idx], sampler, device=device,
                              batch_size=100, shuffle=True)
    # no separate validation subset, use train index instead for validation
    val_loader = DataLoader(g, target_idx[train_idx], sampler, device=device,
                            batch_size=100, shuffle=False)

    # create model
    h_dim = 16  # hidden feature dim
    num_classes = data.num_classes
    num_bases = 20
    model = RGCN(g.num_nodes(), h_dim, num_classes, num_rels, num_bases)
    model = model.to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training
    model.train()
    for epoch in range(100):
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            output_nodes = inv_target[output_nodes]
            for block in blocks:
                block.edata['norm'] = dgl.norm_by_dst(block).unsqueeze(1)
            logits = model(blocks)
            loss = F.cross_entropy(logits, labels[output_nodes])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        acc = evaluate(model, labels, val_loader, inv_target)
        print("Epoch {:05d} | Loss {:.4f} | Val. Accuracy {:.4f} ".format(epoch, total_loss / (it+1), acc))

    # evaluation
    # TODO(tingyu66): should sample all neighbors for test accuracy
    test_sampler = MultiLayerNeighborSampler([100, 100]) # -1 for sampling all neighbors
    test_loader = DataLoader(g, target_idx[test_idx], test_sampler, device=device,
                             batch_size=32, shuffle=False)
    acc = evaluate(model, labels, test_loader, inv_target)
    print("Test accuracy {:.4f}".format(acc))
