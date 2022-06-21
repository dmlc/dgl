import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
import tqdm
# OGB must follow DGL if both DGL and PyG are installed. Otherwise DataLoader will hang.
# (This is a long-standing issue)
from ogb.linkproppred import DglLinkPropPredDataset

parser = argparse.ArgumentParser()
parser.add_argument('--pure-gpu', action='store_true',
                    help='Perform both sampling and training on GPU.')
args = parser.parse_args()

device = 'cuda'

def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``.
    Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts='count', writeback_mapping=True)
    c = g_simple.edata['count']
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(g_simple.num_edges() + 1, dtype=g_simple.idtype)
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges)
    reverse_mapping = mapping[reverse_idx]

    # Correctness check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        feat = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=num_workers)
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden, device=buffer_device,
                            pin_memory=args.pure_gpu)
            feat = feat.to(device)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


def compute_mrr(model, node_emb, src, dst, neg_dst, device, batch_size=500):
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = model.predict(h_src, h_dst).squeeze(-1)
        relevance = torch.zeros(*pred.shape, dtype=torch.bool)
        relevance[:, 0] = True
        rr[start:end] = MF.retrieval_reciprocal_rank(pred, relevance)
    return rr.mean()


def evaluate(model, edge_split, device, num_workers):
    with torch.no_grad():
        node_emb = model.inference(graph, device, 4096, num_workers, 'cpu')
        results = []
        for split in ['valid', 'test']:
            src = edge_split[split]['source_node'].to(device)
            dst = edge_split[split]['target_node'].to(device)
            neg_dst = edge_split[split]['target_node_neg'].to(device)
            results.append(compute_mrr(model, node_emb, src, dst, neg_dst, device))
    return results


dataset = DglLinkPropPredDataset('ogbl-citation2')
graph = dataset[0]
graph, reverse_eids = to_bidirected_with_reverse_mapping(graph)
graph = graph.to('cuda' if args.pure_gpu else 'cpu')
reverse_eids = reverse_eids.to(device)
seed_edges = torch.arange(graph.num_edges()).to(device)
edge_split = dataset.get_edge_split()

model = SAGE(graph.ndata['feat'].shape[1], 256).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

sampler = dgl.dataloading.NeighborSampler([15, 10, 5], prefetch_node_feats=['feat'])
sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, exclude='reverse_id', reverse_eids=reverse_eids,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1))
dataloader = dgl.dataloading.DataLoader(
        graph, seed_edges, sampler,
        device=device, batch_size=512, shuffle=True,
        drop_last=False, num_workers=0, use_uva=not args.pure_gpu)

durations = []
for epoch in range(10):
    model.train()
    t0 = time.time()
    for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
        x = blocks[0].srcdata['feat']
        pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)
        score = torch.cat([pos_score, neg_score])
        labels = torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy_with_logits(score, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 20 == 0:
            mem = torch.cuda.max_memory_allocated() / 1000000
            print('Loss', loss.item(), 'GPU Mem', mem, 'MB')
            if (it + 1) == 1000:
                tt = time.time()
                print(tt - t0)
                durations.append(tt - t0)
                break
    if epoch % 10 == 0:
        model.eval()
        valid_mrr, test_mrr = evaluate(model, edge_split, device, 0 if args.pure_gpu else 12)
        print('Validation MRR:', valid_mrr.item(), 'Test MRR:', test_mrr.item())
print(np.mean(durations[4:]), np.std(durations[4:]))
