from timeit import default_timer
import functools
import torch
import torch.nn as nn
import dgl
import dgl.backend as F
import dgl.function as fn
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
import torch.nn as nn
import time
import numpy as np
import argparse

#####################################
# Timer code from benchmarks folder
#####################################
class Timer:
    def __init__(self, device):
        self.timer = default_timer
        self.device = device

    def __enter__(self):
        if self.device == 'cuda:0':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.tic = self.timer()
        return self

    def __exit__(self, type, value, traceback):
        if self.device == 'cuda:0':
            self.end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            self.elapsed_secs = self.start_event.elapsed_time(
                self.end_event) / 1e3
        else:
            self.elapsed_secs = self.timer() - self.tic


class RGCNHighMem(nn.Module):
    def __init__(self, weight):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, g, feat, etypes):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        g.ndata['h'] = feat
        g.update_all(functools.partial(self.message, etypes=etypes), fn.sum('m', 'h'))
        return g.ndata['h']

    def message(self, edges, etypes):
        return {'m' : torch.bmm(edges.src['h'].unsqueeze(1), self.weight[etypes]).squeeze(1)}

class RGCNLowMem(nn.Module):
    def __init__(self, weight):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, g, feat, etypes):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)

        # sort etypes
        sorted_etypes, index = torch.sort(etypes)
        g = dgl.edge_subgraph(g, index, relabel_nodes=False)
        # Create a new etypes to be an integer list of number of edges.
        num_rels = self.weight.shape[0]
        pos = torch.searchsorted(sorted_etypes, torch.arange(num_rels, device=g.device))
        num = torch.tensor([len(etypes)], device=g.device)
        etypes = (torch.cat([pos[1:], num]) - pos).tolist()

        # message passing
        g.ndata['h'] = feat
        g.update_all(functools.partial(self.message, etypes=etypes), fn.sum('m', 'h'))
        return g.ndata['h']

    def message(self, edges, etypes):
        h_t = torch.split(edges.src['h'], etypes)
        msg = []
        for r in range(self.weight.shape[0]):
            msg.append(torch.matmul(h_t[r], self.weight[r]))
        return {'m' : torch.cat(msg)}


class RGCNGatherMMSorted(nn.Module):
    def __init__(self, weight):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, g, feat, etypes, E_per_rel):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)

        # sort etypes
        etypes, index = torch.sort(etypes)
        g = dgl.edge_subgraph(g, index, relabel_nodes=False)

        # message passing
        g.ndata['h'] = feat
        g.update_all(functools.partial(self.message, etypes=etypes, E_per_rel=E_per_rel),
                     fn.sum('m', 'h'))
        return g.ndata['h']

    def message(self, edges, etypes, E_per_rel):
        h = edges.src['h']
        w = self.weight.view(-1, self.weight.shape[2])
        out = torch.zeros((h.shape[0], self.weight.shape[2]), dtype=torch.float32, device=h.device)
        dgl.sparse._gather_mm(h, w, out, E_per_rel, etypes, sortedE=True)
        return {'m' : out}

class RGCNGatherMMUnsorted(nn.Module):
    def __init__(self, weight):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, g, feat, etypes, E_per_rel):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        g.ndata['h'] = feat
        g.update_all(functools.partial(self.message, etypes=etypes, E_per_rel=E_per_rel),
                     fn.sum('m', 'h'))
        return g.ndata['h']

    def message(self, edges, etypes, E_per_rel):
        h = edges.src['h']
        w = self.weight.view(-1, self.weight.shape[2])
        out = torch.zeros((h.shape[0], self.weight.shape[2]), dtype=torch.float32, device=h.device)
        dgl.sparse._gather_mm(h, w, out, E_per_rel, etypes, sortedE=False)
        return {'m' : out}

class RGCNHetero(nn.Module):
    def __init__(self, weight, etypes):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.ParameterDict({
            etype : nn.Parameter(weight[i]) for i, etype in enumerate(etypes)})

    def forward(self, hg, feat_dict):
        # hg : DGLGraph hetero
        # feat : dict of tensors
        for ntype in hg.ntypes:
            hg.nodes[ntype].data['h'] = feat_dict[ntype]
        fns = {}
        for rel in hg.canonical_etypes:
            fns[rel] = (
                functools.partial(self.message, weight=self.weight[rel[1]]),
                fn.sum('m', 'h')
            )
        hg.multi_update_all(fns, 'sum')
        return {ntype : hg.nodes[ntype].data['h'] for ntype in hg.ntypes}

    def message(self, edges, weight):
        return {'m' : edges.src['h'] @ weight}


def main(args):
    iters = 20

    in_feat = 16 * args.in_feat_scale
    out_feat = 16 * args.out_feat_scale

    dev = "cuda:0"
    torch.cuda.set_device(dev)

    # load graph data
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()

    g = dgl.to_homogeneous(dataset[0]).to(dev)
    etypes = g.edata[dgl.ETYPE].long().to(dev)
    num_rels = len(dataset[0].etypes)
    E_per_rel = torch.histogram(etypes.float().cpu(), bins=num_rels).hist.long()

    print(f"""Dataset: {args.dataset}
    num_nodes: {g.num_nodes()}
    num_edges: {g.num_edges()}
    num_rels: {num_rels}
    in_feat: {in_feat}
    out_feat: {out_feat}
    """)

    feat = torch.randn(g.num_nodes(), in_feat).to(dev)
    weight = torch.randn(num_rels, in_feat, out_feat).to(dev)

    # **** low-mem ******
    conv = RGCNLowMem(weight).to(dev)
    # dry run
    for i in range(3):
        h = conv(g, feat, etypes)
    torch.cuda.synchronize()
    # test
    with Timer(dev) as t:
        for i in range(iters):
            h_lowmem = conv(g, feat, etypes)
    print("low-mem rgcn:", t.elapsed_secs / iters * 1000, "ms")

    # **** high-mem ******
    conv = RGCNHighMem(weight).to(dev)
    # dry run
    for i in range(3):
        h = conv(g, feat, etypes)
    torch.cuda.synchronize()
    # test
    with Timer(dev) as t:
        for i in range(iters):
            h_highmem = conv(g, feat, etypes)
    print("high-mem rgcn:", t.elapsed_secs / iters * 1000, "ms")

    # **** hetero ****
    hg = dataset[0].to(dev)
    conv = RGCNHetero(weight, hg.etypes).to(dev)
    feat_dict = {ntype : torch.randn(hg.num_nodes(ntype), in_feat).to(dev) for ntype in hg.ntypes}
    # dry run
    for i in range(3):
        h_dict = conv(hg, feat_dict)
    torch.cuda.synchronize()
    # test
    with Timer(dev) as t:
        for i in range(iters):
            h_dict = conv(hg, feat_dict)
    print("hetero rgcn:", t.elapsed_secs / iters * 1000, "ms")

    # **** gather_mm sorted ****
    conv = RGCNGatherMMSorted(weight).to(dev)
    # dry run
    for i in range(3):
        h = conv(g, feat, etypes, E_per_rel)
    torch.cuda.synchronize()
    # test
    with Timer(dev) as t:
        for i in range(iters):
            h_gmm_sorted = conv(g, feat, etypes, E_per_rel)
    print("gather_mm_sorted rgcn:", t.elapsed_secs / iters * 1000, "ms")

    # **** gather_mm unsorted ****
    conv = RGCNGatherMMUnsorted(weight).to(dev)
    # dry run
    for i in range(3):
        h = conv(g, feat, etypes, E_per_rel)
    torch.cuda.synchronize()
    # test
    with Timer(dev) as t:
        for i in range(iters):
            h_gmm_unsorted = conv(g, feat, etypes, E_per_rel)
    print("gather_mm_unsorted rgcn:", t.elapsed_secs / iters * 1000, "ms")

    # **** correctness ****
    # assert torch.allclose(h_lowmem, h_highmem, atol=1e-3, rtol=1e-3)
    assert torch.allclose(h_lowmem, h_gmm_sorted, atol=1e-3, rtol=1e-3)
    assert torch.allclose(h_lowmem, h_gmm_unsorted, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='e2e')
    parser.add_argument("-i", "--in_feat_scale", type=int, default=1,
            help="scale input feature length")
    parser.add_argument("-o", "--out_feat_scale", type=int, default=1,
            help="scale output feature length")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    args = parser.parse_args()
    print(args)
    main(args)
