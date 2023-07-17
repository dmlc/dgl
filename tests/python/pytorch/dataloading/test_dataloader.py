import os
import unittest
from collections.abc import Iterator, Mapping
from functools import partial

import backend as F

import dgl
import dgl.ops as OPS
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import parametrize_idtype


@pytest.mark.parametrize("batch_size", [None, 16])
def test_graph_dataloader(batch_size):
    num_batches = 2
    num_samples = num_batches * (batch_size if batch_size is not None else 1)
    minigc_dataset = dgl.data.MiniGCDataset(num_samples, 10, 20)
    data_loader = dgl.dataloading.GraphDataLoader(
        minigc_dataset, batch_size=batch_size, shuffle=True
    )
    assert isinstance(iter(data_loader), Iterator)
    for graph, label in data_loader:
        assert isinstance(graph, dgl.DGLGraph)
        if batch_size is not None:
            assert F.asnumpy(label).shape[0] == batch_size
        else:
            # If batch size is None, the label element will be a single scalar following
            # PyTorch's practice.
            assert F.asnumpy(label).ndim == 0


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@pytest.mark.parametrize("num_workers", [0, 4])
def test_cluster_gcn(num_workers):
    dataset = dgl.data.CoraFullDataset()
    g = dataset[0]
    sampler = dgl.dataloading.ClusterGCNSampler(g, 100)
    dataloader = dgl.dataloading.DataLoader(
        g, torch.arange(100), sampler, batch_size=4, num_workers=num_workers
    )
    assert len(dataloader) == 25
    for i, sg in enumerate(dataloader):
        pass


@pytest.mark.parametrize("num_workers", [0, 4])
def test_shadow(num_workers):
    g = dgl.data.CoraFullDataset()[0]
    sampler = dgl.dataloading.ShaDowKHopSampler([5, 10, 15])
    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(g.num_nodes()),
        sampler,
        batch_size=5,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    for i, (input_nodes, output_nodes, subgraph) in enumerate(dataloader):
        assert torch.equal(input_nodes, subgraph.ndata[dgl.NID])
        assert torch.equal(input_nodes[: output_nodes.shape[0]], output_nodes)
        assert torch.equal(
            subgraph.ndata["label"], g.ndata["label"][input_nodes]
        )
        assert torch.equal(subgraph.ndata["feat"], g.ndata["feat"][input_nodes])
        if i == 5:
            break


@pytest.mark.parametrize("num_workers", [0, 4])
@pytest.mark.parametrize("mode", ["node", "edge", "walk"])
def test_saint(num_workers, mode):
    g = dgl.data.CoraFullDataset()[0]

    if mode == "node":
        budget = 100
    elif mode == "edge":
        budget = 200
    elif mode == "walk":
        budget = (3, 2)

    sampler = dgl.dataloading.SAINTSampler(mode, budget)
    dataloader = dgl.dataloading.DataLoader(
        g, torch.arange(100), sampler, num_workers=num_workers
    )
    assert len(dataloader) == 100
    for sg in dataloader:
        pass


@parametrize_idtype
@pytest.mark.parametrize(
    "mode", ["cpu", "uva_cuda_indices", "uva_cpu_indices", "pure_gpu"]
)
@pytest.mark.parametrize("use_ddp", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_neighbor_nonuniform(idtype, mode, use_ddp, use_mask):
    if mode != "cpu" and F.ctx() == F.cpu():
        pytest.skip("UVA and GPU sampling require a GPU.")
    if mode != "cpu" and use_mask:
        pytest.skip("Masked sampling only works on CPU.")
    if use_ddp:
        if os.name == "nt":
            pytest.skip("PyTorch 1.13.0+ has problems in Windows DDP...")
        dist.init_process_group(
            "gloo" if F.ctx() == F.cpu() else "nccl",
            "tcp://127.0.0.1:12347",
            world_size=1,
            rank=0,
        )
    g = dgl.graph(([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 1, 1, 1, 1])).astype(
        idtype
    )
    g.edata["p"] = torch.FloatTensor([1, 1, 0, 0, 1, 1, 0, 0])
    g.edata["mask"] = g.edata["p"] != 0
    if mode in ("cpu", "uva_cpu_indices"):
        indices = F.copy_to(F.tensor([0, 1], idtype), F.cpu())
    else:
        indices = F.copy_to(F.tensor([0, 1], idtype), F.cuda())
    if mode == "pure_gpu":
        g = g.to(F.cuda())
    use_uva = mode.startswith("uva")

    if use_mask:
        prob, mask = None, "mask"
    else:
        prob, mask = "p", None

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [2], prob=prob, mask=mask
    )
    for num_workers in [0, 1, 2] if mode == "cpu" else [0]:
        dataloader = dgl.dataloading.DataLoader(
            g,
            indices,
            sampler,
            batch_size=1,
            device=F.ctx(),
            num_workers=num_workers,
            use_uva=use_uva,
            use_ddp=use_ddp,
        )
        for input_nodes, output_nodes, blocks in dataloader:
            seed = output_nodes.item()
            neighbors = set(input_nodes[1:].cpu().numpy())
            if seed == 1:
                assert neighbors == {5, 6}
            elif seed == 0:
                assert neighbors == {1, 2}

    g = dgl.heterograph(
        {
            ("B", "BA", "A"): (
                [1, 2, 3, 4, 5, 6, 7, 8],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ),
            ("C", "CA", "A"): (
                [1, 2, 3, 4, 5, 6, 7, 8],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ),
        }
    ).astype(idtype)
    g.edges["BA"].data["p"] = torch.FloatTensor([1, 1, 0, 0, 1, 1, 0, 0])
    g.edges["BA"].data["mask"] = g.edges["BA"].data["p"] != 0
    g.edges["CA"].data["p"] = torch.FloatTensor([0, 0, 1, 1, 0, 0, 1, 1])
    g.edges["CA"].data["mask"] = g.edges["CA"].data["p"] != 0
    if mode == "pure_gpu":
        g = g.to(F.cuda())
    for num_workers in [0, 1, 2] if mode == "cpu" else [0]:
        dataloader = dgl.dataloading.DataLoader(
            g,
            {"A": indices},
            sampler,
            batch_size=1,
            device=F.ctx(),
            num_workers=num_workers,
            use_uva=use_uva,
            use_ddp=use_ddp,
        )
        for input_nodes, output_nodes, blocks in dataloader:
            seed = output_nodes["A"].item()
            # Seed and neighbors are of different node types so slicing is not necessary here.
            neighbors = set(input_nodes["B"].cpu().numpy())
            if seed == 1:
                assert neighbors == {5, 6}
            elif seed == 0:
                assert neighbors == {1, 2}

            neighbors = set(input_nodes["C"].cpu().numpy())
            if seed == 1:
                assert neighbors == {7, 8}
            elif seed == 0:
                assert neighbors == {3, 4}

    if use_ddp:
        dist.destroy_process_group()


def _check_dtype(data, dtype, attr_name):
    if isinstance(data, dict):
        for k, v in data.items():
            assert getattr(v, attr_name) == dtype
    elif isinstance(data, list):
        for v in data:
            assert getattr(v, attr_name) == dtype
    else:
        assert getattr(data, attr_name) == dtype


def _check_device(data):
    if isinstance(data, dict):
        for k, v in data.items():
            assert v.device == F.ctx()
    elif isinstance(data, list):
        for v in data:
            assert v.device == F.ctx()
    else:
        assert data.device == F.ctx()


@pytest.mark.parametrize("sampler_name", ["full", "neighbor"])
@pytest.mark.parametrize(
    "mode", ["cpu", "uva_cuda_indices", "uva_cpu_indices", "pure_gpu"]
)
@pytest.mark.parametrize("nprocs", [1, 4])
@pytest.mark.parametrize("drop_last", [True, False])
def test_ddp_dataloader_decompose_dataset(
    sampler_name, mode, nprocs, drop_last
):
    if torch.cuda.device_count() < nprocs and mode != "cpu":
        pytest.skip(
            "DDP dataloader needs sufficient GPUs for UVA and GPU sampling."
        )
    if mode != "cpu" and F.ctx() == F.cpu():
        pytest.skip("UVA and GPU sampling require a GPU.")

    if os.name == "nt":
        pytest.skip("PyTorch 1.13.0+ has problems in Windows DDP...")
    g, _, _, _ = _create_homogeneous()
    g = g.to(F.cpu())

    sampler = {
        "full": dgl.dataloading.MultiLayerFullNeighborSampler(2),
        "neighbor": dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
    }[sampler_name]
    indices = F.copy_to(F.arange(0, g.num_nodes()), F.cpu())
    data = indices, sampler
    arguments = mode, drop_last
    g.create_formats_()
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    mp.spawn(_ddp_runner, args=(nprocs, g, data, arguments), nprocs=nprocs)


def _ddp_runner(proc_id, nprocs, g, data, args):
    mode, drop_last = args
    indices, sampler = data
    if mode == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(proc_id)
        torch.cuda.set_device(device)
    if mode == "pure_gpu":
        g = g.to(F.cuda())
    if mode in ("cpu", "uva_cpu_indices"):
        indices = indices.cpu()
    else:
        indices = indices.cuda()

    dist.init_process_group(
        "nccl" if mode != "cpu" else "gloo",
        "tcp://127.0.0.1:12347",
        world_size=nprocs,
        rank=proc_id,
    )
    use_uva = mode.startswith("uva")
    batch_size = g.num_nodes()
    shuffle = False
    for num_workers in [1, 4] if mode == "cpu" else [0]:
        dataloader = dgl.dataloading.DataLoader(
            g,
            indices,
            sampler,
            device=device,
            batch_size=batch_size,  # g1.num_nodes(),
            num_workers=num_workers,
            use_uva=use_uva,
            use_ddp=True,
            drop_last=drop_last,
            shuffle=shuffle,
        )
        max_nid = [0]
        for i, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            block = blocks[-1]
            o_src, o_dst = block.edges()
            src_nodes_id = block.srcdata[dgl.NID][o_src]
            dst_nodes_id = block.dstdata[dgl.NID][o_dst]
            max_nid.append(np.max(dst_nodes_id.cpu().numpy()))

        local_max = torch.tensor(np.max(max_nid))
        if torch.distributed.get_backend() == "nccl":
            local_max = local_max.cuda()
        dist.reduce(local_max, 0, op=dist.ReduceOp.MAX)
        if proc_id == 0:
            if drop_last and not shuffle and local_max > 0:
                assert (
                    local_max.item()
                    == len(indices)
                    - len(indices) % nprocs
                    - 1
                    - (len(indices) // nprocs) % batch_size
                )
            elif not drop_last:
                assert local_max == len(indices) - 1
    dist.destroy_process_group()


@parametrize_idtype
@pytest.mark.parametrize(
    "sampler_name", ["full", "neighbor", "neighbor2", "labor"]
)
@pytest.mark.parametrize(
    "mode", ["cpu", "uva_cuda_indices", "uva_cpu_indices", "pure_gpu"]
)
@pytest.mark.parametrize("use_ddp", [False, True])
def test_node_dataloader(idtype, sampler_name, mode, use_ddp):
    if mode != "cpu" and F.ctx() == F.cpu():
        pytest.skip("UVA and GPU sampling require a GPU.")
    if use_ddp:
        if os.name == "nt":
            pytest.skip("PyTorch 1.13.0+ has problems in Windows DDP...")
        dist.init_process_group(
            "gloo" if F.ctx() == F.cpu() else "nccl",
            "tcp://127.0.0.1:12347",
            world_size=1,
            rank=0,
        )
    g1 = dgl.graph(([0, 0, 0, 1, 1], [1, 2, 3, 3, 4])).astype(idtype)
    g1.ndata["feat"] = F.copy_to(F.randn((5, 8)), F.cpu())
    g1.ndata["label"] = F.copy_to(F.randn((g1.num_nodes(),)), F.cpu())
    if mode in ("cpu", "uva_cpu_indices"):
        indices = F.copy_to(F.arange(0, g1.num_nodes(), idtype), F.cpu())
    else:
        indices = F.copy_to(F.arange(0, g1.num_nodes(), idtype), F.cuda())
    if mode == "pure_gpu":
        g1 = g1.to(F.cuda())

    use_uva = mode.startswith("uva")

    sampler = {
        "full": dgl.dataloading.MultiLayerFullNeighborSampler(2),
        "neighbor": dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
        "neighbor2": dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
        "labor": dgl.dataloading.LaborSampler([3, 3]),
    }[sampler_name]
    for num_workers in [0, 1, 2] if mode == "cpu" else [0]:
        dataloader = dgl.dataloading.DataLoader(
            g1,
            indices,
            sampler,
            device=F.ctx(),
            batch_size=g1.num_nodes(),
            num_workers=num_workers,
            use_uva=use_uva,
            use_ddp=use_ddp,
        )
        for input_nodes, output_nodes, blocks in dataloader:
            _check_device(input_nodes)
            _check_device(output_nodes)
            _check_device(blocks)
            _check_dtype(input_nodes, idtype, "dtype")
            _check_dtype(output_nodes, idtype, "dtype")
            _check_dtype(blocks, idtype, "idtype")

    g2 = dgl.heterograph(
        {
            ("user", "follow", "user"): (
                [0, 0, 0, 1, 1, 1, 2],
                [1, 2, 3, 0, 2, 3, 0],
            ),
            ("user", "followed-by", "user"): (
                [1, 2, 3, 0, 2, 3, 0],
                [0, 0, 0, 1, 1, 1, 2],
            ),
            ("user", "play", "game"): ([0, 1, 1, 3, 5], [0, 1, 2, 0, 2]),
            ("game", "played-by", "user"): ([0, 1, 2, 0, 2], [0, 1, 1, 3, 5]),
        }
    ).astype(idtype)
    for ntype in g2.ntypes:
        g2.nodes[ntype].data["feat"] = F.copy_to(
            F.randn((g2.num_nodes(ntype), 8)), F.cpu()
        )
    if mode in ("cpu", "uva_cpu_indices"):
        indices = {nty: F.copy_to(g2.nodes(nty), F.cpu()) for nty in g2.ntypes}
    else:
        indices = {nty: F.copy_to(g2.nodes(nty), F.cuda()) for nty in g2.ntypes}
    if mode == "pure_gpu":
        g2 = g2.to(F.cuda())

    batch_size = max(g2.num_nodes(nty) for nty in g2.ntypes)
    sampler = {
        "full": dgl.dataloading.MultiLayerFullNeighborSampler(2),
        "neighbor": dgl.dataloading.MultiLayerNeighborSampler(
            [{etype: 3 for etype in g2.etypes}] * 2
        ),
        "neighbor2": dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
        "labor": dgl.dataloading.LaborSampler([3, 3]),
    }[sampler_name]
    for num_workers in [0, 1, 2] if mode == "cpu" else [0]:
        dataloader = dgl.dataloading.DataLoader(
            g2,
            indices,
            sampler,
            device=F.ctx(),
            batch_size=batch_size,
            num_workers=num_workers,
            use_uva=use_uva,
            use_ddp=use_ddp,
        )
        assert isinstance(iter(dataloader), Iterator)
        for input_nodes, output_nodes, blocks in dataloader:
            _check_device(input_nodes)
            _check_device(output_nodes)
            _check_device(blocks)
            _check_dtype(input_nodes, idtype, "dtype")
            _check_dtype(output_nodes, idtype, "dtype")
            _check_dtype(blocks, idtype, "idtype")

    if use_ddp:
        dist.destroy_process_group()


@parametrize_idtype
@pytest.mark.parametrize("sampler_name", ["full", "neighbor"])
@pytest.mark.parametrize(
    "neg_sampler",
    [
        dgl.dataloading.negative_sampler.Uniform(2),
        dgl.dataloading.negative_sampler.GlobalUniform(15, False, 3),
        dgl.dataloading.negative_sampler.GlobalUniform(15, True, 3),
    ],
)
@pytest.mark.parametrize("mode", ["cpu", "uva", "pure_gpu"])
@pytest.mark.parametrize("use_ddp", [False, True])
def test_edge_dataloader(idtype, sampler_name, neg_sampler, mode, use_ddp):
    if mode != "cpu" and F.ctx() == F.cpu():
        pytest.skip("UVA and GPU sampling require a GPU.")
    if mode == "uva" and isinstance(
        neg_sampler, dgl.dataloading.negative_sampler.GlobalUniform
    ):
        pytest.skip("GlobalUniform don't support UVA yet.")
    if use_ddp:
        if os.name == "nt":
            pytest.skip("PyTorch 1.13.0+ has problems in Windows DDP...")
        dist.init_process_group(
            "gloo" if F.ctx() == F.cpu() else "nccl",
            "tcp://127.0.0.1:12347",
            world_size=1,
            rank=0,
        )
    g1 = dgl.graph(([0, 0, 0, 1, 1], [1, 2, 3, 3, 4])).astype(idtype)
    g1.ndata["feat"] = F.copy_to(F.randn((5, 8)), F.cpu())
    if mode == "pure_gpu":
        g1 = g1.to(F.cuda())

    sampler = {
        "full": dgl.dataloading.MultiLayerFullNeighborSampler(2),
        "neighbor": dgl.dataloading.MultiLayerNeighborSampler([3, 3]),
    }[sampler_name]

    # no negative sampler
    edge_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    dataloader = dgl.dataloading.DataLoader(
        g1,
        g1.edges(form="eid"),
        edge_sampler,
        device=F.ctx(),
        batch_size=g1.num_edges(),
        use_uva=(mode == "uva"),
        use_ddp=use_ddp,
    )
    for input_nodes, pos_pair_graph, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(pos_pair_graph)
        _check_device(blocks)

    # negative sampler
    edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=neg_sampler
    )
    dataloader = dgl.dataloading.DataLoader(
        g1,
        g1.edges(form="eid"),
        edge_sampler,
        device=F.ctx(),
        batch_size=g1.num_edges(),
        use_uva=(mode == "uva"),
        use_ddp=use_ddp,
    )
    for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(pos_pair_graph)
        _check_device(neg_pair_graph)
        _check_device(blocks)

    g2 = dgl.heterograph(
        {
            ("user", "follow", "user"): (
                [0, 0, 0, 1, 1, 1, 2],
                [1, 2, 3, 0, 2, 3, 0],
            ),
            ("user", "followed-by", "user"): (
                [1, 2, 3, 0, 2, 3, 0],
                [0, 0, 0, 1, 1, 1, 2],
            ),
            ("user", "play", "game"): ([0, 1, 1, 3, 5], [0, 1, 2, 0, 2]),
            ("game", "played-by", "user"): ([0, 1, 2, 0, 2], [0, 1, 1, 3, 5]),
        }
    ).astype(idtype)
    for ntype in g2.ntypes:
        g2.nodes[ntype].data["feat"] = F.copy_to(
            F.randn((g2.num_nodes(ntype), 8)), F.cpu()
        )
    if mode == "pure_gpu":
        g2 = g2.to(F.cuda())

    batch_size = max(g2.num_edges(ety) for ety in g2.canonical_etypes)
    sampler = {
        "full": dgl.dataloading.MultiLayerFullNeighborSampler(2),
        "neighbor": dgl.dataloading.MultiLayerNeighborSampler(
            [{etype: 3 for etype in g2.etypes}] * 2
        ),
    }[sampler_name]

    # no negative sampler
    edge_sampler = dgl.dataloading.as_edge_prediction_sampler(sampler)
    dataloader = dgl.dataloading.DataLoader(
        g2,
        {ety: g2.edges(form="eid", etype=ety) for ety in g2.canonical_etypes},
        edge_sampler,
        device=F.ctx(),
        batch_size=batch_size,
        use_uva=(mode == "uva"),
        use_ddp=use_ddp,
    )
    for input_nodes, pos_pair_graph, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(pos_pair_graph)
        _check_device(blocks)

    # negative sampler
    edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=neg_sampler
    )
    dataloader = dgl.dataloading.DataLoader(
        g2,
        {ety: g2.edges(form="eid", etype=ety) for ety in g2.canonical_etypes},
        edge_sampler,
        device=F.ctx(),
        batch_size=batch_size,
        use_uva=(mode == "uva"),
        use_ddp=use_ddp,
    )

    assert isinstance(iter(dataloader), Iterator)
    for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
        _check_device(input_nodes)
        _check_device(pos_pair_graph)
        _check_device(neg_pair_graph)
        _check_device(blocks)

    if use_ddp:
        dist.destroy_process_group()


def _create_homogeneous():
    s = torch.randint(0, 200, (1000,), device=F.ctx())
    d = torch.randint(0, 200, (1000,), device=F.ctx())
    src = torch.cat([s, d])
    dst = torch.cat([d, s])
    g = dgl.graph((s, d), num_nodes=200)
    reverse_eids = torch.cat(
        [torch.arange(1000, 2000), torch.arange(0, 1000)]
    ).to(F.ctx())
    always_exclude = torch.randint(0, 1000, (50,), device=F.ctx())
    seed_edges = torch.arange(0, 1000, device=F.ctx())
    return g, reverse_eids, always_exclude, seed_edges


def _create_heterogeneous():
    edges = {}
    for utype, etype, vtype in [("A", "AA", "A"), ("A", "AB", "B")]:
        s = torch.randint(0, 200, (1000,), device=F.ctx())
        d = torch.randint(0, 200, (1000,), device=F.ctx())
        edges[utype, etype, vtype] = (s, d)
        edges[vtype, "rev-" + etype, utype] = (d, s)
    g = dgl.heterograph(edges, num_nodes_dict={"A": 200, "B": 200})
    reverse_etypes = {
        "AA": "rev-AA",
        "AB": "rev-AB",
        "rev-AA": "AA",
        "rev-AB": "AB",
    }
    always_exclude = {
        "AA": torch.randint(0, 1000, (50,), device=F.ctx()),
        "AB": torch.randint(0, 1000, (50,), device=F.ctx()),
    }
    seed_edges = {
        "AA": torch.arange(0, 1000, device=F.ctx()),
        "AB": torch.arange(0, 1000, device=F.ctx()),
    }
    return g, reverse_etypes, always_exclude, seed_edges


def _remove_duplicates(s, d):
    s, d = list(zip(*list(set(zip(s.tolist(), d.tolist())))))
    return torch.tensor(s, device=F.ctx()), torch.tensor(d, device=F.ctx())


def _find_edges_to_exclude(g, exclude, always_exclude, pair_eids):
    if exclude == None:
        return always_exclude
    elif exclude == "self":
        return (
            torch.cat([pair_eids, always_exclude])
            if always_exclude is not None
            else pair_eids
        )
    elif exclude == "reverse_id":
        pair_eids = torch.cat([pair_eids, pair_eids + 1000])
        return (
            torch.cat([pair_eids, always_exclude])
            if always_exclude is not None
            else pair_eids
        )
    elif exclude == "reverse_types":
        pair_eids = {g.to_canonical_etype(k): v for k, v in pair_eids.items()}
        if ("A", "AA", "A") in pair_eids:
            pair_eids[("A", "rev-AA", "A")] = pair_eids[("A", "AA", "A")]
        if ("A", "AB", "B") in pair_eids:
            pair_eids[("B", "rev-AB", "A")] = pair_eids[("A", "AB", "B")]
        if always_exclude is not None:
            always_exclude = {
                g.to_canonical_etype(k): v for k, v in always_exclude.items()
            }
            for k in always_exclude.keys():
                if k in pair_eids:
                    pair_eids[k] = torch.cat([pair_eids[k], always_exclude[k]])
                else:
                    pair_eids[k] = always_exclude[k]
        return pair_eids


@pytest.mark.parametrize("always_exclude_flag", [False, True])
@pytest.mark.parametrize(
    "exclude", [None, "self", "reverse_id", "reverse_types"]
)
@pytest.mark.parametrize(
    "sampler",
    [
        dgl.dataloading.MultiLayerFullNeighborSampler(1),
        dgl.dataloading.ShaDowKHopSampler([5]),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 50])
def test_edge_dataloader_excludes(
    exclude, always_exclude_flag, batch_size, sampler
):
    if exclude == "reverse_types":
        g, reverse_etypes, always_exclude, seed_edges = _create_heterogeneous()
    else:
        g, reverse_eids, always_exclude, seed_edges = _create_homogeneous()
    g = g.to(F.ctx())
    if not always_exclude_flag:
        always_exclude = None

    kwargs = {}
    kwargs["exclude"] = (
        partial(_find_edges_to_exclude, g, exclude, always_exclude)
        if always_exclude_flag
        else exclude
    )
    kwargs["reverse_eids"] = reverse_eids if exclude == "reverse_id" else None
    kwargs["reverse_etypes"] = (
        reverse_etypes if exclude == "reverse_types" else None
    )
    sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, **kwargs)

    dataloader = dgl.dataloading.DataLoader(
        g,
        seed_edges,
        sampler,
        batch_size=batch_size,
        device=F.ctx(),
        use_prefetch_thread=False,
    )
    for i, (input_nodes, pair_graph, blocks) in enumerate(dataloader):
        if isinstance(blocks, list):
            subg = blocks[0]
        else:
            subg = blocks
        pair_eids = pair_graph.edata[dgl.EID]
        block_eids = subg.edata[dgl.EID]

        edges_to_exclude = _find_edges_to_exclude(
            g, exclude, always_exclude, pair_eids
        )
        if edges_to_exclude is None:
            continue
        edges_to_exclude = dgl.utils.recursive_apply(
            edges_to_exclude, lambda x: x.cpu().numpy()
        )
        block_eids = dgl.utils.recursive_apply(
            block_eids, lambda x: x.cpu().numpy()
        )

        if isinstance(edges_to_exclude, Mapping):
            for k in edges_to_exclude.keys():
                assert not np.isin(edges_to_exclude[k], block_eids[k]).any()
        else:
            assert not np.isin(edges_to_exclude, block_eids).any()

        if i == 10:
            break


def test_edge_dataloader_exclusion_with_reverse_seed_nodes():
    utype, etype, vtype = ("A", "AB", "B")
    s = torch.randint(0, 20, (500,), device=F.ctx())
    d = torch.randint(0, 20, (500,), device=F.ctx())
    s, d = _remove_duplicates(s, d)
    g = dgl.heterograph({("A", "AB", "B"): (s, d), ("B", "BA", "A"): (d, s)})
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        dgl.dataloading.NeighborSampler(fanouts=[2, 2, 2]),
        exclude="reverse_types",
        reverse_etypes={"AB": "BA", "BA": "AB"},
    )
    seed_edges = {
        "AB": torch.arange(g.number_of_edges("AB"), device=F.ctx()),
        "BA": torch.arange(g.number_of_edges("BA"), device=F.ctx()),
    }
    dataloader = dgl.dataloading.DataLoader(
        g,
        seed_edges,
        sampler,
        batch_size=2,
        device=F.ctx(),
        shuffle=True,
        drop_last=False,
    )
    for _, pos_graph, mfgs in dataloader:
        s, d = pos_graph["AB"].edges()
        AB_pos = list(zip(s.tolist(), d.tolist()))
        s, d = pos_graph["BA"].edges()
        BA_pos = list(zip(s.tolist(), d.tolist()))

        s, d = mfgs[-1]["AB"].edges()
        AB_mfg = list(zip(s.tolist(), d.tolist()))
        s, d = mfgs[-1]["BA"].edges()
        BA_mfg = list(zip(s.tolist(), d.tolist()))

        assert all(edge not in AB_mfg for edge in AB_pos)
        assert all(edge not in BA_mfg for edge in BA_pos)


def test_edge_dataloader_exclusion_without_all_reverses():
    data_dict = {
        ("A", "AB", "B"): (torch.tensor([0, 1]), torch.tensor([0, 1])),
        ("B", "BA", "A"): (torch.tensor([0, 1]), torch.tensor([0, 1])),
        ("B", "BC", "C"): (torch.tensor([0]), torch.tensor([0])),
        ("C", "CA", "A"): (torch.tensor([0, 1]), torch.tensor([0, 1])),
    }
    g = dgl.heterograph(data_dict=data_dict)
    block_sampler = dgl.dataloading.MultiLayerNeighborSampler(
        fanouts=[1], replace=True
    )
    block_sampler = dgl.dataloading.as_edge_prediction_sampler(
        block_sampler,
        exclude="reverse_types",
        reverse_etypes={"AB": "BA"},
    )
    d = dgl.dataloading.DataLoader(
        graph=g,
        indices={
            "AB": torch.tensor([0]),
            "BC": torch.tensor([0]),
        },
        graph_sampler=block_sampler,
        batch_size=2,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        device=F.ctx(),
        use_ddp=False,
    )

    next(iter(d))


def dummy_worker_init_fn(worker_id):
    pass


def test_dataloader_worker_init_fn():
    dataset = dgl.data.CoraFullDataset()
    g = dataset[0]
    sampler = dgl.dataloading.MultiLayerNeighborSampler([2])
    dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(100),
        sampler,
        batch_size=4,
        num_workers=4,
        worker_init_fn=dummy_worker_init_fn,
    )
    for _ in dataloader:
        pass


if __name__ == "__main__":
    # test_node_dataloader(F.int32, 'neighbor', None)
    test_edge_dataloader_excludes(
        "reverse_types", False, 1, dgl.dataloading.ShaDowKHopSampler([5])
    )
    test_edge_dataloader_exclusion_without_all_reverses()
