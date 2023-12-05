import gzip
import io
import os
import tarfile
import tempfile
import unittest

import backend as F

import dgl
import dgl.data as data
import numpy as np
import pandas as pd
import pytest
import yaml
from dgl import DGLError


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Skip MXNet")
def test_reddit():
    # RedditDataset
    g = data.RedditDataset()[0]
    assert g.num_nodes() == 232965
    assert g.num_edges() == 114615892
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))

    transform = dgl.AddSelfLoop(allow_duplicate=True)
    g2 = data.RedditDataset(transform=transform)[0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Skip MXNet")
def test_fakenews():
    transform = dgl.AddSelfLoop(allow_duplicate=True)

    ds = data.FakeNewsDataset("politifact", "bert")
    assert len(ds) == 314
    g = ds[0][0]
    g2 = data.FakeNewsDataset("politifact", "bert", transform=transform)[0][0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()

    ds = data.FakeNewsDataset("gossipcop", "profile")
    assert len(ds) == 5464
    g = ds[0][0]
    g2 = data.FakeNewsDataset("gossipcop", "profile", transform=transform)[0][0]
    assert g2.num_edges() - g.num_edges() == g.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_peptides_structural():
    transform = dgl.AddSelfLoop(allow_duplicate=True)
    dataset1 = data.PeptidesStructuralDataset()
    g1 = dataset1[0][0]
    dataset2 = data.PeptidesStructuralDataset(transform=transform)
    g2 = dataset2[0][0]

    assert g2.num_edges() - g1.num_edges() == g1.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_peptides_functional():
    transform = dgl.AddSelfLoop(allow_duplicate=True)
    dataset1 = data.PeptidesFunctionalDataset()
    g1, label = dataset1[0]
    dataset2 = data.PeptidesFunctionalDataset(transform=transform)
    g2, _ = dataset2[0]

    assert g2.num_edges() - g1.num_edges() == g1.num_nodes()
    assert dataset1.num_classes == label.shape[0]


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_VOC_superpixels():
    transform = dgl.AddSelfLoop(allow_duplicate=True)
    dataset1 = data.VOCSuperpixelsDataset()
    g1 = dataset1[0]
    dataset2 = data.VOCSuperpixelsDataset(transform=transform)
    g2 = dataset2[0]

    assert g2.num_edges() - g1.num_edges() == g1.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_COCO_superpixels():
    transform = dgl.AddSelfLoop(allow_duplicate=True)
    dataset1 = data.COCOSuperpixelsDataset()
    g1 = dataset1[0]
    dataset2 = data.COCOSuperpixelsDataset(transform=transform)
    g2 = dataset2[0]

    assert g2.num_edges() - g1.num_edges() == g1.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_MNIST_SuperPixel():
    transform = dgl.AddSelfLoop(allow_duplicate=True)
    dataset1 = data.MNISTSuperPixelDataset()
    g1, _ = dataset1[0]
    dataset2 = data.MNISTSuperPixelDataset(transform=transform)
    g2, _ = dataset2[0]

    assert g2.num_edges() - g1.num_edges() == g1.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="only supports pytorch"
)
def test_CIFAR10_SuperPixel():
    transform = dgl.AddSelfLoop(allow_duplicate=True)
    dataset1 = data.CIFAR10SuperPixelDataset()
    g1, _ = dataset1[0]
    dataset2 = data.CIFAR10SuperPixelDataset(transform=transform)
    g2, _ = dataset2[0]

    assert g2.num_edges() - g1.num_edges() == g1.num_nodes()


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Skip MXNet")
def test_as_graphpred():
    ds = data.GINDataset(name="MUTAG", self_loop=True)
    new_ds = data.AsGraphPredDataset(ds, [0.8, 0.1, 0.1], verbose=True)
    assert len(new_ds) == 188
    assert new_ds.num_tasks == 1
    assert new_ds.num_classes == 2

    ds = data.FakeNewsDataset("politifact", "profile")
    new_ds = data.AsGraphPredDataset(ds, verbose=True)
    assert len(new_ds) == 314
    assert new_ds.num_tasks == 1
    assert new_ds.num_classes == 2

    ds = data.QM7bDataset()
    new_ds = data.AsGraphPredDataset(ds, [0.8, 0.1, 0.1], verbose=True)
    assert len(new_ds) == 7211
    assert new_ds.num_tasks == 14
    assert new_ds.num_classes is None

    ds = data.QM9Dataset(label_keys=["mu", "gap"])
    new_ds = data.AsGraphPredDataset(ds, [0.8, 0.1, 0.1], verbose=True)
    assert len(new_ds) == 130831
    assert new_ds.num_tasks == 2
    assert new_ds.num_classes is None

    ds = data.QM9EdgeDataset(label_keys=["mu", "alpha"])
    new_ds = data.AsGraphPredDataset(ds, [0.8, 0.1, 0.1], verbose=True)
    assert len(new_ds) == 130831
    assert new_ds.num_tasks == 2
    assert new_ds.num_classes is None

    ds = data.TUDataset("DD")
    new_ds = data.AsGraphPredDataset(ds, [0.8, 0.1, 0.1], verbose=True)
    assert len(new_ds) == 1178
    assert new_ds.num_tasks == 1
    assert new_ds.num_classes == 2

    ds = data.LegacyTUDataset("DD")
    new_ds = data.AsGraphPredDataset(ds, [0.8, 0.1, 0.1], verbose=True)
    assert len(new_ds) == 1178
    assert new_ds.num_tasks == 1
    assert new_ds.num_classes == 2

    ds = data.BA2MotifDataset()
    new_ds = data.AsGraphPredDataset(ds, [0.8, 0.1, 0.1], verbose=True)
    assert len(new_ds) == 1000
    assert new_ds.num_tasks == 1
    assert new_ds.num_classes == 2


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="ogb only supports pytorch"
)
def test_as_linkpred_ogb():
    from ogb.linkproppred import DglLinkPropPredDataset

    ds = data.AsLinkPredDataset(
        DglLinkPropPredDataset("ogbl-collab"), split_ratio=None, verbose=True
    )
    # original dataset has 46329 test edges
    assert ds.test_edges[0][0].shape[0] == 46329
    # force generate new split
    ds = data.AsLinkPredDataset(
        DglLinkPropPredDataset("ogbl-collab"),
        split_ratio=[0.7, 0.2, 0.1],
        verbose=True,
    )
    assert ds.test_edges[0][0].shape[0] == 235812


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="ogb only supports pytorch"
)
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Skip MXNet")
def test_as_nodepred_ogb():
    from ogb.nodeproppred import DglNodePropPredDataset

    ds = data.AsNodePredDataset(
        DglNodePropPredDataset("ogbn-arxiv"), split_ratio=None, verbose=True
    )
    split = DglNodePropPredDataset("ogbn-arxiv").get_idx_split()
    train_idx, val_idx, test_idx = split["train"], split["valid"], split["test"]
    assert F.array_equal(ds.train_idx, F.tensor(train_idx))
    assert F.array_equal(ds.val_idx, F.tensor(val_idx))
    assert F.array_equal(ds.test_idx, F.tensor(test_idx))
    # force generate new split
    ds = data.AsNodePredDataset(
        DglNodePropPredDataset("ogbn-arxiv"),
        split_ratio=[0.7, 0.2, 0.1],
        verbose=True,
    )


@unittest.skipIf(
    dgl.backend.backend_name != "pytorch", reason="ogb only supports pytorch"
)
def test_as_graphpred_ogb():
    from ogb.graphproppred import DglGraphPropPredDataset

    ds = data.AsGraphPredDataset(
        DglGraphPropPredDataset("ogbg-molhiv"), split_ratio=None, verbose=True
    )
    assert len(ds.train_idx) == 32901
    # force generate new split
    ds = data.AsGraphPredDataset(
        DglGraphPropPredDataset("ogbg-molhiv"),
        split_ratio=[0.6, 0.2, 0.2],
        verbose=True,
    )
    assert len(ds.train_idx) == 24676
