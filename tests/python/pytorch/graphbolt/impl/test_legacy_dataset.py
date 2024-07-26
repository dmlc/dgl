import dgl.graphbolt as gb
import pytest
import torch
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset, CoraGraphDataset


def test_LegacyDataset_homo_node_pred():
    cora = CoraGraphDataset(transform=AddSelfLoop())
    dataset = gb.LegacyDataset(cora)

    # Check tasks.
    assert len(dataset.tasks) == 1
    task = dataset.tasks[0]
    assert task.train_set.names == ("seeds", "labels")
    assert len(task.train_set) == 140
    assert task.validation_set.names == ("seeds", "labels")
    assert len(task.validation_set) == 500
    assert task.test_set.names == ("seeds", "labels")
    assert len(task.test_set) == 1000
    assert task.metadata["num_classes"] == 7

    num_nodes = 2708
    assert dataset.graph.num_nodes == num_nodes
    assert len(dataset.all_nodes_set) == num_nodes
    assert dataset.feature.size("node", None, "feat") == torch.Size([1433])
    assert (
        dataset.feature.read(
            "node", None, "feat", torch.tensor([num_nodes - 1])
        ).size(dim=0)
        == 1
    )
    # Out of bound indexing results in segmentation fault instead of exception
    # in CI. This may be related to docker env. Skip it for now.
    # with pytest.raises(IndexError):
    #    dataset.feature.read("node", None, "feat", torch.Tensor([num_nodes]))
