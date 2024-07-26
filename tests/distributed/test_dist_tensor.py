import operator
import os
import unittest

import backend as F

import dgl
import pytest
from utils import create_random_graph, generate_ip_config, reset_envs

dist_g = None


def rand_mask(shape, dtype):
    return F.randn(shape) > 0


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support some of operations in DistGraph",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support"
)
def setup_module():
    global dist_g

    reset_envs()
    os.environ["DGL_DIST_MODE"] = "standalone"

    dist_g = create_random_graph(10000)
    # Partition the graph.
    num_parts = 1
    graph_name = "dist_graph_test_3"
    dist_g.ndata["features"] = F.unsqueeze(F.arange(0, dist_g.num_nodes()), 1)
    dist_g.edata["features"] = F.unsqueeze(F.arange(0, dist_g.num_edges()), 1)
    dgl.distributed.partition_graph(
        dist_g, graph_name, num_parts, "/tmp/dist_graph"
    )

    dgl.distributed.initialize("kv_ip_config.txt")
    dist_g = dgl.distributed.DistGraph(
        graph_name, part_config="/tmp/dist_graph/{}.json".format(graph_name)
    )
    dist_g.edata["mask1"] = dgl.distributed.DistTensor(
        (dist_g.num_edges(),), F.bool, init_func=rand_mask
    )
    dist_g.edata["mask2"] = dgl.distributed.DistTensor(
        (dist_g.num_edges(),), F.bool, init_func=rand_mask
    )


def check_binary_op(key1, key2, key3, op):
    for i in range(0, dist_g.num_edges(), 1000):
        i_end = min(i + 1000, dist_g.num_edges())
        assert F.array_equal(
            dist_g.edata[key3][i:i_end],
            op(dist_g.edata[key1][i:i_end], dist_g.edata[key2][i:i_end]),
        )
        # Test with different index dtypes. int32 is not supported.
        with pytest.raises(
            dgl.utils.internal.InconsistentDtypeException,
            match="DGL now requires the input tensor to have",
        ):
            _ = dist_g.edata[key3][F.tensor([100, 20, 10], F.int32)]
        _ = dist_g.edata[key3][F.tensor([100, 20, 10], F.int64)]


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support some of operations in DistGraph",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support"
)
def test_op():
    dist_g.edata["mask3"] = dist_g.edata["mask1"] | dist_g.edata["mask2"]
    check_binary_op("mask1", "mask2", "mask3", operator.or_)


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support some of operations in DistGraph",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support"
)
def teardown_module():
    # Since there are two tests in one process, this is needed to make sure
    # the client exits properly.
    dgl.distributed.exit_client()


if __name__ == "__main__":
    setup_module()
    test_op()
    teardown_module()
