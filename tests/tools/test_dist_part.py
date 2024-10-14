import json
import os
import tempfile

import dgl
import dgl.backend as F

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from dgl.data.utils import load_graphs, load_tensors
from dgl.distributed.partition import (
    _etype_tuple_to_str,
    _get_inner_edge_mask,
    _get_inner_node_mask,
    load_partition,
    RESERVED_FIELD_DTYPE,
)

from distpartitioning import array_readwriter
from distpartitioning.utils import generate_read_list
from pytest_utils import chunk_graph, create_chunked_dataset
from scipy import sparse as spsp

from tools.verification_utils import (
    verify_graph_feats,
    verify_partition_data_types,
    verify_partition_formats,
)


def _test_chunk_graph(
    num_chunks,
    data_fmt="numpy",
    edges_fmt="csv",
    vector_rows=False,
    num_chunks_nodes=None,
    num_chunks_edges=None,
    num_chunks_node_data=None,
    num_chunks_edge_data=None,
):
    with tempfile.TemporaryDirectory() as root_dir:
        g = create_chunked_dataset(
            root_dir,
            num_chunks,
            data_fmt=data_fmt,
            edges_fmt=edges_fmt,
            vector_rows=vector_rows,
            num_chunks_nodes=num_chunks_nodes,
            num_chunks_edges=num_chunks_edges,
            num_chunks_node_data=num_chunks_node_data,
            num_chunks_edge_data=num_chunks_edge_data,
        )

        # check metadata.json
        output_dir = os.path.join(root_dir, "chunked-data")
        json_file = os.path.join(output_dir, "metadata.json")
        assert os.path.isfile(json_file)
        with open(json_file, "rb") as f:
            meta_data = json.load(f)
        assert meta_data["graph_name"] == "mag240m"
        assert len(meta_data["num_nodes_per_chunk"][0]) == num_chunks

        # check edge_index
        output_edge_index_dir = os.path.join(output_dir, "edge_index")
        for c_etype in g.canonical_etypes:
            c_etype_str = _etype_tuple_to_str(c_etype)
            if num_chunks_edges is None:
                n_chunks = num_chunks
            else:
                n_chunks = num_chunks_edges
            for i in range(n_chunks):
                fname = os.path.join(
                    output_edge_index_dir, f"{c_etype_str}{i}.txt"
                )
                assert os.path.isfile(fname)
                if edges_fmt == "csv":
                    with open(fname, "r") as f:
                        header = f.readline()
                        num1, num2 = header.rstrip().split(" ")
                        assert isinstance(int(num1), int)
                        assert isinstance(int(num2), int)
                elif edges_fmt == "parquet":
                    metadata = pq.read_metadata(fname)
                    assert metadata.num_columns == 2
                else:
                    assert False, f"Invalid edges_fmt: {edges_fmt}"

        # check node/edge_data
        suffix = "npy" if data_fmt == "numpy" else "parquet"
        reader_fmt_meta = {"name": data_fmt}

        def test_data(sub_dir, feat, expected_data, expected_shape, num_chunks):
            data = []
            for i in range(num_chunks):
                fname = os.path.join(sub_dir, f"{feat}-{i}.{suffix}")
                assert os.path.isfile(fname), f"{fname} cannot be found."
                feat_array = array_readwriter.get_array_parser(
                    **reader_fmt_meta
                ).read(fname)
                assert feat_array.shape[0] == expected_shape
                data.append(feat_array)
            data = np.concatenate(data, 0)
            assert torch.equal(torch.from_numpy(data), expected_data)

        output_node_data_dir = os.path.join(output_dir, "node_data")
        for ntype in g.ntypes:
            sub_dir = os.path.join(output_node_data_dir, ntype)
            if isinstance(num_chunks_node_data, int):
                chunks_data = num_chunks_node_data
            elif isinstance(num_chunks_node_data, dict):
                chunks_data = num_chunks_node_data.get(ntype, num_chunks)
            else:
                chunks_data = num_chunks
            for feat, data in g.nodes[ntype].data.items():
                if isinstance(chunks_data, dict):
                    n_chunks = chunks_data.get(feat, num_chunks)
                else:
                    n_chunks = chunks_data
                test_data(
                    sub_dir,
                    feat,
                    data,
                    g.num_nodes(ntype) // n_chunks,
                    n_chunks,
                )

        output_edge_data_dir = os.path.join(output_dir, "edge_data")
        for c_etype in g.canonical_etypes:
            c_etype_str = _etype_tuple_to_str(c_etype)
            sub_dir = os.path.join(output_edge_data_dir, c_etype_str)
            if isinstance(num_chunks_edge_data, int):
                chunks_data = num_chunks_edge_data
            elif isinstance(num_chunks_edge_data, dict):
                chunks_data = num_chunks_edge_data.get(c_etype, num_chunks)
            else:
                chunks_data = num_chunks
            for feat, data in g.edges[c_etype].data.items():
                if isinstance(chunks_data, dict):
                    n_chunks = chunks_data.get(feat, num_chunks)
                else:
                    n_chunks = chunks_data
                test_data(
                    sub_dir,
                    feat,
                    data,
                    g.num_edges(c_etype) // n_chunks,
                    n_chunks,
                )


@pytest.mark.parametrize("num_chunks", [1, 8])
@pytest.mark.parametrize("data_fmt", ["numpy", "parquet"])
@pytest.mark.parametrize("edges_fmt", ["csv", "parquet"])
def test_chunk_graph_basics(num_chunks, data_fmt, edges_fmt):
    _test_chunk_graph(num_chunks, data_fmt=data_fmt, edges_fmt=edges_fmt)


@pytest.mark.parametrize("num_chunks", [1, 8])
@pytest.mark.parametrize("vector_rows", [True, False])
def test_chunk_graph_vector_rows(num_chunks, vector_rows):
    _test_chunk_graph(
        num_chunks,
        data_fmt="parquet",
        edges_fmt="parquet",
        vector_rows=vector_rows,
    )


@pytest.mark.parametrize(
    "num_chunks, "
    "num_chunks_nodes, "
    "num_chunks_edges, "
    "num_chunks_node_data, "
    "num_chunks_edge_data",
    [
        [1, None, None, None, None],
        [8, None, None, None, None],
        [4, 4, 4, 8, 12],
        [4, 4, 4, {"paper": 10}, {("author", "writes", "paper"): 24}],
        [
            4,
            4,
            4,
            {"paper": {"feat": 10}},
            {("author", "writes", "paper"): {"year": 24}},
        ],
    ],
)
def test_chunk_graph_arbitrary_chunks(
    num_chunks,
    num_chunks_nodes,
    num_chunks_edges,
    num_chunks_node_data,
    num_chunks_edge_data,
):
    _test_chunk_graph(
        num_chunks,
        num_chunks_nodes=num_chunks_nodes,
        num_chunks_edges=num_chunks_edges,
        num_chunks_node_data=num_chunks_node_data,
        num_chunks_edge_data=num_chunks_edge_data,
    )


def create_mini_chunked_dataset(
    root_dir,
    num_chunks,
    data_fmt,
    edges_fmt,
    vector_rows,
    few_entity="node",
    **kwargs,
):
    num_nodes = {"n1": 1000, "n2": 1010, "n3": 1020}
    etypes = [
        ("n1", "r1", "n2"),
        ("n2", "r1", "n1"),
        ("n1", "r2", "n3"),
        ("n2", "r3", "n3"),
    ]
    node_items = ["n1", "n2", "n3"]
    edges_coo = {}
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(
            num_nodes[src_ntype],
            num_nodes[dst_ntype],
            density=0.001,
            format="coo",
            random_state=100,
        )
        edges_coo[etype] = (arr.row, arr.col)
    edge_items = []
    if few_entity == "edge":
        edges_coo[("n1", "a0", "n2")] = (
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
        )
        edges_coo[("n1", "a1", "n3")] = (
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
        )
        edge_items.append(("n1", "a0", "n2"))
        edge_items.append(("n1", "a1", "n3"))
    elif few_entity == "node":
        edges_coo[("n1", "r_few", "n_few")] = (
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
        )
        edges_coo[("a0", "a01", "n_1")] = (
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
        )
        edge_items.append(("n1", "r_few", "n_few"))
        edge_items.append(("a0", "a01", "n_1"))
        node_items.append("n_few")
        node_items.append("n_1")
        num_nodes["n_few"] = 2
        num_nodes["n_1"] = 2
    g = dgl.heterograph(edges_coo)

    node_data = {}
    edge_data = {}
    # save feature
    input_dir = os.path.join(root_dir, "data_test")

    for ntype in node_items:
        os.makedirs(os.path.join(input_dir, ntype))
        feat = np.random.randn(num_nodes[ntype], 3)
        feat_path = os.path.join(input_dir, f"{ntype}/feat.npy")
        with open(feat_path, "wb") as f:
            np.save(f, feat)
        g.nodes[ntype].data["feat"] = torch.from_numpy(feat)
        node_data[ntype] = {"feat": feat_path}

    for etype in set(edge_items):
        os.makedirs(os.path.join(input_dir, etype[1]))
        num_edge = len(edges_coo[etype][0])
        feat = np.random.randn(num_edge, 4)
        feat_path = os.path.join(input_dir, f"{etype[1]}/feat.npy")
        with open(feat_path, "wb") as f:
            np.save(f, feat)
        g.edges[etype].data["feat"] = torch.from_numpy(feat)
        edge_data[etype] = {"feat": feat_path}

    output_dir = os.path.join(root_dir, "chunked-data")
    chunk_graph(
        g,
        "mag240m",
        node_data,
        edge_data,
        num_chunks=num_chunks,
        output_path=output_dir,
        data_fmt=data_fmt,
        edges_fmt=edges_fmt,
        vector_rows=vector_rows,
        **kwargs,
    )
    return g


def _test_pipeline(
    num_chunks,
    num_parts,
    world_size,
    graph_formats=None,
    data_fmt="numpy",
    num_chunks_nodes=None,
    num_chunks_edges=None,
    num_chunks_node_data=None,
    num_chunks_edge_data=None,
    use_verify_partitions=False,
):

    if num_parts % world_size != 0:
        # num_parts should be a multiple of world_size
        return

    with tempfile.TemporaryDirectory() as root_dir:
        g = create_chunked_dataset(
            root_dir,
            num_chunks,
            data_fmt=data_fmt,
            num_chunks_nodes=num_chunks_nodes,
            num_chunks_edges=num_chunks_edges,
            num_chunks_node_data=num_chunks_node_data,
            num_chunks_edge_data=num_chunks_edge_data,
        )

        # Step1: graph partition
        in_dir = os.path.join(root_dir, "chunked-data")
        output_dir = os.path.join(root_dir, "parted_data")
        os.system(
            "python3 tools/partition_algo/random_partition.py "
            "--in_dir {} --out_dir {} --num_partitions {}".format(
                in_dir, output_dir, num_parts
            )
        )
        for ntype in ["author", "institution", "paper"]:
            fname = os.path.join(output_dir, "{}.txt".format(ntype))
            with open(fname, "r") as f:
                header = f.readline().rstrip()
                assert isinstance(int(header), int)

        # Step2: data dispatch
        partition_dir = os.path.join(root_dir, "parted_data")
        out_dir = os.path.join(root_dir, "partitioned")
        ip_config = os.path.join(root_dir, "ip_config.txt")
        with open(ip_config, "w") as f:
            for i in range(world_size):
                f.write(f"127.0.0.{i + 1}\n")

        cmd = "python3 tools/dispatch_data.py"
        cmd += f" --in-dir {in_dir}"
        cmd += f" --partitions-dir {partition_dir}"
        cmd += f" --out-dir {out_dir}"
        cmd += f" --ip-config {ip_config}"
        cmd += " --ssh-port 22"
        cmd += " --process-group-timeout 60"
        cmd += " --save-orig-nids"
        cmd += " --save-orig-eids"
        cmd += f" --graph-formats {graph_formats}" if graph_formats else ""
        os.system(cmd)

        # check if verify_partitions.py is used for validation.
        if use_verify_partitions:
            cmd = "python3 tools/verify_partitions.py "
            cmd += f" --orig-dataset-dir {in_dir}"
            cmd += f" --part-graph {out_dir}"
            cmd += f" --partitions-dir {output_dir}"
            os.system(cmd)
            return

        # read original node/edge IDs
        def read_orig_ids(fname):
            orig_ids = {}
            for i in range(num_parts):
                ids_path = os.path.join(out_dir, f"part{i}", fname)
                part_ids = load_tensors(ids_path)
                for type, data in part_ids.items():
                    if type not in orig_ids:
                        orig_ids[type] = data
                    else:
                        orig_ids[type] = torch.cat((orig_ids[type], data))
            return orig_ids

        orig_nids = read_orig_ids("orig_nids.dgl")
        orig_eids = read_orig_ids("orig_eids.dgl")

        # load partitions and verify
        part_config = os.path.join(out_dir, "metadata.json")
        for i in range(num_parts):
            part_g, node_feats, edge_feats, gpb, _, _, _ = load_partition(
                part_config, i
            )
            verify_partition_data_types(part_g)
            verify_partition_formats(part_g, graph_formats)
            verify_graph_feats(
                g, gpb, part_g, node_feats, edge_feats, orig_nids, orig_eids
            )


@pytest.mark.parametrize(
    "num_chunks, num_parts, world_size",
    [[4, 4, 4], [8, 4, 2], [8, 4, 4], [9, 6, 3], [11, 11, 1], [11, 4, 1]],
)
def test_pipeline_basics(num_chunks, num_parts, world_size):
    _test_pipeline(num_chunks, num_parts, world_size)
    _test_pipeline(
        num_chunks, num_parts, world_size, use_verify_partitions=False
    )


@pytest.mark.parametrize(
    "graph_formats", [None, "csc", "coo,csc", "coo,csc,csr"]
)
def test_pipeline_formats(graph_formats):
    _test_pipeline(4, 4, 4, graph_formats)


@pytest.mark.parametrize(
    "num_chunks, "
    "num_parts, "
    "world_size, "
    "num_chunks_node_data, "
    "num_chunks_edge_data",
    [
        # Test cases where no. of chunks more than
        # no. of partitions
        [8, 4, 4, 8, 8],
        [8, 4, 2, 8, 8],
        [9, 7, 5, 9, 9],
        [8, 8, 4, 8, 8],
        # Test cases where no. of chunks smaller
        # than no. of partitions
        [7, 8, 4, 7, 7],
        [1, 8, 4, 1, 1],
        [1, 4, 4, 1, 1],
        [3, 4, 4, 3, 3],
        [1, 4, 2, 1, 1],
        [3, 4, 2, 3, 3],
        [1, 5, 3, 1, 1],
    ],
)
def test_pipeline_arbitrary_chunks(
    num_chunks,
    num_parts,
    world_size,
    num_chunks_node_data,
    num_chunks_edge_data,
):
    _test_pipeline(
        num_chunks,
        num_parts,
        world_size,
        num_chunks_node_data=num_chunks_node_data,
        num_chunks_edge_data=num_chunks_edge_data,
    )


@pytest.mark.parametrize(
    "graph_formats", [None, "csc", "coo,csc", "coo,csc,csr"]
)
def test_pipeline_formats(graph_formats):
    _test_pipeline(4, 4, 4, graph_formats)


@pytest.mark.parametrize("data_fmt", ["numpy", "parquet"])
def test_pipeline_feature_format(data_fmt):
    _test_pipeline(4, 4, 4, data_fmt=data_fmt)


@pytest.mark.parametrize(
    "num_chunks, num_parts, world_size",
    [[4, 4, 4], [8, 4, 2], [8, 4, 4], [9, 6, 3], [11, 11, 1], [11, 4, 1]],
)
@pytest.mark.parametrize("few_entity", ["node", "edge"])
def test_partition_hetero_few_entity(
    num_chunks,
    num_parts,
    world_size,
    few_entity,
    graph_formats=None,
    data_fmt="numpy",
    edges_fmt="csv",
    vector_rows=False,
    num_chunks_nodes=None,
    num_chunks_edges=None,
    num_chunks_node_data=None,
    num_chunks_edge_data=None,
):
    with tempfile.TemporaryDirectory() as root_dir:
        g = create_mini_chunked_dataset(
            root_dir,
            num_chunks,
            few_entity=few_entity,
            data_fmt=data_fmt,
            edges_fmt=edges_fmt,
            vector_rows=vector_rows,
            num_chunks_nodes=num_chunks_nodes,
            num_chunks_edges=num_chunks_edges,
            num_chunks_node_data=num_chunks_node_data,
            num_chunks_edge_data=num_chunks_edge_data,
        )

        # Step1: graph partition
        in_dir = os.path.join(root_dir, "chunked-data")
        output_dir = os.path.join(root_dir, "parted_data")
        os.system(
            "python3 tools/partition_algo/random_partition.py "
            "--in_dir {} --out_dir {} --num_partitions {}".format(
                in_dir, output_dir, num_parts
            )
        )

        # Step2: data dispatch
        partition_dir = os.path.join(root_dir, "parted_data")
        out_dir = os.path.join(root_dir, "partitioned")
        ip_config = os.path.join(root_dir, "ip_config.txt")
        with open(ip_config, "w") as f:
            for i in range(world_size):
                f.write(f"127.0.0.{i + 1}\n")

        cmd = "python3 tools/dispatch_data.py"
        cmd += f" --in-dir {in_dir}"
        cmd += f" --partitions-dir {partition_dir}"
        cmd += f" --out-dir {out_dir}"
        cmd += f" --ip-config {ip_config}"
        cmd += " --ssh-port 22"
        cmd += " --process-group-timeout 60"
        cmd += " --save-orig-nids"
        cmd += " --save-orig-eids"
        cmd += f" --graph-formats {graph_formats}" if graph_formats else ""
        os.system(cmd)

        # read original node/edge IDs
        def read_orig_ids(fname):
            orig_ids = {}
            for i in range(num_parts):
                ids_path = os.path.join(out_dir, f"part{i}", fname)
                part_ids = load_tensors(ids_path)
                for type, data in part_ids.items():
                    if type not in orig_ids:
                        orig_ids[type] = data
                    else:
                        orig_ids[type] = torch.cat((orig_ids[type], data))
            return orig_ids

        orig_nids = read_orig_ids("orig_nids.dgl")
        orig_eids = read_orig_ids("orig_eids.dgl")

        # load partitions and verify
        part_config = os.path.join(out_dir, "metadata.json")
        for i in range(num_parts):
            part_g, node_feats, edge_feats, gpb, _, _, _ = load_partition(
                part_config, i
            )
            verify_partition_data_types(part_g)
            verify_partition_formats(part_g, graph_formats)
            verify_graph_feats(
                g, gpb, part_g, node_feats, edge_feats, orig_nids, orig_eids
            )


def test_utils_generate_read_list():
    read_list = generate_read_list(10, 4)
    assert np.array_equal(read_list[0], np.array([0, 1, 2]))
    assert np.array_equal(read_list[1], np.array([3, 4, 5]))
    assert np.array_equal(read_list[2], np.array([6, 7]))
    assert np.array_equal(read_list[3], np.array([8, 9]))
