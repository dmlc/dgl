import json
import os
import tempfile

import dgl

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
from pytest_utils import create_chunked_dataset

from tools.verification_utils import (
    verify_graph_feats,
    verify_partition_data_types,
    verify_partition_formats,
)


def _test_pipeline_graphbolt(
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
        cmd += " --use-graphbolt"
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
                part_config, i, use_graphbolt=True
            )
            verify_partition_data_types(part_g, use_graphbolt=True)
            verify_graph_feats(
                g,
                gpb,
                part_g,
                node_feats,
                edge_feats,
                orig_nids,
                orig_eids,
                use_graphbolt=True,
            )


@pytest.mark.parametrize(
    "num_chunks, num_parts, world_size",
    [[4, 4, 4], [8, 4, 2], [8, 4, 4], [9, 6, 3], [11, 11, 1], [11, 4, 1]],
)
def test_pipeline_basics(num_chunks, num_parts, world_size):
    _test_pipeline_graphbolt(num_chunks, num_parts, world_size)
    _test_pipeline_graphbolt(
        num_chunks, num_parts, world_size, use_verify_partitions=False
    )


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

    _test_pipeline_graphbolt(
        num_chunks,
        num_parts,
        world_size,
        num_chunks_node_data=num_chunks_node_data,
        num_chunks_edge_data=num_chunks_edge_data,
    )


@pytest.mark.parametrize("data_fmt", ["numpy", "parquet"])
def test_pipeline_feature_format(data_fmt):
    _test_pipeline_graphbolt(4, 4, 4, data_fmt=data_fmt)
