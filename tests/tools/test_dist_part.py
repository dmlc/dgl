import json
import os
import tempfile
import unittest

import numpy as np
import pytest
import torch
from chunk_graph import chunk_graph
from create_chunked_dataset import create_chunked_dataset
from distpartitioning import array_readwriter

import dgl
from dgl.data.utils import load_graphs, load_tensors
from dgl.distributed.partition import (RESERVED_FIELD_DTYPE,
                                       _etype_tuple_to_str,
                                       _get_inner_edge_mask,
                                       _get_inner_node_mask, load_partition)


def _verify_partition_data_types(part_g):
    for k, dtype in RESERVED_FIELD_DTYPE.items():
        if k in part_g.ndata:
            assert part_g.ndata[k].dtype == dtype
        if k in part_g.edata:
            assert part_g.edata[k].dtype == dtype

def _verify_partition_formats(part_g, formats):
    # Verify saved graph formats
    if formats is None:
        assert "coo" in part_g.formats()["created"]
    else:
        formats = formats.split(',')
        for format in formats:
            assert format in part_g.formats()["created"]


def _verify_graph_feats(
    g, gpb, part, node_feats, edge_feats, orig_nids, orig_eids
):
    for ntype in g.ntypes:
        ntype_id = g.get_ntype_id(ntype)
        inner_node_mask = _get_inner_node_mask(part, ntype_id)
        inner_nids = part.ndata[dgl.NID][inner_node_mask]
        ntype_ids, inner_type_nids = gpb.map_to_per_ntype(inner_nids)
        partid = gpb.nid2partid(inner_type_nids, ntype)
        assert np.all(ntype_ids.numpy() == ntype_id)
        assert np.all(partid.numpy() == gpb.partid)

        orig_id = orig_nids[ntype][inner_type_nids]
        local_nids = gpb.nid2localnid(inner_type_nids, gpb.partid, ntype)

        for name in g.nodes[ntype].data:
            if name in [dgl.NID, "inner_node"]:
                continue
            true_feats = g.nodes[ntype].data[name][orig_id]
            ndata = node_feats[ntype + "/" + name][local_nids]
            assert torch.equal(ndata, true_feats)

    for etype in g.canonical_etypes:
        etype_id = g.get_etype_id(etype)
        inner_edge_mask = _get_inner_edge_mask(part, etype_id)
        inner_eids = part.edata[dgl.EID][inner_edge_mask]
        etype_ids, inner_type_eids = gpb.map_to_per_etype(inner_eids)
        partid = gpb.eid2partid(inner_type_eids, etype)
        assert np.all(etype_ids.numpy() == etype_id)
        assert np.all(partid.numpy() == gpb.partid)

        orig_id = orig_eids[_etype_tuple_to_str(etype)][inner_type_eids]
        local_eids = gpb.eid2localeid(inner_type_eids, gpb.partid, etype)

        for name in g.edges[etype].data:
            if name in [dgl.EID, "inner_edge"]:
                continue
            true_feats = g.edges[etype].data[name][orig_id]
            edata = edge_feats[_etype_tuple_to_str(etype) + "/" + name][local_eids]
            assert torch.equal(edata, true_feats)


@pytest.mark.parametrize("num_chunks", [1, 8])
@pytest.mark.parametrize("data_fmt", ['numpy', 'parquet'])
def test_chunk_graph(num_chunks, data_fmt):

    with tempfile.TemporaryDirectory() as root_dir:

        g = create_chunked_dataset(root_dir, num_chunks, data_fmt=data_fmt)

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
            for i in range(num_chunks):
                fname = os.path.join(
                    output_edge_index_dir, f'{c_etype_str}{i}.txt'
                )
                assert os.path.isfile(fname)
                with open(fname, "r") as f:
                    header = f.readline()
                    num1, num2 = header.rstrip().split(" ")
                    assert isinstance(int(num1), int)
                    assert isinstance(int(num2), int)

        # check node/edge_data
        suffix = 'npy' if data_fmt=='numpy' else 'parquet'
        reader_fmt_meta = {"name": data_fmt}
        def test_data(sub_dir, feat, expected_data, expected_shape):
            data = []
            for i in range(num_chunks):
                fname = os.path.join(sub_dir, f'{feat}-{i}.{suffix}')
                assert os.path.isfile(fname)
                feat_array =  array_readwriter.get_array_parser(
                            **reader_fmt_meta
                        ).read(fname)
                assert feat_array.shape[0] == expected_shape
                data.append(feat_array)
            data = np.concatenate(data, 0)
            assert torch.equal(torch.from_numpy(data), expected_data)

        output_node_data_dir = os.path.join(output_dir, "node_data")
        for ntype in g.ntypes:
            sub_dir = os.path.join(output_node_data_dir, ntype)
            for feat, data in g.nodes[ntype].data.items():
                test_data(sub_dir, feat, data, g.num_nodes(ntype) // num_chunks)

        output_edge_data_dir = os.path.join(output_dir, "edge_data")
        for c_etype in g.canonical_etypes:
            c_etype_str = _etype_tuple_to_str(c_etype)
            sub_dir = os.path.join(output_edge_data_dir, c_etype_str)
            for feat, data in g.edges[c_etype].data.items():
                test_data(sub_dir, feat, data, g.num_edges(c_etype) // num_chunks)


def _test_pipeline(num_chunks, num_parts, world_size, graph_formats=None, data_fmt='numpy'):
    if num_chunks < num_parts:
        # num_parts should less/equal than num_chunks
        return

    if num_parts % world_size != 0:
        # num_parts should be a multiple of world_size
        return

    with tempfile.TemporaryDirectory() as root_dir:

        g = create_chunked_dataset(root_dir, num_chunks, data_fmt=data_fmt)

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
        partition_dir = os.path.join(root_dir, 'parted_data')
        out_dir = os.path.join(root_dir, 'partitioned')
        ip_config = os.path.join(root_dir, 'ip_config.txt')
        with open(ip_config, 'w') as f:
            for i in range(world_size):
                f.write(f'127.0.0.{i + 1}\n')

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
            _verify_partition_data_types(part_g)
            _verify_partition_formats(part_g, graph_formats)
            _verify_graph_feats(
                g, gpb, part_g, node_feats, edge_feats, orig_nids, orig_eids
            )


@pytest.mark.parametrize("num_chunks, num_parts, world_size", [[8, 4, 2], [9, 6, 3], [11, 11, 1], [11, 4, 2], [5, 3, 1]])
def test_pipeline_basics(num_chunks, num_parts, world_size):
    _test_pipeline(num_chunks, num_parts, world_size)


@pytest.mark.parametrize(
    "graph_formats", [None, "csc", "coo,csc", "coo,csc,csr"]
)
def test_pipeline_formats(graph_formats):
    _test_pipeline(4, 4, 4, graph_formats)


@pytest.mark.parametrize(
    "data_fmt", ['numpy', "parquet"]
)
def test_pipeline_feature_format(data_fmt):
    _test_pipeline(4, 4, 4, data_fmt=data_fmt)
