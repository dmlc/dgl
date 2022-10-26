import json
import os
import tempfile
import unittest

import numpy as np
import pytest
import torch
from chunk_graph import chunk_graph
from create_chunked_dataset import create_chunked_dataset

import dgl
from dgl.data.utils import load_graphs, load_tensors
from dgl.distributed.partition import (
    RESERVED_FIELD_DTYPE,
    load_partition,
    _get_inner_node_mask,
    _get_inner_edge_mask,
)


def _verify_partition_data_types(part_g):
    for k, dtype in RESERVED_FIELD_DTYPE.items():
        if k in part_g.ndata:
            assert part_g.ndata[k].dtype == dtype
        if k in part_g.edata:
            assert part_g.edata[k].dtype == dtype


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

    for etype in g.etypes:
        etype_id = g.get_etype_id(etype)
        inner_edge_mask = _get_inner_edge_mask(part, etype_id)
        inner_eids = part.edata[dgl.EID][inner_edge_mask]
        etype_ids, inner_type_eids = gpb.map_to_per_etype(inner_eids)
        partid = gpb.eid2partid(inner_type_eids, etype)
        assert np.all(etype_ids.numpy() == etype_id)
        assert np.all(partid.numpy() == gpb.partid)

        orig_id = orig_eids[etype][inner_type_eids]
        local_eids = gpb.eid2localeid(inner_type_eids, gpb.partid, etype)

        for name in g.edges[etype].data:
            if name in [dgl.EID, "inner_edge"]:
                continue
            true_feats = g.edges[etype].data[name][orig_id]
            edata = edge_feats[etype + "/" + name][local_eids]
            assert torch.equal(edata == true_feats)


@pytest.mark.parametrize("num_chunks", [1, 8])
def test_chunk_graph(num_chunks):

    with tempfile.TemporaryDirectory() as root_dir:

        g = create_chunked_dataset(root_dir, num_chunks)

        num_cite_edges = g.number_of_edges("cites")
        num_write_edges = g.number_of_edges("writes")
        num_affiliate_edges = g.number_of_edges("affiliated_with")

        num_institutions = g.number_of_nodes("institution")
        num_authors = g.number_of_nodes("author")
        num_papers = g.number_of_nodes("paper")

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
        for utype, etype, vtype in g.canonical_etypes:
            fname = ":".join([utype, etype, vtype])
            for i in range(num_chunks):
                chunk_f_name = os.path.join(
                    output_edge_index_dir, fname + str(i) + ".txt"
                )
                assert os.path.isfile(chunk_f_name)
                with open(chunk_f_name, "r") as f:
                    header = f.readline()
                    num1, num2 = header.rstrip().split(" ")
                    assert isinstance(int(num1), int)
                    assert isinstance(int(num2), int)

        # check node_data
        output_node_data_dir = os.path.join(output_dir, "node_data", "paper")
        for feat in ["feat", "label", "year"]:
            for i in range(num_chunks):
                chunk_f_name = "{}-{}.npy".format(feat, i)
                chunk_f_name = os.path.join(output_node_data_dir, chunk_f_name)
                assert os.path.isfile(chunk_f_name)
                feat_array = np.load(chunk_f_name)
                assert feat_array.shape[0] == num_papers // num_chunks

        # check edge_data
        num_edges = {
            "paper:cites:paper": num_cite_edges,
            "author:writes:paper": num_write_edges,
            "paper:rev_writes:author": num_write_edges,
        }
        output_edge_data_dir = os.path.join(output_dir, "edge_data")
        for etype, feat in [
            ["paper:cites:paper", "count"],
            ["author:writes:paper", "year"],
            ["paper:rev_writes:author", "year"],
        ]:
            output_edge_sub_dir = os.path.join(output_edge_data_dir, etype)
            for i in range(num_chunks):
                chunk_f_name = "{}-{}.npy".format(feat, i)
                chunk_f_name = os.path.join(output_edge_sub_dir, chunk_f_name)
                assert os.path.isfile(chunk_f_name)
                feat_array = np.load(chunk_f_name)
            assert feat_array.shape[0] == num_edges[etype] // num_chunks


@pytest.mark.parametrize("num_chunks", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_parts", [1, 2, 3, 4, 8])
def test_part_pipeline(num_chunks, num_parts):
    if num_chunks < num_parts:
        # num_parts should less/equal than num_chunks
        return

    with tempfile.TemporaryDirectory() as root_dir:

        g = create_chunked_dataset(root_dir, num_chunks)

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
            for i in range(num_parts):
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
            _verify_graph_feats(
                g, gpb, part_g, node_feats, edge_feats, orig_nids, orig_eids
            )
