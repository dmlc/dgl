import argparse
import logging
import os
import platform

import constants

import dgl
import numpy as np

import pyarrow
import pyarrow.parquet as pq
import torch as th
from dgl.data.utils import load_graphs, load_tensors

from dgl.distributed.partition import (
    _etype_str_to_tuple,
    _etype_tuple_to_str,
    _get_inner_edge_mask,
    _get_inner_node_mask,
    load_partition,
    RESERVED_FIELD_DTYPE,
)
from utils import get_idranges, read_json
from verification_utils import (
    get_node_partids,
    read_file,
    read_orig_ids,
    verify_graph_feats,
    verify_metadata_counts,
    verify_node_partitionids,
    verify_partition_data_types,
    verify_partition_formats,
)


def _read_graph(schema):
    """Read a DGL Graph object from storage using metadata schema, which is
    a json object describing the DGL graph on disk.

    Parameters:
    -----------
    schema : json object
        json object describing the input graph to read from the disk

    Returns:
    --------
    DGL Graph Object :
        DGL Graph object is created which is read from the disk storage.
    """
    edges = {}
    edge_types = schema[constants.STR_EDGE_TYPE]
    for etype in edge_types:
        efiles = schema[constants.STR_EDGES][etype][constants.STR_DATA]
        src = []
        dst = []
        for fname in efiles:
            if (
                schema[constants.STR_EDGES][etype][constants.STR_FORMAT][
                    constants.STR_NAME
                ]
                == constants.STR_CSV
            ):
                data = read_file(fname, constants.STR_CSV)
            elif (
                schema[constants.STR_EDGES][etype][constants.STR_FORMAT][
                    constants.STR_NAME
                ]
                == constants.STR_PARQUET
            ):
                data = read_file(fname)
            else:
                raise ValueError(
                    f"Unknown edge format for {etype} - {schema[constants.STR_EDGES][etype][constants.STR_FORMAT]}"
                )

                src.append(data[:, 0])
            dst.append(data[:, 1])
        src = np.concatenate(src)
        dst = np.concatenate(dst)
        edges[_etype_str_to_tuple(etype)] = (src, dst)

    g = dgl.heterograph(edges)
    # g = dgl.to_homogeneous(g)

    g.ndata["orig_id"] = g.ndata[dgl.NID]
    g.edata["orig_id"] = g.edata[dgl.EID]

    # read features here.
    for ntype in schema[constants.STR_NODE_TYPE]:
        if ntype in schema[constants.STR_NODE_DATA]:
            for featname, featdata in schema[constants.STR_NODE_DATA][
                ntype
            ].items():
                files = fdata[constants.STR_DATA]
                feats = []
                for fname in files:
                    feats.append(read_file(fname, constants.STR_NUMPY))
                if len(feats) > 0:
                    g.nodes[ntype].data[featname] = th.from_numpy(
                        np.concatenate(feats)
                    )

    # read edge features here.
    for etype in schema[constants.STR_EDGE_TYPE]:
        if etype in schema[constants.STR_EDGE_DATA]:
            for featname, fdata in schema[constants.STR_EDGE_DATA][etype]:
                files = fdata[constants.STR_DATA]
                feats = []
                for fname in files:
                    feats.append(read_file(fname))
                if len(feats) > 0:
                    g.edges[etype].data[featname] = th.from_numpy(
                        np.concatenate(feats)
                    )

    # print from graph
    logging.info(f"|V|= {g.num_nodes()}")
    logging.info(f"|E|= {g.num_edges()}")
    for ntype in g.ntypes:
        for name, data in g.nodes[ntype].data.items():
            if isinstance(data, th.Tensor):
                logging.info(
                    f"Input Graph: nfeat - {ntype}/{name} - data - {data.size()}"
                )

    for c_etype in g.canonical_etypes:
        for name, data in g.edges[c_etype].data.items():
            if isinstance(data, th.Tensor):
                logging.info(
                    f"Input Graph: efeat - {etype}/{name} - data - {g.edges[etype].data[name].size()}"
                )

    return g


def _read_part_graphs(part_config, part_metafile):
    """Read partitioned graph objects from disk storage.

    Parameters:
    ----------
    part_config : json object
        json object created using the metadata file for the partitioned graph.
    part_metafile : string
        absolute path of the metadata.json file for the partitioned graph.

    Returns:
    --------
    list of tuples :
        where each tuple contains 4 objects in the following order:
            partitioned graph object
            global partition book
            node features
            edge features
    """
    part_graph_data = []
    for i in range(part_config["num_parts"]):
        part_g, node_feats, edge_feats, gpb, _, _, _ = load_partition(
            part_metafile, i
        )
        part_graph_data.append((part_g, node_feats, edge_feats, gpb))
    return part_graph_data


def _validate_results(params):
    """Main function to verify the graph partitions

    Parameters:
    -----------
    params : argparser object
        to access the command line arguments
    """
    logging.info(f"loading config files...")
    part_config = os.path.join(params.part_graph_dir, "metadata.json")
    part_schema = read_json(part_config)
    num_parts = part_schema["num_parts"]

    logging.info(f"loading config files of the original dataset...")
    graph_config = os.path.join(params.orig_dataset_dir, "metadata.json")
    graph_schema = read_json(graph_config)

    logging.info(f"loading original ids from the dgl files...")
    orig_nids = read_orig_ids(params.part_graph_dir, "orig_nids.dgl", num_parts)
    orig_eids = read_orig_ids(params.part_graph_dir, "orig_eids.dgl", num_parts)

    logging.info(f"loading node to partition-ids from files... ")
    node_partids = get_node_partids(params.partitions_dir, graph_schema)

    logging.info(f"loading the original dataset...")
    g = _read_graph(graph_schema)

    logging.info(f"Beginning the verification process...")
    for i in range(num_parts):
        part_g, node_feats, edge_feats, gpb, _, _, _ = load_partition(
            part_config, i
        )

        verify_partition_data_types(part_g)
        verify_partition_formats(part_g, None)
        verify_graph_feats(
            g, gpb, part_g, node_feats, edge_feats, orig_nids, orig_eids
        )
        verify_metadata_counts(part_schema, part_g, graph_schema, g, i)
        verify_node_partitionids(
            node_partids, part_g, g, gpb, graph_schema, orig_nids, i
        )
        logging.info(f"Verification of partitioned graph - {i}... SUCCESS !!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct graph partitions")
    parser.add_argument(
        "--orig-dataset-dir",
        required=True,
        type=str,
        help="The directory path that contains the original graph input files.",
    )
    parser.add_argument(
        "--part-graph-dir",
        required=True,
        type=str,
        help="The directory path that contains the partitioned graph files.",
    )
    parser.add_argument(
        "--partitions-dir",
        required=True,
        type=str,
        help="The directory path that contains metis/random partitions results.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="To enable log level for debugging purposes. Available options: \
                          (Critical, Error, Warning, Info, Debug, Notset), default value \
                          is: Info",
    )
    params = parser.parse_args()

    numeric_level = getattr(logging, params.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format=f"[{platform.node()} %(levelname)s %(asctime)s PID:%(process)d] %(message)s",
    )

    _validate_results(params)
