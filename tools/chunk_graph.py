# See the __main__ block for usage of chunk_graph().
import json
import logging
import os
import pathlib
from contextlib import contextmanager

import torch
from utils import array_readwriter, setdir

import dgl

def chunk_numpy_array(arr, fmt_meta, chunk_sizes, path_fmt):
    paths = []
    offset = 0

    for j, n in enumerate(chunk_sizes):
        path = os.path.abspath(path_fmt % j)
        arr_chunk = arr[offset : offset + n]
        logging.info("Chunking %d-%d" % (offset, offset + n))
        array_readwriter.get_array_parser(**fmt_meta).write(path, arr_chunk)
        offset += n
        paths.append(path)

    return paths

def _initialize_num_chunks(g, num_chunks, kwargs=None):
    '''Initialize num_chunks for each node/edge.

    Parameters
    ----------
    g: DGLGraph
        Graph to be chunked.
    num_chunks: int
        Default number of chunks to be applied onto node/edge data.
    kwargs: dict
        Key word arguments to specify details for each node/edge data.

    Returns
    -------
    num_chunks_data: dict
        Detailed number of chunks for each node/edge.
    '''
    def _init(g, num_chunks, key, kwargs=None):
        chunks_data = kwargs.get(key, None)
        is_node = '_node' in key
        data_types = g.ntypes if is_node else g.canonical_etypes
        if isinstance(chunks_data, int):
            chunks_data = {
                data_type: chunks_data for data_type in data_types
            }
        elif isinstance(chunks_data, dict):
            for data_type in data_types:
                if data_type not in chunks_data:
                    chunks_data[data_type] = num_chunks
        else:
            chunks_data = {
                data_type: num_chunks for data_type in data_types
            }
        for _, data in chunks_data.items():
            if isinstance(data, dict):
                n_chunks = list(data.values())
            else:
                n_chunks = [data]
            assert all(isinstance(v, int) for v in n_chunks), (
                    "num_chunks for each data type should be int."
                )
        return chunks_data
    num_chunks_data = {}
    for key in [
        'num_chunks_nodes', 'num_chunks_edges', 'num_chunks_node_data',
        'num_chunks_edge_data'
    ]:
        num_chunks_data[key] = _init(g, num_chunks, key, kwargs=kwargs)
    return num_chunks_data

def _chunk_graph(g, name, ndata_paths, edata_paths, num_chunks, **kwargs):
    # First deal with ndata and edata that are homogeneous (i.e. not a dict-of-dict)
    if len(g.ntypes) == 1 and not isinstance(
        next(iter(ndata_paths.values())), dict
    ):
        ndata_paths = {g.ntypes[0]: ndata_paths}
    if len(g.etypes) == 1 and not isinstance(
        next(iter(edata_paths.values())), dict
    ):
        edata_paths = {g.etypes[0]: ndata_paths}
    # Then convert all edge types to canonical edge types
    etypestrs = {etype: ":".join(etype) for etype in g.canonical_etypes}
    edata_paths = {
        ":".join(g.to_canonical_etype(k)): v for k, v in edata_paths.items()
    }

    metadata = {}

    metadata["graph_name"] = name
    metadata["node_type"] = g.ntypes

    # Initialize num_chunks for each node/edge.
    num_chunks_details = _initialize_num_chunks(g, num_chunks, kwargs=kwargs)

    # Compute the number of nodes per chunk per node type
    metadata["num_nodes_per_chunk"] = num_nodes_per_chunk = []
    num_chunks_nodes = num_chunks_details['num_chunks_nodes']
    for ntype in g.ntypes:
        num_nodes = g.num_nodes(ntype)
        num_nodes_list = []
        n_chunks = num_chunks_nodes[ntype]
        for i in range(n_chunks):
            n = num_nodes // n_chunks + (i < num_nodes % n_chunks)
            num_nodes_list.append(n)
        num_nodes_per_chunk.append(num_nodes_list)

    metadata["edge_type"] = [etypestrs[etype] for etype in g.canonical_etypes]

    # Compute the number of edges per chunk per edge type
    metadata["num_edges_per_chunk"] = num_edges_per_chunk = []
    num_chunks_edges = num_chunks_details['num_chunks_edges']
    for etype in g.canonical_etypes:
        num_edges = g.num_edges(etype)
        num_edges_list = []
        n_chunks = num_chunks_edges[etype]
        for i in range(n_chunks):
            n = num_edges // n_chunks + (i < num_edges % n_chunks)
            num_edges_list.append(n)
        num_edges_per_chunk.append(num_edges_list)
    num_edges_per_chunk_dict = {
        k: v for k, v in zip(g.canonical_etypes, num_edges_per_chunk)
    }

    # Split edge index
    metadata["edges"] = {}
    with setdir("edge_index"):
        for etype in g.canonical_etypes:
            etypestr = etypestrs[etype]
            logging.info("Chunking edge index for %s" % etypestr)
            edges_meta = {}
            fmt_meta = {"name": "csv", "delimiter": " "}
            edges_meta["format"] = fmt_meta

            srcdst = torch.stack(g.edges(etype=etype), 1)
            edges_meta["data"] = chunk_numpy_array(
                srcdst.numpy(),
                fmt_meta,
                num_edges_per_chunk_dict[etype],
                etypestr + "%d.txt",
            )
            metadata["edges"][etypestr] = edges_meta

    # Chunk node data
    metadata["node_data"] = {}
    num_chunks_node_data = num_chunks_details['num_chunks_node_data']
    with setdir("node_data"):
        for ntype, ndata_per_type in ndata_paths.items():
            ndata_meta = {}
            with setdir(ntype):
                for key, path in ndata_per_type.items():
                    logging.info(
                        "Chunking node data for type %s key %s" % (ntype, key)
                    )
                    chunk_sizes = []
                    num_nodes = g.num_nodes(ntype)
                    n_chunks = num_chunks_node_data[ntype]
                    if isinstance(n_chunks, dict):
                        n_chunks = n_chunks.get(key, num_chunks)
                    assert isinstance(n_chunks, int), (
                        f"num_chunks for {ntype}/{key} should be int while "
                        f"{type(n_chunks)} is got."
                    )
                    for i in range(n_chunks):
                        n = num_nodes // n_chunks + (i < num_nodes % n_chunks)
                        chunk_sizes.append(n)
                    ndata_key_meta = {}
                    reader_fmt_meta = writer_fmt_meta = {"name": "numpy"}
                    arr = array_readwriter.get_array_parser(
                        **reader_fmt_meta
                    ).read(path)
                    ndata_key_meta["format"] = writer_fmt_meta
                    ndata_key_meta["data"] = chunk_numpy_array(
                        arr,
                        writer_fmt_meta,
                        chunk_sizes,
                        key + "-%d.npy",
                    )
                    ndata_meta[key] = ndata_key_meta

            metadata["node_data"][ntype] = ndata_meta

    # Chunk edge data
    metadata["edge_data"] = {}
    num_chunks_edge_data = num_chunks_details['num_chunks_edge_data']
    with setdir("edge_data"):
        for etypestr, edata_per_type in edata_paths.items():
            edata_meta = {}
            etype = tuple(etypestr.split(":"))
            with setdir(etypestr):
                for key, path in edata_per_type.items():
                    logging.info(
                        "Chunking edge data for type %s key %s"
                        % (etypestr, key)
                    )
                    chunk_sizes = []
                    num_edges = g.num_edges(etype)
                    n_chunks = num_chunks_edge_data[etype]
                    if isinstance(n_chunks, dict):
                        n_chunks = n_chunks.get(key, num_chunks)
                    assert isinstance(n_chunks, int), (
                        f"num_chunks for {etype}/{key} should be int while "
                        f"{type(n_chunks)} is got."
                    )
                    for i in range(n_chunks):
                        n = num_edges // n_chunks + (i < num_edges % n_chunks)
                        chunk_sizes.append(n)
                    edata_key_meta = {}
                    reader_fmt_meta = writer_fmt_meta = {"name": "numpy"}
                    arr = array_readwriter.get_array_parser(
                        **reader_fmt_meta
                    ).read(path)
                    edata_key_meta["format"] = writer_fmt_meta
                    edata_key_meta["data"] = chunk_numpy_array(
                        arr,
                        writer_fmt_meta,
                        chunk_sizes,
                        key + "-%d.npy",
                    )
                    edata_meta[key] = edata_key_meta

            metadata["edge_data"][etypestr] = edata_meta

    metadata_path = "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=4)
    logging.info("Saved metadata in %s" % os.path.abspath(metadata_path))


def chunk_graph(g, name, ndata_paths, edata_paths, num_chunks, output_path,
        **kwargs):
    """
    Split the graph into multiple chunks.

    A directory will be created at :attr:`output_path` with the metadata and chunked
    edge list as well as the node/edge data.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    name : str
        The name of the graph, to be used later in DistDGL training.
    ndata_paths : dict[str, pathlike] or dict[ntype, dict[str, pathlike]]
        The dictionary of paths pointing to the corresponding numpy array file for each
        node data key.
    edata_paths : dict[etype, pathlike] or dict[etype, dict[str, pathlike]]
        The dictionary of paths pointing to the corresponding numpy array file for each
        edge data key. ``etype`` could be canonical or non-canonical.
    num_chunks : int
        The number of chunks
    output_path : pathlike
        The output directory saving the chunked graph.
    kwargs : dict
        Key word arguments to control chunk details.
    """
    for ntype, ndata in ndata_paths.items():
        for key in ndata.keys():
            ndata[key] = os.path.abspath(ndata[key])
    for etype, edata in edata_paths.items():
        for key in edata.keys():
            edata[key] = os.path.abspath(edata[key])
    with setdir(output_path):
        _chunk_graph(g, name, ndata_paths, edata_paths, num_chunks, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    input_dir = "/data"
    output_dir = "/chunked-data"
    (g,), _ = dgl.load_graphs(os.path.join(input_dir, "graph.dgl"))
    chunk_graph(
        g,
        "mag240m",
        {
            "paper": {
                "feat": os.path.join(input_dir, "paper/feat.npy"),
                "label": os.path.join(input_dir, "paper/label.npy"),
                "year": os.path.join(input_dir, "paper/year.npy"),
            }
        },
        {
            "cites": {"count": os.path.join(input_dir, "cites/count.npy")},
            "writes": {"year": os.path.join(input_dir, "writes/year.npy")},
            # you can put the same data file if they indeed share the features.
            "rev_writes": {"year": os.path.join(input_dir, "writes/year.npy")},
        },
        4,
        output_dir,
    )
# The generated metadata goes as in tools/sample-config/mag240m-metadata.json.
