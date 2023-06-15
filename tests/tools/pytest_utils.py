import json
import logging
import os

import dgl
import numpy as np
import torch
from distpartitioning import array_readwriter
from distpartitioning.array_readwriter.parquet import ParquetArrayParser
from files import setdir


def _chunk_numpy_array(arr, fmt_meta, chunk_sizes, path_fmt, vector_rows=False):
    paths = []
    offset = 0

    for j, n in enumerate(chunk_sizes):
        path = os.path.abspath(path_fmt % j)
        arr_chunk = arr[offset : offset + n]
        shape = arr_chunk.shape
        logging.info("Chunking %d-%d" % (offset, offset + n))
        # If requested we write multi-column arrays as single-column vector Parquet files
        array_parser = array_readwriter.get_array_parser(**fmt_meta)
        if (
            isinstance(array_parser, ParquetArrayParser)
            and len(shape) > 1
            and shape[1] > 1
        ):
            array_parser.write(path, arr_chunk, vector_rows=vector_rows)
        else:
            array_parser.write(path, arr_chunk)
        offset += n
        paths.append(path)

    return paths


def _initialize_num_chunks(g, num_chunks, kwargs=None):
    """Initialize num_chunks for each node/edge.

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
    """

    def _init(g, num_chunks, key, kwargs=None):
        chunks_data = kwargs.get(key, None)
        is_node = "_node" in key
        data_types = g.ntypes if is_node else g.canonical_etypes
        if isinstance(chunks_data, int):
            chunks_data = {data_type: chunks_data for data_type in data_types}
        elif isinstance(chunks_data, dict):
            for data_type in data_types:
                if data_type not in chunks_data:
                    chunks_data[data_type] = num_chunks
        else:
            chunks_data = {data_type: num_chunks for data_type in data_types}
        for _, data in chunks_data.items():
            if isinstance(data, dict):
                n_chunks = list(data.values())
            else:
                n_chunks = [data]
            assert all(
                isinstance(v, int) for v in n_chunks
            ), "num_chunks for each data type should be int."
        return chunks_data

    num_chunks_data = {}
    for key in [
        "num_chunks_nodes",
        "num_chunks_edges",
        "num_chunks_node_data",
        "num_chunks_edge_data",
    ]:
        num_chunks_data[key] = _init(g, num_chunks, key, kwargs=kwargs)
    return num_chunks_data


def _chunk_graph(
    g,
    name,
    ndata_paths,
    edata_paths,
    num_chunks,
    data_fmt,
    edges_format,
    vector_rows=False,
    **kwargs,
):
    # First deal with ndata and edata that are homogeneous
    # (i.e. not a dict-of-dict)
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

    # add node_type_counts
    metadata["num_nodes_per_type"] = [g.num_nodes(ntype) for ntype in g.ntypes]

    # Initialize num_chunks for each node/edge.
    num_chunks_details = _initialize_num_chunks(g, num_chunks, kwargs=kwargs)

    # Compute the number of nodes per chunk per node type
    metadata["num_nodes_per_chunk"] = num_nodes_per_chunk = []
    num_chunks_nodes = num_chunks_details["num_chunks_nodes"]
    for ntype in g.ntypes:
        num_nodes = g.num_nodes(ntype)
        num_nodes_list = []
        n_chunks = num_chunks_nodes[ntype]
        for i in range(n_chunks):
            n = num_nodes // n_chunks + (i < num_nodes % n_chunks)
            num_nodes_list.append(n)
        num_nodes_per_chunk.append(num_nodes_list)

    metadata["edge_type"] = [etypestrs[etype] for etype in g.canonical_etypes]
    metadata["num_edges_per_type"] = [
        g.num_edges(etype) for etype in g.canonical_etypes
    ]

    # Compute the number of edges per chunk per edge type
    metadata["num_edges_per_chunk"] = num_edges_per_chunk = []
    num_chunks_edges = num_chunks_details["num_chunks_edges"]
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

    idxes_etypestr = {
        idx: (etype, etypestrs[etype])
        for idx, etype in enumerate(g.canonical_etypes)
    }
    idxes = np.arange(len(idxes_etypestr))

    # Split edge index
    metadata["edges"] = {}
    with setdir("edge_index"):
        np.random.shuffle(idxes)
        for idx in idxes:
            etype = idxes_etypestr[idx][0]
            etypestr = idxes_etypestr[idx][1]
            logging.info("Chunking edge index for %s" % etypestr)
            edges_meta = {}
            if edges_format == "csv":
                fmt_meta = {"name": edges_format, "delimiter": " "}
            elif edges_format == "parquet":
                fmt_meta = {"name": edges_format}
            else:
                raise RuntimeError(f"Invalid edges_fmt: {edges_format}")
            edges_meta["format"] = fmt_meta

            srcdst = torch.stack(g.edges(etype=etype), 1)
            edges_meta["data"] = _chunk_numpy_array(
                srcdst.numpy(),
                fmt_meta,
                num_edges_per_chunk_dict[etype],
                etypestr + "%d.txt",
            )
            metadata["edges"][etypestr] = edges_meta

    # Chunk node data
    reader_fmt_meta, writer_fmt_meta = {"name": "numpy"}, {"name": data_fmt}
    file_suffix = "npy" if data_fmt == "numpy" else "parquet"
    metadata["node_data"] = {}
    num_chunks_node_data = num_chunks_details["num_chunks_node_data"]
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
                    arr = array_readwriter.get_array_parser(
                        **reader_fmt_meta
                    ).read(path)
                    ndata_key_meta["format"] = writer_fmt_meta
                    ndata_key_meta["data"] = _chunk_numpy_array(
                        arr,
                        writer_fmt_meta,
                        chunk_sizes,
                        key + "-%d." + file_suffix,
                        vector_rows=vector_rows,
                    )
                    ndata_meta[key] = ndata_key_meta

            metadata["node_data"][ntype] = ndata_meta

    # Chunk edge data
    metadata["edge_data"] = {}
    num_chunks_edge_data = num_chunks_details["num_chunks_edge_data"]
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
                    arr = array_readwriter.get_array_parser(
                        **reader_fmt_meta
                    ).read(path)
                    edata_key_meta["format"] = writer_fmt_meta
                    edata_key_meta["data"] = _chunk_numpy_array(
                        arr,
                        writer_fmt_meta,
                        chunk_sizes,
                        key + "-%d." + file_suffix,
                        vector_rows=vector_rows,
                    )
                    edata_meta[key] = edata_key_meta

            metadata["edge_data"][etypestr] = edata_meta

    metadata_path = "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=4)
    logging.info("Saved metadata in %s" % os.path.abspath(metadata_path))


def chunk_graph(
    g,
    name,
    ndata_paths,
    edata_paths,
    num_chunks,
    output_path,
    data_fmt="numpy",
    edges_fmt="csv",
    vector_rows=False,
    **kwargs,
):
    """
    Split the graph into multiple chunks.

    A directory will be created at :attr:`output_path` with the metadata and
    chunked edge list as well as the node/edge data.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    name : str
        The name of the graph, to be used later in DistDGL training.
    ndata_paths : dict[str, pathlike] or dict[ntype, dict[str, pathlike]]
        The dictionary of paths pointing to the corresponding numpy array file
        for each node data key.
    edata_paths : dict[etype, pathlike] or dict[etype, dict[str, pathlike]]
        The dictionary of paths pointing to the corresponding numpy array file
        for each edge data key. ``etype`` could be canonical or non-canonical.
    num_chunks : int
        The number of chunks
    output_path : pathlike
        The output directory saving the chunked graph.
    data_fmt : str
        Format of node/edge data: 'numpy' or 'parquet'.
    edges_fmt : str
        Format of edges files: 'csv' or 'parquet'.
    vector_rows : str
        When true will write parquet files as single-column vector row files.
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
        _chunk_graph(
            g,
            name,
            ndata_paths,
            edata_paths,
            num_chunks,
            data_fmt,
            edges_fmt,
            vector_rows,
            **kwargs,
        )


def create_chunked_dataset(
    root_dir,
    num_chunks,
    data_fmt="numpy",
    edges_fmt="csv",
    vector_rows=False,
    **kwargs,
):
    """
    This function creates a sample dataset, based on MAG240 dataset.

    Parameters:
    -----------
    root_dir : string
        directory in which all the files for the chunked dataset will be stored.
    """
    # Step0: prepare chunked graph data format.
    # A synthetic mini MAG240.
    num_institutions = 1200
    num_authors = 1200
    num_papers = 1200

    def rand_edges(num_src, num_dst, num_edges):
        eids = np.random.choice(num_src * num_dst, num_edges, replace=False)
        src = torch.from_numpy(eids // num_dst)
        dst = torch.from_numpy(eids % num_dst)

        return src, dst

    num_cite_edges = 24 * 1000
    num_write_edges = 12 * 1000
    num_affiliate_edges = 2400

    # Structure.
    data_dict = {
        ("paper", "cites", "paper"): rand_edges(
            num_papers, num_papers, num_cite_edges
        ),
        ("author", "writes", "paper"): rand_edges(
            num_authors, num_papers, num_write_edges
        ),
        ("author", "affiliated_with", "institution"): rand_edges(
            num_authors, num_institutions, num_affiliate_edges
        ),
        ("institution", "writes", "paper"): rand_edges(
            num_institutions, num_papers, num_write_edges
        ),
    }
    src, dst = data_dict[("author", "writes", "paper")]
    data_dict[("paper", "rev_writes", "author")] = (dst, src)
    g = dgl.heterograph(data_dict)

    # paper feat, label, year
    num_paper_feats = 3
    paper_feat = np.random.randn(num_papers, num_paper_feats)
    num_classes = 4
    paper_label = np.random.choice(num_classes, num_papers)
    paper_year = np.random.choice(2022, num_papers)
    paper_orig_ids = np.arange(0, num_papers)
    writes_orig_ids = np.arange(0, num_write_edges)

    # masks.
    paper_train_mask = np.random.choice([True, False], num_papers)
    paper_test_mask = np.random.choice([True, False], num_papers)
    paper_val_mask = np.random.choice([True, False], num_papers)

    author_train_mask = np.random.choice([True, False], num_authors)
    author_test_mask = np.random.choice([True, False], num_authors)
    author_val_mask = np.random.choice([True, False], num_authors)

    inst_train_mask = np.random.choice([True, False], num_institutions)
    inst_test_mask = np.random.choice([True, False], num_institutions)
    inst_val_mask = np.random.choice([True, False], num_institutions)

    write_train_mask = np.random.choice([True, False], num_write_edges)
    write_test_mask = np.random.choice([True, False], num_write_edges)
    write_val_mask = np.random.choice([True, False], num_write_edges)

    # Edge features.
    cite_count = np.random.choice(10, num_cite_edges)
    write_year = np.random.choice(2022, num_write_edges)
    write2_year = np.random.choice(2022, num_write_edges)

    # Save features.
    input_dir = os.path.join(root_dir, "data_test")
    os.makedirs(input_dir)
    for sub_d in ["paper", "cites", "writes", "writes2"]:
        os.makedirs(os.path.join(input_dir, sub_d))

    paper_feat_path = os.path.join(input_dir, "paper/feat.npy")
    with open(paper_feat_path, "wb") as f:
        np.save(f, paper_feat)
    g.nodes["paper"].data["feat"] = torch.from_numpy(paper_feat)

    paper_label_path = os.path.join(input_dir, "paper/label.npy")
    with open(paper_label_path, "wb") as f:
        np.save(f, paper_label)
    g.nodes["paper"].data["label"] = torch.from_numpy(paper_label)

    paper_year_path = os.path.join(input_dir, "paper/year.npy")
    with open(paper_year_path, "wb") as f:
        np.save(f, paper_year)
    g.nodes["paper"].data["year"] = torch.from_numpy(paper_year)

    paper_orig_ids_path = os.path.join(input_dir, "paper/orig_ids.npy")
    with open(paper_orig_ids_path, "wb") as f:
        np.save(f, paper_orig_ids)
    g.nodes["paper"].data["orig_ids"] = torch.from_numpy(paper_orig_ids)

    cite_count_path = os.path.join(input_dir, "cites/count.npy")
    with open(cite_count_path, "wb") as f:
        np.save(f, cite_count)
    g.edges["cites"].data["count"] = torch.from_numpy(cite_count)

    write_year_path = os.path.join(input_dir, "writes/year.npy")
    with open(write_year_path, "wb") as f:
        np.save(f, write_year)
    g.edges[("author", "writes", "paper")].data["year"] = torch.from_numpy(
        write_year
    )
    g.edges["rev_writes"].data["year"] = torch.from_numpy(write_year)

    writes_orig_ids_path = os.path.join(input_dir, "writes/orig_ids.npy")
    with open(writes_orig_ids_path, "wb") as f:
        np.save(f, writes_orig_ids)
    g.edges[("author", "writes", "paper")].data["orig_ids"] = torch.from_numpy(
        writes_orig_ids
    )

    write2_year_path = os.path.join(input_dir, "writes2/year.npy")
    with open(write2_year_path, "wb") as f:
        np.save(f, write2_year)
    g.edges[("institution", "writes", "paper")].data["year"] = torch.from_numpy(
        write2_year
    )

    etype = ("author", "writes", "paper")
    write_train_mask_path = os.path.join(input_dir, "writes/train_mask.npy")
    with open(write_train_mask_path, "wb") as f:
        np.save(f, write_train_mask)
    g.edges[etype].data["train_mask"] = torch.from_numpy(write_train_mask)

    write_test_mask_path = os.path.join(input_dir, "writes/test_mask.npy")
    with open(write_test_mask_path, "wb") as f:
        np.save(f, write_test_mask)
    g.edges[etype].data["test_mask"] = torch.from_numpy(write_test_mask)

    write_val_mask_path = os.path.join(input_dir, "writes/val_mask.npy")
    with open(write_val_mask_path, "wb") as f:
        np.save(f, write_val_mask)
    g.edges[etype].data["val_mask"] = torch.from_numpy(write_val_mask)

    for sub_d in ["author", "institution"]:
        os.makedirs(os.path.join(input_dir, sub_d))
    paper_train_mask_path = os.path.join(input_dir, "paper/train_mask.npy")
    with open(paper_train_mask_path, "wb") as f:
        np.save(f, paper_train_mask)
    g.nodes["paper"].data["train_mask"] = torch.from_numpy(paper_train_mask)

    paper_test_mask_path = os.path.join(input_dir, "paper/test_mask.npy")
    with open(paper_test_mask_path, "wb") as f:
        np.save(f, paper_test_mask)
    g.nodes["paper"].data["test_mask"] = torch.from_numpy(paper_test_mask)

    paper_val_mask_path = os.path.join(input_dir, "paper/val_mask.npy")
    with open(paper_val_mask_path, "wb") as f:
        np.save(f, paper_val_mask)
    g.nodes["paper"].data["val_mask"] = torch.from_numpy(paper_val_mask)

    author_train_mask_path = os.path.join(input_dir, "author/train_mask.npy")
    with open(author_train_mask_path, "wb") as f:
        np.save(f, author_train_mask)
    g.nodes["author"].data["train_mask"] = torch.from_numpy(author_train_mask)

    author_test_mask_path = os.path.join(input_dir, "author/test_mask.npy")
    with open(author_test_mask_path, "wb") as f:
        np.save(f, author_test_mask)
    g.nodes["author"].data["test_mask"] = torch.from_numpy(author_test_mask)

    author_val_mask_path = os.path.join(input_dir, "author/val_mask.npy")
    with open(author_val_mask_path, "wb") as f:
        np.save(f, author_val_mask)
    g.nodes["author"].data["val_mask"] = torch.from_numpy(author_val_mask)

    inst_train_mask_path = os.path.join(input_dir, "institution/train_mask.npy")
    with open(inst_train_mask_path, "wb") as f:
        np.save(f, inst_train_mask)
    g.nodes["institution"].data["train_mask"] = torch.from_numpy(
        inst_train_mask
    )

    inst_test_mask_path = os.path.join(input_dir, "institution/test_mask.npy")
    with open(inst_test_mask_path, "wb") as f:
        np.save(f, inst_test_mask)
    g.nodes["institution"].data["test_mask"] = torch.from_numpy(inst_test_mask)

    inst_val_mask_path = os.path.join(input_dir, "institution/val_mask.npy")
    with open(inst_val_mask_path, "wb") as f:
        np.save(f, inst_val_mask)
    g.nodes["institution"].data["val_mask"] = torch.from_numpy(inst_val_mask)

    node_data = {
        "paper": {
            "feat": paper_feat_path,
            "train_mask": paper_train_mask_path,
            "test_mask": paper_test_mask_path,
            "val_mask": paper_val_mask_path,
            "label": paper_label_path,
            "year": paper_year_path,
            "orig_ids": paper_orig_ids_path,
        },
        "author": {
            "train_mask": author_train_mask_path,
            "test_mask": author_test_mask_path,
            "val_mask": author_val_mask_path,
        },
        "institution": {
            "train_mask": inst_train_mask_path,
            "test_mask": inst_test_mask_path,
            "val_mask": inst_val_mask_path,
        },
    }

    edge_data = {
        "cites": {"count": cite_count_path},
        ("author", "writes", "paper"): {
            "year": write_year_path,
            "orig_ids": writes_orig_ids_path,
            "train_mask": write_train_mask_path,
            "test_mask": write_test_mask_path,
            "val_mask": write_val_mask_path,
        },
        "rev_writes": {"year": write_year_path},
        ("institution", "writes", "paper"): {"year": write2_year_path},
    }

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
    logging.debug("Done with creating chunked graph")

    return g
