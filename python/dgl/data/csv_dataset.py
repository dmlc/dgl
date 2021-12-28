import os
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np
from typing import List, Optional
import pydantic as dt
from .dgl_dataset import DGLDataset
from ..convert import heterograph as dgl_heterograph
from .. import backend as F
from .utils import save_graphs, load_graphs
from ..base import dgl_warning, DGLError
import abc
import ast


class MetaNode(dt.BaseModel):
    """ Class of node_data in YAML. Internal use only. """
    file_name: str
    ntype: Optional[str] = '_V'
    graph_id_field: Optional[str] = 'graph_id'
    node_id_field: Optional[str] = 'node_id'


class MetaEdge(dt.BaseModel):
    """ Class of edge_data in YAML. Internal use only. """
    file_name: str
    etype: Optional[List[str]] = ['_V', '_E', '_V']
    graph_id_field: Optional[str] = 'graph_id'
    src_id_field: Optional[str] = 'src_id'
    dst_id_field: Optional[str] = 'dst_id'


class MetaGraph(dt.BaseModel):
    """ Class of graph_data in YAML. Internal use only. """
    file_name: str
    graph_id_field: Optional[str] = 'graph_id'


class MetaYaml(dt.BaseModel):
    """ Class of YAML. Internal use only. """
    version: str
    dataset_name: str
    separator: Optional[str] = ','
    node_data: List[MetaNode]
    edge_data: List[MetaEdge]
    graph_data: Optional[MetaGraph] = None


def load_yaml_with_sanity_check(yaml_file):
    """ Load yaml and do sanity check. Internal use only. """
    with open(yaml_file) as f:
        yaml_data = yaml.load(f, Loader=SafeLoader)
        meta_yaml = MetaYaml(**yaml_data)
        if meta_yaml.version != '1.0.0':
            raise DGLError("Invalid version: {}. '1.0.0' is supported only for now.".format(
                meta_yaml.version))
        return meta_yaml


class NodeData:
    """ Class of node data which is used for DGLGraph construction. Internal use only. """

    def __init__(self, node_id, data, type=None, graph_id=None):
        self.id = np.array(node_id, dtype=np.int)
        self.data = data
        self.type = type if type is not None else '_V'
        self.graph_id = np.array(graph_id, dtype=np.int) if graph_id is not None else np.full(
            len(node_id), 0)
        assert len(self.id) == len(self.graph_id)
        for k, v in self.data.items():
            assert len(self.id) == len(v)


class EdgeData:
    """ Class of edge data which is used for DGLGraph construction. Internal use only. """

    def __init__(self, src_id, dst_id, data, type=None, graph_id=None):
        self.src = np.array(src_id, dtype=np.int)
        self.dst = np.array(dst_id, dtype=np.int)
        self.data = data
        self.type = type if type is not None else ('_V', '_E', '_V')
        self.graph_id = np.array(graph_id, dtype=np.int) if graph_id is not None else np.full(
            len(src_id), 0)
        assert len(self.src) == len(self.dst)
        assert len(self.src) == len(self.graph_id)
        for k, v in self.data.items():
            assert len(self.src) == len(v)


class GraphData:
    """ Class of graph data which is used for DGLGraph construction. Internal use only. """

    def __init__(self, graph_id, data):
        self.graph_id = np.array(graph_id, dtype=np.int)
        self.data = data
        for k, v in self.data.items():
            assert len(self.graph_id) == len(v)


class CSVDataLoader:
    """ Class for loading data from csv. Internal use only. """
    @staticmethod
    def read_csv(file_name, base_dir, separator):
        csv_path = file_name
        if base_dir is not None:
            csv_path = os.path.join(base_dir, csv_path)
        return pd.read_csv(csv_path, sep=separator)

    @staticmethod
    def load_node_data_from_csv(meta_node, data_parser, base_dir=None, separator=','):
        df = CSVDataLoader.read_csv(meta_node.file_name, base_dir, separator)
        ntype = meta_node.ntype
        ndata = data_parser(df)
        if meta_node.node_id_field not in ndata:
            raise DGLError(f"node_id_field[{meta_node.node_id_field}] does not exist in parsed node data dict.")
        node_ids = ndata[meta_node.node_id_field]
        ndata.pop(meta_node.node_id_field)
        if meta_node.graph_id_field in ndata:
            graph_ids = ndata[meta_node.graph_id_field]
            ndata.pop(meta_node.graph_id_field)
        else:
            graph_ids = None
        return NodeData(node_ids, ndata, type=ntype, graph_id=graph_ids)

    @staticmethod
    def load_edge_data_from_csv(meta_edge, data_parser, base_dir=None, separator=','):
        df = CSVDataLoader.read_csv(meta_edge.file_name, base_dir, separator)
        etype = tuple(meta_edge.etype)
        edata = data_parser(df)
        if meta_edge.src_id_field not in edata:
            raise DGLError(f"src_id_field[{meta_edge.src_id_field}] does not exist in parsed edge data dict.")
        if meta_edge.dst_id_field not in edata:
            raise DGLError(f"dst_id_field[{meta_edge.dst_id_field}] does not exist in parsed edge data dict.")
        src_ids = edata[meta_edge.src_id_field]
        dst_ids = edata[meta_edge.dst_id_field]
        graph_ids = edata[meta_edge.graph_id_field] if meta_edge.graph_id_field in edata else None
        for key in [meta_edge.src_id_field, meta_edge.dst_id_field, meta_edge.graph_id_field]:
            if key in edata:
                edata.pop(key)
        return EdgeData(src_ids, dst_ids, edata, type=etype, graph_id=graph_ids)

    @staticmethod
    def load_graph_data_from_csv(meta_graph, data_parser, base_dir=None, separator=','):
        df = CSVDataLoader.read_csv(meta_graph.file_name, base_dir, separator)
        gdata = data_parser(df)
        if meta_graph.graph_id_field not in gdata:
            raise DGLError(f"graph_id_field[{meta_graph.graph_id_field}] does not exist in parsed graph data dict.")
        graph_ids = gdata[meta_graph.graph_id_field]
        gdata.pop(meta_graph.graph_id_field)
        return GraphData(graph_ids, gdata)


class DGLGraphConstructor:
    """ Class for constructing DGLGraph from Node/Edge/Graph data. Internal use only. """
    @staticmethod
    def construct_graphs(node_data, edge_data, graph_data=None):
        if not isinstance(node_data, list):
            node_data = [node_data]
        if not isinstance(edge_data, list):
            edge_data = [edge_data]
        node_dict = DGLGraphConstructor._parse_node_data(node_data)
        edge_dict = DGLGraphConstructor._parse_edge_data(edge_data, node_dict)
        graph_dict = DGLGraphConstructor._construct_graphs(
            node_dict, edge_dict)
        if graph_data is None:
            graph_data = GraphData(np.full(1, 0), {})
        graphs, data = DGLGraphConstructor._parse_graph_data(
            graph_data, graph_dict)
        return graphs, data

    @staticmethod
    def _parse_node_data(node_data):
        # node_ids could be arbitrary numeric values, namely non-sorted, duplicated, not labeled from 0 to num_nodes-1
        node_dict = {}
        for n_data in node_data:
            graph_ids = np.unique(n_data.graph_id)
            for graph_id in graph_ids:
                if graph_id in node_dict and n_data.type in node_dict[graph_id]:
                    raise DGLError(f"Duplicate node type[{n_data.type}] for same graph[{graph_id}], please place the same node_type for same graph into single NodeData.")
                idx = n_data.graph_id == graph_id
                ids = n_data.id[idx]
                u_ids, u_indices = np.unique(ids, return_index=True)
                if len(ids) > len(u_ids):
                    dgl_warning(
                        "There exist duplicated ids and only the first ones are kept.")
                ndata = {}
                ndata['mapping'] = {index: i for i,
                                    index in enumerate(ids[u_indices])}
                data = {}
                for key, value in n_data.data.items():
                    data[key] = F.tensor(value[idx][u_indices])
                ndata['data'] = data
                if graph_id not in node_dict:
                    node_dict[graph_id] = {}
                node_dict[graph_id][n_data.type] = ndata
        return node_dict

    @staticmethod
    def _parse_edge_data(edge_data, node_dict):
        edge_dict = {}
        for e_data in edge_data:
            (src_type, e_type, dst_type) = e_data.type
            graph_ids = np.unique(e_data.graph_id)
            for graph_id in graph_ids:
                if graph_id in edge_dict and e_data.type in edge_dict[graph_id]:
                    raise DGLError(f"Duplicate edge type[{e_data.type}] for same graph[{graph_id}], please place the same edge_type for same graph into single EdgeData.")
                idx = e_data.graph_id == graph_id
                edata = {}
                src_mapping = node_dict[graph_id][src_type]['mapping']
                dst_mapping = node_dict[graph_id][dst_type]['mapping']
                src_ids = [src_mapping[index] for index in e_data.src[idx]]
                dst_ids = [dst_mapping[index] for index in e_data.dst[idx]]
                edata['edges'] = (
                    F.tensor(src_ids), F.tensor(dst_ids))
                data = {}
                for key, value in e_data.data.items():
                    data[key] = F.tensor(value[idx])
                edata['data'] = data
                if graph_id not in edge_dict:
                    edge_dict[graph_id] = {}
                edge_dict[graph_id][e_data.type] = edata
        return edge_dict

    @staticmethod
    def _parse_graph_data(graph_data, graphs_dict):
        if len(graphs_dict) > len(graph_data.graph_id):
            raise DGLError(
                "More graph ids are found in node/edge data than graph data. Please specify all graph ids in graph data CSV.")
        graph_ids = graph_data.graph_id
        graphs = []
        for graph_id in graph_ids:
            if graph_id not in graphs_dict:
                graphs_dict[graph_id] = dgl_heterograph(
                    {('_V', '_E', '_V'): ([], [])})
        for graph_id in graph_ids:
            graphs.append(graphs_dict[graph_id])
        data = {}
        for k, v in graph_data.data.items():
            data[k] = F.tensor(v)
        return graphs, data

    @staticmethod
    def _construct_graphs(node_dict, edge_dict):
        graph_dict = {}
        for graph_id in node_dict:
            if graph_id not in edge_dict:
                edata = {}
                edata['edges'] = ([], [])
                edge_dict[graph_id][('_V', '_E', '_V')] = edata

            edges = {}
            for etype, edata in edge_dict[graph_id].items():
                edges[etype] = edata['edges']
            nodes = {}
            for ntype, ndata in node_dict[graph_id].items():
                nodes[ntype] = len(ndata['mapping'])
            graph = dgl_heterograph(
                edges, num_nodes_dict=nodes)

            def assign_data(type, src_data, dst_data):
                for key, value in src_data.items():
                    dst_data[type].data[key] = value
            for type, data in node_dict[graph_id].items():
                assign_data(type, data['data'], graph.nodes)
            for (type), data in edge_dict[graph_id].items():
                assign_data(type, data['data'], graph.edges)
            graph_dict[graph_id] = graph
        return graph_dict


class DefaultDataParser:
    """ Default data parser for DGLCSVDataset. It
        1. ignores any columns which does not have a header.
        2. tries to convert to list of numeric values if cell data is a str.
        3. read data and infer data type directly, otherwise.
    """

    def __init__(self, separator='|'):
        self.separator = separator

    def __call__(self, df):
        data = {}
        for header in df:
            if 'Unnamed' in header:
                dgl_warning("Unamed column is found. Ignored...")
                continue
            dt = df[header].to_numpy().squeeze()
            if isinstance(dt[0], str):
                #probably consists of list of numeric values
                dt = np.array([ast.literal_eval(row) for row in dt])
            data[header] = dt
        return data


class DGLCSVDataset(DGLDataset):
    """ This class aims to parse data from CSV files, construct DGLGraph
        and behaves as a DGLDataset.

    Parameters
    ----------
    data_path : str
        Directory which contains 'meta.yaml' and CSV files
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.
    data_parser : dict
        A dictionary used for data parsing when loading data from CSV files.
        The key is node/edge/graph type which specifies the header in CSV file
        and the value is a callable object which is used to parse corresponding
        column data. Especially, the graph level data parser is required
        to be placed under key '_GRAPH_'. Default: None. If None, a default
        data parser is applied which load data directly and tries to convert
        list into array.

    Attributes
    ----------
    graphs : :class:`dgl.DGLGraph`
        Graphs of the dataset
    data : dict
        any available graph-level data such as graph-level feature, labels.

    Examples
    --------
    Single heterograph dataset
    >>> csv_dataset = data.DGLCSVDataset(test_dir)
    >>> g = csv_dataset[0]

    Multiple graphs dataset
    >>> csv_dataset = data.DGLCSVDataset(test_dir)
    >>> g, label = csv_dataset[0]
    """
    META_YAML_NAME = 'meta.yaml'

    def __init__(self, data_path, force_reload=False, verbose=True, data_parser=None):
        """
        data_parser : dict
            {'user' : NodeDataParser,
             ('user', 'like', 'item') : EdgeDataParser,
             '_GRAPH_' : GraphDataParser,
            }
        """
        self.graphs = None
        self.data = None
        self.data_parser = {} if data_parser is None else data_parser
        meta_yaml_path = os.path.join(data_path, DGLCSVDataset.META_YAML_NAME)
        if not os.path.exists(meta_yaml_path):
            raise DGLError(
                "'{}' cannot be found under {}.".format(DGLCSVDataset.META_YAML_NAME, data_path))
        self.meta_yaml = load_yaml_with_sanity_check(meta_yaml_path)
        ds_name = self.meta_yaml.dataset_name
        super().__init__(ds_name, raw_dir=os.path.dirname(
            meta_yaml_path), force_reload=force_reload, verbose=verbose)

    def process(self):
        """Parse node/edge data from CSV files and construct DGL.Graphs
        """
        meta_yaml = self.meta_yaml
        base_dir = self.raw_dir
        node_data = []
        for meta_node in meta_yaml.node_data:
            if meta_node is None:
                continue
            ntype = meta_node.ntype
            data_parser = DefaultDataParser() if ntype not in self.data_parser else self.data_parser[
                ntype]
            ndata = CSVDataLoader.load_node_data_from_csv(
                meta_node, base_dir=base_dir, separator=meta_yaml.separator, data_parser=data_parser)
            node_data.append(ndata)
        edge_data = []
        for meta_edge in meta_yaml.edge_data:
            if meta_edge is None:
                continue
            etype = tuple(meta_edge.etype)
            data_parser = DefaultDataParser() if etype not in self.data_parser else self.data_parser[
                etype]
            edata = CSVDataLoader.load_edge_data_from_csv(
                meta_edge, base_dir=base_dir, separator=meta_yaml.separator, data_parser=data_parser)
            edge_data.append(edata)
        graph_data = None
        if meta_yaml.graph_data is not None:
            meta_graph = meta_yaml.graph_data
            data_parser = DefaultDataParser(
            ) if '_GRAPH_' not in self.data_parser else self.data_parser['_GRAPH_']
            graph_data = CSVDataLoader.load_graph_data_from_csv(
                meta_graph, base_dir=base_dir, separator=meta_yaml.separator, data_parser=data_parser)
        # construct graphs
        self.graphs, self.data = DGLGraphConstructor.construct_graphs(
            node_data, edge_data, graph_data)

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.name + '.bin')
        if os.path.exists(graph_path):
            return True

        return False

    def save(self):
        if self.graphs is None:
            raise DGLError("No graphs available in dataset")
        graph_path = os.path.join(self.save_path,
                                  self.name + '.bin')
        save_graphs(graph_path, self.graphs,
                    labels=self.data)

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.name + '.bin')
        self.graphs, self.data = load_graphs(graph_path)

    def __getitem__(self, i):
        if 'label' in self.data:
            return self.graphs[i], self.data['label'][i]
        else:
            return self.graphs[i]

    def __len__(self):
        return len(self.graphs)
