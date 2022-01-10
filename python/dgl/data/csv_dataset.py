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
from typing import Callable


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
    version: Optional[str] = '1.0.0'
    dataset_name: str
    separator: Optional[str] = ','
    node_data: List[MetaNode]
    edge_data: List[MetaEdge]
    graph_data: Optional[MetaGraph] = None


def load_yaml_with_sanity_check(yaml_file):
    """ Load yaml and do sanity check. Internal use only. """
    with open(yaml_file) as f:
        yaml_data = yaml.load(f, Loader=SafeLoader)
        try:
            meta_yaml = MetaYaml(**yaml_data)
        except dt.ValidationError as e:
            print(
                "Details of pydantic.ValidationError:\n{}".format(e.json()))
            raise DGLError(
                "Validation Error for YAML fields. Details are shown above.")
        if meta_yaml.version != '1.0.0':
            raise DGLError("Invalid CSVDataset version {}. Supported versions: '1.0.0'".format(
                meta_yaml.version))
        ntypes = [meta.ntype for meta in meta_yaml.node_data]
        if len(ntypes) > len(set(ntypes)):
            raise DGLError(
                "Each node CSV file must have a unique node type name, but found duplicate node type: {}.".format(ntypes))
        etypes = [tuple(meta.etype) for meta in meta_yaml.edge_data]
        if len(etypes) > len(set(etypes)):
            raise DGLError(
                "Each edge CSV file must have a unique edge type name, but found duplicate edge type: {}.".format(etypes))
        return meta_yaml


def _validate_data_length(data_dict):
    len_dict = {k: len(v) for k, v in data_dict.items()}
    lst = list(len_dict.values())
    res = lst.count(lst[0]) == len(lst)
    if not res:
        raise DGLError(
            "All data are required to have same length while some of them does not. Length of data={}".format(str(len_dict)))


class BaseData:
    """ Class of base data which is inherited by Node/Edge/GraphData. Internal use only. """
    @staticmethod
    def read_csv(file_name, base_dir, separator):
        csv_path = file_name
        if base_dir is not None:
            csv_path = os.path.join(base_dir, csv_path)
        return pd.read_csv(csv_path, sep=separator)

    @staticmethod
    def pop_from_dataframe(df: pd.DataFrame, item: str):
        ret = None
        try:
            ret = df.pop(item).to_numpy().squeeze()
        except KeyError:
            pass
        return ret


class NodeData(BaseData):
    """ Class of node data which is used for DGLGraph construction. Internal use only. """

    def __init__(self, node_id, data, type=None, graph_id=None):
        self.id = np.array(node_id, dtype=np.int64)
        self.data = data
        self.type = type if type is not None else '_V'
        self.graph_id = np.array(graph_id, dtype=np.int) if graph_id is not None else np.full(
            len(node_id), 0)
        _validate_data_length({**{'id': self.id, 'graph_id': self.graph_id}, **self.data})

    @staticmethod
    def load_from_csv(meta: MetaNode, data_parser: Callable, base_dir=None, separator=','):
        df = BaseData.read_csv(meta.file_name, base_dir, separator)
        node_ids = BaseData.pop_from_dataframe(df, meta.node_id_field)
        graph_ids = BaseData.pop_from_dataframe(df, meta.graph_id_field)
        if node_ids is None:
            raise DGLError("Missing node id field [{}] in file [{}].".format(
                meta.node_id_field, meta.file_name))
        ntype = meta.ntype
        ndata = data_parser(df)
        return NodeData(node_ids, ndata, type=ntype, graph_id=graph_ids)

    @staticmethod
    def to_dict(node_data: List['NodeData']) -> dict:
        # node_ids could be arbitrary numeric values, namely non-sorted, duplicated, not labeled from 0 to num_nodes-1
        node_dict = {}
        for n_data in node_data:
            graph_ids = np.unique(n_data.graph_id)
            for graph_id in graph_ids:
                idx = n_data.graph_id == graph_id
                ids = n_data.id[idx]
                u_ids, u_indices = np.unique(ids, return_index=True)
                if len(ids) > len(u_ids):
                    dgl_warning(
                        "There exist duplicated ids and only the first ones are kept.")
                if graph_id not in node_dict:
                    node_dict[graph_id] = {}
                node_dict[graph_id][n_data.type] = {'mapping': {index: i for i,
                                                                index in enumerate(ids[u_indices])},
                                                    'data': {k: F.tensor(v[idx][u_indices])
                                                             for k, v in n_data.data.items()}}
        return node_dict


class EdgeData(BaseData):
    """ Class of edge data which is used for DGLGraph construction. Internal use only. """

    def __init__(self, src_id, dst_id, data, type=None, graph_id=None):
        self.src = np.array(src_id, dtype=np.int64)
        self.dst = np.array(dst_id, dtype=np.int64)
        self.data = data
        self.type = type if type is not None else ('_V', '_E', '_V')
        self.graph_id = np.array(graph_id, dtype=np.int) if graph_id is not None else np.full(
            len(src_id), 0)
        _validate_data_length({**{'src': self.src, 'dst': self.dst, 'graph_id': self.graph_id}, **self.data})

    @staticmethod
    def load_from_csv(meta: MetaEdge, data_parser: Callable, base_dir=None, separator=','):
        df = BaseData.read_csv(meta.file_name, base_dir, separator)
        src_ids = BaseData.pop_from_dataframe(df, meta.src_id_field)
        if src_ids is None:
            raise DGLError("Missing src id field [{}] in file [{}].".format(
                meta.src_id_field, meta.file_name))
        dst_ids = BaseData.pop_from_dataframe(df, meta.dst_id_field)
        if dst_ids is None:
            raise DGLError("Missing dst id field [{}] in file [{}].".format(
                meta.dst_id_field, meta.file_name))
        graph_ids = BaseData.pop_from_dataframe(df, meta.graph_id_field)
        etype = tuple(meta.etype)
        edata = data_parser(df)
        return EdgeData(src_ids, dst_ids, edata, type=etype, graph_id=graph_ids)

    @staticmethod
    def to_dict(edge_data: List['EdgeData'], node_dict: dict) -> dict:
        edge_dict = {}
        for e_data in edge_data:
            (src_type, e_type, dst_type) = e_data.type
            graph_ids = np.unique(e_data.graph_id)
            for graph_id in graph_ids:
                if graph_id in edge_dict and e_data.type in edge_dict[graph_id]:
                    raise DGLError(f"Duplicate edge type[{e_data.type}] for same graph[{graph_id}], please place the same edge_type for same graph into single EdgeData.")
                idx = e_data.graph_id == graph_id
                src_mapping = node_dict[graph_id][src_type]['mapping']
                dst_mapping = node_dict[graph_id][dst_type]['mapping']
                src_ids = [src_mapping[index] for index in e_data.src[idx]]
                dst_ids = [dst_mapping[index] for index in e_data.dst[idx]]
                if graph_id not in edge_dict:
                    edge_dict[graph_id] = {}
                edge_dict[graph_id][e_data.type] = {'edges': (F.tensor(src_ids), F.tensor(dst_ids)),
                                                    'data': {k: F.tensor(v[idx])
                                                             for k, v in e_data.data.items()}}
        return edge_dict


class GraphData(BaseData):
    """ Class of graph data which is used for DGLGraph construction. Internal use only. """

    def __init__(self, graph_id, data):
        self.graph_id = np.array(graph_id, dtype=np.int64)
        self.data = data
        _validate_data_length({**{'graph_id': self.graph_id}, **self.data})

    @staticmethod
    def load_from_csv(meta: MetaGraph, data_parser: Callable, base_dir=None, separator=','):
        df = BaseData.read_csv(meta.file_name, base_dir, separator)
        graph_ids = BaseData.pop_from_dataframe(df, meta.graph_id_field)
        if graph_ids is None:
            raise DGLError("Missing graph id field [{}] in file [{}].".format(
                meta.graph_id_field, meta.file_name))
        gdata = data_parser(df)
        return GraphData(graph_ids, gdata)

    @staticmethod
    def to_dict(graph_data: 'GraphData', graphs_dict: dict) -> dict:
        missing_ids = np.setdiff1d(
            np.array(list(graphs_dict.keys())), graph_data.graph_id)
        if len(missing_ids) > 0:
            raise DGLError(
                "Found following graph ids in node/edge CSVs but not in graph CSV: {}.".format(missing_ids))
        graph_ids = graph_data.graph_id
        graphs = []
        for graph_id in graph_ids:
            if graph_id not in graphs_dict:
                graphs_dict[graph_id] = dgl_heterograph(
                    {('_V', '_E', '_V'): ([], [])})
        for graph_id in graph_ids:
            graphs.append(graphs_dict[graph_id])
        data = {k: F.tensor(v) for k, v in graph_data.data.items()}
        return graphs, data


class DGLGraphConstructor:
    """ Class for constructing DGLGraph from Node/Edge/Graph data. Internal use only. """
    @staticmethod
    def construct_graphs(node_data, edge_data, graph_data=None):
        if not isinstance(node_data, list):
            node_data = [node_data]
        if not isinstance(edge_data, list):
            edge_data = [edge_data]
        node_dict = NodeData.to_dict(node_data)
        edge_dict = EdgeData.to_dict(edge_data, node_dict)
        graph_dict = DGLGraphConstructor._construct_graphs(
            node_dict, edge_dict)
        if graph_data is None:
            graph_data = GraphData(np.full(1, 0), {})
        graphs, data = GraphData.to_dict(
            graph_data, graph_dict)
        return graphs, data

    @staticmethod
    def _construct_graphs(node_dict, edge_dict):
        graph_dict = {}
        for graph_id in node_dict:
            if graph_id not in edge_dict:
                edge_dict[graph_id][('_V', '_E', '_V')] = {'edges': ([], [])}
            graph = dgl_heterograph({etype: edata['edges']
                                     for etype, edata in edge_dict[graph_id].items()},
                                    num_nodes_dict={ntype: len(ndata['mapping'])
                                                    for ntype, ndata in node_dict[graph_id].items()})

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
        2. tries to convert to list of numeric values(generated by
            np.array().tolist()) if cell data is a str separated by ','.
        3. read data and infer data type directly, otherwise.
    """

    def __call__(self, df: pd.DataFrame):
        data = {}
        for header in df:
            if 'Unnamed' in header:
                dgl_warning("Unamed column is found. Ignored...")
                continue
            dt = df[header].to_numpy().squeeze()
            if len(dt) > 0 and isinstance(dt[0], str):
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
    force_reload : bool, optional
        Whether to reload the dataset. Default: False
    verbose: bool, optional
        Whether to print out progress information. Default: True.
    node_data_parser : dict[str, callable], optional
        A dictionary used for node data parsing when loading from CSV files.
        The key is node type which specifies the header in CSV file and the
        value is a callable object which is used to parse corresponding
        column data. Default: None. If None, a default data parser is applied
        which load data directly and tries to convert list into array.
    edge_data_parser : dict[(str, str, str), callable], optional
        A dictionary used for edge data parsing when loading from CSV files.
        The key is edge type which specifies the header in CSV file and the
        value is a callable object which is used to parse corresponding
        column data. Default: None. If None, a default data parser is applied
        which load data directly and tries to convert list into array.
    graph_data_parser : callable, optional
        A callable object which is used to parse corresponding column graph
        data. Default: None. If None, a default data parser is applied
        which load data directly and tries to convert list into array.

    Attributes
    ----------
    graphs : :class:`dgl.DGLGraph`
        Graphs of the dataset
    data : dict
        any available graph-level data such as graph-level feature, labels.

    Examples
    [TODO]: link to a detailed web page.
    """
    META_YAML_NAME = 'meta.yaml'

    def __init__(self, data_path, force_reload=False, verbose=True, node_data_parser=None, edge_data_parser=None, graph_data_parser=None):
        self.graphs = None
        self.data = None
        self.node_data_parser = {} if node_data_parser is None else node_data_parser
        self.edge_data_parser = {} if edge_data_parser is None else edge_data_parser
        self.graph_data_parser = graph_data_parser
        self.default_data_parser = DefaultDataParser()
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
            data_parser = self.node_data_parser.get(
                ntype, self.default_data_parser)
            ndata = NodeData.load_from_csv(
                meta_node, base_dir=base_dir, separator=meta_yaml.separator, data_parser=data_parser)
            node_data.append(ndata)
        edge_data = []
        for meta_edge in meta_yaml.edge_data:
            if meta_edge is None:
                continue
            etype = tuple(meta_edge.etype)
            data_parser = self.edge_data_parser.get(
                etype, self.default_data_parser)
            edata = EdgeData.load_from_csv(
                meta_edge, base_dir=base_dir, separator=meta_yaml.separator, data_parser=data_parser)
            edge_data.append(edata)
        graph_data = None
        if meta_yaml.graph_data is not None:
            meta_graph = meta_yaml.graph_data
            data_parser = self.default_data_parser if self.graph_data_parser is None else self.graph_data_parser
            graph_data = GraphData.load_from_csv(
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
