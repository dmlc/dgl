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
from .utils import save_graphs, load_graphs, save_info, load_info
from ..base import dgl_warning, DGLError


class MetaLabel(dt.BaseModel):
    type: str
    field: Optional[str] = 'label'
    num_classes: Optional[int] = 0


class MetaFeat(dt.BaseModel):
    field: Optional[str] = 'feat'
    separator: Optional[str] = ','


class MetaEdge(dt.BaseModel):
    file_name: str
    separator: Optional[str] = ','
    etype: Optional[List[str]] = ['_V', '_E', '_V']
    graph_id_field: Optional[str] = 'graph_id'
    src_id_field: Optional[str] = 'src'
    dst_id_field: Optional[str] = 'dst'
    feats: Optional[MetaFeat] = None
    labels: Optional[MetaLabel] = None
    split_type_field: Optional[str] = 'split_type'


class MetaNode(dt.BaseModel):
    file_name: str
    separator: Optional[str] = ','
    ntype: Optional[str] = '_V'
    graph_id_field: Optional[str] = 'graph_id'
    node_id_field: Optional[str] = 'id'
    feats: Optional[MetaFeat] = None
    labels: Optional[MetaLabel] = None
    split_type_field: Optional[str] = 'split_type'


class MetaGraph(dt.BaseModel):
    file_name: str
    separator: Optional[str] = ','
    graph_id_field: Optional[str] = 'graph_id'
    feats: Optional[MetaFeat] = None
    labels: Optional[MetaLabel] = None
    split_type_field: Optional[str] = 'split_type'


class MetaYaml(dt.BaseModel):
    version: str
    dataset_name: str
    edge_data: List[MetaEdge]
    node_data: List[MetaNode]
    graph_data: Optional[MetaGraph] = None


def _yaml_sanity_check(yaml_file):
    with open(yaml_file) as f:
        yaml_data = yaml.load(f, Loader=SafeLoader)
        meta_yaml = MetaYaml(**yaml_data)
        assert meta_yaml.version == '1.0.0'
        for meta_edge in meta_yaml.edge_data:
            if meta_edge is None:
                continue
            if meta_edge.labels is not None:
                meta_label = meta_edge.labels
                assert meta_label.type in ['classification', 'regression']
                if meta_label.type == 'classification':
                    assert meta_label.num_classes is not None
                assert meta_edge.split_type_field is not None
        for meta_node in meta_yaml.node_data:
            if meta_node is None:
                continue
            if meta_node.labels is not None:
                meta_label = meta_node.labels
                assert meta_label.type in ['classification', 'regression']
                if meta_label.type == 'classification':
                    assert meta_label.num_classes is not None
                assert meta_node.split_type_field is not None
        return meta_yaml


class EdgeData:
    """Parse edge data from edges_xxx.csv according to YAML::edge_data
    """

    def __init__(self, root_path, meta_edge):
        self.type = None
        self.graph_id = None
        self.src = None
        self.dst = None
        self.label = None
        self.num_classes = None
        self.split_type = None
        self.feat = None
        self._process(root_path, meta_edge)

    def _process(self, root_path, meta_edge):
        if meta_edge is None:
            return
        file_path = os.path.join(root_path, meta_edge.file_name)
        csv_header = pd.read_csv(
            file_path, nrows=0).columns.tolist()
        separator = meta_edge.separator
        self.type = meta_edge.etype
        if meta_edge.graph_id_field in csv_header:
            self.graph_id = pd.read_csv(file_path, sep=separator, usecols=[
                meta_edge.graph_id_field]).to_numpy().squeeze()
        self.src = pd.read_csv(file_path, sep=separator, usecols=[
            meta_edge.src_id_field]).to_numpy().squeeze()
        self.dst = pd.read_csv(file_path, sep=separator, usecols=[
            meta_edge.dst_id_field]).to_numpy().squeeze()
        if self.graph_id is None:
            self.graph_id = np.full(len(self.src), 0)
        if meta_edge.labels is not None:
            meta_label = meta_edge.labels
            self.label = pd.read_csv(file_path, sep=separator, usecols=[
                meta_label.field]).to_numpy().squeeze()
            if meta_label.type == 'classification':
                self.num_classes = np.full(
                    self.label.shape, meta_label.num_classes)
        if meta_edge.split_type_field in csv_header:
            self.split_type = pd.read_csv(file_path, sep=separator, usecols=[
                meta_edge.split_type_field]).to_numpy().squeeze()
        if meta_edge.feats is not None:
            meta_feat = meta_edge.feats
            feat_data = pd.read_csv(
                file_path, sep=separator, usecols=[meta_feat.field]).to_numpy().squeeze()
            self.feat = np.array([row.split(meta_feat.separator)
                                 for row in feat_data]).astype(np.float)


class NodeData:
    """Parse node data from nodes_xxx.csv according to YAML::node_data
    """

    def __init__(self, root_path, meta_node):
        self.graph_id = None
        self.id = None
        self.type = None
        self.label = None
        self.num_classes = None
        self.split_type = None
        self.feat = None
        self._process(root_path, meta_node)

    def _process(self, root_path, meta_node):
        if meta_node is None:
            return
        file_path = os.path.join(root_path, meta_node.file_name)
        csv_header = pd.read_csv(
            file_path, nrows=0).columns.tolist()
        separator = meta_node.separator
        if meta_node.graph_id_field in csv_header:
            self.graph_id = pd.read_csv(file_path, sep=separator, usecols=[
                meta_node.graph_id_field]).to_numpy().squeeze()
        self.id = pd.read_csv(file_path, sep=separator, usecols=[
            meta_node.node_id_field]).to_numpy().squeeze()
        if self.graph_id is None:
            self.graph_id = np.full(len(self.id), 0)
        self.type = meta_node.ntype
        if meta_node.labels is not None:
            meta_label = meta_node.labels
            self.label = pd.read_csv(file_path, sep=separator, usecols=[
                meta_label.field]).to_numpy().squeeze()
            if meta_label.type == 'classification':
                self.num_classes = np.full(
                    self.label.shape, meta_label.num_classes)
        if meta_node.split_type_field in csv_header:
            self.split_type = pd.read_csv(file_path, sep=separator, usecols=[
                meta_node.split_type_field]).to_numpy().squeeze()
        if meta_node.feats is not None:
            meta_feat = meta_node.feats
            feat_data = pd.read_csv(
                file_path, sep=separator, usecols=[meta_feat.field]).to_numpy().squeeze()
            self.feat = np.array([row.split(meta_feat.separator)
                                 for row in feat_data]).astype(np.float)


class GraphData:
    """Parse graph data from graphs.csv according to YAML::graph_data
    """

    def __init__(self, root_path, meta_graph):
        self.graph_id = None
        self.feat = None
        self.label = None
        self.num_classes = None
        self.split_type = None
        self._process(root_path, meta_graph)

    def _process(self, root_path, meta_graph):
        if meta_graph is None:
            return
        file_path = os.path.join(root_path, meta_graph.file_name)
        csv_header = pd.read_csv(
            file_path, nrows=0).columns.tolist()
        separator = meta_graph.separator
        if meta_graph.graph_id_field in csv_header:
            self.graph_id = pd.read_csv(file_path, sep=separator, usecols=[
                meta_graph.graph_id_field]).to_numpy().squeeze()
        if self.graph_id is None:
            self.graph_id = np.full(1, 0)
        if meta_graph.labels is not None:
            meta_label = meta_graph.labels
            if meta_label.field in csv_header:
                self.label = pd.read_csv(file_path, sep=separator, usecols=[
                    meta_label.field]).to_numpy().squeeze()
                self.num_classes = np.full(
                    self.label.shape, meta_label.num_classes)
        if meta_graph.split_type_field in csv_header:
            self.split_type = pd.read_csv(file_path, sep=separator, usecols=[
                meta_graph.split_type_field]).to_numpy().squeeze()
        if meta_graph.feats is not None:
            meta_feat = meta_graph.feats
            feat_data = pd.read_csv(
                file_path, sep=separator, usecols=[meta_feat.field]).to_numpy().squeeze()
            self.feat = np.array([row.split(meta_feat.separator)
                                 for row in feat_data]).astype(np.float)


class CSVDataset(DGLDataset):
    """The CSV graph dataset read data from CSV files according to passed in YAML config.

    Parameters
    -----------
    data_path: str
        Absolute path where YAML config file and related CSV files lie.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.
    """
    META_YAML_NAME = 'meta.yaml'

    def __init__(self, data_path, force_reload=False, verbose=True):
        self.graphs = None
        self.labels = None
        self.feats = None
        self.mask = None
        self.num_classes = None
        meta_yaml = os.path.join(data_path, CSVDataset.META_YAML_NAME)
        if not os.path.exists(meta_yaml):
            raise DGLError(
                "'{}' cannot be found under {}.".format(CSVDataset.META_YAML_NAME, data_path))
        meta = _yaml_sanity_check(meta_yaml)
        self.meta = meta
        ds_name = meta.dataset_name
        super().__init__(ds_name, raw_dir=os.path.dirname(
            meta_yaml), force_reload=force_reload, verbose=verbose)

    def process(self):
        """Parse node/edge data from CSV files and construct DGL.Graphs
        """
        meta = self.meta
        root = self.raw_dir
        edge_data = []
        for meta_edge in meta.edge_data:
            edge_data.append(EdgeData(root, meta_edge))

        #parse NodeData
        node_data = []
        for meta_node in meta.node_data:
            node_data.append(NodeData(root, meta_node))

        #parse GraphData
        graph_data = GraphData(
            root, meta.graph_data) if meta.graph_data is not None else None

        #construct dgl.heterograph
        edge_dict = {}
        for e_data in edge_data:
            assert len(e_data.type) == 3
            graph_ids = np.unique(e_data.graph_id)
            for graph_id in graph_ids:
                idx = e_data.graph_id == graph_id
                edata = {}
                edata['edges'] = (
                    F.tensor(e_data.src[idx]), F.tensor(e_data.dst[idx]))
                for key in ['feat', 'label', 'num_classes', 'split_type']:
                    if getattr(e_data, key) is not None:
                        edata[key] = F.tensor(
                            getattr(e_data, key)[idx])
                if graph_id not in edge_dict:
                    edge_dict[graph_id] = {}
                edge_dict[graph_id][(
                    e_data.type[0], e_data.type[1], e_data.type[2])] = edata
        node_dict = {}
        for n_data in node_data:
            graph_ids = np.unique(n_data.graph_id)
            for graph_id in graph_ids:
                idx = n_data.graph_id == graph_id
                ndata = {}
                ndata['num_nodes'] = len(n_data.id[idx])
                for key in ['feat', 'label', 'num_classes', 'split_type']:
                    if getattr(n_data, key) is not None:
                        ndata[key] = F.tensor(
                            getattr(n_data, key)[idx])
                if graph_id not in node_dict:
                    node_dict[graph_id] = {}
                node_dict[graph_id][n_data.type] = ndata
        graphs = {}
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
                nodes[ntype] = ndata['num_nodes']
            graph = dgl_heterograph(
                edges, num_nodes_dict=nodes)

            def assign_data(type, src_data, dst_data):
                for key in ['feat', 'label', 'num_classes']:
                    if key in src_data:
                        dst_data[type].data[key] = src_data[key]
                if 'split_type' in src_data:
                    data = src_data['split_type']
                    dst_data[type].data['train_mask'] = F.tensor(
                        data == 0)
                    dst_data[type].data['val_mask'] = F.tensor(
                        data == 1)
                    dst_data[type].data['test_mask'] = F.tensor(
                        data == 2)
            for type, data in node_dict[graph_id].items():
                assign_data(type, data, graph.nodes)
            for (type), data in edge_dict[graph_id].items():
                assign_data(type, data, graph.edges)
            graphs[graph_id] = graph
        if graph_data is not None:
            if len(graphs) > len(graph_data.graph_id):
                raise DGLError(
                    "More graph ids are found in node/edge data than graph data. Please specify all graph ids in graph data CSV.")
            for graph_id in graph_data.graph_id:
                if graph_id not in graphs:
                    graphs[graph_id] = dgl_heterograph(
                        {('_V', '_E', '_V'): ([], [])})
            if graph_data.feat is not None:
                self.feats = F.tensor(graph_data.feat)
            if graph_data.label is not None:
                self.labels = F.tensor(graph_data.label)
            if graph_data.split_type is not None:
                self.mask = {}
                self.mask['train_mask'] = F.tensor(graph_data.split_type == 0)
                self.mask['val_mask'] = F.tensor(graph_data.split_type == 1)
                self.mask['test_mask'] = F.tensor(graph_data.split_type == 2)
            self.num_classes = graph_data.num_classes
        self.graphs = []
        for graph_id in sorted(graphs):
            self.graphs.append(graphs[graph_id])
        self._sanity_check_after_process()

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.name + '.pkl')
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True

        return False

    def save(self):
        graph_path = os.path.join(self.save_path,
                                  self.name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.name + '.pkl')
        assert self.graphs is not None

        save_graphs(graph_path, self.graphs,
                    labels={'labels': self.labels} if self.labels is not None else None)
        info = {}
        for key in ['feats', 'mask', 'num_classes']:
            if getattr(self, key) is not None:
                info[key] = getattr(self, key)
        save_info(info_path, info)

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.name + '.pkl')
        graphs, labels = load_graphs(graph_path)
        self.graphs = graphs
        if len(labels)>0:
            self.labels = labels['labels']
        info = load_info(info_path)
        for k, v in info.items():
            setattr(self, k, v)

    def __getitem__(self, i):
        if self.labels is None:
            return self.graphs[i]
        else:
            return (self.graphs[i], self.labels[i])

    def __len__(self):
        return len(self.graphs)

    def _sanity_check_after_process(self):
        #TODO:
        pass
