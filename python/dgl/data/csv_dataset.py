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
from ..base import dgl_warning


class MetaLabel(dt.BaseModel):
    type: str
    field: Optional[str] = 'label'
    num_classes: Optional[int] = 0


class MetaEdge(dt.BaseModel):
    file_name: str
    separator: Optional[str] = ','
    etype: Optional[List[str]] = ['_V', '_E', '_V']
    graph_id_field: Optional[str] = 'graph_id'
    src_id_field: Optional[str] = 'src'
    dst_id_field: Optional[str] = 'dst'
    feat_field_prefix: Optional[str] = 'feat_'
    labels: Optional[MetaLabel] = None
    split_type_field: Optional[str] = 'split_type'


class MetaNode(dt.BaseModel):
    file_name: str
    separator: Optional[str] = ','
    ntype: Optional[str] = '_V'
    graph_id_field: Optional[str] = 'graph_id'
    node_id_field: Optional[str] = 'id'
    feat_field_prefix: Optional[str] = 'feat_'
    labels: Optional[MetaLabel] = None
    split_type_field: Optional[str] = 'split_type'


class MetaGraph(dt.BaseModel):
    file_name: str
    separator: Optional[str] = ','
    graph_id_field: Optional[str] = 'graph_id'
    feat_field_prefix: Optional[str] = 'feat_'
    labels: MetaLabel
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
        assert meta_yaml.version == '0.0.1'
        has_multi_graphs = meta_yaml.graph_data is not None
        if has_multi_graphs:
            meta_graph = meta_yaml.graph_data
            meta_label = meta_graph.labels
            assert meta_label.type == 'classification'
            assert meta_label.num_classes > 0
        for meta_edge in meta_yaml.edge_data:
            if meta_edge is None:
                continue
            if has_multi_graphs:
                assert meta_edge.graph_id_field is not None
            if meta_edge.labels is not None:
                meta_label = meta_edge.labels
                assert meta_label.type in ['classification', 'regression']
                if meta_label.type == 'classification':
                    assert meta_label.num_classes is not None
                assert meta_edge.split_type_field is not None
        for meta_node in meta_yaml.node_data:
            if meta_node is None:
                continue
            if has_multi_graphs:
                assert meta_node.graph_id_field is not None
            if meta_node.labels is not None:
                meta_label = meta_node.labels
                assert meta_label.type in ['classification', 'regression']
                if meta_label.type == 'classification':
                    assert meta_label.num_classes is not None
                assert meta_node.split_type_field is not None
        return meta_yaml


class EdgeData:
    """Parse edge data from edges_xxx.csv according to YAML::edges
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
            file_path, index_col=0, nrows=0).columns.tolist()
        separator = meta_edge.separator
        self.type = meta_edge.etype
        if meta_edge.graph_id_field in csv_header:
            self.graph_id = pd.read_csv(file_path, sep=separator, usecols=[
                meta_edge.graph_id_field]).to_numpy().squeeze()
        self.src = pd.read_csv(file_path, sep=separator, usecols=[
            meta_edge.src_id_field]).to_numpy().squeeze()
        self.dst = pd.read_csv(file_path, sep=separator, usecols=[
            meta_edge.dst_id_field]).to_numpy().squeeze()
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
        feat_prefix = meta_edge.feat_field_prefix
        feat_data = pd.read_csv(
            file_path, sep=separator, usecols=lambda x: feat_prefix in x)
        if not feat_data.empty:
            self.feat = np.stack([feat_data[key].to_numpy()
                                 for key in feat_data], axis=1)


class NodeData:
    """Parse node data from nodes_xxx.csv according to YAML::nodes
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
            file_path, index_col=0, nrows=0).columns.tolist()
        separator = meta_node.separator
        if meta_node.graph_id_field in csv_header:
            self.graph_id = pd.read_csv(file_path, sep=separator, usecols=[
                meta_node.graph_id_field]).to_numpy().squeeze()
        self.id = pd.read_csv(file_path, sep=separator, usecols=[
            meta_node.node_id_field]).to_numpy().squeeze()
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
        feat_prefix = meta_node.feat_field_prefix
        feat_data = pd.read_csv(
            file_path, sep=separator, usecols=lambda x: feat_prefix in x)
        if not feat_data.empty:
            self.feat = np.stack([feat_data[key].to_numpy()
                                 for key in feat_data], axis=1)


class GraphData:
    """Parse graph data from graphs.csv according to YAML::graphs
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
            file_path, index_col=0, nrows=0).columns.tolist()
        separator = meta_graph.separator
        self.graph_id = pd.read_csv(file_path, sep=separator, usecols=[
            meta_graph.graph_id_field]).to_numpy().squeeze()
        meta_label = meta_graph.labels
        self.label = pd.read_csv(file_path, sep=separator, usecols=[
            meta_label.field]).to_numpy().squeeze()
        self.num_classes = meta_label.num_classes
        if meta_graph.split_type_field in csv_header:
            self.split_type = pd.read_csv(file_path, sep=separator, usecols=[
                meta_graph.split_type_field]).to_numpy().squeeze()
        feat_prefix = meta_graph.feat_field_prefix
        feat_data = pd.read_csv(
            file_path, sep=separator, usecols=lambda x: feat_prefix in x)
        if not feat_data.empty:
            self.feat = np.stack([feat_data[key].to_numpy()
                                 for key in feat_data], axis=1)


class CSVDataset(DGLDataset):
    """The CSV graph dataset read data from CSV files according to passed in YAML config.

    Parameters
    -----------
    meta_yaml: str
        Absolute path of YAML config file.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.
    """

    def __init__(self, meta_yaml, force_reload=False, verbose=True):
        self.graph = None
        self.graphs = None
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
        if graph_data is None:
            edge_dict = {}
            for e_data in edge_data:
                assert len(e_data.type) == 3
                edge_dict[(e_data.type[0], e_data.type[1], e_data.type[2])] = (
                    F.tensor(e_data.src), F.tensor(e_data.dst))
            node_dict = {}
            for n_data in node_data:
                node_dict[n_data.type] = len(n_data.id)
            self.graph = dgl_heterograph(edge_dict, num_nodes_dict=node_dict)

            def assign_data(src_data, dst_data):
                type = (src_data.type[0], src_data.type[1], src_data.type[2]) if len(
                    src_data.type) == 3 else src_data.type
                if src_data.feat is not None:
                    dst_data[type].data['feat'] = F.tensor(src_data.feat)
                if src_data.label is not None:
                    dst_data[type].data['label'] = F.tensor(src_data.label)
                if src_data.num_classes is not None:
                    dst_data[type].data['num_classes'] = F.tensor(
                        src_data.num_classes)
                if src_data.split_type is not None:
                    dst_data[type].data['train_mask'] = F.tensor(
                        src_data.split_type == 0)
                    dst_data[type].data['val_mask'] = F.tensor(
                        src_data.split_type == 1)
                    dst_data[type].data['test_mask'] = F.tensor(
                        src_data.split_type == 2)
            for n_data in node_data:
                assign_data(n_data, self.graph.nodes)
            for e_data in edge_data:
                assign_data(e_data, self.graph.edges)
        else:
            edge_dict = {}
            for e_data in edge_data:
                graph_id_arr = np.unique(e_data.graph_id)
                for g_id in graph_id_arr:
                    idx = e_data.graph_id == g_id
                    edge_dict[g_id] = {}
                    edge_dict[g_id]['edge'] = (
                        F.tensor(e_data.src[idx]), F.tensor(e_data.dst[idx]))
                    if e_data.feat is not None:
                        edge_dict[g_id]['feat'] = F.tensor(e_data.feat[idx])
            node_dict = {}
            for n_data in node_data:
                graph_id_arr = np.unique(n_data.graph_id)
                for g_id in graph_id_arr:
                    node_dict[g_id] = {}
                    idx = n_data.graph_id == g_id
                    node_dict[g_id]['node'] = len(n_data.id[idx])
                    if n_data.feat is not None:
                        node_dict[g_id]['feat'] = F.tensor(n_data.feat[idx])
            self.graphs = []
            self.feat = F.tensor(
                graph_data.feat) if graph_data.feat is not None else None
            self.labels = F.tensor(
                graph_data.label) if graph_data.label is not None else None
            for g_id in graph_data.graph_id:
                if g_id not in node_dict:
                    raise RuntimeError(
                        "No node data is found for graph_id~{}.".format(g_id))
                e_data = (F.tensor([]), F.tensor(
                    [])) if g_id not in edge_dict else edge_dict[g_id]['edge']
                src_, dst_ = e_data
                g = dgl_heterograph({('_V', '_E', '_V'): e_data}, num_nodes_dict={
                                    '_V': node_dict[g_id]['node']})
                if 'feat' in node_dict[g_id]:
                    g.nodes['_V'].data['feat'] = node_dict[g_id]['feat']
                if 'feat' in edge_dict[g_id]:
                    g.edges['_E'].data['feat'] = edge_dict[g_id]['feat']
                self.graphs.append(g)
            self.mask = {}
            if graph_data.split_type is not None:
                self.mask['train_mask'] = F.tensor(graph_data.split_type == 0)
                self.mask['val_mask'] = F.tensor(graph_data.split_type == 1)
                self.mask['test_mask'] = F.tensor(graph_data.split_type == 2)
            self.num_classes = graph_data.num_classes

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
        assert self.graph is not None or self.graphs is not None
        if self.graph is not None:
            save_graphs(graph_path, self.graph)
        else:
            labels_dict = {'labels': self.labels}
            if self.feat is not None:
                labels_dict['feat'] = self.feat
            save_graphs(graph_path, self.graphs,
                        labels=labels_dict)
        info = {}
        info['is_graph_list'] = self.graphs is not None
        if info['is_graph_list']:
            info['mask'] = self.mask
            info['num_classes'] = self.num_classes
        save_info(info_path, info)

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.name + '.pkl')
        graphs, labels = load_graphs(graph_path)
        info = load_info(info_path)
        assert 'is_graph_list' in info
        if info['is_graph_list']:
            self.graphs = graphs
            self.mask = info['mask']
            self.num_classes = info['num_classes']
            self.labels = labels['labels']
            self.feat = labels['feat'] if 'feat' in labels else None
        else:
            self.graph = graphs[0]

    def __getitem__(self, i):
        if self.graph is not None:
            return self.graph
        elif self.feat is None:
            return (self.graphs[i], self.labels[i])
        else:
            return (self.graphs[i], self.labels[i], self.feat[i])

    def __len__(self):
        return 1 if self.graph is not None else len(self.graphs)

    def _sanity_check_after_process(self):
        #TODO:
        pass
