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


class MetaLabel(dt.BaseModel):
    type: str
    field: str
    num_classes: Optional[int] = None


class MetaEdge(dt.BaseModel):
    file_name: str
    separator: str
    etype: Optional[List[str]] = None
    graph_id_field: Optional[str] = None
    src_id_field: str
    dst_id_field: str
    feat_prefix_field: Optional[str] = None
    labels: Optional[MetaLabel] = None
    split_type_field: Optional[str] = None


class MetaNode(dt.BaseModel):
    file_name: str
    separator: str
    ntype: Optional[str] = None
    graph_id_field: Optional[str] = None
    node_id_field: str
    feat_prefix_field: Optional[str] = None
    labels: Optional[MetaLabel] = None
    split_type_field: Optional[str] = None


class MetaGraph(dt.BaseModel):
    file_name: str
    separator: str
    graph_id_field: str
    labels: MetaLabel
    split_type_field: str


class MetaYaml(dt.BaseModel):
    version: str
    dataset_type: str
    dataset_name: Optional[str] = None
    edges: List[MetaEdge]
    nodes: List[MetaNode]
    graphs: Optional[MetaGraph] = None


def _yaml_sanity_check(yaml_file):
    with open(yaml_file) as f:
        yaml_data = yaml.load(f, Loader=SafeLoader)
        meta_yaml = MetaYaml(**yaml_data)
        assert meta_yaml.version == '0.0.1'
        assert meta_yaml.dataset_type in ['user_defined', 'dgl_data']
        if meta_yaml.dataset_type == 'dgl_data':
            assert meta_yaml.dataset_name is not None
        has_multi_graphs = meta_yaml.graphs is not None
        if has_multi_graphs:
            meta_graph = meta_yaml.graphs
            meta_label = meta_graph.labels
            assert meta_label.type == 'classification'
            assert meta_label.num_classes is not None
        for meta_edge in meta_yaml.edges:
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
        for meta_node in meta_yaml.nodes:
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
        separator = meta_edge.separator
        self.type = meta_edge.etype if meta_edge.etype is not None else [
            '_V', '_E', '_V']
        if meta_edge.graph_id_field is not None:
            self.graph_id = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
                meta_edge.graph_id_field]).to_numpy().squeeze())
        self.src = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
            meta_edge.src_id_field]).to_numpy().squeeze())
        self.dst = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
            meta_edge.dst_id_field]).to_numpy().squeeze())
        if meta_edge.labels is not None:
            meta_label = meta_edge.labels
            self.label = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
                meta_label.field]).to_numpy().squeeze())
            if meta_label.type == 'classification':
                self.num_classes = F.tensor(
                    np.full(self.label.shape, meta_label.num_classes))
            self.split_type = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
                meta_edge.split_type_field]).to_numpy().squeeze())
        if meta_edge.feat_prefix_field is not None:
            feat_prefix = meta_edge.feat_prefix_field
            feat_data = pd.read_csv(
                file_path, sep=separator, usecols=lambda x: feat_prefix in x)
            self.feat = F.tensor(np.stack([feat_data[key].to_numpy()
                                 for key in feat_data], axis=1))


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
        separator = meta_node.separator
        if meta_node.graph_id_field is not None:
            self.graph_id = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
                meta_node.graph_id_field]).to_numpy().squeeze())
        self.id = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
            meta_node.node_id_field]).to_numpy().squeeze())
        self.type = meta_node.ntype if meta_node.ntype is not None else '_V'
        if meta_node.labels is not None:
            meta_label = meta_node.labels
            self.label = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
                meta_label.field]).to_numpy().squeeze())
            if meta_label.type == 'classification':
                self.num_classes = F.tensor(
                    np.full(self.label.shape, meta_label.num_classes))
            self.split_type = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
                meta_node.split_type_field]).to_numpy().squeeze())
        if meta_node.feat_prefix_field is not None:
            feat_prefix = meta_node.feat_prefix_field
            feat_data = pd.read_csv(
                file_path, sep=separator, usecols=lambda x: feat_prefix in x)
            self.feat = F.tensor(np.stack([feat_data[key].to_numpy()
                                 for key in feat_data], axis=1))


class GraphData:
    """Parse graph data from graphs.csv according to YAML::graphs
    """

    def __init__(self, root_path, meta_graph):
        self.graph_id = None
        self.label = None
        self.num_classes = None
        self.split_type = None
        self._process(root_path, meta_graph)

    def _process(self, root_path, meta_graph):
        if meta_graph is None:
            return
        file_path = os.path.join(root_path, meta_graph.file_name)
        separator = meta_graph.separator
        self.graph_id = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
            meta_graph.graph_id_field]).to_numpy().squeeze())
        meta_label = meta_graph.labels
        self.label = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
            meta_label.field]).to_numpy().squeeze())
        self.num_classes = meta_label.num_classes
        self.split_type = F.tensor(pd.read_csv(file_path, sep=separator, usecols=[
            meta_graph.split_type_field]).to_numpy().squeeze())


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
        ds_name = meta.dataset_name if meta.dataset_name is not None else 'csv_dataset'
        super().__init__(ds_name, raw_dir=os.path.dirname(
            meta_yaml), force_reload=force_reload, verbose=verbose)

    def process(self):
        """Parse node/edge data from CSV files and construct DGL.Graphs
        """
        meta = self.meta
        root = self.raw_dir
        edge_data = []
        for meta_edge in meta.edges:
            edge_data.append(EdgeData(root, meta_edge))

        #parse NodeData
        node_data = []
        for meta_node in meta.nodes:
            node_data.append(NodeData(root, meta_node))

        #parse GraphData
        graph_data = GraphData(
            root, meta.graphs) if meta.graphs is not None else None

        #construct dgl.heterograph
        if graph_data is None:
            edge_dict = {}
            for e_data in edge_data:
                assert len(e_data.type) == 3
                edge_dict[(e_data.type[0], e_data.type[1], e_data.type[2])] = (
                    e_data.src, e_data.dst)
            node_dict = {}
            for n_data in node_data:
                node_dict[n_data.type] = len(n_data.id)
            self.graph = dgl_heterograph(edge_dict, num_nodes_dict=node_dict)

            def assign_data(src_data, dst_data):
                type = (src_data.type[0], src_data.type[1], src_data.type[2]) if len(
                    src_data.type) == 3 else src_data.type
                dst_data[type].data['feat'] = src_data.feat
                dst_data[type].data['label'] = src_data.label
                dst_data[type].data['num_classes'] = src_data.num_classes
                dst_data[type].data['train_mask'] = src_data.split_type == 0
                dst_data[type].data['val_mask'] = src_data.split_type == 1
                dst_data[type].data['test_mask'] = src_data.split_type == 2
            for n_data in node_data:
                assign_data(n_data, self.graph.nodes)
            for e_data in edge_data:
                assign_data(e_data, self.graph.edges)
        else:
            edge_dict = {}
            for e_data in edge_data:
                graph_id_arr = np.unique(F.asnumpy(e_data.graph_id))
                for g_id in graph_id_arr:
                    idx = e_data.graph_id == g_id
                    edge_dict[g_id] = {}
                    edge_dict[g_id]['edge'] = (
                        e_data.src[idx], e_data.dst[idx])
                    edge_dict[g_id]['feat'] = e_data.feat[idx]
            node_dict = {}
            for n_data in node_data:
                graph_id_arr = np.unique(F.asnumpy(n_data.graph_id))
                for g_id in graph_id_arr:
                    idx = n_data.graph_id == g_id
                    node_dict[g_id] = {}
                    node_dict[g_id]['node'] = len(n_data.id[idx])
                    node_dict[g_id]['feat'] = n_data.feat[idx]
            self.graphs = []
            self.labels = graph_data.label
            for g_id in F.asnumpy(graph_data.graph_id):
                if g_id not in node_dict:
                    raise RuntimeError(
                        "No node data is found for graph_id~{}.".format(g_id))
                e_data = (F.tensor([]), F.tensor(
                    [])) if g_id not in edge_dict else edge_dict[g_id]['edge']
                g = dgl_heterograph({('_V', '_E', '_V'): e_data}, num_nodes_dict={
                                    '_V': node_dict[g_id]['node']})
                g.nodes['_V'].data['feat'] = node_dict[g_id]['feat']
                g.edges['_E'].data['feat'] = edge_dict[g_id]['feat']
                self.graphs.append(g)
            self.mask = {}
            self.mask['train_mask'] = graph_data.split_type == 0
            self.mask['val_mask'] = graph_data.split_type == 1
            self.mask['test_mask'] = graph_data.split_type == 2
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
            save_graphs(graph_path, self.graphs,
                        labels={'labels': self.labels})
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
        else:
            self.graph = graphs[0]

    def __getitem__(self, i):
        return self.graph if self.graph is not None else (self.graphs[i], self.labels[i])

    def __len__(self):
        return 1 if self.graph is not None else len(self.graphs)

    def _sanity_check_after_process(self):
        #TODO:
        pass
