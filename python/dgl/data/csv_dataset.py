import os
import numpy as np
from .dgl_dataset import DGLDataset
from .utils import save_graphs, load_graphs
from ..base import DGLError


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
        from .csv_dataset_base import load_yaml_with_sanity_check, DefaultDataParser
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
        from .csv_dataset_base import NodeData, EdgeData, GraphData, DGLGraphConstructor
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
