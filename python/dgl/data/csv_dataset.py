import os

import numpy as np

from .. import backend as F
from ..base import DGLError
from .dgl_dataset import DGLDataset
from .utils import load_graphs, save_graphs, Subset


class CSVDataset(DGLDataset):
    """Dataset class that loads and parses graph data from CSV files.

    This class requires the following additional packages:

        - pyyaml >= 5.4.1
        - pandas >= 1.1.5
        - pydantic >= 1.9.0

    The parsed graph and feature data will be cached for faster reloading. If
    the source CSV files are modified, please specify ``force_reload=True``
    to re-parse from them.

    Parameters
    ----------
    data_path : str
        Directory which contains 'meta.yaml' and CSV files
    force_reload : bool, optional
        Whether to reload the dataset. Default: False
    verbose: bool, optional
        Whether to print out progress information. Default: True.
    ndata_parser : dict[str, callable] or callable, optional
        Callable object which takes in the ``pandas.DataFrame`` object created from
        CSV file, parses node data and returns a dictionary of parsed data. If given a
        dictionary, the key is node type and the value is a callable object which is
        used to parse data of corresponding node type. If given a single callable
        object, such object is used to parse data of all node type data. Default: None.
        If None, a default data parser is applied which load data directly and tries to
        convert list into array.
    edata_parser : dict[(str, str, str), callable], or callable, optional
        Callable object which takes in the ``pandas.DataFrame`` object created from
        CSV file, parses edge data and returns a dictionary of parsed data. If given a
        dictionary, the key is edge type and the value is a callable object which is
        used to parse data of corresponding edge type. If given a single callable
        object, such object is used to parse data of all edge type data. Default: None.
        If None, a default data parser is applied which load data directly and tries to
        convert list into array.
    gdata_parser : callable, optional
        Callable object which takes in the ``pandas.DataFrame`` object created from
        CSV file, parses graph data and returns a dictionary of parsed data. Default:
        None. If None, a default data parser is applied which load data directly and
        tries to convert list into array.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    graphs : :class:`dgl.DGLGraph`
        Graphs of the dataset
    data : dict
        any available graph-level data such as graph-level feature, labels.

    Examples
    --------
    Please refer to :ref:`guide-data-pipeline-loadcsv`.

    """

    META_YAML_NAME = "meta.yaml"

    def __init__(
        self,
        data_path,
        force_reload=False,
        verbose=True,
        ndata_parser=None,
        edata_parser=None,
        gdata_parser=None,
        transform=None,
    ):
        from .csv_dataset_base import (
            DefaultDataParser,
            load_yaml_with_sanity_check,
        )

        self.graphs = None
        self.data = None
        self.ndata_parser = {} if ndata_parser is None else ndata_parser
        self.edata_parser = {} if edata_parser is None else edata_parser
        self.gdata_parser = gdata_parser
        self.default_data_parser = DefaultDataParser()
        meta_yaml_path = os.path.join(data_path, CSVDataset.META_YAML_NAME)
        if not os.path.exists(meta_yaml_path):
            raise DGLError(
                "'{}' cannot be found under {}.".format(
                    CSVDataset.META_YAML_NAME, data_path
                )
            )
        self.meta_yaml = load_yaml_with_sanity_check(meta_yaml_path)
        ds_name = self.meta_yaml.dataset_name
        super().__init__(
            ds_name,
            raw_dir=os.path.dirname(meta_yaml_path),
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        """Parse node/edge data from CSV files and construct DGL.Graphs"""
        from .csv_dataset_base import (
            DGLGraphConstructor,
            EdgeData,
            GraphData,
            NodeData,
        )

        meta_yaml = self.meta_yaml
        base_dir = self.raw_dir
        node_data = []
        for meta_node in meta_yaml.node_data:
            if meta_node is None:
                continue
            ntype = meta_node.ntype
            data_parser = (
                self.ndata_parser
                if callable(self.ndata_parser)
                else self.ndata_parser.get(ntype, self.default_data_parser)
            )
            ndata = NodeData.load_from_csv(
                meta_node,
                base_dir=base_dir,
                separator=meta_yaml.separator,
                data_parser=data_parser,
            )
            node_data.append(ndata)
        edge_data = []
        for meta_edge in meta_yaml.edge_data:
            if meta_edge is None:
                continue
            etype = tuple(meta_edge.etype)
            data_parser = (
                self.edata_parser
                if callable(self.edata_parser)
                else self.edata_parser.get(etype, self.default_data_parser)
            )
            edata = EdgeData.load_from_csv(
                meta_edge,
                base_dir=base_dir,
                separator=meta_yaml.separator,
                data_parser=data_parser,
            )
            edge_data.append(edata)
        graph_data = None
        if meta_yaml.graph_data is not None:
            meta_graph = meta_yaml.graph_data
            data_parser = (
                self.default_data_parser
                if self.gdata_parser is None
                else self.gdata_parser
            )
            graph_data = GraphData.load_from_csv(
                meta_graph,
                base_dir=base_dir,
                separator=meta_yaml.separator,
                data_parser=data_parser,
            )
        # construct graphs
        self.graphs, self.data = DGLGraphConstructor.construct_graphs(
            node_data, edge_data, graph_data
        )
        if len(self.data) == 1:
            self.labels = list(self.data.values())[0]

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.name + ".bin")
        if os.path.exists(graph_path):
            return True

        return False

    def save(self):
        if self.graphs is None:
            raise DGLError("No graphs available in dataset")
        graph_path = os.path.join(self.save_path, self.name + ".bin")
        save_graphs(graph_path, self.graphs, labels=self.data)

    def load(self):
        graph_path = os.path.join(self.save_path, self.name + ".bin")
        self.graphs, self.data = load_graphs(graph_path)
        if len(self.data) == 1:
            self.labels = list(self.data.values())[0]

    def __getitem__(self, i):
        if F.is_tensor(i) and F.ndim(i) == 1:
            return Subset(self, F.copy_to(i, F.cpu()))

        if self._transform is None:
            g = self.graphs[i]
        else:
            g = self._transform(self.graphs[i])

        if len(self.data) == 1:
            return g, self.labels[i]
        elif len(self.data) > 0:
            data = {k: v[i] for (k, v) in self.data.items()}
            return g, data
        else:
            return g

    def __len__(self):
        return len(self.graphs)
