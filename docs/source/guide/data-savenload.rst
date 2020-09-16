.. _guide-data-pipeline-savenload:

4.4 Save and load data
----------------------

We recommend to implement saving and loading functions to cache the
processed data in local disk. This saves a lot of data processing time
in most cases. We provide four functions to make things simple:

-  :func:`dgl.save_graphs` and :func:`dgl.load_graphs`: save/load DGLGraph objects and labels to/from local disk.
-  :func:`dgl.data.utils.save_info` and :func:`dgl.data.utils.load_info`: save/load useful information of the dataset (python ``dict`` object) to/from local disk.

The following example shows how to save and load a list of graphs and
dataset information.

.. code:: 

    import os
    from dgl import save_graphs, load_graphs
    from dgl.data.utils import makedirs, save_info, load_info
    
    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})
    
    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']
    
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

Note that there are cases not suitable to save processed data. For
example, in the builtin dataset :class:`dgl.data.GDELTDataset`,
the processed data is quite large, so it’s more effective to process
each data example in ``__getitem__(idx)``.

Loading OGB datasets using ``ogb`` package
----------------------------------------------

`Open Graph Benchmark (OGB) <https://ogb.stanford.edu/docs/home/>`__ is
a collection of benchmark datasets. The official OGB package
`ogb <https://github.com/snap-stanford/ogb>`__ provides APIs for
downloading and processing OGB datasets into :class:`dgl.data.DGLGraph` objects. We
introduce their basic usage here.

First install ogb package using pip:

.. code:: 

    pip install ogb

The following code shows how to load datasets for *Graph Property
Prediction* tasks.

.. code:: 

    # Load Graph Property Prediction datasets in OGB
    import dgl
    import torch
    from ogb.graphproppred import DglGraphPropPredDataset
    from torch.utils.data import DataLoader
    
    
    def _collate_fn(batch):
        # batch is a list of tuple (graph, label)
        graphs = [e[0] for e in batch]
        g = dgl.batch(graphs)
        labels = [e[1] for e in batch]
        labels = torch.stack(labels, 0)
        return g, labels
    
    # load dataset
    dataset = DglGraphPropPredDataset(name='ogbg-molhiv')
    split_idx = dataset.get_idx_split()
    # dataloader
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=_collate_fn)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)

Loading *Node Property Prediction* datasets is similar, but note that
there is only one graph object in this kind of dataset.

.. code:: 

    # Load Node Property Prediction datasets in OGB
    from ogb.nodeproppred import DglNodePropPredDataset
    
    dataset = DglNodePropPredDataset(name='ogbn-proteins')
    split_idx = dataset.get_idx_split()
    
    # there is only one graph in Node Property Prediction datasets
    g, labels = dataset[0]
    # get split labels
    train_label = dataset.labels[split_idx['train']]
    valid_label = dataset.labels[split_idx['valid']]
    test_label = dataset.labels[split_idx['test']]

*Link Property Prediction* datasets also contain one graph per dataset:

.. code:: 

    # Load Link Property Prediction datasets in OGB
    from ogb.linkproppred import DglLinkPropPredDataset
    
    dataset = DglLinkPropPredDataset(name='ogbl-ppa')
    split_edge = dataset.get_edge_split()
    
    graph = dataset[0]
    print(split_edge['train'].keys())
    print(split_edge['valid'].keys())
    print(split_edge['test'].keys())
