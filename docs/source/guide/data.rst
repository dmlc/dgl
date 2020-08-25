.. _guide-data-pipeline:

Chapter 4: Graph Data Pipeline
====================================================

DGL implements many commonly used graph datasets in :ref:`apidata`. They
follow a standard pipeline defined in class :class:`dgl.data.DGLDataset`. We highly
recommend processing graph data into a :class:`dgl.data.DGLDataset` subclass, as the
pipeline provides simple and clean solution for loading, processing and
saving graph data.

This chapter introduces how to create a DGL-Dataset for our own graph
data. The following contents explain how the pipeline works, and
show how to implement each component of it.

DGLDataset class
--------------------

:class:`dgl.data.DGLDataset` is the base class for processing, loading and saving
graph datasets defined in :ref:`apidata`. It implements the basic pipeline
for processing graph data. The following flow chart shows how the
pipeline works.

To process a graph dataset located in a remote server or local disk, we
define a class, say ``MyDataset``, inherits from :class:`dgl.data.DGLDataset`. The
template of ``MyDataset`` is as follows.

.. figure:: https://data.dgl.ai/asset/image/userguide_data_flow.png
    :align: center

    Flow chart for graph data input pipeline defined in class DGLDataset.

.. code:: 

    from dgl.data import DGLDataset
    
    class MyDataset(DGLDataset):
        """ Template for customizing graph datasets in DGL.
    
        Parameters
        ----------
        url : str
            URL to download the raw dataset
        raw_dir : str
            Specifying the directory that will store the 
            downloaded data or the directory that
            already stores the input data.
            Default: ~/.dgl/
        save_dir : str
            Directory to save the processed dataset.
            Default: the value of `raw_dir`
        force_reload : bool
            Whether to reload the dataset. Default: False
        verbose : bool
            Whether to print out progress information
        """
        def __init__(self, 
                     url=None, 
                     raw_dir=None, 
                     save_dir=None, 
                     force_reload=False, 
                     verbose=False):
            super(MyDataset, self).__init__(name='dataset_name',
                                            url=url,
                                            raw_dir=raw_dir,
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)
    
        def download(self):
            # download raw data to local disk
            pass
    
        def process(self):
            # process raw data to graphs, labels, splitting masks
            pass
        
        def __getitem__(self, idx):
            # get one example by index
            pass
    
        def __len__(self):
            # number of data examples
            pass
    
        def save(self):
            # save processed data to directory `self.save_path`
            pass
    
        def load(self):
            # load processed data from directory `self.save_path`
            pass
    
        def has_cache(self):
            # check whether there are processed data in `self.save_path`
            pass


:class:`dgl.data.DGLDataset` class has abstract functions ``process()``,
``__getitem__(idx)`` and ``__len__()`` that must be implemented in the
subclass. But we recommend to implement saving and loading as well,
since they can save significant time for processing large datasets, and
there are several APIs making it easy (see :ref:`ref-save-load-data`).

Note that the purpose of :class:`dgl.data.DGLDataset` is to provide a standard and
convenient way to load graph data. We can store graphs, features,
labels, masks and basic information about the dataset, such as number of
classes, number of labels, etc. Operations such as sampling, partition
or feature normalization are done outside of the :class:`dgl.data.DGLDataset`
subclass.

The rest of this chapter shows the best practices to implement the
functions in the pipeline.

Download raw data (optional)
--------------------------------

If our dataset is already in local disk, make sure it’s in directory
``raw_dir``. If we want to run our code anywhere without bothering to
download and move data to the right directory, we can do it
automatically by implementing function ``download()``.

If the dataset is a zip file, make ``MyDataset`` inherit from
:class:`dgl.data.DGLBuiltinDataset` class, which handles the zip file extraction for us. Otherwise,
implement ``download()`` like in
:class:`dgl.data.QM7bDataset`:

.. code:: 

    import os
    from dgl.data.utils import download
    
    def download(self):
        # path to store the file
        file_path = os.path.join(self.raw_dir, self.name + '.mat')
        # download file
        download(self.url, path=file_path)

The above code downloads a .mat file to directory ``self.raw_dir``. If
the file is a .gz, .tar, .tar.gz or .tgz file, use :func:`dgl.data.utils.extract_archive`
function to extract. The following code shows how to download a .gz file
in :class:`dgl.data.BitcoinOTCDataset`:

.. code:: 

    from dgl.data.utils import download, extract_archive
    
    def download(self):
        # path to store the file
        # make sure to use the same suffix as the original file name's
        gz_file_path = os.path.join(self.raw_dir, self.name + '.csv.gz')
        # download file
        download(self.url, path=gz_file_path)
        # check SHA-1
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name + '.csv.gz'))
        # extract file to directory `self.name` under `self.raw_dir`
        self._extract_gz(gz_file_path, self.raw_path)

The above code will extract the file into directory ``self.name`` under
``self.raw_dir``. If the class inherits from :class:`dgl.data.DGLBuiltinDataset`
to handle zip file, it will extract the file into directory ``self.name`` 
as well.

Optionally, we can check SHA-1 string of the downloaded file as the
example above does, in case the author changed the file in the remote
server some day.

Process data
----------------

We implement the data processing code in function ``process()``, and it
assumes that the raw data is located in ``self.raw_dir`` already. There
are typically three types of tasks in machine learning on graphs: graph
classification, node classification, and link prediction. We will show
how to process datasets related to these tasks.

Here we focus on the standard way to process graphs, features and masks.
We will use builtin datasets as examples and skip the implementations
for building graphs from files, but add links to the detailed
implementations. Please refer to :ref:`guide-graph-external` to see a
complete guide on how to build graphs from external sources.

Processing Graph Classification datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Graph classification datasets are almost the same as most datasets in
typical machine learning tasks, where mini-batch training is used. So we
process the raw data to a list of :class:`dgl.DGLGraph` objects and a list of
label tensors. In addition, if the raw data has been splitted into
several files, we can add a parameter ``split`` to load specific part of
the data.

Take :class:`dgl.data.QM7bDataset` as example:

.. code:: 

    class QM7bDataset(DGLDataset):
        _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
               'datasets/qm7b.mat'
        _sha1_str = '4102c744bb9d6fd7b40ac67a300e49cd87e28392'
    
        def __init__(self, raw_dir=None, force_reload=False, verbose=False):
            super(QM7bDataset, self).__init__(name='qm7b',
                                              url=self._url,
                                              raw_dir=raw_dir,
                                              force_reload=force_reload,
                                              verbose=verbose)
    
        def process(self):
            mat_path = self.raw_path + '.mat'
            # process data to a list of graphs and a list of labels
            self.graphs, self.label = self._load_graph(mat_path)
        
        def __getitem__(self, idx):
            """ Get graph and label by index
    
            Parameters
            ----------
            idx : int
                Item index
    
            Returns
            -------
            (dgl.DGLGraph, Tensor)
            """
            return self.graphs[idx], self.label[idx]
    
        def __len__(self):
            """Number of graphs in the dataset"""
            return len(self.graphs)


In ``process()``, the raw data is processed to a list of graphs and a
list of labels. We must implement ``__getitem__(idx)`` and ``__len__()``
for iteration. We recommend to make ``__getitem__(idx)`` to return a
tuple ``(graph, label)`` as above. Please check the `QM7bDataset source
code <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/qm7b.html#QM7bDataset>`__
for details of ``self._load_graph()`` and ``__getitem__``.

We can also add properties to the class to indicate some useful
information of the dataset. In :class:`dgl.data.QM7bDataset`, we can add a property
``num_labels`` to indicate the total number of prediction tasks in this
multi-task dataset:

.. code:: 

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 14

After all these coding, we can finally use the :class:`dgl.data.QM7bDataset` as
follows:

.. code:: 

    from torch.utils.data import DataLoader
    
    # load data
    dataset = QM7bDataset()
    num_labels = dataset.num_labels
    
    # create collate_fn
    def _collate_fn(batch):
        graphs, labels = batch
        g = dgl.batch(graphs)
        labels = torch.tensor(labels, dtype=torch.long)
        return g, labels
    
    # create dataloaders
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate_fn)
    
    # training
    for epoch in range(100):
        for g, labels in dataloader:
            # your training code here
            pass

A complete guide for training graph classification models can be found
in :ref:`guide-training-graph-classification`.

For more examples of graph classification datasets, please refer to our builtin graph classification
datasets: 

* :ref:`gindataset`

* :ref:`minigcdataset`

* :ref:`qm7bdata`

* :ref:`tudata`

Processing Node Classification datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different from graph classification, node classification is typically on
a single graph. As such, splits of the dataset are on the nodes of the
graph. We recommend using node masks to specify the splits. We use
builtin dataset `CitationGraphDataset <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/citation_graph.html#CitationGraphDataset>`__ as an example:

.. code:: 

    import dgl
    from dgl.data import DGLBuiltinDataset
    
    class CitationGraphDataset(DGLBuiltinDataset):
        _urls = {
            'cora_v2' : 'dataset/cora_v2.zip',
            'citeseer' : 'dataset/citeseer.zip',
            'pubmed' : 'dataset/pubmed.zip',
        }
    
        def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
            assert name.lower() in ['cora', 'citeseer', 'pubmed']
            if name.lower() == 'cora':
                name = 'cora_v2'
            url = _get_dgl_url(self._urls[name])
            super(CitationGraphDataset, self).__init__(name,
                                                       url=url,
                                                       raw_dir=raw_dir,
                                                       force_reload=force_reload,
                                                       verbose=verbose)
    
        def process(self):
            # Skip some processing code
            # === data processing skipped ===
    
            # build graph
            g = dgl.graph(graph)
            # splitting masks
            g.ndata['train_mask'] = generate_mask_tensor(train_mask)
            g.ndata['val_mask'] = generate_mask_tensor(val_mask)
            g.ndata['test_mask'] = generate_mask_tensor(test_mask)
            # node labels
            g.ndata['label'] = F.tensor(labels)
            # node features
            g.ndata['feat'] = F.tensor(_preprocess_features(features), 
                                       dtype=F.data_type_dict['float32'])
            self._num_labels = onehot_labels.shape[1]
            self._labels = labels
            self._g = g
    
        def __getitem__(self, idx):
            assert idx == 0, "This dataset has only one graph"
            return self._g
    
        def __len__(self):
            return 1

For brevity, we skip some code in ``process()`` to highlight the key
part for processing node classification dataset: spliting masks, node
features and node labels are stored in ``g.ndata``. For detailed
implementation, please refer to `CitationGraphDataset source
code <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/citation_graph.html#CitationGraphDataset>`__.

Notice that the implementations of ``__getitem__(idx)`` and
``__len__()`` are changed as well, since there is often only one graph
for node classification tasks. The masks are ``bool tensors`` in PyTorch
and TensorFlow, and ``float tensors`` in MXNet.

We use a subclass of ``CitationGraphDataset``, :class:`dgl.data.CiteseerGraphDataset`,
to show the usage of it:

.. code:: 

    # load data
    dataset = CiteseerGraphDataset(raw_dir='')
    graph = dataset[0]
    
    # get split masks
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    
    # get node features
    feats = graph.ndata['feat']
    
    # get labels
    labels = graph.ndata['label']

A complete guide for training node classification models can be found in
:ref:`guide-training-node-classification`.

For more examples of node classification datasets, please refer to our
builtin datasets:

* :ref:`citationdata`

* :ref:`corafulldata`

* :ref:`amazoncobuydata`

* :ref:`coauthordata`

* :ref:`karateclubdata`

* :ref:`ppidata`

* :ref:`redditdata`

* :ref:`sbmdata`

* :ref:`sstdata`

* :ref:`rdfdata`

Processing dataset for Link Prediction datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The processing of link prediction datasets is similar to that for node
classification’s, there is often one graph in the dataset.

We use builtin dataset
`KnowledgeGraphDataset <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/knowledge_graph.html#KnowledgeGraphDataset>`__
as example, and still skip the detailed data processing code to
highlight the key part for processing link prediction datasets:

.. code:: 

    # Example for creating Link Prediction datasets
    class KnowledgeGraphDataset(DGLBuiltinDataset):
        def __init__(self, name, reverse=True, raw_dir=None, force_reload=False, verbose=True):
            self._name = name
            self.reverse = reverse
            url = _get_dgl_url('dataset/') + '{}.tgz'.format(name)
            super(KnowledgeGraphDataset, self).__init__(name,
                                                        url=url,
                                                        raw_dir=raw_dir,
                                                        force_reload=force_reload,
                                                        verbose=verbose)
    
        def process(self):
            # Skip some processing code
            # === data processing skipped ===
    
            # splitting mask
            g.edata['train_mask'] = train_mask
            g.edata['val_mask'] = val_mask
            g.edata['test_mask'] = test_mask
            # edge type
            g.edata['etype'] = etype
            # node type
            g.ndata['ntype'] = ntype
            self._g = g
    
        def __getitem__(self, idx):
            assert idx == 0, "This dataset has only one graph"
            return self._g
    
        def __len__(self):
            return 1

As shown in the code, we add splitting masks into ``edata`` field of the
graph. Check `KnowledgeGraphDataset source
code <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/knowledge_graph.html#KnowledgeGraphDataset>`__
to see the complete code. We use a subclass of ``KnowledgeGraphDataset``, :class:`dgl.data.FB15k237Dataset`,
to show the usage of it:

.. code:: 

    import torch
    
    # load data
    dataset = FB15k237Dataset()
    graph = dataset[0]
    
    # get training mask
    train_mask = graph.edata['train_mask']
    train_idx = torch.nonzero(train_mask).squeeze()
    src, dst = graph.edges(train_idx)
    # get edge types in training set
    rel = graph.edata['etype'][train_idx]


A complete guide for training link prediction models can be found in
:ref:`guide-training-link-prediction`.

For more examples of link prediction datasets, please refer to our
builtin datasets: 

* :ref:`kgdata`

* :ref:`bitcoinotcdata`

.. _ref-save-load-data:

Save and load data
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
