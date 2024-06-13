.. code:: ipython3

    # !pip install pytorch-lightning
    # !pip install torchlayers
    # !pip install torchmetric
    # !pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
    # !pip install rich --upgrade

.. code:: ipython3

    """
    Training a GNN for Graph Classification
    =======================================
    By the end of this tutorial, you will be able to
    -  Load a DGL-provided graph classification dataset.
    -  Understand what *readout* function does.
    -  Understand how to create and use a minibatch of graphs.
    -  Build a GNN-based graph classification model.
    -  Train and evaluate the model on a DGL-provided dataset.
    (Time estimate: 18 minutes)
    """
    
    import dgl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pytorch_lightning as pl
    import torchmetrics

.. code:: ipython3

    ######################################################################
    # Overview of Graph Classification with GNN
    # -----------------------------------------
    # 
    # Graph classification or regression requires a model to predict certain
    # graph-level properties of a single graph given its node and edge
    # features.  Molecular property prediction is one particular application.
    # 
    # This tutorial shows how to train a graph classification model for a
    # small dataset from the paper `How Powerful Are Graph Neural
    # Networks <https://arxiv.org/abs/1810.00826>`__.
    # 
    # Loading Data
    # ------------
    # 
    
    import dgl.data
    
    # Generate a synthetic dataset with 10000 graphs, ranging from 10 to 500 nodes.
    dataset = dgl.data.GINDataset('PROTEINS', self_loop=True)
    
    
    ######################################################################
    # The dataset is a set of graphs, each with node features and a single
    # label. One can see the node feature dimensionality and the number of
    # possible graph categories of ``GINDataset`` objects in ``dim_nfeats``
    # and ``gclasses`` attributes.
    # 
    
    print('Node feature dimensionality:', dataset.dim_nfeats)
    print('Number of graph categories:', dataset.gclasses)


.. parsed-literal::

    Node feature dimensionality: 3
    Number of graph categories: 2
    

.. code:: ipython3

    
    ######################################################################
    # Defining Data Loader
    # --------------------
    # 
    # A graph classification dataset usually contains two types of elements: a
    # set of graphs, and their graph-level labels. Similar to an image
    # classification task, when the dataset is large enough, we need to train
    # with mini-batches. When you train a model for image classification or
    # language modeling, you will use a ``DataLoader`` to iterate over the
    # dataset. In DGL, you can use the ``GraphDataLoader``.
    # 
    # You can also use various dataset samplers provided in
    # `torch.utils.data.sampler <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`__.
    # For example, this tutorial creates a training ``GraphDataLoader`` and
    # test ``GraphDataLoader``, using ``SubsetRandomSampler`` to tell PyTorch
    # to sample from only a subset of the dataset.
    # 
    from dgl.dataloading import GraphDataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    
    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)
    
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))
    
    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=50, drop_last=False)
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=50, drop_last=False)

.. code:: ipython3

    ######################################################################
    # You can try to iterate over the created ``GraphDataLoader`` and see what it
    # gives:
    # 
    # small graphs are merged into one big graph
    it = iter(train_dataloader)
    batch = next(it)
    print(batch)


.. parsed-literal::

    [Graph(num_nodes=2387, num_edges=11251,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), tensor([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 1])]
    

.. code:: ipython3

    
    ######################################################################
    # As each element in ``dataset`` has a graph and a label, the
    # ``GraphDataLoader`` will return two objects for each iteration. The
    # first element is the batched graph, and the second element is simply a
    # label vector representing the category of each graph in the mini-batch.
    # Next, we’ll talked about the batched graph.
    # 
    # A Batched Graph in DGL
    # ----------------------
    # 
    # In each mini-batch, the sampled graphs are combined into a single bigger
    # batched graph via ``dgl.batch``. The single bigger batched graph merges
    # all original graphs as separately connected components, with the node
    # and edge features concatenated. This bigger graph is also a ``DGLGraph``
    # instance (so you can
    # still treat it as a normal ``DGLGraph`` object as in
    # `here <2_dglgraph.ipynb>`__). It however contains the information
    # necessary for recovering the original graphs, such as the number of
    # nodes and edges of each graph element.
    # 
    
    batched_graph, labels = batch
    print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
    print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())
    
    # Recover the original graph elements from the minibatch
    graphs = dgl.unbatch(batched_graph)
    print('The original graphs in the minibatch:')
    print(graphs)
    
    


.. parsed-literal::

    Number of nodes for each graph element in the batch: tensor([ 32,  42,  59,   6,  46,  29,  60,  88,  36,  43,  66,  43,  18, 113,
             36,  59,   7,  27,  24,  46,  31, 148,  54,  16,  20,  44,  40,  58,
             54,  86,  42,  39,  61,   7,  11,  46,  23,  50,  40,   9,  34,  22,
            140,  40,  43, 105, 131,  38,  55,  20])
    Number of edges for each graph element in the batch: tensor([160, 204, 299,  28, 238, 143, 256, 408, 164, 199, 336, 223,  94, 503,
            158, 259,  19, 117, 126, 180, 157, 714, 300,  72,  84, 222, 200, 296,
            278, 426, 216, 187, 279,  35,  53, 218, 117, 264, 214,  43, 150, 100,
            506, 192, 221, 461, 591, 202, 249,  90])
    The original graphs in the minibatch:
    [Graph(num_nodes=32, num_edges=160,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=42, num_edges=204,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=59, num_edges=299,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=6, num_edges=28,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=46, num_edges=238,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=29, num_edges=143,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=60, num_edges=256,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=88, num_edges=408,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=36, num_edges=164,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=43, num_edges=199,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=66, num_edges=336,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=43, num_edges=223,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=18, num_edges=94,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=113, num_edges=503,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=36, num_edges=158,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=59, num_edges=259,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=7, num_edges=19,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=27, num_edges=117,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=24, num_edges=126,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=46, num_edges=180,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=31, num_edges=157,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=148, num_edges=714,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=54, num_edges=300,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=16, num_edges=72,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=20, num_edges=84,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=44, num_edges=222,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=40, num_edges=200,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=58, num_edges=296,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=54, num_edges=278,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=86, num_edges=426,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=42, num_edges=216,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=39, num_edges=187,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=61, num_edges=279,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=7, num_edges=35,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=11, num_edges=53,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=46, num_edges=218,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=23, num_edges=117,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=50, num_edges=264,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=40, num_edges=214,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=9, num_edges=43,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=34, num_edges=150,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=22, num_edges=100,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=140, num_edges=506,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=40, num_edges=192,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=43, num_edges=221,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=105, num_edges=461,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=131, num_edges=591,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=38, num_edges=202,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=55, num_edges=249,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={}), Graph(num_nodes=20, num_edges=90,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={})]
    

.. code:: ipython3

    graphs = dgl.batch(graphs)
    print('The batched graphs in the minibatch:')
    print(graphs)


.. parsed-literal::

    The batched graphs in the minibatch:
    Graph(num_nodes=2387, num_edges=11251,
          ndata_schemes={'attr': Scheme(shape=(3,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={})
    

.. code:: ipython3

    
    ######################################################################
    # Define Model
    # ------------
    # 
    # This tutorial will build a two-layer `Graph Convolutional Network
    # (GCN) <http://tkipf.github.io/graph-convolutional-networks/>`__. Each of
    # its layer computes new node representations by aggregating neighbor
    # information. If you have gone through the
    # :doc:`introduction <1_introduction>`, you will notice two
    # differences:
    # 
    # -  Since the task is to predict a single category for the *entire graph*
    #    instead of for every node, you will need to aggregate the
    #    representations of all the nodes and potentially the edges to form a
    #    graph-level representation. Such process is more commonly referred as
    #    a *readout*. A simple choice is to average the node features of a
    #    graph with ``dgl.mean_nodes()``.
    #
    # -  The input graph to the model will be a batched graph yielded by the
    #    ``GraphDataLoader``. The readout functions provided by DGL can handle
    #    batched graphs so that they will return one representation for each
    #    minibatch element.
    # 
    
    from dgl.nn import GraphConv
    
    class GCN(pl.LightningModule):
        def __init__(self, in_feats, h_feats, num_classes):
            super(GCN, self).__init__()
    
            self.conv1 = GraphConv(in_feats, h_feats)
            self.conv2 = GraphConv(h_feats, num_classes)
        
        def forward(self, g, in_feat):
            h = self.conv1(g, in_feat)
            h = F.relu(h)
            h = self.conv2(g, h)
            g.ndata['h'] = h
            return dgl.mean_nodes(g, 'h') # simple average pooling
    
        def training_step(self,batch,batch_idx):
          batched_graph,labels = batch
          pred = self(batched_graph, batched_graph.ndata['attr'].float())
          loss = F.cross_entropy(pred, labels)
          self.log('train_loss',loss)
          return loss
    
    
        def configure_optimizers(self):
          optimizer = torch.optim.Adam(self.parameters(),lr=0.01)
          return [optimizer]
    
    GCN4GraphCLS = GCN(dataset.dim_nfeats, 16, dataset.gclasses)
    

.. code:: ipython3

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=10,
        gpus=1,
        precision=32,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        num_sanity_val_steps=0,
      )
    trainer.fit(GCN4GraphCLS,train_dataloader)


.. parsed-literal::

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    `Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
    Missing logger folder: /content/lightning_logs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name  | Type      | Params
    ------------------------------------
    0 | conv1 | GraphConv | 64    
    1 | conv2 | GraphConv | 34    
    ------------------------------------
    98        Trainable params
    0         Non-trainable params
    98        Total params
    0.000     Total estimated model params size (MB)
    /usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/connectors/data_connector.py:245: PossibleUserWarning: The dataloader, train_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
      category=PossibleUserWarning,
    /usr/local/lib/python3.7/dist-packages/pytorch_lightning/trainer/trainer.py:1937: PossibleUserWarning: The number of training batches (18) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
      category=PossibleUserWarning,
    


.. parsed-literal::

    Training: 0it [00:00, ?it/s]


.. parsed-literal::

    /usr/local/lib/python3.7/dist-packages/torchmetrics/utilities/prints.py:36: UserWarning: Torchmetrics v0.9 introduced a new argument class property called `full_state_update` that has
                    not been set for this class (_ResultMetric). The property determines if `update` by
                    default needs access to the full metric state. If this is not the case, significant speedups can be
                    achieved and we recommend setting this to `False`.
                    We provide an checking function
                    `from torchmetrics.utilities import check_forward_no_full_state`
                    that can be used to check if the `full_state_update=True` (old and potential slower behaviour,
                    default for now) or if `full_state_update=False` can be used safely.
                    
      warnings.warn(*args, **kwargs)
    




