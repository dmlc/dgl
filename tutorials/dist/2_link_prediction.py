"""
Distributed Link Prediction
===============================

In this tutorial, we will walk through the steps of performing distributed GNN training
for a link prediction task. This tutorial assumes that you have read the `Distributed Node Classification <https://docs.dgl.ai/en/latest/tutorials/dist/1_node_classification.html>`_ and `Stochastic Training of GNN for Link Prediction <https://docs.dgl.ai/en/latest/tutorials/large/L2_large_link_prediction.html#sphx-glr-tutorials-large-l2-large-link-prediction-py>`_. The general pipeline is shown below.

.. figure:: https://data.dgl.ai/tutorial/link.png
   :alt: Imgur


Partition a graph
-----------------

In this tutorial, we will use `OGBL citation2 graph <https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2>`_
as an example to illustrate the graph partitioning. Let’s first load the graph into a DGL graph and convert it 
into a training graph, validation edges and test edges with :class:`~dgl.data.AsLinkPredDataset`.

.. code-block:: python


    import os
    os.environ['DGLBACKEND'] = 'pytorch'
    import dgl
    import torch as th
    from ogb.linkproppred import DglLinkPropPredDataset
    data = DglLinkPropPredDataset(name='ogbl-citation2')
    graph = data[0]
    data = dgl.data.AsLinkPredDataset(data, [0.8, 0.1, 0.1])
    graph_train = data[0]
    dgl.distributed.partition_graph(graph_train, graph_name='ogbl-citation2', num_parts=4,
                                out_path='4part_data',
                                balance_edges=True)



Then, we store the validation and test edges with the graph partitions.



.. code-block:: python


    import pickle
    with open('4part_data/val.pkl', 'wb') as f:
        pickle.dump(data.val_edges, f)
    with open('4part_data/test.pkl', 'wb') as f:
        pickle.dump(data.test_edges, f)



Distributed training script
---------------------------

The distributed link prediction script is very similar to distributed node classification script with just a few modifications.


Initialize network communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first initialize the network communication and Pytorch's distributed communication. 

.. code-block:: python

    import dgl
    import torch as th
    dgl.distributed.initialize(ip_config='ip_config.txt')
    th.distributed.init_process_group(backend='gloo')


The configuration file `ip_config.txt` has the following format:

.. code-block:: shell

  ip_addr1 [port1]
  ip_addr2 [port2]

Each row is a machine. The first column is the IP address and the second column is the port for
connecting to the DGL server on the machine. The port is optional and the default port is 30050.

    


Reference to the distributed graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DGL's servers load the graph partitions automatically. After the servers load the partitions,
trainers connect to the servers and can start to reference to the distributed graph in the cluster as below.


.. code-block:: python

    g = dgl.distributed.DistGraph('ogbl-citation2')


As shown in the code, we refer to a distributed graph by its name. This name is basically the one passed
to the `partition_graph` function as shown in the section above.

Get training and validation node IDs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For distributed training, each trainer can run its own set of training nodes. We can get the current graph in the trainer with its node ids and edge ids 
by invoking `node_split` and `edge_split`. We can also get the valid edges and test edges by loading the pickle files.



.. code-block:: python

    train_eids = dgl.distributed.edge_split(th.ones((g.num_edges(),), dtype=th.bool), g.get_partition_book(), force_even=True)
    train_nids = dgl.distributed.node_split(th.ones((g.num_nodes(),), dtype=th.bool), g.get_partition_book())
    with open('4part_data/val.pkl', 'rb') as f:
        global_valid_eid = pickle.load(f)
    with open('4part_data/test.pkl', 'rb') as f:
        global_test_eid = pickle.load(f)


Define a GNN model
^^^^^^^^^^^^^^^^^^

For distributed training, we define a GNN model exactly in the same way as
`mini-batch training <https://doc.dgl.ai/guide/minibatch.html#>`_ or
`full-graph training <https://doc.dgl.ai/guide/training-node.html#guide-training-node-classification>`_.
The code below defines the GraphSage model.


.. code-block:: python

    import torch.nn as nn
    import torch.nn.functional as F
    import dgl.nn as dglnn
    import torch.optim as optim

    class SAGE(nn.Module):
        def __init__(self, in_feats, n_hidden, n_classes, n_layers):
            super().__init__()
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.n_classes = n_classes
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))

        def forward(self, blocks, x):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                x = layer(block, x)
                if l != self.n_layers - 1:
                    x = F.relu(x)
            return x

    num_hidden = 256
    num_labels = len(th.unique(g.ndata['labels'][0:g.num_nodes()]))
    num_layers = 2
    lr = 0.001
    model = SAGE(g.ndata['feat'].shape[1], num_hidden, num_labels, num_layers)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


For distributed training, we need to convert the model into a distributed model with
Pytorch's `DistributedDataParallel`.


.. code-block:: python

    model = th.nn.parallel.DistributedDataParallel(model)

We also define an edge predictor :class:`~dgl.nn.pytorch.link.EdgePredictor` to predict the edge scores of pairs of node representations

.. code-block:: python

    from dgl.nn import EdgePredictor
    predictor = EdgePredictor('dot')

Distributed mini-batch sampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use :class:`~dgl.dataloading.pytorch.DistEdgeDataLoader`, the distributed counterpart
of :class:`~dgl.dataloading.pytorch.EdgeDataLoader`, to create a distributed mini-batch sampler for
link prediction. 



.. code-block:: python
    sampler = dgl.dataloading.MultiLayerNeighborSampler([25,10])
    dataloader = dgl.dataloading.DistEdgeDataLoader(
        g=g,
        eids=train_eids.numpy(),
        graph_sampler=sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)


Training loop
^^^^^^^^^^^^^

The training loop for distributed training is also exactly the same as the single-process training.


.. code-block:: python

    import sklearn.metrics
    import numpy as np

    epoch = 0
    for epoch in range(10):
        for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(dataloader):
            pos_graph = pos_graph
            neg_graph = neg_graph
            node_inputs = mfgs[0].srcdata[dgl.NID]
            batch_inputs = g.ndata['feat'][node_inputs]

            batch_pred = model(mfgs, batch_inputs)
            pos_feature = batch_pred
            pos_graph.ndata['h'] = batch_pred
            pos_src, pos_dst = pos_graph.edges()
            pos_score = predictor(pos_feature[pos_src], pos_feature[pos_dst])

            neg_feature = batch_pred
            neg_graph.ndata['h'] = batch_pred
            neg_src, neg_dst = neg_graph.edges()
            neg_score = predictor(neg_feature[pos_src], neg_feature[pos_dst])

            score = th.cat([pos_score, neg_score])
            label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)])
            loss = F.binary_cross_entropy_with_logits(score, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

Inference
^^^^^^^^^^^^^

In the inference stage, we use the model after training loop to get the embedding of nodes.

.. code-block:: python

    def inference(model, graph, node_features, args):
        with th.no_grad():
            sampler = dgl.dataloading.MultiLayerNeighborSampler([25,10])
            train_dataloader = dgl.dataloading.DistNodeDataLoader(
                graph, th.arange(graph.num_nodes()), sampler,
                batch_size=1024,
                shuffle=False,
                drop_last=False)

            result = []
            for input_nodes, output_nodes, mfgs in train_dataloader:
                node_inputs = mfgs[0].srcdata[dgl.NID]
                inputs = node_features[node_inputs]
                result.append(model(mfgs, inputs))

            return th.cat(result)

    node_reprs = inference(model, g, g.ndata['feat'], args)

The test edges is encoded as ((positive_edge_src, positive_edge_dst), (negative_edge_src, negative_edge_dst)). Therefore, we can get the ground truth with positive pairs and negative pairs. 

.. code-block:: python

    test_pos_src = global_test_eid[0][0]
    test_pos_dst = global_test_eid[0][1]
    test_neg_src = global_test_eid[1][0]
    test_neg_dst = global_test_eid[1][1]
    test_labels = th.cat([th.ones_like(test_pos_src), th.zeros_like(test_neg_src)]).cpu().numpy()

Then, we use the dot product predictor to get the score of positive and negative test pairs to compute metrics such as AUC:

.. code-block:: python

    h_pos_src = node_reprs[test_pos_src]
    h_pos_dst = node_reprs[test_pos_dst]
    h_neg_src = node_reprs[test_neg_src]
    h_neg_dst = node_reprs[test_neg_dst]
    score_pos = predictor(h_pos_src, h_pos_dst)
    score_neg = predictor(h_neg_src, h_neg_dst)

    test_preds = th.cat([score_pos, score_neg]).cpu().numpy()
    auc = skm.roc_auc_score(test_labels, test_preds)


Set up distributed training environment
---------------------------------------

The distributed training environment set up is similar to the distributed node classification. Please refer here for more details:
`Set up distributed training environment <https://docs.dgl.ai/en/latest/tutorials/dist/1_node_classification.html#set-up-distributed-training-environment>`_
"""
