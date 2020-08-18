.. _guide-training-graph-classification:

5.4 Graph Classification
----------------------------------

Instead of a big single graph, sometimes we might have the data in the
form of multiple graphs, for example a list of different types of
communities of people. By characterizing the friendships among people in
the same community by a graph, we get a list of graphs to classify. In
this scenario, a graph classification model could help identify the type
of the community, i.e. to classify each graph based on the structure and
overall information.

Overview
~~~~~~~~

The major difference between graph classification and node
classification or link prediction is that the prediction result
characterize the property of the entire input graph. We perform the
message passing over nodes/edges just like the previous tasks, but also
try to retrieve a graph-level representation.

The graph classification proceeds as follows:

.. figure:: https://data.dgl.ai/tutorial/batch/graph_classifier.png
   :alt: Graph Classification Process

   Graph Classification Process

From left to right, the common practice is:

-  Prepare graphs in to a batch of graphs
-  Message passing on the batched graphs to update node/edge features
-  Aggregate node/edge features into a graph-level representation
-  Classification head for the task

Batch of Graphs
^^^^^^^^^^^^^^^

Usually a graph classification task trains on a lot of graphs, and it
will be very inefficient if we use only one graph at a time when
training the model. Borrowing the idea of mini-batch training from
common deep learning practice, we can build a batch of multiple graphs
and send them together for one training iteration.

In DGL, we can build a single batched graph of a list of graphs. This
batched graph can be simply used as a single large graph, with separated
components representing the corresponding original small graphs.

.. figure:: https://data.dgl.ai/tutorial/batch/batch.png
   :alt: Batched Graph

   Batched Graph

Graph Readout
^^^^^^^^^^^^^

Every graph in the data may have its unique structure, as well as its
node and edge features. In order to make a single prediction, we usually
aggregate and summarize over the possibly abundant information. This
type of operation is named *Readout*. Common aggregations include
summation, average, maximum or minimum over all node or edge features.

Given a graph :math:`g`, we can define the average readout aggregation
as

.. math:: h_g = \frac{1}{|\mathcal{V}|}\sum_{v\in \mathcal{V}}h_v

In DGL the corresponding function call is :func:`dgl.readout_nodes`.

Once :math:`h_g` is available, we can pass it through an MLP layer for
classification output.

Writing neural network model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input to the model is the batched graph with node and edge features.
One thing to note is the node and edge features in the batched graph
have no batch dimension. A little special care should be put in the
model:

Computation on a batched graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we discuss the computational properties of a batched graph.

First, different graphs in a batch are entirely separated, i.e. no edge
connecting two graphs. With this nice property, all message passing
functions still have the same results.

Second, the readout function on a batched graph will be conducted over
each graph separately. Assume the batch size is :math:`B` and the
feature to be aggregated has dimension :math:`D`, the shape of the
readout result will be :math:`(B, D)`.

.. code:: python

    g1 = dgl.graph(([0, 1], [1, 0]))
    g1.ndata['h'] = torch.tensor([1., 2.])
    g2 = dgl.graph(([0, 1], [1, 2]))
    g2.ndata['h'] = torch.tensor([1., 2., 3.])
    
    dgl.readout_nodes(g1, 'h')
    # tensor([3.])  # 1 + 2
    
    bg = dgl.batch([g1, g2])
    dgl.readout_nodes(bg, 'h')
    # tensor([3., 6.])  # [1 + 2, 1 + 2 + 3]

Finally, each node/edge feature tensor on a batched graph is in the
format of concatenating the corresponding feature tensor from all
graphs.

.. code:: python

    bg.ndata['h']
    # tensor([1., 2., 1., 2., 3.])

Model definition
^^^^^^^^^^^^^^^^

Being aware of the above computation rules, we can define a very simple
model.

.. code:: python

    class Classifier(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_classes):
            super(Classifier, self).__init__()
            self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
            self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
            self.classify = nn.Linear(hidden_dim, n_classes)
    
        def forward(self, g, feat):
            # Apply graph convolution and activation.
            h = F.relu(self.conv1(g, h))
            h = F.relu(self.conv2(g, h))
            with g.local_scope():
                g.ndata['h'] = h
                # Calculate graph representation by average readout.
                hg = dgl.mean_nodes(g, 'h')
                return self.classify(hg)

Training loop
~~~~~~~~~~~~~

Data Loading
^^^^^^^^^^^^

Once the model’s defined, we can start training. Since graph
classification deals with lots of relative small graphs instead of a big
single one, we usually can train efficiently on stochastic mini-batches
of graphs, without the need to design sophisticated graph sampling
algorithms.

Assuming that we have a graph classification dataset as introduced in
:ref:`guide-data-pipeline`.

.. code:: python

    import dgl.data
    dataset = dgl.data.GINDataset('MUTAG', False)

Each item in the graph classification dataset is a pair of a graph and
its label. We can speed up the data loading process by taking advantage
of the DataLoader, by customizing the collate function to batch the
graphs:

.. code:: python

    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

Then one can create a DataLoader that iterates over the dataset of
graphs in minibatches.

.. code:: python

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)

Loop
^^^^

Training loop then simply involves iterating over the dataloader and
updating the model.

.. code:: python

    model = Classifier(10, 20, 5)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(20):
        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['feats']
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

DGL implements
`GIN <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin>`__
as an example of graph classification. The training loop is inside the
function ``train`` in
```main.py`` <https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/main.py>`__.
The model implementation is inside
```gin.py`` <https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/gin.py>`__
with more components such as using
:class:`dgl.nn.pytorch.GINConv` (also available in MXNet and Tensorflow)
as the graph convolution layer, batch normalization, etc.

Heterogeneous graph
~~~~~~~~~~~~~~~~~~~

Graph classification with heterogeneous graphs is a little different
from that with homogeneous graphs. Except that you need heterogeneous
graph convolution modules, yoyu also need to aggregate over the nodes of
different types in the readout function.

The following shows an example of summing up the average of node
representations for each node type.

.. code:: python

    class RGCN(nn.Module):
        def __init__(self, in_feats, hid_feats, out_feats, rel_names):
            super().__init__()
            
            self.conv1 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names}, aggregate='sum')
            self.conv2 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(hid_feats, out_feats)
                for rel in rel_names}, aggregate='sum')
      
        def forward(self, graph, inputs):
            # inputs are features of nodes
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            return h
    
    class HeteroClassifier(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
            super().__init__()
            
            self.conv1 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names}, aggregate='sum')
            self.conv2 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(hid_feats, out_feats)
                for rel in rel_names}, aggregate='sum')
            self.classify = nn.Linear(hidden_dim, n_classes)
    
        def forward(self, g):
            h = g.ndata['feat']
            # Apply graph convolution and activation.
            h = F.relu(self.conv1(g, h))
            h = F.relu(self.conv2(g, h))
    
            with g.local_scope():
                g.ndata['h'] = h
                # Calculate graph representation by average readout.
                hg = 0
                for ntype in g.ntypes:
                    hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                return self.classify(hg)

The rest of the code is not different from that for homogeneous graphs.

.. code:: python

    model = HeteroClassifier(10, 20, 5)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(20):
        for batched_graph, labels in dataloader:
            logits = model(batched_graph)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
