"""
Training GNN with Neighbor Sampling for Node Classification
===========================================================

This tutorial shows how to train a multi-layer GraphSAGE for node
classification on ``ogbn-arxiv`` provided by `Open Graph
Benchmark (OGB) <https://ogb.stanford.edu/>`__. The dataset contains around
170 thousand nodes and 1 million edges.

By the end of this tutorial, you will be able to

-  Train a GNN model for node classification on a single GPU with DGL's
   neighbor sampling components.

This tutorial assumes that you have read the :doc:`Introduction of Neighbor
Sampling for GNN Training <L0_neighbor_sampling_overview>`.

"""


######################################################################
# Loading Dataset
# ---------------
#
# OGB already prepared the data as DGL graph.
#

import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

dataset = DglNodePropPredDataset("ogbn-arxiv")
device = "cpu"  # change to 'cuda' for GPU


######################################################################
# OGB dataset is a collection of graphs and their labels. ``ogbn-arxiv``
# dataset only contains a single graph. So you can
# simply get the graph and its node labels like this:
#

graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
graph.ndata["label"] = node_labels[:, 0]
print(graph)
print(node_labels)

node_features = graph.ndata["feat"]
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
print("Number of classes:", num_classes)


######################################################################
# You can get the training-validation-test split of the nodes with
# ``get_split_idx`` method.
#

idx_split = dataset.get_idx_split()
train_nids = idx_split["train"]
valid_nids = idx_split["valid"]
test_nids = idx_split["test"]


######################################################################
# How DGL Handles Computation Dependency
# --------------------------------------
#
# In the :doc:`previous tutorial <L0_neighbor_sampling_overview>`, you
# have seen that the computation dependency for message passing of a
# single node can be described as a series of *message flow graphs* (MFG).
#
# |image1|
#
# .. |image1| image:: https://data.dgl.ai/tutorial/img/bipartite.gif
#


######################################################################
# Defining Neighbor Sampler and Data Loader in DGL
# ------------------------------------------------
#
# DGL provides tools to iterate over the dataset in minibatches
# while generating the computation dependencies to compute their outputs
# with the MFGs above. For node classification, you can use
# ``dgl.dataloading.DataLoader`` for iterating over the dataset.
# It accepts a sampler object to control how to generate the computation
# dependencies in the form of MFGs.  DGL provides
# implementations of common sampling algorithms such as
# ``dgl.dataloading.NeighborSampler`` which randomly picks
# a fixed number of neighbors for each node.
#
# .. note::
#
#    To write your own neighbor sampler, please refer to :ref:`this user
#    guide section <guide-minibatch-customizing-neighborhood-sampler>`.
#
# The syntax of ``dgl.dataloading.DataLoader`` is mostly similar to a
# PyTorch ``DataLoader``, with the addition that it needs a graph to
# generate computation dependency from, a set of node IDs to iterate on,
# and the neighbor sampler you defined.
#
# Let’s say that each node will gather messages from 4 neighbors on each
# layer. The code defining the data loader and neighbor sampler will look
# like the following.
#

sampler = dgl.dataloading.NeighborSampler([4, 4])
train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DGL's DataLoader.
    graph,  # The graph
    train_nids,  # The node IDs to iterate over in minibatches
    sampler,  # The neighbor sampler
    device=device,  # Put the sampled MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=1024,  # Batch size
    shuffle=True,  # Whether to shuffle the nodes for every epoch
    drop_last=False,  # Whether to drop the last incomplete batch
    num_workers=0,  # Number of sampler processes
)


######################################################################
# .. note::
#
#    Since DGL 0.7 neighborhood sampling on GPU is supported.  Please
#    refer to :ref:`guide-minibatch-gpu-sampling` if you are
#    interested.
#


######################################################################
# You can iterate over the data loader and see what it yields.
#

input_nodes, output_nodes, mfgs = example_minibatch = next(
    iter(train_dataloader)
)
print(example_minibatch)
print(
    "To compute {} nodes' outputs, we need {} nodes' input features".format(
        len(output_nodes), len(input_nodes)
    )
)


######################################################################
# DGL's ``DataLoader`` gives us three items per iteration.
#
# -  An ID tensor for the input nodes, i.e., nodes whose input features
#    are needed on the first GNN layer for this minibatch.
# -  An ID tensor for the output nodes, i.e. nodes whose representations
#    are to be computed.
# -  A list of MFGs storing the computation dependencies
#    for each GNN layer.
#


######################################################################
# You can get the source and destination node IDs of the MFGs
# and verify that the first few source nodes are always the same as the destination
# nodes.  As we described in the :doc:`overview <L0_neighbor_sampling_overview>`,
# destination nodes' own features from the previous layer may also be necessary in
# the computation of the new features.
#

mfg_0_src = mfgs[0].srcdata[dgl.NID]
mfg_0_dst = mfgs[0].dstdata[dgl.NID]
print(mfg_0_src)
print(mfg_0_dst)
print(torch.equal(mfg_0_src[: mfgs[0].num_dst_nodes()], mfg_0_dst))


######################################################################
# Defining Model
# --------------
#
# Let’s consider training a 2-layer GraphSAGE with neighbor sampling. The
# model can be written as follows:
#

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type="mean")
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[: mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h_dst = h[: mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(mfgs[1], (h, h_dst))  # <---
        return h


model = Model(num_features, 128, num_classes).to(device)


######################################################################
# If you compare against the code in the
# :doc:`introduction <../blitz/1_introduction>`, you will notice several
# differences:
#
# -  **DGL GNN layers on MFGs**. Instead of computing on the
#    full graph:
#
#    .. code:: python
#
#       h = self.conv1(g, x)
#
#    you only compute on the sampled MFG:
#
#    .. code:: python
#
#       h = self.conv1(mfgs[0], (x, h_dst))
#
#    All DGL’s GNN modules support message passing on MFGs,
#    where you supply a pair of features, one for source nodes and another
#    for destination nodes.
#
# -  **Feature slicing for self-dependency**. There are statements that
#    perform slicing to obtain the previous-layer representation of the
#     nodes:
#
#    .. code:: python
#
#       h_dst = x[:mfgs[0].num_dst_nodes()]
#
#    ``num_dst_nodes`` method works with MFGs, where it will
#    return the number of destination nodes.
#
#    Since the first few source nodes of the yielded MFG are
#    always the same as the destination nodes, these statements obtain the
#    representations of the destination nodes on the previous layer. They are
#    then combined with neighbor aggregation in ``dgl.nn.SAGEConv`` layer.
#
# .. note::
#
#    See the :doc:`custom message passing
#    tutorial <L4_message_passing>` for more details on how to
#    manipulate MFGs produced in this way, such as the usage
#    of ``num_dst_nodes``.
#


######################################################################
# Defining Training Loop
# ----------------------
#
# The following initializes the model and defines the optimizer.
#

opt = torch.optim.Adam(model.parameters())


######################################################################
# When computing the validation score for model selection, usually you can
# also do neighbor sampling. To do that, you need to define another data
# loader.
#

valid_dataloader = dgl.dataloading.DataLoader(
    graph,
    valid_nids,
    sampler,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    device=device,
)


######################################################################
# The following is a training loop that performs validation every epoch.
# It also saves the model with the best validation accuracy into a file.
#

import tqdm
import sklearn.metrics

best_accuracy = 0
best_model_path = "model.pt"
for epoch in range(10):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata["feat"]
            labels = mfgs[-1].dstdata["label"]

            predictions = model(mfgs, inputs)

            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            accuracy = sklearn.metrics.accuracy_score(
                labels.cpu().numpy(),
                predictions.argmax(1).detach().cpu().numpy(),
            )

            tq.set_postfix(
                {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy},
                refresh=False,
            )

    model.eval()

    predictions = []
    labels = []
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata["feat"]
            labels.append(mfgs[-1].dstdata["label"].cpu().numpy())
            predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print("Epoch {} Validation Accuracy {}".format(epoch, accuracy))
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)

        # Note that this tutorial do not train the whole model to the end.
        break


######################################################################
# Conclusion
# ----------
#
# In this tutorial, you have learned how to train a multi-layer GraphSAGE
# with neighbor sampling.
#
# What’s next?
# ------------
#
# -  :doc:`Stochastic training of GNN for link
#    prediction <L2_large_link_prediction>`.
# -  :doc:`Adapting your custom GNN module for stochastic
#    training <L4_message_passing>`.
# -  During inference you may wish to disable neighbor sampling. If so,
#    please refer to the :ref:`user guide on exact offline
#    inference <guide-minibatch-inference>`.
#


# Thumbnail credits: Stanford CS224W Notes
# sphinx_gallery_thumbnail_path = '_static/blitz_1_introduction.png'
