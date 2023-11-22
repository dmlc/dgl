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
# `ogbn-arxiv` is already prepared as ``BuiltinDataset`` in GraphBolt.
#

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.graphbolt as gb
import numpy as np
import torch

dataset = gb.BuiltinDataset("ogbn-arxiv").load()
device = "cpu"  # change to 'cuda' for GPU


######################################################################
# Dataset consists of graph, feature and tasks. You can get the
# training-validation-test set from the tasks. Seed nodes and corresponding
# labels are already stored in each training-validation-test set. Other
# metadata such as number of classes are also stored in the tasks. In this
# dataset, there is only one task: `node classification``.
#

graph = dataset.graph
feature = dataset.feature
train_set = dataset.tasks[0].train_set
valid_set = dataset.tasks[0].validation_set
test_set = dataset.tasks[0].test_set
task_name = dataset.tasks[0].metadata["name"]
num_classes = dataset.tasks[0].metadata["num_classes"]
print(f"Task: {task_name}. Number of classes: {num_classes}")


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
# ``dgl.graphbolt.MultiProcessDataLoader`` for iterating over the dataset.
# It accepts a data pipe that generates minibatches of nodes and their
# labels, sample neighbors for each node, and generate the computation
# dependencies in the form of MFGs. Feature fetching, block creation and
# copying to target device are also supported. All these operations are
# split into separate stages in the data pipe, so that you can customize
# the data pipeline by inserting your own operations.
#
# .. note::
#
#    To write your own neighbor sampler, please refer to :ref:`this user
#    guide section <guide-minibatch-customizing-neighborhood-sampler>`.
#
#
# Let’s say that each node will gather messages from 4 neighbors on each
# layer. The code defining the data loader and neighbor sampler will look
# like the following.
#

datapipe = gb.ItemSampler(train_set, batch_size=1024, shuffle=True)
datapipe = datapipe.sample_neighbor(graph, [4, 4])
datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
datapipe = datapipe.to_dgl()
datapipe = datapipe.copy_to(device)
train_dataloader = gb.MultiProcessDataLoader(datapipe, num_workers=0)


######################################################################
# .. note::
#
#    In this example, neighborhood sampling runs on CPU, If you are
#    interested in running it on GPU, please refer to
#    :ref:`guide-minibatch-gpu-sampling`.
#


######################################################################
# You can iterate over the data loader and a ``DGLMiniBatch`` object
# is yielded.
#

data = next(iter(train_dataloader))
print(data)


######################################################################
# You can get the input node IDs from MFGs.
#

mfgs = data.blocks
input_nodes = mfgs[0].srcdata[dgl.NID]
print(f"Input nodes: {input_nodes}.")

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


in_size = feature.size("node", None, "feat")[0]
model = Model(in_size, 64, num_classes).to(device)


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

datapipe = gb.ItemSampler(valid_set, batch_size=1024, shuffle=False)
datapipe = datapipe.sample_neighbor(graph, [4, 4])
datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
datapipe = datapipe.to_dgl()
datapipe = datapipe.copy_to(device)
valid_dataloader = gb.MultiProcessDataLoader(datapipe, num_workers=0)


import sklearn.metrics

######################################################################
# The following is a training loop that performs validation every epoch.
# It also saves the model with the best validation accuracy into a file.
#

import tqdm

best_accuracy = 0
best_model_path = "model.pt"
for epoch in range(10):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, data in enumerate(tq):
            x = data.node_features["feat"]
            labels = data.labels

            predictions = model(data.blocks, x)

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
        for data in tq:
            x = data.node_features["feat"]
            labels.append(data.labels.cpu().numpy())
            predictions.append(model(data.blocks, x).argmax(1).cpu().numpy())
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
