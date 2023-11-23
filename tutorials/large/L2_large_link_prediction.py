"""
Stochastic Training of GNN for Link Prediction
==============================================

This tutorial will show how to train a multi-layer GraphSAGE for link
prediction on `CoraGraphDataset <https://data.dgl.ai/dataset/cora_v2.zip>`__.
The dataset contains 2708 nodes and 10556 edges.

By the end of this tutorial, you will be able to

-  Train a GNN model for link prediction on target device with DGL's
   neighbor sampling components.

This tutorial assumes that you have read the :doc:`Introduction of Neighbor
Sampling for GNN Training <L0_neighbor_sampling_overview>` and :doc:`Neighbor
Sampling for Node Classification <L1_large_node_classification>`.

"""


######################################################################
# Link Prediction Overview
# ------------------------
#
# Unlike node classification predicts labels for nodes based on their
# local neighborhoods, link prediction assesses the likelihood of an edge
# existing between two nodes, necessitating different sampling strategies
# that account for pairs of nodes and their joint neighborhoods.
#


######################################################################
# Loading Dataset
# ---------------
#
# `cora` is already prepared as ``BuiltinDataset`` in GraphBolt.
#

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl.graphbolt as gb
import numpy as np
import torch
import tqdm

dataset = gb.BuiltinDataset("cora").load()
device = torch.device("cpu")  # change to 'cuda' for GPU


######################################################################
# Dataset consists of graph, feature and tasks. You can get the
# training-validation-test set from the tasks. Seed nodes and corresponding
# labels are already stored in each training-validation-test set. This
# dataset contains 2 tasks, one for node classification and the other for
# link prediction. We will use the link prediction task.
#

graph = dataset.graph
feature = dataset.feature
train_set = dataset.tasks[1].train_set
test_set = dataset.tasks[1].test_set
task_name = dataset.tasks[1].metadata["name"]
print(f"Task: {task_name}.")


######################################################################
# Defining Neighbor Sampler and Data Loader in DGL
# ------------------------------------------------
#
# Different from the :doc:`link prediction tutorial for full
# graph <../blitz/4_link_predict>`, a common practice to train GNN on large graphs is
# to iterate over the edges
# in minibatches, since computing the probability of all edges is usually
# impossible. For each minibatch of edges, you compute the output
# representation of their incident nodes using neighbor sampling and GNN,
# in a similar fashion introduced in the :doc:`large-scale node classification
# tutorial <L1_large_node_classification>`.
#
# To perform link prediction, you need to specify a negative sampler. DGL
# provides builtin negative samplers such as
# ``dgl.graphbolt.UniformNegativeSampler``.  Here this tutorial uniformly
# draws 5 negative examples per positive example.
#
# Except for the negative sampler, the rest of the code is identical to
# the :doc:`node classification tutorial <L1_large_node_classification>`.
#

datapipe = gb.ItemSampler(train_set, batch_size=256, shuffle=True)
datapipe = datapipe.sample_uniform_negative(graph, 5)
datapipe = datapipe.sample_neighbor(graph, [5, 5, 5])
datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
datapipe = datapipe.to_dgl()
datapipe = datapipe.copy_to(device)
train_dataloader = gb.MultiProcessDataLoader(datapipe, num_workers=0)


######################################################################
# You can peek one minibatch from ``train_dataloader`` and see what it
# will give you.
#

data = next(iter(train_dataloader))
print(f"DGLMiniBatch: {data}")


######################################################################
# Defining Model for Node Representation
# --------------------------------------
#

import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.hidden_size = hidden_size
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
        return hidden_x


######################################################################
# Defining Training Loop
# ----------------------
#
# The following initializes the model and defines the optimizer.
#

in_size = feature.size("node", None, "feat")[0]
model = SAGE(in_size, 128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#####################################################################
# Convert the minibatch to a training pair and a label tensor.
#


def to_binary_link_dgl_computing_pack(data: gb.DGLMiniBatch):
    """Convert the minibatch to a training pair and a label tensor."""
    pos_src, pos_dst = data.positive_node_pairs
    neg_src, neg_dst = data.negative_node_pairs
    node_pairs = (
        torch.cat((pos_src, neg_src), dim=0),
        torch.cat((pos_dst, neg_dst), dim=0),
    )
    pos_label = torch.ones_like(pos_src)
    neg_label = torch.zeros_like(neg_src)
    labels = torch.cat([pos_label, neg_label], dim=0)
    return (node_pairs, labels.float())


######################################################################
# The following is the training loop for link prediction and
# evaluation.
#

for epoch in range(10):
    model.train()
    total_loss = 0
    for step, data in tqdm.tqdm(enumerate(train_dataloader)):
        # Unpack MiniBatch.
        compacted_pairs, labels = to_binary_link_dgl_computing_pack(data)
        node_feature = data.node_features["feat"]
        # Convert sampled subgraphs to DGL blocks.
        blocks = data.blocks

        # Get the embeddings of the input nodes.
        y = model(blocks, node_feature)
        logits = model.predictor(
            y[compacted_pairs[0]] * y[compacted_pairs[1]]
        ).squeeze()

        # Compute loss.
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:03d} | Loss {total_loss / (step + 1):.3f}")


######################################################################
# Evaluating Performance with Link Prediction
# -------------------------------------------
#


model.eval()

datapipe = gb.ItemSampler(test_set, batch_size=256, shuffle=False)
# Since we need to use all neghborhoods for evaluation, we set the fanout
# to -1.
datapipe = datapipe.sample_neighbor(graph, [-1, -1])
datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
datapipe = datapipe.to_dgl()
datapipe = datapipe.copy_to(device)
eval_dataloader = gb.MultiProcessDataLoader(datapipe, num_workers=0)

logits = []
labels = []
for step, data in enumerate(eval_dataloader):
    # Unpack MiniBatch.
    compacted_pairs, label = to_binary_link_dgl_computing_pack(data)

    # The features of sampled nodes.
    x = data.node_features["feat"]

    # Forward.
    y = model(data.blocks, x)
    logit = (
        model.predictor(y[compacted_pairs[0]] * y[compacted_pairs[1]])
        .squeeze()
        .detach()
    )

    logits.append(logit)
    labels.append(label)

logits = torch.cat(logits, dim=0)
labels = torch.cat(labels, dim=0)


# Compute the AUROC score.
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(labels, logits)
print("Link Prediction AUC:", auc)


######################################################################
# Conclusion
# ----------
#
# In this tutorial, you have learned how to train a multi-layer GraphSAGE
# for link prediction with neighbor sampling.
#
