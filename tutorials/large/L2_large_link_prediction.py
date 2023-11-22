"""
Stochastic Training of GNN for Link Prediction
==============================================

This tutorial will show how to train a multi-layer GraphSAGE for link
prediction on ``ogbl-citation2`` provided by `Open Graph Benchmark
(OGB) <https://ogb.stanford.edu/>`__. The dataset
contains around 3 million nodes and 30 million edges.

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
# Link prediction requires the model to predict the probability of
# existence of an edge. This tutorial does so by computing a Mean
# Reciprocal Rank (MRR) score for each edge, which is a common metric for
# link prediction.
#


######################################################################
# Loading Dataset
# ---------------
#
# `ogbl-citation2` is already prepared as ``BuiltinDataset`` in GraphBolt.
#

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl.graphbolt as gb
import numpy as np
import torch
import tqdm

dataset = gb.BuiltinDataset("ogbl-citation2").load()
device = "cpu"  # change to 'cuda' for GPU


######################################################################
# Dataset consists of graph, feature and tasks. You can get the
# training-validation-test set from the tasks. Seed nodes and corresponding
# labels are already stored in each training-validation-test set. Other
# metadata such as number of classes are also stored in the tasks. In this
# dataset, there is only one task: `link prediction``.
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

datapipe = gb.ItemSampler(train_set, batch_size=1024, shuffle=True)
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
# Defining Model for Link Prediction
# --------------------------------------
#

import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.hidden_size = hidden_size
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
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

    def inference(self, graph, features, dataloader, device):
        """Conduct layer-wise inference to get all the node embeddings."""
        feature = features.read("node", None, "feat")

        buffer_device = torch.device("cpu")
        # Enable pin_memory for faster CPU to GPU data transfer if the
        # model is running on a GPU.
        pin_memory = buffer_device != device

        print("Start node embedding inference.")
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1

            y = torch.empty(
                graph.total_num_nodes,
                self.hidden_size,
                dtype=torch.float32,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feature = feature.to(device)
            for step, data in tqdm.tqdm(enumerate(dataloader)):
                x = feature[data.input_nodes]
                hidden_x = layer(data.blocks[0], x)  # len(blocks) = 1
                if not is_last_layer:
                    hidden_x = F.relu(hidden_x)
                # By design, our output nodes are contiguous.
                y[
                    data.output_nodes[0] : data.output_nodes[-1] + 1
                ] = hidden_x.to(buffer_device, non_blocking=True)
                if step == 100:
                    print("Stop inference after 100 steps.")
                    break
            feature = y
        return y


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

for epoch in range(1):
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

        if step % 20 == 0:
            print(f"Epoch {epoch} Loss {loss.item()}")

        if step == 100:
            print("Stop training after 100 steps.")
            break


######################################################################
# Evaluating Performance with Link Prediction
# -------------------------------------------
#

##############################################
# Define MRR computation function
#


@torch.no_grad()
def compute_mrr(device, model, evaluator, node_emb, src, dst, neg_dst):
    """Compute the Mean Reciprocal Rank (MRR) for given source and destination
    nodes.

    This function computes the MRR for a set of node pairs, dividing the task
    into batches to handle potentially large graphs.
    """
    rr = torch.zeros(src.shape[0])
    # Loop over node pairs in batches.
    for start in tqdm.trange(
        0, src.shape[0], 4096, desc="Evaluate"
    ):
        end = min(start + 4096, src.shape[0])

        # Concatenate positive and negative destination nodes.
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)

        # Fetch embeddings for current batch of source and destination nodes.
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = (
            node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        )

        # Compute prediction scores using the model.
        pred = model.predictor(h_src * h_dst).squeeze(-1)

        # Evaluate the predictions to obtain MRR values.
        input_dict = {"y_pred_pos": pred[:, 0], "y_pred_neg": pred[:, 1:]}
        rr[start:end] = evaluator.eval(input_dict)["mrr_list"]
    return rr.mean()


#####################################
# Evaluate the trained model.
#

model.eval()
from ogb.linkproppred import Evaluator
evaluator = Evaluator(name="ogbl-citation2")

all_nodes_set = dataset.all_nodes_set
datapipe = gb.ItemSampler(all_nodes_set, batch_size=4096, shuffle=False)
# Since we need to use all neghborhoods for evaluation, we set the fanout
# to -1.
datapipe = datapipe.sample_neighbor(graph, [-1])
datapipe = datapipe.to_dgl()
datapipe = datapipe.copy_to(device)
eval_dataloader = gb.MultiProcessDataLoader(datapipe, num_workers=0)

# Compute node embeddings for the entire graph.
node_emb = model.inference(graph, feature, eval_dataloader, device)
results = []

# Loop over both validation and test sets.
for split in [valid_set, test_set]:
    # Unpack the item set.
    src = split._items[0][:, 0].to(node_emb.device)
    dst = split._items[0][:, 1].to(node_emb.device)
    neg_dst = split._items[1].to(node_emb.device)

    # Compute MRR values for the current split.
    results.append(
        compute_mrr(device, model, evaluator, node_emb, src, dst, neg_dst)
    )

valid_mrr, test_mrr = results
print(
    f"Validation MRR {valid_mrr.item():.4f}, "
    f"Test MRR {test_mrr.item():.4f}"
)


######################################################################
# Conclusion
# ----------
#
# In this tutorial, you have learned how to train a multi-layer GraphSAGE
# for link prediction with neighbor sampling.
#
