"""
Stochastic Training of GNN for Link Prediction
==============================================

This tutorial will show how to train a multi-layer GraphSAGE for link
prediction on ``ogbn-arxiv`` provided by `Open Graph Benchmark
(OGB) <https://ogb.stanford.edu/>`__. The dataset
contains around 170 thousand nodes and 1 million edges.

By the end of this tutorial, you will be able to

-  Train a GNN model for link prediction on a single GPU with DGL's
   neighbor sampling components.

This tutorial assumes that you have read the :doc:`Introduction of Neighbor
Sampling for GNN Training <L0_neighbor_sampling_overview>` and :doc:`Neighbor
Sampling for Node Classification <L1_large_node_classification>`.

"""


######################################################################
# Link Prediction Overview
# ------------------------
#
# Link prediction requires the model to predict the probability of
# existence of an edge. This tutorial does so by computing a dot product
# between the representations of both incident nodes.
#
# .. math::
#
#
#    \hat{y}_{u\sim v} = \sigma(h_u^T h_v)
#
# It then minimizes the following binary cross entropy loss.
#
# .. math::
#
#
#    \mathcal{L} = -\sum_{u\sim v\in \mathcal{D}}\left( y_{u\sim v}\log(\hat{y}_{u\sim v}) + (1-y_{u\sim v})\log(1-\hat{y}_{u\sim v})) \right)
#
# This is identical to the link prediction formulation in :doc:`the previous
# tutorial on link prediction <../blitz/4_link_predict>`.
#


######################################################################
# Loading Dataset
# ---------------
#
# This tutorial loads the dataset from the ``ogb`` package as in the
# :doc:`previous tutorial <L1_large_node_classification>`.
#

import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

dataset = DglNodePropPredDataset("ogbn-arxiv")
device = "cpu"  # change to 'cuda' for GPU

graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
print(graph)
print(node_labels)

node_features = graph.ndata["feat"]
node_labels = node_labels[:, 0]
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
print("Number of classes:", num_classes)

idx_split = dataset.get_idx_split()
train_nids = idx_split["train"]
valid_nids = idx_split["valid"]
test_nids = idx_split["test"]


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
# DGL provides ``dgl.dataloading.as_edge_prediction_sampler`` to
# iterate over edges for edge classification or link prediction tasks.
#
# To perform link prediction, you need to specify a negative sampler. DGL
# provides builtin negative samplers such as
# ``dgl.dataloading.negative_sampler.Uniform``.  Here this tutorial uniformly
# draws 5 negative examples per positive example.
#

negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)


######################################################################
# After defining the negative sampler, one can then define the edge data
# loader with neighbor sampling.  To create an ``DataLoader`` for
# link prediction, provide a neighbor sampler object as well as the negative
# sampler object created above.
#

sampler = dgl.dataloading.NeighborSampler([4, 4])
sampler = dgl.dataloading.as_edge_prediction_sampler(
    sampler, negative_sampler=negative_sampler
)
train_dataloader = dgl.dataloading.DataLoader(
    # The following arguments are specific to DataLoader.
    graph,  # The graph
    torch.arange(graph.number_of_edges()),  # The edges to iterate over
    sampler,  # The neighbor sampler
    device=device,  # Put the MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=1024,  # Batch size
    shuffle=True,  # Whether to shuffle the nodes for every epoch
    drop_last=False,  # Whether to drop the last incomplete batch
    num_workers=0,  # Number of sampler processes
)


######################################################################
# You can peek one minibatch from ``train_dataloader`` and see what it
# will give you.
#

input_nodes, pos_graph, neg_graph, mfgs = next(iter(train_dataloader))
print("Number of input nodes:", len(input_nodes))
print(
    "Positive graph # nodes:",
    pos_graph.number_of_nodes(),
    "# edges:",
    pos_graph.number_of_edges(),
)
print(
    "Negative graph # nodes:",
    neg_graph.number_of_nodes(),
    "# edges:",
    neg_graph.number_of_edges(),
)
print(mfgs)


######################################################################
# The example minibatch consists of four elements.
#
# The first element is an ID tensor for the input nodes, i.e., nodes
# whose input features are needed on the first GNN layer for this minibatch.
#
# The second element and the third element are the positive graph and the
# negative graph for this minibatch.
# The concept of positive and negative graphs have been introduced in the
# :doc:`full-graph link prediction tutorial <../blitz/4_link_predict>`.  In minibatch
# training, the positive graph and the negative graph only contain nodes
# necessary for computing the pair-wise scores of positive and negative examples
# in the current minibatch.
#
# The last element is a list of :doc:`MFGs <L0_neighbor_sampling_overview>`
# storing the computation dependencies for each GNN layer.
# The MFGs are used to compute the GNN outputs of the nodes
# involved in positive/negative graph.
#


######################################################################
# Defining Model for Node Representation
# --------------------------------------
#
# The model is almost identical to the one in the :doc:`node classification
# tutorial <L1_large_node_classification>`. The only difference is
# that since you are doing link prediction, the output dimension will not
# be the number of classes in the dataset.
#

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class Model(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type="mean")
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[: mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[: mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h


model = Model(num_features, 128).to(device)


######################################################################
# Defining the Score Predictor for Edges
# --------------------------------------
#
# After getting the node representation necessary for the minibatch, the
# last thing to do is to predict the score of the edges and non-existent
# edges in the sampled minibatch.
#
# The following score predictor, copied from the :doc:`link prediction
# tutorial <../blitz/4_link_predict>`, takes a dot product between the
# incident nodes’ representations.
#

import dgl.function as fn


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]


######################################################################
# Evaluating Performance with Unsupervised Learning (Optional)
# ------------------------------------------------------------
#
# There are various ways to evaluate the performance of link prediction.
# This tutorial follows the practice of `GraphSAGE
# paper <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>`__.
# Basically, it first trains a GNN via link prediction, and get an embedding
# for each node.  Then it trains a downstream classifier on top of this
# embedding and compute the accuracy as an assessment of the embedding
# quality.
#


######################################################################
# To obtain the representations of all the nodes, this tutorial uses
# neighbor sampling as introduced in the :doc:`node classification
# tutorial <L1_large_node_classification>`.
#
# .. note::
#
#    If you would like to obtain node representations without
#    neighbor sampling during inference, please refer to this :ref:`user
#    guide <guide-minibatch-inference>`.
#


def inference(model, graph, node_features):
    with torch.no_grad():
        nodes = torch.arange(graph.number_of_nodes())

        sampler = dgl.dataloading.NeighborSampler([4, 4])
        train_dataloader = dgl.dataloading.DataLoader(
            graph,
            torch.arange(graph.number_of_nodes()),
            sampler,
            batch_size=1024,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            device=device,
        )

        result = []
        for input_nodes, output_nodes, mfgs in train_dataloader:
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata["feat"]
            result.append(model(mfgs, inputs))

        return torch.cat(result)


import sklearn.metrics


def evaluate(emb, label, train_nids, valid_nids, test_nids):
    classifier = nn.Linear(emb.shape[1], num_classes).to(device)
    opt = torch.optim.LBFGS(classifier.parameters())

    def compute_loss():
        pred = classifier(emb[train_nids].to(device))
        loss = F.cross_entropy(pred, label[train_nids].to(device))
        return loss

    def closure():
        loss = compute_loss()
        opt.zero_grad()
        loss.backward()
        return loss

    prev_loss = float("inf")
    for i in range(1000):
        opt.step(closure)
        with torch.no_grad():
            loss = compute_loss().item()
            if np.abs(loss - prev_loss) < 1e-4:
                print("Converges at iteration", i)
                break
            else:
                prev_loss = loss

    with torch.no_grad():
        pred = classifier(emb.to(device)).cpu()
        label = label
        valid_acc = sklearn.metrics.accuracy_score(
            label[valid_nids].numpy(), pred[valid_nids].numpy().argmax(1)
        )
        test_acc = sklearn.metrics.accuracy_score(
            label[test_nids].numpy(), pred[test_nids].numpy().argmax(1)
        )
    return valid_acc, test_acc


######################################################################
# Defining Training Loop
# ----------------------
#
# The following initializes the model and defines the optimizer.
#

model = Model(node_features.shape[1], 128).to(device)
predictor = DotPredictor().to(device)
opt = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()))


######################################################################
# The following is the training loop for link prediction and
# evaluation, and also saves the model that performs the best on the
# validation set:
#

import tqdm
import sklearn.metrics

best_accuracy = 0
best_model_path = "model.pt"
for epoch in range(1):
    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(tq):
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata["feat"]

            outputs = model(mfgs, inputs)
            pos_score = predictor(pos_graph, outputs)
            neg_score = predictor(neg_graph, outputs)

            score = torch.cat([pos_score, neg_score])
            label = torch.cat(
                [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
            )
            loss = F.binary_cross_entropy_with_logits(score, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tq.set_postfix({"loss": "%.03f" % loss.item()}, refresh=False)

            if (step + 1) % 500 == 0:
                model.eval()
                emb = inference(model, graph, node_features)
                valid_acc, test_acc = evaluate(
                    emb, node_labels, train_nids, valid_nids, test_nids
                )
                print(
                    "Epoch {} Validation Accuracy {} Test Accuracy {}".format(
                        epoch, valid_acc, test_acc
                    )
                )
                if best_accuracy < valid_acc:
                    best_accuracy = valid_acc
                    torch.save(model.state_dict(), best_model_path)
                model.train()

                # Note that this tutorial do not train the whole model to the end.
                break


######################################################################
# Evaluating Performance with Link Prediction (Optional)
# ------------------------------------------------------
#
# In practice, it is more common to evaluate the link prediction
# model to see whether it can predict new edges. There are different
# evaluation metrics such as
# `AUC <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`__
# or `various metrics from information retrieval <https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)>`__.
# Ultimately, they require the model to predict one scalar score given
# a node pair among a set of node pairs.
#
# Assuming that you have the following test set with labels, where
# ``test_pos_src`` and ``test_pos_dst`` are ground truth node pairs
# with edges in between (or *positive* pairs), and ``test_neg_src``
# and ``test_neg_dst`` are ground truth node pairs without edges
# in between (or *negative* pairs).
#

# Positive pairs
# These are randomly generated as an example.  You will need to
# replace them with your own ground truth.
n_test_pos = 1000
test_pos_src, test_pos_dst = (
    torch.randint(0, graph.num_nodes(), (n_test_pos,)),
    torch.randint(0, graph.num_nodes(), (n_test_pos,)),
)
# Negative pairs.  Likewise, you will need to replace them with your
# own ground truth.
test_neg_src = test_pos_src
test_neg_dst = torch.randint(0, graph.num_nodes(), (n_test_pos,))


######################################################################
# First you need to compute the node representations for all the nodes
# with the ``inference`` method above:
#

node_reprs = inference(model, graph, node_features)

######################################################################
# Since the predictor is a dot product, you can now easily compute the
# score of positive and negative test pairs to compute metrics such
# as AUC:
#

h_pos_src = node_reprs[test_pos_src]
h_pos_dst = node_reprs[test_pos_dst]
h_neg_src = node_reprs[test_neg_src]
h_neg_dst = node_reprs[test_neg_dst]
score_pos = (h_pos_src * h_pos_dst).sum(1)
score_neg = (h_neg_src * h_neg_dst).sum(1)
test_preds = torch.cat([score_pos, score_neg]).cpu().numpy()
test_labels = (
    torch.cat([torch.ones_like(score_pos), torch.zeros_like(score_neg)])
    .cpu()
    .numpy()
)

auc = sklearn.metrics.roc_auc_score(test_labels, test_preds)
print("Link Prediction AUC:", auc)


######################################################################
# Conclusion
# ----------
#
# In this tutorial, you have learned how to train a multi-layer GraphSAGE
# for link prediction with neighbor sampling.
#


# Thumbnail credits: Link Prediction with Neo4j, Mark Needham
# sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'
