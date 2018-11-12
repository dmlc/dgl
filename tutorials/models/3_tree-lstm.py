"""
.. _model-tree-lstm:

Tree LSTM DGL Tutorial
=========================

**Author**: `Zihao Ye`, `Qipeng Guo`, `Minjie Wang`, `Zheng Zhang`

"""
 
##############################################################################
#
# Tree-LSTM structure was first introduced by Kai et. al in their ACL 2015
# paper: `Improved Semantic Representations From Tree-Structured Long
# Short-Term Memory Networks <https://arxiv.org/pdf/1503.00075.pdf>`__,
# aiming to introduce syntactic information in the network by extending
# chain structured LSTM to tree structured LSTM, and uses Dependency
# Tree/Constituency Tree as the latent tree structure.
#
# The difficulty of training Tree-LSTM is that trees have different shape,
# making it difficult to parallelize. DGL offers a neat alternative. The
# key points are pooling all the trees into one graph, and then induce
# message passing over them.
#
# The task and the dataset
# ------------------------
#
# We will use Tree-LSTM for sentiment analysis task. We have wrapped the
# `Stanford Sentiment Treebank <https://nlp.stanford.edu/sentiment/>`__ in
# ``dgl.data``. The dataset provides a fine-grained tree level sentiment
# annotation: 5 classes(very negative, negative, neutral, positive, and
# very positive) that indicates the sentiment in current subtree. Non-leaf
# nodes in constituency tree does not contain words, we use a special
# ``PAD_WORD`` token to denote them, during the training/inferencing,
# their embeddings would be masked to all-zero.
#
# .. figure:: https://i.loli.net/2018/11/08/5be3d4bfe031b.png
#    :alt: 
#
# The figure displays one sample of the SST dataset, which is a
# constituency parse tree with their nodes labeled with sentiment. To
# speed up things, let's build a tiny set with 5 sentences and take a look
# at the first one:
#

import dgl
import dgl.data as data

# Each sample in the dataset is a constituency tree. The leaf nodes
# represent words. The word is a int value stored in the "x" field.
# The non-leaf nodes has a special word PAD_WORD. The sentiment
# label is stored in the "y" feature field.
trainset = data.SST(mode='tiny')  # the "tiny" set has only 5 trees
tiny_sst = trainset.trees
num_vocabs = trainset.num_vocabs
num_classes = trainset.num_classes

vocab = trainset.vocab # vocabulary dict: key -> id
inv_vocab = {v: k for k, v in vocab.items()} # inverted vocabulary dict: id -> word

a_tree = tiny_sst[0]
for token in a_tree.ndata['x'].tolist():
    if token != trainset.PAD_WORD:
        print(inv_vocab[token], end=" ")

##############################################################################
# Step 1: batching
# ----------------
#
# The first step is to throw all the trees into one graph, using
# ``dgl.batch`` API.
#

import networkx as nx
import matplotlib.pyplot as plt

graph = dgl.batch(tiny_sst)
def plot_tree(g):
    # this plot requires pygraphviz package
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=False, node_size=10,
            node_color=[[.5, .5, .5]], arrowsize=4)
    plt.show()

plot_tree(graph.to_networkx())

##############################################################################
# You can read more about the definition of ``dgl.batch``, or can skip
# ahead to the next step (link):
# 
# .. note::
#    **Definition**: a ``BatchedDGLGraph`` is a ``DGLGraph`` that
#    unions a list of ``DGLGraph``\ s. 
#    
#    - The union includes all the nodes,
#      edges, and their features. The order of nodes, edges and features are
#      preserved. 
#     
#        - Given that we have :math:`V_i` nodes for graph
#          :math:`\mathcal{G}_i`, the node ID :math:`j` in graph
#          :math:`\mathcal{G}_i` correspond to node ID
#          :math:`j + \sum_{k=1}^{i-1} V_k` in the batched graph. 
#    
#        - Therefore, performing feature transformation and message passing on
#          ``BatchedDGLGraph`` is equivalent to doing those on all ``DGLGraph``
#          constituents in parallel. 
#    - Duplicate references to the same graph are
#      treated as deep copies; the nodes, edges, and features are duplicated,
#      and mutation on one reference does not affect the other. 
#    - Currently, ``BatchedDGLGraph`` is immutable in graph structure (i.e. one can't add
#      nodes and edges to it). We need to support mutable batched graphs in
#      (far) future. 
#    - The ``BatchedDGLGraph`` keeps track of the meta
#      information of the constituents so it can be ``unbatch``\ ed to list of
#      ``DGLGraph``\ s.
#
# For more details about the ``BatchedDGLGraph`` module in DGL, please
# read the :doc:`API reference <../../api/python/batch>`.
#
# Step 2: Tree-LSTM Cell with message-passing APIs
# ------------------------------------------------
#
# .. note::
#    The paper proposed two types of Tree LSTM: Child-Sum
#    Tree-LSTMs, and :math:`N`-arr Tree-LSTMs. In this tutorial we focus on
#    the later one. We use PyTorch as our backend framework to set up the
#    network.
#
# In Tree LSTM, each unit at node :math:`j` maintains a hidden
# representation :math:`h_j` and a memory cell :math:`c_j`. The unit
# :math:`j` takes the input vector :math:`x_j` and the hidden
# representations of the their child units: :math:`h_k, k\in C(j)` as
# input, then compute its new hidden representation :math:`h_j` and memory
# cell :math:`c_j` in the following way.
#
# .. math::
#
#    i_j = \sigma\left(W^{(i)}x_j + \sum_{l=1}^{N}U^{(i)}_l h_{jl} + b^{(i)}\right), \\
#    f_{jk} = \sigma\left(W^{(f)}x_j + \sum_{l=1}^{N}U_{kl}^{(f)} h_{jl} + b^{(f)} \right), \\
#    o_j = \sigma\left(W^{(o)}x_j + \sum_{l=1}^{N}U_{l}^{(o)} h_{jl} + b^{(o)} \right), \\
#    u_j = \textrm{tanh}\left(W^{(u)}x_j + \sum_{l=1}^{N} U_l^{(u)}h_{jl} + b^{(u)} \right) , \\
#    c_j = i_j \odot u_j + \sum_{l=1}^{N} f_{jl} \odot c_{jl}, \\
#    h_j = o_j \cdot \textrm{tanh}(c_j), \\
#
# The process can be decomposed into three phases: ``message_func``,
# ``reduce_func`` and ``apply_node_func``.
#
# ``apply_node_func`` is a new node UDF we have not introduced before. In
# ``apply_node_func``, user specifies what to do with node features,
# without considering edge features and messages. In Tree-LSTM case, ``apply_node_func`` is a must, since there exists (leaf) nodes with
# :math:`0` incoming edges, which would not be updated via
# ``reduce_func``.
#

import torch as th
import torch.nn as nn

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size)
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c_f = th.sum(f * nodes.mailbox['c'], 1)
        return {'h_cat': h_cat, 'c_f': c_f}

    def apply_node_func(self, nodes):
        is_leaf = nodes.data['is_leaf']
        # Treat leaf node and non-leaf node differently.
        iou = is_leaf * self.W_iou(nodes.data['x']) + (1 - is_leaf) * self.U_iou(nodes.data['h_cat'])
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c_f']
        h = o * th.tanh(c)
        return {'h' : h, 'c' : c}

##############################################################################
# Step 3: define traversal
# ------------------------
#
# After defining the message passing functions, we then need to induce the
# right order to trigger them. This is a significant departure from models
# such as GCN, where all nodes are pulling messages from upstream ones
# *simultaneously*.
#
# In the case of Tree-LSTM, messages start from leaves of the tree, and
# propogate/processed upwards until they reach the roots. A visulization
# is as follows:
#
# .. figure:: https://i.loli.net/2018/11/09/5be4b5d2df54d.gif
#    :alt:
#
# DGL defines a generator to perform the topological sort, each item is a
# tensor recording the nodes from bottom level to the roots. One can
# appreciate the degree of parallelism by inspecting the difference of the
# followings:
#

print(dgl.topological_nodes_generator(a_tree))
print(dgl.topological_nodes_generator(graph))

##############################################################################
# The ``graph.prop_nodes`` then call triggers the message passing:
#
# .. note::
#    **notice**: Before we call `graph.prop_nodes`, we must specify a `message_func` and `reduce_func` in advance, here we use built-in copy-from-source and sum function as our message function and reduce 
#    function for demonstration.

import dgl.function as fn
import torch as th

graph.ndata['a'] = th.ones(graph.number_of_nodes(), 1)
graph.register_message_func(fn.copy_src('a', 'a'))
graph.register_reduce_func(fn.sum('a', 'a'))

traversal_order = dgl.topological_nodes_generator(graph)
graph.prop_nodes(traversal_order)

# the following is a syntax sugar that does the same
# dgl.prop_nodes_topo(graph)

print(graph.ndata['a'])

##############################################################################
# Putting it together
# -------------------
#
# Here is the complete code that specifies the ``Tree-LSTM`` class:
#

class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, batch, x, h, c, is_leaf):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        x : Tensor
            Initial node input.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        is_leaf: Tensor
            Indicator of whether a node is leaf or not.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g = batch.graph
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        embeds = self.embedding(batch.wordid)
        x = x.index_copy(0, batch.nid_with_word, embeds)
        g.ndata['is_leaf'] = is_leaf
        g.ndata['x'] = x
        g.ndata['h'] = h
        g.ndata['c'] = c
        # init h_cat and c_f for message passing
        g.ndata['h_cat'] = th.zeros_like(h).repeat(1, 2)
        g.ndata['c_f'] = th.zeros_like(c)
        # propagate
        dgl.prop_nodes_topo(g)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        logits = self.linear(h)
        return logits

##############################################################################
# Main Loop
# ---------
#
# Finally, we could write a training paradigm in PyTorch:
#

from torch.utils.data import DataLoader
import torch.nn.functional as F

# hyper parameters
x_size = 256
h_size = 256
dropout = 0.5
lr = 0.05
weight_decay = 1e-4
epochs = 10

# create the model
model = TreeLSTM(trainset.num_vocabs,
                 x_size,
                 h_size,
                 trainset.num_classes,
                 dropout)
print(model)

# create the optimizer
optimizer = th.optim.Adagrad(model.parameters(),
                          lr=lr,
                          weight_decay=weight_decay)
                          
train_loader = DataLoader(dataset=tiny_sst,
                          batch_size=5,
                          collate_fn=data.SST.batcher,
                          shuffle=False,
                          num_workers=0)

# training loop
for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        g = batch.graph
        n = g.number_of_nodes()
        is_leaf = batch.is_leaf
        x = th.zeros((n, x_size))
        h = th.zeros((n, h_size))
        c = th.zeros((n, h_size))
        logits = model(batch, x, h, c, is_leaf)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp, batch.label, reduction='elementwise_mean') 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = th.argmax(logits, 1)
        acc = th.sum(th.eq(batch.label, pred))
        print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} |".format(epoch, step, loss.item(), 1.0*acc.item()/len(batch.label)))

##############################################################################
# To train the model on full dataset with different settings(CPU/GPU,
# etc.), please refer to our repo's
# `example <https://github.com/jermainewang/dgl/tree/master/examples/pytorch/tree_lstm>`__.
