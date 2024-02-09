Build Model
===========

**GraphTransformer** is a graph neural network that uses multi-head self-attention (sparse or dense) to encode the graph structure and node features. It is a generalization of the `Transformer <https://arxiv.org/abs/1706.03762>`_ architecture to arbitrary graphs. 

In this tutorial, we will show how to build a graph transformer model with DGL using the `Graphormer <https://arxiv.org/abs/2106.05234>`_ model as an example.

Graphormer is a Transformer model designed for graph-structured data, which encodes the structural information of a graph into the standard Transformer. Specifically, Graphormer utilizes degree encoding to measure the importance of nodes, spatial and path Encoding to measure the relation between node pairs. The degree encoding and the node features serve as input to Graphormer, while the spatial and path encoding act as bias terms in the self-attention module.

Degree Encoding
-------------------
The degree encoder is a learnable embedding layer that encodes the degree of each node into a vector. It takes as input the batched input and output degrees of graph nodes, and outputs the degree embeddings of the nodes.

.. code:: python

    degree_encoder = dgl.nn.DegreeEncoder(
        max_degree=8,  # the maximum degree to cut off
        embedding_dim=512  # the dimension of the degree embedding
    )

Path Encoding
-------------
The path encoder encodes the edge features on the shortest path between two nodes to get attention bias for the self-attention module. It takes as input the batched edge features in shape  and outputs the attention bias based on path encoding.

.. code:: python

    path_encoder = PathEncoder(
        max_len=5,  # the maximum length of the shortest path
        feat_dim=512,  # the dimension of the edge feature
        num_heads=8,  # the number of attention heads
    )

Spatial Encoding
----------------
The spatial encoder encodes the shortest distance between two nodes to get attention bias for the self-attention module. It takes as input the shortest distance between two nodes and outputs the attention bias based on spatial encoding.

.. code:: python

    spatial_encoder = SpatialEncoder(
        max_dist=5,  # the maximum distance between two nodes
        num_heads=8,  # the number of attention heads
    )


Graphormer Layer
----------------
The Graphormer layer is like a Transformer encoder layer with the Multi-head Attention part replaced with :class:`~dgl.nn.BiasedMHA`. It takes in not only the input node features, but also the attention bias computed computed above, and outputs the updated node features.

We can stack multiple Graphormer layers as a list just like implementing a Transformer encoder in PyTorch.

.. code:: python

    layers = th.nn.ModuleList([
        GraphormerLayer(
            feat_size=512,  # the dimension of the input node features
            hidden_size=1024,  # the dimension of the hidden layer
            num_heads=8,  # the number of attention heads
            dropout=0.1,  # the dropout rate
            activation=th.nn.ReLU(),  # the activation function
            norm_first=False,  # whether to put the normalization before attention and feedforward
        )
        for _ in range(6)
    ])

Model Forward
-------------
Grouping the modules above defines the primary components of the Graphormer model. We then can define the forward process as follows:

.. code:: python

    node_feat, in_degree, out_degree, attn_mask, path_data, dist = \
        next(iter(dataloader))  #  we will use the first batch as an example
    num_graphs, max_num_nodes, _ = node_feat.shape
    deg_emb = degree_encoder(th.stack((in_degree, out_degree)))

    # node feature + degree encoding as input
    node_feat = node_feat + deg_emb

    # spatial encoding and path encoding serve as attention bias
    path_encoding = path_encoder(dist, path_data)
    spatial_encoding = spatial_encoder(dist)
    attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

    # graphormer layers
    for layer in layers:
        x = layer(
            x,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
        )

For simplicity, we omit some details in the forward process. For the complete implementation, please refer to the `Graphormer example <https://github.com/dmlc/dgl/tree/master/examples/core/Graphormer>`_.

You can also explore other `utility modules <https://docs.dgl.ai/api/python/nn-pytorch.html#utility-modules-for-graph-transformer>`_ to customize your own graph transformer model. In the next section, we will show how to prepare the data for training.
