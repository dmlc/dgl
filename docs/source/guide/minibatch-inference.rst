.. _guide-minibatch-inference:

6.7 Exact Offline Inference on Large Graphs
------------------------------------------------------

:ref:`(中文版) <guide_cn-minibatch-inference>`

Both subgraph sampling and neighborhood sampling are to reduce the
memory and time consumption for training GNNs with GPUs. When performing
inference it is usually better to truly aggregate over all neighbors
instead to get rid of the randomness introduced by sampling. However,
full-graph forward propagation is usually infeasible on GPU due to
limited memory, and slow on CPU due to slow computation. This section
introduces the methodology of full-graph forward propagation with
limited GPU memory via minibatch and neighborhood sampling.

The inference algorithm is different from the training algorithm, as the
representations of all nodes should be computed layer by layer, starting
from the first layer. Specifically, for a particular layer, we need to
compute the output representations of all nodes from this GNN layer in
minibatches. The consequence is that the inference algorithm will have
an outer loop iterating over the layers, and an inner loop iterating
over the minibatches of nodes. In contrast, the training algorithm has
an outer loop iterating over the minibatches of nodes, and an inner loop
iterating over the layers for both neighborhood sampling and message
passing.

The following animation shows how the computation would look like (note
that for every layer only the first three minibatches are drawn).

.. figure:: https://data.dgl.ai/asset/image/guide_6_6_0.gif
   :alt: Imgur



Implementing Offline Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the two-layer GCN we have mentioned in Section 6.1
:ref:`guide-minibatch-node-classification-model`. The way
to implement offline inference still involves using
:class:`~dgl.graphbolt.NeighborSampler`, but sampling for
only one layer at a time.

.. code:: python

    datapipe = gb.ItemSampler(all_nodes_set, batch_size=1024, shuffle=True)
    datapipe = datapipe.sample_neighbor(g, [-1]) # 1 layers.
    datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)


Note that offline inference is implemented as a method of the GNN module
because the computation on one layer depends on how messages are aggregated
and combined as well.

.. code:: python

    class SAGE(nn.Module):
        def __init__(self, in_size, hidden_size, out_size):
            super().__init__()
            self.layers = nn.ModuleList()
            # Three-layer GraphSAGE-mean.
            self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hidden_size, out_size, "mean"))
            self.dropout = nn.Dropout(0.5)
            self.hidden_size = hidden_size
            self.out_size = out_size

        def forward(self, blocks, x):
            hidden_x = x
            for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
                hidden_x = layer(block, hidden_x)
                is_last_layer = layer_idx == len(self.layers) - 1
                if not is_last_layer:
                    hidden_x = F.relu(hidden_x)
                    hidden_x = self.dropout(hidden_x)
            return hidden_x
    
        def inference(self, graph, features, dataloader, device):
            """
            Offline inference with this module
            """
            feature = features.read("node", None, "feat")

            # Compute representations layer by layer
            for layer_idx, layer in enumerate(self.layers):
                is_last_layer = layer_idx == len(self.layers) - 1

                y = torch.empty(
                    graph.total_num_nodes,
                    self.out_size if is_last_layer else self.hidden_size,
                    dtype=torch.float32,
                    device=buffer_device,
                    pin_memory=pin_memory,
                )
                feature = feature.to(device)

                for step, data in tqdm(enumerate(dataloader)):
                    x = feature[data.input_nodes]
                    hidden_x = layer(data.blocks[0], x)  # len(blocks) = 1
                    if not is_last_layer:
                        hidden_x = F.relu(hidden_x)
                        hidden_x = self.dropout(hidden_x)
                    # By design, our output nodes are contiguous.
                    y[
                        data.seeds[0] : data.seeds[-1] + 1
                    ] = hidden_x.to(device)
                feature = y

            return y


Note that for the purpose of computing evaluation metric on the
validation set for model selection we usually don’t have to compute
exact offline inference. The reason is that we need to compute the
representation for every single node on every single layer, which is
usually very costly especially in the semi-supervised regime with a lot
of unlabeled data. Neighborhood sampling will work fine for model
selection and validation.

One can see
`GraphSAGE <https://github.com/dmlc/dgl/blob/master/examples/graphbolt/node_classification.py>`__
and
`RGCN <https://github.com/dmlc/dgl/blob/master/examples/graphbolt/rgcn/hetero_rgcn.py>`__
for examples of offline inference.
