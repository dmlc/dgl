.. _guide-minibatch-sparse:

6.5 Training GNN with DGL sparse
---------------------------------

This tutorial demonstrates how to use dgl sparse library to sample on graph and
train model. It trains and tests a GraphSAGE model using the sparse sample and
compact operators to sample submatrix from the whole matrix.

Training GNN with DGL sparse is quite similar to
:ref:`guide-minibatch-node-classification-sampler`. The major difference is
the customized sampler and matrix that represents graph.

We have cutomized one sampler in
:ref:`guide-minibatch-customizing-neighborhood-sampler`. In this tutorial, we
will customize another sampler with DGL sparse library as shown below.

.. code:: python

    @functional_datapipe("sample_sparse_neighbor")
    class SparseNeighborSampler(SubgraphSampler):
        def __init__(self, datapipe, matrix, fanouts):
            super().__init__(datapipe)
            self.matrix = matrix
            # Convert fanouts to a list of tensors.
            self.fanouts = []
            for fanout in fanouts:
                if not isinstance(fanout, torch.Tensor):
                    fanout = torch.LongTensor([int(fanout)])
                self.fanouts.insert(0, fanout)

        def sample_subgraphs(self, seeds):
            sampled_matrices = []
            src = seeds

            #####################################################################
            # (HIGHLIGHT) Using the sparse sample operator to preform random
            # sampling on the neighboring nodes of the seeds nodes. The sparse
            # compact operator is then employed to compact and relabel the sampled
            # matrix, resulting in the sampled matrix and the relabel index.
            #####################################################################
            for fanout in self.fanouts:
                # Sample neighbors.
                sampled_matrix = self.matrix.sample(1, fanout, ids=src).coalesce()
                # Compact the sampled matrix.
                compacted_mat, row_ids = sampled_matrix.compact(0)
                sampled_matrices.insert(0, compacted_mat)
                src = row_ids

            return src, sampled_matrices

Another major difference is the matrix that represents graph. Previously we use
:class:`~dgl.graphbolt.FusedCSCSamplingGraph` for sampling. In this tutorial,
we use :class:`~dgl.sparse.SparseMatrix` to represent graph.

.. code:: python

    dataset = gb.BuiltinDataset("ogbn-products").load()
    g = dataset.graph
    # Create sparse.
    N = g.num_nodes
    A = dglsp.from_csc(g.csc_indptr, g.indices, shape=(N, N))


The remaining code is almost same as node classification tutorial.

To use this sampler with :class:`~dgl.graphbolt.DataLoader`:

.. code:: python

    datapipe = gb.ItemSampler(ids, batch_size=1024)
    # Customize graphbolt sampler by sparse.
    datapipe = datapipe.sample_sparse_neighbor(A, fanouts)
    # Use grapbolt to fetch features.
    datapipe = datapipe.fetch_feature(features, node_feature_keys=["feat"])
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)

Model definition is shown below:

.. code:: python

    class SAGEConv(nn.Module):
        r"""GraphSAGE layer from `Inductive Representation Learning on
        Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__
        """

        def __init__(
            self,
            in_feats,
            out_feats,
        ):
            super(SAGEConv, self).__init__()
            self._in_src_feats, self._in_dst_feats = in_feats, in_feats
            self._out_feats = out_feats

            self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=True)
            self.reset_parameters()

        def reset_parameters(self):
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

        def forward(self, A, feat):
            feat_src = feat
            feat_dst = feat[: A.shape[1]]

            # Aggregator type: mean.
            srcdata = self.fc_neigh(feat_src)
            # Divided by degree.
            D_hat = dglsp.diag(A.sum(0)) ** -1
            A_div = A @ D_hat
            # Conv neighbors.
            dstdata = A_div.T @ srcdata

            rst = self.fc_self(feat_dst) + dstdata
            return rst


    class SAGE(nn.Module):
        def __init__(self, in_size, hid_size, out_size):
            super().__init__()
            self.layers = nn.ModuleList()
            # Three-layer GraphSAGE-gcn.
            self.layers.append(SAGEConv(in_size, hid_size))
            self.layers.append(SAGEConv(hid_size, hid_size))
            self.layers.append(SAGEConv(hid_size, out_size))
            self.dropout = nn.Dropout(0.5)
            self.hid_size = hid_size
            self.out_size = out_size

        def forward(self, sampled_matrices, x):
            hidden_x = x
            for layer_idx, (layer, sampled_matrix) in enumerate(
                zip(self.layers, sampled_matrices)
            ):
                hidden_x = layer(sampled_matrix, hidden_x)
                if layer_idx != len(self.layers) - 1:
                    hidden_x = F.relu(hidden_x)
                    hidden_x = self.dropout(hidden_x)
            return hidden_x


Launch training:

.. code:: python

    features = dataset.feature
    # Create GraphSAGE model.
    in_size = features.size("node", None, "feat")[0]
    num_classes = dataset.tasks[0].metadata["num_classes"]
    out_size = num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, data in enumerate(dataloader):
            node_feature = data.node_features["feat"].float()
            blocks = data.sampled_subgraphs
            y = data.labels
            y_hat = model(blocks, node_feature)
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

For more details, please refer to the `full example
<https://github.com/dmlc/dgl/blob/master/examples/graphbolt/sparse/graphsage.py>`__.
