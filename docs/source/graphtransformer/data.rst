Prepare Data
============

In this section, we will prepare the data for the Graphormer model introduced before. We can use any dataset containing :class:`~dgl.DGLGraph` objects and standard PyTorch dataloader to feed the data to the model. The key is to define a collate function to group features of multiple graphs into batches. We show an example of the collate function as follows:


.. code:: python

    def collate(graphs):
        # compute shortest path features, can be done in advance
        for g in graphs:
            spd, path = dgl.shortest_dist(g, root=None, return_paths=True)
            g.ndata["spd"] = spd
            g.ndata["path"] = path

        num_graphs = len(graphs)
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        attn_mask = th.zeros(num_graphs, max_num_nodes, max_num_nodes)
        node_feat = []
        in_degree, out_degree = [], []
        path_data = []
        # Since shortest_dist returns -1 for unreachable node pairs and padded
        # nodes are unreachable to others, distance relevant to padded nodes
        # use -1 padding as well.
        dist = -th.ones(
            (num_graphs, max_num_nodes, max_num_nodes), dtype=th.long
        )

        for i in range(num_graphs):
            # A binary mask where invalid positions are indicated by True.
            # Avoid the case where all positions are invalid.
            attn_mask[i, :, num_nodes[i] + 1 :] = 1

            # +1 to distinguish padded non-existing nodes from real nodes
            node_feat.append(graphs[i].ndata["feat"] + 1)

            # 0 for padding
            in_degree.append(
                th.clamp(graphs[i].in_degrees() + 1, min=0, max=512)
            )
            out_degree.append(
                th.clamp(graphs[i].out_degrees() + 1, min=0, max=512)
            )

            # Path padding to make all paths to the same length "max_len".
            path = graphs[i].ndata["path"]
            path_len = path.size(dim=2)
            # shape of shortest_path: [n, n, max_len]
            max_len = 5
            if path_len >= max_len:
                shortest_path = path[:, :, :max_len]
            else:
                p1d = (0, max_len - path_len)
                # Use the same -1 padding as shortest_dist for
                # invalid edge IDs.
                shortest_path = th.nn.functional.pad(path, p1d, "constant", -1)
            pad_num_nodes = max_num_nodes - num_nodes[i]
            p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
            shortest_path = th.nn.functional.pad(shortest_path, p3d, "constant", -1)
            # +1 to distinguish padded non-existing edges from real edges
            edata = graphs[i].edata["feat"] + 1

            # shortest_dist pads non-existing edges (at the end of shortest
            # paths) with edge IDs -1, and th.zeros(1, edata.shape[1]) stands
            # for all padded edge features.
            edata = th.cat(
                (edata, th.zeros(1, edata.shape[1]).to(edata.device)), dim=0
            )
            path_data.append(edata[shortest_path])

            dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

        # node feat padding
        node_feat = th.nn.utils.rnn.pad_sequence(node_feat, batch_first=True)

        # degree padding
        in_degree = th.nn.utils.rnn.pad_sequence(in_degree, batch_first=True)
        out_degree = th.nn.utils.rnn.pad_sequence(out_degree, batch_first=True)

        return (
            node_feat,
            in_degree,
            out_degree,
            attn_mask,
            th.stack(path_data),
            dist,
        )

In this example, we also omit details like the addition of a virtual node. For more details, please refer to the `Graphormer example <https://github.com/dmlc/dgl/tree/master/examples/core/Graphormer>`_.
