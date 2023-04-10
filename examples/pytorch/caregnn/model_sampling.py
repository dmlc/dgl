import dgl
import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn


def _l1_dist(edges):
    # formula 2
    ed = th.norm(edges.src["nd"] - edges.dst["nd"], 1, 1)
    return {"ed": ed}


class CARESampler(dgl.dataloading.BlockSampler):
    def __init__(self, p, dists, num_layers):
        super().__init__()
        self.p = p
        self.dists = dists
        self.num_layers = num_layers

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        with g.local_scope():
            new_edges_masks = {}
            for etype in g.canonical_etypes:
                edge_mask = th.zeros(g.num_edges(etype))
                # extract each node from dict because of single node type
                for node in seed_nodes:
                    edges = g.in_edges(node, form="eid", etype=etype)
                    num_neigh = (
                        th.ceil(
                            g.in_degrees(node, etype=etype)
                            * self.p[block_id][etype]
                        )
                        .int()
                        .item()
                    )
                    neigh_dist = self.dists[block_id][etype][edges]
                    if neigh_dist.shape[0] > num_neigh:
                        neigh_index = np.argpartition(neigh_dist, num_neigh)[
                            :num_neigh
                        ]
                    else:
                        neigh_index = np.arange(num_neigh)
                    edge_mask[edges[neigh_index]] = 1
                new_edges_masks[etype] = edge_mask.bool()

            return dgl.edge_subgraph(g, new_edges_masks, relabel_nodes=False)

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for block_id in reversed(range(self.num_layers)):
            frontier = self.sample_frontier(block_id, g, seed_nodes)
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

    def __len__(self):
        return self.num_layers


class CAREConv(nn.Module):
    """One layer of CARE-GNN."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_classes,
        edges,
        activation=None,
        step_size=0.02,
    ):
        super(CAREConv, self).__init__()

        self.activation = activation
        self.step_size = step_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.edges = edges

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.MLP = nn.Linear(self.in_dim, self.num_classes)

        self.p = {}
        self.last_avg_dist = {}
        self.f = {}
        # indicate whether the RL converges
        self.cvg = {}
        for etype in edges:
            self.p[etype] = 0.5
            self.last_avg_dist[etype] = 0
            self.f[etype] = []
            self.cvg[etype] = False

    def forward(self, g, feat):
        g.srcdata["h"] = feat

        # formula 8
        hr = {}
        for etype in g.canonical_etypes:
            g.update_all(fn.copy_u("h", "m"), fn.mean("m", "hr"), etype=etype)
            hr[etype] = g.dstdata["hr"]
            if self.activation is not None:
                hr[etype] = self.activation(hr[etype])

        # formula 9 using mean as inter-relation aggregator
        p_tensor = (
            th.Tensor(list(self.p.values())).view(-1, 1, 1).to(feat.device)
        )
        h_homo = th.sum(th.stack(list(hr.values())) * p_tensor, dim=0)
        h_homo += feat[: g.number_of_dst_nodes()]
        if self.activation is not None:
            h_homo = self.activation(h_homo)

        return self.linear(h_homo)


class CAREGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes,
        hid_dim=64,
        edges=None,
        num_layers=2,
        activation=None,
        step_size=0.02,
    ):
        super(CAREGNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.edges = edges
        self.num_layers = num_layers
        self.activation = activation
        self.step_size = step_size

        self.layers = nn.ModuleList()

        if self.num_layers == 1:
            # Single layer
            self.layers.append(
                CAREConv(
                    self.in_dim,
                    self.num_classes,
                    self.num_classes,
                    self.edges,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )

        else:
            # Input layer
            self.layers.append(
                CAREConv(
                    self.in_dim,
                    self.hid_dim,
                    self.num_classes,
                    self.edges,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )

            # Hidden layers with n - 2 layers
            for i in range(self.num_layers - 2):
                self.layers.append(
                    CAREConv(
                        self.hid_dim,
                        self.hid_dim,
                        self.num_classes,
                        self.edges,
                        activation=self.activation,
                        step_size=self.step_size,
                    )
                )

            # Output layer
            self.layers.append(
                CAREConv(
                    self.hid_dim,
                    self.num_classes,
                    self.num_classes,
                    self.edges,
                    activation=self.activation,
                    step_size=self.step_size,
                )
            )

    def forward(self, blocks, feat):
        # formula 4
        sim = th.tanh(self.layers[0].MLP(blocks[-1].dstdata["feature"].float()))

        # Forward of n layers of CARE-GNN
        for block, layer in zip(blocks, self.layers):
            feat = layer(block, feat)
        return feat, sim

    def RLModule(self, graph, epoch, idx, dists):
        for i, layer in enumerate(self.layers):
            for etype in self.edges:
                if not layer.cvg[etype]:
                    # formula 5
                    eid = graph.in_edges(idx, form="eid", etype=etype)
                    avg_dist = th.mean(dists[i][etype][eid])

                    # formula 6
                    if layer.last_avg_dist[etype] < avg_dist:
                        layer.p[etype] -= self.step_size
                        layer.f[etype].append(-1)
                        # avoid overflow, follow the author's implement
                        if layer.p[etype] < 0:
                            layer.p[etype] = 0.001
                    else:
                        layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                        if layer.p[etype] > 1:
                            layer.p[etype] = 0.999
                    layer.last_avg_dist[etype] = avg_dist

                    # formula 7
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True
