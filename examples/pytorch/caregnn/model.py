import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn


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
        self.dist = {}

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.MLP = nn.Linear(self.in_dim, self.num_classes)

        self.p = {}
        self.last_avg_dist = {}
        self.f = {}
        self.cvg = {}
        for etype in edges:
            self.p[etype] = 0.5
            self.last_avg_dist[etype] = 0
            self.f[etype] = []
            self.cvg[etype] = False

    def _calc_distance(self, edges):
        # formula 2
        d = th.norm(
            th.tanh(self.MLP(edges.src["h"]))
            - th.tanh(self.MLP(edges.dst["h"])),
            1,
            1,
        )
        return {"d": d}

    def _top_p_sampling(self, g, p):
        # this implementation is low efficient
        # optimization requires dgl.sampling.select_top_p requested in issue #3100
        dist = g.edata["d"]
        neigh_list = []
        for node in g.nodes():
            edges = g.in_edges(node, form="eid")
            num_neigh = th.ceil(g.in_degrees(node) * p).int().item()
            neigh_dist = dist[edges]
            if neigh_dist.shape[0] > num_neigh:
                neigh_index = np.argpartition(
                    neigh_dist.cpu().detach(), num_neigh
                )[:num_neigh]
            else:
                neigh_index = np.arange(num_neigh)
            neigh_list.append(edges[neigh_index])
        return th.cat(neigh_list)

    def forward(self, g, feat):
        with g.local_scope():
            g.ndata["h"] = feat

            hr = {}
            for i, etype in enumerate(g.canonical_etypes):
                g.apply_edges(self._calc_distance, etype=etype)
                self.dist[etype] = g.edges[etype].data["d"]
                sampled_edges = self._top_p_sampling(g[etype], self.p[etype])

                # formula 8
                g.send_and_recv(
                    sampled_edges,
                    fn.copy_u("h", "m"),
                    fn.mean("m", "h_%s" % etype[1]),
                    etype=etype,
                )
                hr[etype] = g.ndata["h_%s" % etype[1]]
                if self.activation is not None:
                    hr[etype] = self.activation(hr[etype])

            # formula 9 using mean as inter-relation aggregator
            p_tensor = (
                th.Tensor(list(self.p.values())).view(-1, 1, 1).to(g.device)
            )
            h_homo = th.sum(th.stack(list(hr.values())) * p_tensor, dim=0)
            h_homo += feat
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
        self.activation = activation
        self.step_size = step_size
        self.num_layers = num_layers

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

    def forward(self, graph, feat):
        # For full graph training, directly use the graph
        # formula 4
        sim = th.tanh(self.layers[0].MLP(feat))

        # Forward of n layers of CARE-GNN
        for layer in self.layers:
            feat = layer(graph, feat)

        return feat, sim

    def RLModule(self, graph, epoch, idx):
        for layer in self.layers:
            for etype in self.edges:
                if not layer.cvg[etype]:
                    # formula 5
                    eid = graph.in_edges(idx, form="eid", etype=etype)
                    avg_dist = th.mean(layer.dist[etype][eid])

                    # formula 6
                    if layer.last_avg_dist[etype] < avg_dist:
                        if layer.p[etype] - self.step_size > 0:
                            layer.p[etype] -= self.step_size
                        layer.f[etype].append(-1)
                    else:
                        if layer.p[etype] + self.step_size <= 1:
                            layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                    layer.last_avg_dist[etype] = avg_dist

                    # formula 7
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True
