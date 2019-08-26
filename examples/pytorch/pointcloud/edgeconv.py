import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class EdgeConv(nn.Module):
    def __init__(self, in_features, out_features, batch_norm=False):
        super(EdgeConv, self).__init__()
        self.batch_norm = batch_norm

        self.theta = nn.Linear(in_features, out_features)
        self.phi = nn.Linear(in_features, out_features)

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features)

    def apply_edges(self, edges):
        theta_x = self.theta(edges.dst['x'] - edges.src['x'])
        phi_x = self.phi(edges.src['x'])
        return {'e': theta_x + phi_x}

    def forward(self, g, h):
        with g.local_scope():
            n_samples, n_points, n_dims = h.shape
            g.ndata['x'] = h.view(-1, n_dims)
            if not self.batch_norm:
                g.update_all(self.apply_edges, fn.max('e', 'x'))
            else:
                g.apply_edges(self.apply_edges)
                # Although the official implementation includes a per-edge
                # batch norm within EdgeConv, I choose to replace it with a
                # global batch norm for a number of reasons:
                #
                # (1) When the point clouds within each batch do not have the
                #     same number of points, batch norm would not work.
                #
                # (2) Even if the point clouds always have the same number of
                #     points, the points may as well be shuffled even with the
                #     same (type of) object (and the official implementation
                #     *does* shuffle the points of the same example for each
                #     epoch).
                #
                #     For example, the first point of a point cloud of an
                #     airplane does not always necessarily reside at its nose.
                #
                #     In this case, the learned statistics of each position
                #     by batch norm is not as meaningful as those learned from
                #     images.
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(fn.copy_e('e', 'e'), fn.max('e', 'x'))
            return g.ndata['x'].view(n_samples, n_points, -1)
