import torch
from torch import nn, einsum
from einops import repeat

import dgl
import dgl.function as fn
from dgl.nn.pytorch.factory import KNNGraph
from dgl.geometry import farthest_point_sampler
from pointnet2 import RelativePositionMessage, PointNetConv

'''
Part of the code are adapted from
https://github.com/lucidrains/point-transformer-pytorch and 
https://github.com/qq456cvb/Point-Transformers
'''

# helpers


def exists(val):
    return val is not None


def max_value(t):
    return torch.finfo(t.dtype).max


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(
        lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *
                     ((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# classes


class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_neighbors
    ):
        super().__init__()
        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(input_dim, input_dim * 3, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x, pos, mask=None):
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # prepare mask
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]

        # expand values
        v = repeat(v, 'b j d -> b i j d', i=n)

        # determine k nearest neighbors for each point, if specified
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim=-1)

            if exists(mask):
                mask_value = max_value(rel_dist)
                rel_dist.masked_fill_(~mask, mask_value)

            dist, indices = rel_dist.topk(num_neighbors, largest=False)

            v = batched_index_select(v, indices, dim=2)
            qk_rel = batched_index_select(qk_rel, indices, dim=2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim=2)
            mask = batched_index_select(
                mask, indices, dim=2) if exists(mask) else None

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # masking
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # attention
        attn = sim.softmax(dim=-2)

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        return agg, attn


class PointTransformerBlock(nn.Module):
    def __init__(self, input_dim, n_neigbors):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.pt = PointTransformerLayer(input_dim, num_neighbors=n_neigbors)
        self.linear2 = nn.Linear(input_dim, input_dim)

    def forward(self, x, pos):
        h = self.linear1(x)
        h, attn = self.pt(h, pos)
        h = self.linear2(h)
        h = h + x
        return h, attn


class TransitionDown(nn.Module):
    """
    The Transition Downn Layer
    """

    def __init__(self, mlp_sizes, batch_size, downsampling_rate, n_neighbors):
        super(TransitionDown, self).__init__()
        self.knn_graph_transformer = KNNGraph(n_neighbors)
        self.message = RelativePositionMessage(n_neighbors)
        self.conv = PointNetConv(mlp_sizes, batch_size)
        self.downsampling_rate = downsampling_rate
        self.batch_size = batch_size

    def forward(self, feat, pos):
        n_points = feat.shape[1]
        centroids = farthest_point_sampler(
            pos, n_points // self.downsampling_rate)
        g = self.knn_graph_transformer(pos)
        g.update_all(self.message, self.conv)

        mask = g.ndata['center'] == 1
        pos_dim = g.ndata['pos'].shape[-1]
        feat_dim = g.ndata['new_feat'].shape[-1]
        pos_res = g.ndata['pos'][mask].view(self.batch_size, -1, pos_dim)
        feat_res = g.ndata['new_feat'][mask].view(
            self.batch_size, -1, feat_dim)
        return pos_res, feat_res


class PointTransformer(nn.Module):
    def __init__(self, batch_size, feature_dim=3, n_blocks=4, downsampling_rate=4, hidden_dim=32, n_neigbors=16):
        super(PointTransformer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ptb = PointTransformerBlock(hidden_dim, n_neigbors)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(n_blocks):
            block_hidden_dim = hidden_dim * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(
                [block_hidden_dim // 2 + 3, block_hidden_dim, block_hidden_dim], batch_size, downsampling_rate=downsampling_rate, n_neighbors=n_neigbors))
            self.transformers.append(
                PointTransformerBlock(hidden_dim, n_neigbors))

    def forward(self, x):
        if x.shape[-1] > 3:
            pos = x[:, :, :3]
            feat = x[:, :, 3:]
        else:
            pos = x
            feat = None
        h = self.fc(feat)
        h, _ = self.ptb(h, pos)

        for td, tf in zip(self.transition_downs, self.transformers):
            h, pos = td(h, pos)
            h, _ = tf(h, pos)

        return h


class PointTransformerCLS(nn.Module):
    def __init__(self, out_classes, batch_size, feature_dim=3, n_blocks=4, downsampling_rate=4, hidden_dim=32, n_neigbors=16):
        super(PointTransformerCLS, self).__init__()
        self.backbone = PointTransformer(
            batch_size, feature_dim, n_blocks, downsampling_rate, hidden_dim, n_neigbors)
        self.out_layer = nn.Linear(hidden_dim * 2 ** (n_blocks), out_classes)

    def forward(self, x):
        h = self.backbone(x)
        out = self.out_layer(h)
        return out
