import torch
from torch import nn

import numpy as np

from helper import square_distance, index_points, TransitionDown

'''
Part of the code are adapted from
https://github.com/qq456cvb/Point-Transformers
'''


class PointTransformerBlock(nn.Module):
    def __init__(self, input_dim, transformer_dim, n_neigbors):
        super(PointTransformerBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, transformer_dim)
        self.fc2 = nn.Linear(transformer_dim, input_dim)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim)
        )
        self.w_qs = nn.Linear(transformer_dim, transformer_dim, bias=False)
        self.w_ks = nn.Linear(transformer_dim, transformer_dim, bias=False)
        self.w_vs = nn.Linear(transformer_dim, transformer_dim, bias=False)
        self.k = n_neigbors

    def forward(self, x, pos):
        dists = square_distance(pos, pos)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_pos = index_points(pos, knn_idx)

        h = self.fc1(x)
        q, k, v = self.w_qs(h), index_points(
            self.w_ks(h), knn_idx), index_points(self.w_vs(h), knn_idx)

        pos_enc = self.fc_delta(pos[:, :, None] - knn_pos)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = torch.softmax(attn / np.sqrt(k.size(-1)),
                             dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + x
        return res, attn


class PointTransformer(nn.Module):
    def __init__(self, n_points, batch_size, feature_dim=3, n_blocks=4, downsampling_rate=4, hidden_dim=32, transformer_dim=512, n_neigbors=16):
        super(PointTransformer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ptb = PointTransformerBlock(
            hidden_dim, transformer_dim, n_neigbors)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(n_blocks):
            block_hidden_dim = hidden_dim * 2 ** (i + 1)
            block_n_points = n_points // (downsampling_rate ** (i + 1))
            self.transition_downs.append(TransitionDown(block_n_points, batch_size, [
                                         block_hidden_dim // 2 + 3, block_hidden_dim, block_hidden_dim], n_neighbors=n_neigbors))
            self.transformers.append(
                PointTransformerBlock(block_hidden_dim, transformer_dim, n_neigbors))

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
            pos, h = td(pos, h)
            h, _ = tf(h, pos)

        return h


class PointTransformerCLS(nn.Module):
    def __init__(self, out_classes, batch_size, feature_dim=3, n_blocks=4, downsampling_rate=4, hidden_dim=32, n_neigbors=16):
        super(PointTransformerCLS, self).__init__()
        self.backbone = PointTransformer(
            1024, batch_size, feature_dim, n_blocks, downsampling_rate, hidden_dim, n_neigbors)
        self.out_layer = nn.Linear(hidden_dim * 2 ** (n_blocks), out_classes)

    def forward(self, x):
        h = self.backbone(x)
        out = self.out_layer(torch.mean(h, dim=1))
        return out


class PointTransformerCLS(nn.Module):
    def __init__(self, out_classes, batch_size, feature_dim=3, n_blocks=4, downsampling_rate=4, hidden_dim=32, n_neigbors=16):
        super(PointTransformerCLS, self).__init__()
        self.backbone = PointTransformer(
            1024, batch_size, feature_dim, n_blocks, downsampling_rate, hidden_dim, n_neigbors)
        self.out_layer = nn.Linear(hidden_dim * 2 ** (n_blocks), out_classes)

    def forward(self, x):
        h = self.backbone(x)
        out = self.out_layer(torch.mean(h, dim=1))
        return out


# class PointTransformerSeg(nn.Module):
#     def __init__(self, out_classes, batch_size, feature_dim=3, n_blocks=4, downsampling_rate=4, hidden_dim=32, n_neigbors=16):
#         super().__init__()
#         self.backbone = PointTransformer(
#             1024, batch_size, feature_dim, n_blocks, downsampling_rate, hidden_dim, n_neigbors)
#         npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
#         self.fc2 = nn.Sequential(
#             nn.Linear(32 * 2 ** nblocks, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 32 * 2 ** nblocks)
#         )
#         self.transformer2 = TransformerBlock(
#             32 * 2 ** nblocks, cfg.model.transformer_dim, nneighbor)
#         self.nblocks = nblocks
#         self.transition_ups = nn.ModuleList()
#         self.transformers = nn.ModuleList()
#         for i in reversed(range(nblocks)):
#             channel = 32 * 2 ** i
#             self.transition_ups.append(
#                 TransitionUp(channel * 2, channel, channel))
#             self.transformers.append(TransformerBlock(
#                 channel, cfg.model.transformer_dim, nneighbor))

#         self.fc3 = nn.Sequential(
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, n_c)
#         )

#     def forward(self, x):
#         points, xyz_and_feats = self.backbone(x)
#         xyz = xyz_and_feats[-1][0]
#         points = self.transformer2(xyz, self.fc2(points))[0]

#         for i in range(self.nblocks):
#             points = self.transition_ups[i](
#                 xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
#             xyz = xyz_and_feats[- i - 2][0]
#             points = self.transformers[i](xyz, points)[0]

#         return self.fc3(points)
