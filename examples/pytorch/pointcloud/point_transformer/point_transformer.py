import numpy as np
import torch
from helper import index_points, square_distance, TransitionDown, TransitionUp
from torch import nn

"""
Part of the code are adapted from
https://github.com/qq456cvb/Point-Transformers
"""


class PointTransformerBlock(nn.Module):
    def __init__(self, input_dim, n_neighbors, transformer_dim=None):
        super(PointTransformerBlock, self).__init__()
        if transformer_dim is None:
            transformer_dim = input_dim
        self.fc1 = nn.Linear(input_dim, transformer_dim)
        self.fc2 = nn.Linear(transformer_dim, input_dim)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
        )
        self.w_qs = nn.Linear(transformer_dim, transformer_dim, bias=False)
        self.w_ks = nn.Linear(transformer_dim, transformer_dim, bias=False)
        self.w_vs = nn.Linear(transformer_dim, transformer_dim, bias=False)
        self.n_neighbors = n_neighbors

    def forward(self, x, pos):
        dists = square_distance(pos, pos)
        knn_idx = dists.argsort()[:, :, : self.n_neighbors]  # b x n x k
        knn_pos = index_points(pos, knn_idx)

        h = self.fc1(x)
        q, k, v = (
            self.w_qs(h),
            index_points(self.w_ks(h), knn_idx),
            index_points(self.w_vs(h), knn_idx),
        )

        pos_enc = self.fc_delta(pos[:, :, None] - knn_pos)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = torch.softmax(
            attn / np.sqrt(k.size(-1)), dim=-2
        )  # b x n x k x f

        res = torch.einsum("bmnf,bmnf->bmf", attn, v + pos_enc)
        res = self.fc2(res) + x
        return res, attn


class PointTransformer(nn.Module):
    def __init__(
        self,
        n_points,
        batch_size,
        feature_dim=3,
        n_blocks=4,
        downsampling_rate=4,
        hidden_dim=32,
        transformer_dim=None,
        n_neighbors=16,
    ):
        super(PointTransformer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ptb = PointTransformerBlock(
            hidden_dim, n_neighbors, transformer_dim
        )
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(n_blocks):
            block_hidden_dim = hidden_dim * 2 ** (i + 1)
            block_n_points = n_points // (downsampling_rate ** (i + 1))
            self.transition_downs.append(
                TransitionDown(
                    block_n_points,
                    batch_size,
                    [
                        block_hidden_dim // 2 + 3,
                        block_hidden_dim,
                        block_hidden_dim,
                    ],
                    n_neighbors=n_neighbors,
                )
            )
            self.transformers.append(
                PointTransformerBlock(
                    block_hidden_dim, n_neighbors, transformer_dim
                )
            )

    def forward(self, x):
        if x.shape[-1] > 3:
            pos = x[:, :, :3]
        else:
            pos = x

        feat = x
        h = self.fc(feat)
        h, _ = self.ptb(h, pos)

        hidden_state = [(pos, h)]
        for td, tf in zip(self.transition_downs, self.transformers):
            pos, h = td(pos, h)
            h, _ = tf(h, pos)
            hidden_state.append((pos, h))

        return h, hidden_state


class PointTransformerCLS(nn.Module):
    def __init__(
        self,
        out_classes,
        batch_size,
        n_points=1024,
        feature_dim=3,
        n_blocks=4,
        downsampling_rate=4,
        hidden_dim=32,
        transformer_dim=None,
        n_neighbors=16,
    ):
        super(PointTransformerCLS, self).__init__()
        self.backbone = PointTransformer(
            n_points,
            batch_size,
            feature_dim,
            n_blocks,
            downsampling_rate,
            hidden_dim,
            transformer_dim,
            n_neighbors,
        )
        self.out = self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim * 2 ** (n_blocks), 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, out_classes),
        )

    def forward(self, x):
        h, _ = self.backbone(x)
        out = self.out(torch.mean(h, dim=1))
        return out


class PointTransformerSeg(nn.Module):
    def __init__(
        self,
        out_classes,
        batch_size,
        n_points=2048,
        feature_dim=3,
        n_blocks=4,
        downsampling_rate=4,
        hidden_dim=32,
        transformer_dim=None,
        n_neighbors=16,
    ):
        super().__init__()
        self.backbone = PointTransformer(
            n_points,
            batch_size,
            feature_dim,
            n_blocks,
            downsampling_rate,
            hidden_dim,
            transformer_dim,
            n_neighbors,
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 2**n_blocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2**n_blocks),
        )
        self.ptb = PointTransformerBlock(
            32 * 2**n_blocks, n_neighbors, transformer_dim
        )

        self.n_blocks = n_blocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(n_blocks)):
            block_hidden_dim = 32 * 2**i
            self.transition_ups.append(
                TransitionUp(
                    block_hidden_dim * 2, block_hidden_dim, block_hidden_dim
                )
            )
            self.transformers.append(
                PointTransformerBlock(
                    block_hidden_dim, n_neighbors, transformer_dim
                )
            )

        self.out = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_classes),
        )

    def forward(self, x, cat_vec=None):
        _, hidden_state = self.backbone(x)
        pos, h = hidden_state[-1]
        h, _ = self.ptb(self.fc(h), pos)

        for i in range(self.n_blocks):
            h = self.transition_ups[i](
                pos, h, hidden_state[-i - 2][0], hidden_state[-i - 2][1]
            )
            pos = hidden_state[-i - 2][0]
            h, _ = self.transformers[i](h, pos)
        return self.out(torch.cat([h, cat_vec], dim=-1))


class PartSegLoss(nn.Module):
    def __init__(self, eps=0.2):
        super(PartSegLoss, self).__init__()
        self.eps = eps
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, y):
        num_classes = logits.shape[1]
        logits = logits.permute(0, 2, 1).contiguous().view(-1, num_classes)
        loss = self.loss(logits, y)
        return loss
