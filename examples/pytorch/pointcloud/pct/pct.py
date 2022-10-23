import torch
from helper import TransitionDown
from torch import nn

"""
Part of the code are adapted from
https://github.com/MenghaoGuo/PCT
"""


class PCTPositionEmbedding(nn.Module):
    def __init__(self, channels=256):
        super(PCTPositionEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv_pos = nn.Conv1d(3, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)

        self.sa1 = SALayerCLS(channels)
        self.sa2 = SALayerCLS(channels)
        self.sa3 = SALayerCLS(channels)
        self.sa4 = SALayerCLS(channels)

        self.relu = nn.ReLU()

    def forward(self, x, xyz):
        # add position embedding
        xyz = xyz.permute(0, 2, 1)
        xyz = self.conv_pos(xyz)

        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N

        x1 = self.sa1(x, xyz)
        x2 = self.sa2(x1, xyz)
        x3 = self.sa3(x2, xyz)
        x4 = self.sa4(x3, xyz)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class SALayerCLS(nn.Module):
    def __init__(self, channels):
        super(SALayerCLS, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, xyz):
        x = x + xyz
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class SALayerSeg(nn.Module):
    def __init__(self, channels):
        super(SALayerSeg, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class PointTransformerCLS(nn.Module):
    def __init__(self, output_channels=40):
        super(PointTransformerCLS, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.g_op0 = TransitionDown(
            in_channels=128, out_channels=128, n_neighbor=32
        )
        self.g_op1 = TransitionDown(
            in_channels=256, out_channels=256, n_neighbor=32
        )

        self.pt_last = PCTPositionEmbedding()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        xyz = x[..., :3]
        x = x[..., 3:].permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = x.permute(0, 2, 1)

        new_xyz, feature_0 = self.g_op0(xyz, x, n_point=512)
        new_xyz, feature_1 = self.g_op1(new_xyz, feature_0, n_point=256)

        # add position embedding on each layer
        x = self.pt_last(feature_1, new_xyz)

        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x, _ = torch.max(x, 2)
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x


class PointTransformerSeg(nn.Module):
    def __init__(self, part_num=50):
        super(PointTransformerSeg, self).__init__()
        self.part_num = part_num
        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SALayerSeg(128)
        self.sa2 = SALayerSeg(128)
        self.sa3 = SALayerSeg(128)
        self.sa4 = SALayerSeg(128)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x, cls_label):
        x = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x_max, _ = torch.max(x, 2)
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_feature = self.label_conv(cls_label).repeat(1, 1, N)
        x_global_feature = torch.cat(
            (x_max_feature, x_avg_feature, cls_label_feature), 1
        )
        x = torch.cat((x, x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        return x


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
