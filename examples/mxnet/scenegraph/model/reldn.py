import pickle

import dgl

import gluoncv as gcv
import mxnet as mx
import numpy as np
from dgl.nn.mxnet import GraphConv
from dgl.utils import toindex
from mxnet import nd
from mxnet.gluon import nn

__all__ = ["RelDN"]


class EdgeConfMLP(nn.Block):
    """compute the confidence for edges"""

    def __init__(self):
        super(EdgeConfMLP, self).__init__()

    def forward(self, edges):
        score_pred = nd.log_softmax(edges.data["preds"])[:, 1:].max(axis=1)
        score_phr = (
            score_pred
            + edges.src["node_class_logit"]
            + edges.dst["node_class_logit"]
        )
        return {"score_pred": score_pred, "score_phr": score_phr}


class EdgeBBoxExtend(nn.Block):
    """encode the bounding boxes"""

    def __init__(self):
        super(EdgeBBoxExtend, self).__init__()

    def bbox_delta(self, bbox_a, bbox_b):
        n = bbox_a.shape[0]
        result = nd.zeros((n, 4), ctx=bbox_a.context)
        result[:, 0] = bbox_a[:, 0] - bbox_b[:, 0]
        result[:, 1] = bbox_a[:, 1] - bbox_b[:, 1]
        result[:, 2] = nd.log(
            (bbox_a[:, 2] - bbox_a[:, 0] + 1e-8)
            / (bbox_b[:, 2] - bbox_b[:, 0] + 1e-8)
        )
        result[:, 3] = nd.log(
            (bbox_a[:, 3] - bbox_a[:, 1] + 1e-8)
            / (bbox_b[:, 3] - bbox_b[:, 1] + 1e-8)
        )
        return result

    def forward(self, edges):
        ctx = edges.src["pred_bbox"].context
        n = edges.src["pred_bbox"].shape[0]
        delta_src_obj = self.bbox_delta(
            edges.src["pred_bbox"], edges.dst["pred_bbox"]
        )
        delta_src_rel = self.bbox_delta(
            edges.src["pred_bbox"], edges.data["rel_bbox"]
        )
        delta_rel_obj = self.bbox_delta(
            edges.data["rel_bbox"], edges.dst["pred_bbox"]
        )
        result = nd.zeros((n, 12), ctx=ctx)
        result[:, 0:4] = delta_src_obj
        result[:, 4:8] = delta_src_rel
        result[:, 8:12] = delta_rel_obj
        return {"pred_bbox_additional": result}


class EdgeFreqPrior(nn.Block):
    """make use of the pre-trained frequency prior"""

    def __init__(self, prior_pkl):
        super(EdgeFreqPrior, self).__init__()
        with open(prior_pkl, "rb") as f:
            freq_prior = pickle.load(f)
        self.freq_prior = freq_prior

    def forward(self, edges):
        ctx = edges.src["node_class_pred"].context
        src_ind = edges.src["node_class_pred"].asnumpy().astype(int)
        dst_ind = edges.dst["node_class_pred"].asnumpy().astype(int)
        prob = self.freq_prior[src_ind, dst_ind]
        out = nd.array(prob, ctx=ctx)
        return {"freq_prior": out}


class EdgeSpatial(nn.Block):
    """spatial feature branch"""

    def __init__(self, n_classes):
        super(EdgeSpatial, self).__init__()
        self.mlp = nn.Sequential()
        self.mlp.add(nn.Dense(64))
        self.mlp.add(nn.LeakyReLU(0.1))
        self.mlp.add(nn.Dense(64))
        self.mlp.add(nn.LeakyReLU(0.1))
        self.mlp.add(nn.Dense(n_classes))

    def forward(self, edges):
        feat = nd.concat(
            edges.src["pred_bbox"],
            edges.dst["pred_bbox"],
            edges.data["rel_bbox"],
            edges.data["pred_bbox_additional"],
        )
        out = self.mlp(feat)
        return {"spatial": out}


class EdgeVisual(nn.Block):
    """visual feature branch"""

    def __init__(self, n_classes, vis_feat_dim=7 * 7 * 3):
        super(EdgeVisual, self).__init__()
        self.dim_in = vis_feat_dim
        self.mlp_joint = nn.Sequential()
        self.mlp_joint.add(nn.Dense(vis_feat_dim // 2))
        self.mlp_joint.add(nn.LeakyReLU(0.1))
        self.mlp_joint.add(nn.Dense(vis_feat_dim // 3))
        self.mlp_joint.add(nn.LeakyReLU(0.1))
        self.mlp_joint.add(nn.Dense(n_classes))

        self.mlp_sub = nn.Dense(n_classes)
        self.mlp_ob = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(
            edges.src["node_feat"],
            edges.dst["node_feat"],
            edges.data["edge_feat"],
        )
        out_joint = self.mlp_joint(feat)
        out_sub = self.mlp_sub(edges.src["node_feat"])
        out_ob = self.mlp_ob(edges.dst["node_feat"])
        out = out_joint + out_sub + out_ob
        return {"visual": out}


class RelDN(nn.Block):
    """The RelDN Model"""

    def __init__(self, n_classes, prior_pkl, semantic_only=False):
        super(RelDN, self).__init__()
        # output layers
        self.edge_bbox_extend = EdgeBBoxExtend()
        # semantic through mlp encoding
        if prior_pkl is not None:
            self.freq_prior = EdgeFreqPrior(prior_pkl)

        # with predicate class and a link class
        self.spatial = EdgeSpatial(n_classes + 1)
        # with visual features
        self.visual = EdgeVisual(n_classes + 1)
        self.edge_conf_mlp = EdgeConfMLP()
        self.semantic_only = semantic_only

    def forward(self, g):
        if g is None or g.number_of_nodes() == 0:
            return g
        # predictions
        g.apply_edges(self.freq_prior)
        if self.semantic_only:
            g.edata["preds"] = g.edata["freq_prior"]
        else:
            # bbox extension
            g.apply_edges(self.edge_bbox_extend)
            g.apply_edges(self.spatial)
            g.apply_edges(self.visual)
            g.edata["preds"] = (
                g.edata["freq_prior"] + g.edata["spatial"] + g.edata["visual"]
            )
        # subgraph for gconv
        g.apply_edges(self.edge_conf_mlp)
        return g
