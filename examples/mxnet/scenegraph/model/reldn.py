import dgl
import gluoncv as gcv
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from dgl.utils import toindex
import pickle

from dgl.nn.mxnet import GraphConv

__all__ = ['RelDN']

class EdgeConfMLP(nn.Block):
    def __init__(self):
        super(EdgeConfMLP, self).__init__()

    def forward(self, edges):
        score_pred = nd.softmax(edges.data['preds'])[:,1:].max(axis=1)
        score_phr = score_pred * \
                    edges.src['node_class_prob'].max(axis=1) * \
                    edges.dst['node_class_prob'].max(axis=1)
        return {'score_pred': score_pred,
                'score_phr': score_phr}

class EdgeBBoxExtend(nn.Block):
    def __init__(self):
        super(EdgeBBoxExtend, self).__init__()

    def bbox_delta(self, bbox_a, bbox_b):
        n = bbox_a.shape[0]
        result = nd.zeros((n, 4), ctx=bbox_a.context)
        result[:,0] = bbox_a[:,0] - bbox_b[:,0]
        result[:,1] = bbox_a[:,1] - bbox_b[:,1]
        result[:,2] = nd.log((bbox_a[:,2] - bbox_a[:,0]) / (bbox_b[:,2] - bbox_b[:,0]))
        result[:,3] = nd.log((bbox_a[:,3] - bbox_a[:,1]) / (bbox_b[:,3] - bbox_b[:,1]))
        return result

    def forward(self, edges):
        ctx = edges.src['pred_bbox'].context
        n = edges.src['pred_bbox'].shape[0]
        delta_src_obj = self.bbox_delta(edges.src['pred_bbox'], edges.dst['pred_bbox'])
        delta_src_rel = self.bbox_delta(edges.src['pred_bbox'], edges.data['rel_bbox'])
        delta_rel_obj = self.bbox_delta(edges.data['rel_bbox'], edges.dst['pred_bbox'])
        result = nd.zeros((n, 12), ctx=ctx)
        # result[:,0:5] = edge_bbox
        result[:,0:4] = delta_src_obj
        result[:,4:8] = delta_src_rel
        result[:,8:12] = delta_rel_obj
        return {'pred_bbox_additional': result}

class EdgeSemantic(nn.Block):
    def __init__(self, n_classes, use_prior=False):
        super(EdgeSemantic, self).__init__()
        '''
        self.mlp = nn.Dense(n_classes)
        self.use_prior = use_prior
        '''

    def forward(self, edges):
        '''
        if self.use_prior:
            feat = nd.concat(edges.src['node_class_prob'], edges.dst['node_class_prob'], edges.data['freq_prior'])
        else:
            feat = nd.concat(edges.src['node_class_prob'], edges.dst['node_class_prob'])
        out = self.mlp(feat)
        '''
        out = edges.data['freq_prior']
        return {'semantic': out}

class EdgeFreqPrior(nn.Block):
    def __init__(self, prior_pkl):
        super(EdgeFreqPrior, self).__init__()
        with open(prior_pkl, 'rb') as f:
            freq_prior = pickle.load(f)
        freq_prior[:,:,0] = 1 - freq_prior.sum(axis=2)
        self.freq_prior = freq_prior

    def forward(self, edges):
        ctx = edges.src['node_class'].context
        src_ind = edges.src['node_class'][:,0].asnumpy().astype(int)
        dst_ind = edges.dst['node_class'][:,0].asnumpy().astype(int)
        prob = self.freq_prior[src_ind, dst_ind]
        out = nd.array(prob, ctx=ctx)
        return {'freq_prior': out}

class EdgeSpatial(nn.Block):
    def __init__(self, n_classes):
        super(EdgeSpatial, self).__init__()
        self.mlp = nn.Sequential()
        self.mlp.add(nn.Dense(64))
        self.mlp.add(nn.Activation('relu'))
        self.mlp.add(nn.Dense(64))
        self.mlp.add(nn.Activation('relu'))
        self.mlp.add(nn.Dense(n_classes))

    def forward(self, edges):
        feat = nd.concat(edges.src['pred_bbox'], edges.dst['pred_bbox'], 
                         edges.data['rel_bbox'], edges.data['pred_bbox_additional'])
        out = self.mlp(feat)
        return {'spatial': out}

class EdgeVisual(nn.Block):
    def __init__(self, n_classes):
        super(EdgeVisual, self).__init__()
        self.mlp_joint = nn.Dense(n_classes)
        self.mlp_sub = nn.Dense(n_classes)
        self.mlp_ob = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_feat'], edges.dst['node_feat'], edges.data['edge_feat'])
        out_joint = self.mlp_joint(feat)
        out_sub = self.mlp_sub(edges.src['node_feat'])
        out_ob = self.mlp_ob(edges.dst['node_feat'])
        out = out_joint + out_sub + out_ob
        return {'visual': out}

class RelDN(nn.Block):
    def __init__(self, n_classes, prior_pkl=None, semantic_only=False):
        super(RelDN, self).__init__()
        # output layers
        self.edge_bbox_extend = EdgeBBoxExtend()
        # semantic through mlp encoding
        if prior_pkl is not None:
            self.freq_prior = EdgeFreqPrior(prior_pkl)
            use_prior = True
        else:
            use_prior = False

        self.semantic = EdgeSemantic(n_classes + 1, use_prior)
        self.use_prior = use_prior
        # with predicate class and a link class
        self.spatial = EdgeSpatial(n_classes + 1)
        # with visual features
        self.visual = EdgeVisual(n_classes + 1)
        self.edge_conf_mlp = EdgeConfMLP()
        self.semantic_only = semantic_only

    def forward(self, g): 
        if g is None or g.number_of_nodes() == 0:
            return g
        cls = g.ndata['node_class_pred']
        g.ndata['node_class_prob'] = nd.softmax(cls)
        # bbox extension
        g.apply_edges(self.edge_bbox_extend)
        # predictions
        if self.use_prior:
            g.apply_edges(self.freq_prior)
        g.apply_edges(self.semantic)
        g.apply_edges(self.spatial)
        g.apply_edges(self.visual)
        if self.semantic_only:
            g.edata['preds'] = g.edata['freq_prior']
        else:
            g.edata['preds'] = g.edata['semantic'] + g.edata['spatial'] + g.edata['visual']
        # subgraph for gconv
        g.apply_edges(self.edge_conf_mlp)
        return g
