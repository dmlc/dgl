import dgl
import gluoncv as gcv
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from dgl.utils import toindex
from dgl.nn.mxnet import GraphConv

__all__ = ['EdgeGCN']

class EdgeLinkMLP(nn.Block):
    def __init__(self, n_hidden, n_classes):
        super(EdgeLinkMLP, self).__init__()
        self.mlp1 = nn.Dense(n_hidden)
        self.relu1 = nn.Activation('relu')
        self.mlp2 = nn.Dense(n_hidden)
        self.relu2 = nn.Activation('relu')
        self.mlp3 = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_class_prob'], edges.src['pred_bbox'],
                         edges.dst['node_class_prob'], edges.dst['pred_bbox'], 
                         edges.data['pred_bbox_additional'])
        out = self.relu1(self.mlp1(feat))
        out = self.relu2(self.mlp2(out))
        out = self.mlp3(out)
        return {'link_preds': out}

class EdgeMLP(nn.Block):
    def __init__(self, n_hidden, n_classes):
        super(EdgeMLP, self).__init__()
        self.mlp1 = nn.Dense(n_hidden)
        self.relu1 = nn.Activation('relu')
        self.mlp2 = nn.Dense(n_hidden)
        self.relu2 = nn.Activation('relu')
        self.mlp3 = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_class_prob'], edges.src['emb'], edges.src['pred_bbox'],
                         edges.dst['node_class_prob'], edges.dst['emb'], edges.dst['pred_bbox'],
                         edges.data['pred_bbox_additional'])
        out = self.relu1(self.mlp1(feat))
        out = self.relu2(self.mlp2(out))
        out = self.mlp3(out)
        return {'preds': out}

class EdgeConfMLP(nn.Block):
    def __init__(self):
        super(EdgeConfMLP, self).__init__()

    def forward(self, edges):
        score_pred = nd.softmax(edges.data['link_preds'])[:,1] * \
                     nd.softmax(edges.data['preds']).max(axis=1)
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
        edge_bbox = nd.zeros((n, 5), ctx=ctx)
        stack_bbox = nd.stack(edges.src['pred_bbox'], edges.dst['pred_bbox'])
        edge_bbox[:,0] = nd.stack(edges.src['pred_bbox'][:,0], edges.dst['pred_bbox'][:,0]).min(axis=0)
        edge_bbox[:,1] = nd.stack(edges.src['pred_bbox'][:,1], edges.dst['pred_bbox'][:,1]).min(axis=0)
        edge_bbox[:,2] = nd.stack(edges.src['pred_bbox'][:,2], edges.dst['pred_bbox'][:,2]).min(axis=0)
        edge_bbox[:,3] = nd.stack(edges.src['pred_bbox'][:,3], edges.dst['pred_bbox'][:,3]).min(axis=0)
        edge_bbox[:,4] = (edge_bbox[:,2] - edge_bbox[:,0]) * (edge_bbox[:,3] - edge_bbox[:,1])
        delta_src_obj = self.bbox_delta(edges.src['pred_bbox'], edges.dst['pred_bbox'])
        delta_src_rel = self.bbox_delta(edges.src['pred_bbox'], edge_bbox)
        delta_rel_obj = self.bbox_delta(edge_bbox, edges.dst['pred_bbox'])
        result = nd.zeros((n, 17), ctx=ctx)
        result[:,0:5] = edge_bbox
        result[:,5:9] = delta_src_obj
        result[:,9:13] = delta_src_rel
        result[:,13:17] = delta_rel_obj
        return {'pred_bbox_additional': result}

class EdgeGCN(nn.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 pretrained_base,
                 ctx):
        super(EdgeGCN, self).__init__()
        self.layers = nn.Sequential()
        # input layer
        self.layers.add(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.edge_link_mlp = EdgeLinkMLP(50, 2)
        self.edge_mlp = EdgeMLP(100, n_classes)
        self.edge_conf_mlp = EdgeConfMLP()
        self.edge_bbox_extend = EdgeBBoxExtend()

    def forward(self, g):
        if g is None or g.number_of_nodes() == 0:
            return g
        cls = g.ndata['node_class_pred']
        g.ndata['node_class_prob'] = nd.softmax(cls)
        # bbox extension
        g.apply_edges(self.edge_bbox_extend)
        # link pred
        g.apply_edges(self.edge_link_mlp)
        # subgraph for gconv
        if mx.autograd.is_training():
            eids = np.where(g.edata['link'].asnumpy() > 0)[0]
            sub_g = g.edge_subgraph(toindex(eids.tolist()))
            sub_g.copy_from_parent()
            # graph conv
            x = sub_g.ndata['node_feat']
            for i, layer in enumerate(self.layers):
                x = layer(sub_g, x)
            sub_g.ndata['emb'] = x
            sub_g.copy_to_parent()
            # link classification
            g.apply_edges(self.edge_mlp)
        else:
            # graph conv
            x = g.ndata['node_feat']
            for i, layer in enumerate(self.layers):
                x = layer(g, x)
            g.ndata['emb'] = x
            # link classification
            g.apply_edges(self.edge_mlp)
        g.apply_edges(self.edge_conf_mlp)
        return g
