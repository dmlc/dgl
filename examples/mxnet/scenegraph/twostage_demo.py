import dgl
import gluoncv as gcv
import mxnet as mx
import numpy as np
import logging, time
from operator import itemgetter
from mxnet import nd, gluon
from mxnet.gluon import nn
from dgl.utils import toindex
from dgl.nn.mxnet import GraphConv
from gluoncv.model_zoo import get_model
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import crop_resize_normalize

class SoftmaxHD(nn.HybridBlock):
    """Softmax on multiple dimensions
    Parameters
    ----------
    axis : the axis for softmax normalization
    """
    def __init__(self, axis=(2, 3), **kwargs):
        super(SoftmaxHD, self).__init__(**kwargs)
        self.axis = axis

    def hybrid_forward(self, F, x):
        x_max = F.max(x, axis=self.axis, keepdims=True)
        x_exp = F.exp(F.broadcast_minus(x, x_max))
        norm = F.sum(x_exp, axis=self.axis, keepdims=True)
        res = F.broadcast_div(x_exp, norm)
        return res

class EdgeLinkMLP(nn.Block):
    def __init__(self, n_hidden, n_classes):
        super(EdgeLinkMLP, self).__init__()
        self.mlp1 = nn.Dense(n_hidden)
        self.relu = nn.Activation('relu')
        self.mlp2 = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_class_prob'], edges.src['bbox'],
                         edges.dst['node_class_prob'], edges.dst['bbox'])
        out = self.relu(self.mlp1(feat))
        out = self.mlp2(out)
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
        feat = nd.concat(edges.src['node_class_prob'], edges.src['emb'], edges.src['bbox'],
                         edges.dst['node_class_prob'], edges.dst['emb'], edges.dst['bbox'])
        out = self.relu1(self.mlp1(feat))
        out = self.relu2(self.mlp2(out))
        out = self.mlp3(out)
        return {'preds': out}

class EdgeGCN(nn.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_obj_classes,
                 n_layers,
                 activation,
                 box_feat_ext,
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
        self._box_feat_ext = get_model(box_feat_ext, pretrained=True, ctx=ctx).features
        self._box_cls = nn.Dense(n_obj_classes)
        self._softmax = SoftmaxHD(axis=(1))

    def forward(self, g, topk=20):
        # extract node visual feature
        x = self._box_feat_ext(g.ndata['images'])
        g.ndata['node_feat'] = x 
        cls = self._box_cls(x)
        g.ndata['node_class_pred'] = cls
        g.ndata['node_class_prob'] = self._softmax(cls)
        # link pred
        g.apply_edges(self.edge_link_mlp)
        '''
        # subgraph for gconv
        tmp = self._softmax(g.edata['link_preds'])[:,1].asnumpy()
        eids = tmp.argsort()[0:topk]
        sub_g = g.edge_subgraph(toindex(eids[0].tolist()))
        sub_g.copy_from_parent()
        # graph conv
        x = sub_g.ndata['node_feat']
        for i, layer in enumerate(self.layers):
            x = layer(sub_g, x)
        sub_g.ndata['emb'] = x
        # link classification
        sub_g.apply_edges(self.edge_mlp)
        sub_g.copy_to_parent()
        '''
        x = g.ndata['node_feat']
        for i, layer in enumerate(self.layers):
            x = layer(g, x)
        g.ndata['emb'] = x
        # link classification
        g.apply_edges(self.edge_mlp)
        return g

def _build_complete_graph(bbox, scores, img, thr=0.5):
    bbox_list = []
    bbox_np_list = []
    for i in range(bbox.shape[0]):
        if scores[i] > thr:
            bbox_list.append(bbox[i])
            bbox_np_list.append(bbox[i].asnumpy())

    N = len(bbox_list)
    bbox = mx.nd.stack(*bbox_list)
    g = dgl.DGLGraph()
    g.add_nodes(N)
    edge_list = []
    for i in range(N-1):
        for j in range(i+1, N):
            edge_list.append((i, j))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)

    img_list = crop_resize_normalize(img, bbox_np_list, (224, 224))
    node_imgs = mx.nd.stack(*img_list)

    img_shape = x.shape
    bbox[:,0] /= img_shape[1]
    bbox[:,1] /= img_shape[0]
    bbox[:,2] /= img_shape[1]
    bbox[:,3] /= img_shape[0]
    g.ndata['bbox'] = bbox
    g.ndata['images'] = node_imgs
    return g

# Hyperparams
ctx = mx.cpu()
N_relations = 50
N_objects = 150

# vg = gcv.data.VGRelation(top_frequent_rel=N_relations, top_frequent_obj=N_objects,
#                          balancing='weight')

# network
net = EdgeGCN(in_feats=1024, n_hidden=100, n_classes=N_relations, n_obj_classes=N_objects,
              n_layers=3, activation=nd.relu,
              box_feat_ext='mobilenet1.0', ctx=ctx)
net.load_parameters('params/model-24.params')
vg_obj_classes = ['airplane', 'animal', 'arm', 'back', 'background', 'bag', 'ball', 'banana', 'band', 'base', 'bench', 'bicycle', 'bird', 'board', 'boat', 'book', 'bottle', 'bowl', 'box', 'branch', 'brick', 'building', 'bus', 'button', 'cabinet', 'cap', 'car', 'cat', 'ceiling', 'chair', 'child', 'clock', 'cloud', 'coat', 'container', 'contemplation', 'counter', 'cow', 'cup', 'curtain', 'design', 'dog', 'door', 'ear', 'edge', 'elephant', 'eye', 'face', 'fence', 'field', 'finger', 'flag', 'floor', 'flower', 'food', 'foot', 'frame', 'giraffe', 'girl', 'glass', 'glove', 'grass', 'hair', 'hand', 'handle', 'hat', 'head', 'headlight', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kite', 'lamp', 'land', 'leaf', 'leg', 'letter', 'light', 'line', 'logo', 'male_child', 'man', 'mirror', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'numeral', 'paper', 'part', 'path', 'people', 'person', 'picture', 'pillow', 'plant', 'plate', 'pole', 'post', 'railing', 'road', 'rock', 'roof', 'sand', 'screen', 'seat', 'shadow', 'sheep', 'shelf', 'shirt', 'shoe', 'short_pants', 'shrub', 'sidewalk', 'sign', 'skateboard', 'ski', 'sky', 'snow', 'sock', 'soil', 'spectacles', 'street', 'sunglasses', 'table', 'tail', 'tile', 'tire', 'topographic_point', 'train', 'tree', 'trouser', 'truck', 'trunk', 'umbrella', 'wall', 'water', 'wave', 'wheel', 'window', 'wing', 'wire', 'woman', 'word', 'writing', 'zebra']
vg_rel_classes = ['about', 'above', 'across', 'along', 'approximately', 'arrive', 'attach', 'away', 'back', 'be', 'behind', 'belong_to', 'below', 'between', 'by', 'depart', 'depend_on', 'down', 'eat', 'fly', 'hang', 'have', 'in', 'incorporate', 'inside', 'leave', 'lie', 'look', 'make', 'next', 'outside', 'over', 'play', 'put', 'run', 'show', 'sit', 'stand', 'state', 'stay', 'swing', 'tend', 'transport', 'traverse', 'turn', 'under', 'use', 'walk', 'watch', 'wear']
detector = get_model('yolo3_mobilenet1.0_custom', classes=vg_obj_classes)
params_path = '/home/ubuntu/gluon-cv/scripts/detection/visualgenome/' + \
              'yolo3_mobilenet1.0_custom_0190_0.0000.params'
detector.load_parameters(params_path)

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/pose/soccer.png?raw=true', path='soccer.png')
x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)
class_IDs, scores, bounding_boxs = detector(x)

g = _build_complete_graph(bounding_boxs[0], scores[0], img)

g = net(g)
import pdb; pdb.set_trace()
softmax_net = SoftmaxHD(axis=(1))
link_probs = softmax_net(g.edata['link_preds'])[:,1].asnumpy()
thresh = 0.2
eids = np.where(link_probs > thresh)[0]
rel_probs = softmax_net(g.edata['preds']).asnumpy()
obj_probs = g.ndata['node_class_prob'].asnumpy()

node_ids = g.find_edges(eids)
node_src = node_ids[0].asnumpy()
node_dst = node_ids[1].asnumpy()
for i, eid in enumerate(eids):
    rel_prob = rel_probs[eid]
    rel_prob_ind = rel_prob.argsort()[::-1][0]

    sub_prob = obj_probs[node_src[i]]
    sub_prob_ind = sub_prob.argsort()[::-1][0]
    obj_prob = obj_probs[node_dst[i]]
    obj_prob_ind = obj_prob.argsort()[::-1][0]
    print("<%s %s %s>"%(vg_obj_classes[sub_prob_ind],
                        vg_rel_classes[rel_prob_ind],
                        vg_obj_classes[obj_prob_ind]))
