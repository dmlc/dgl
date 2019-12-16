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

class EdgeLinkMLP(nn.Block):
    def __init__(self, n_hidden, n_classes):
        super(EdgeLinkMLP, self).__init__()
        self.mlp1 = nn.Dense(n_hidden)
        self.relu = nn.Activation('relu')
        self.mlp2 = nn.Dense(n_classes)

    def forward(self, edges):
        feat = nd.concat(edges.src['node_class_vec'], edges.src['bbox'], edges.dst['node_class_vec'], edges.dst['bbox'])
        out = self.mlp1(feat)
        out = self.mlp2(self.relu(out))
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
        feat = nd.concat(edges.src['node_class_vec'], edges.src['emb'], edges.src['bbox'],
                         edges.dst['node_class_vec'], edges.dst['emb'], edges.dst['bbox'])
        out = self.relu1(self.mlp1(feat))
        out = self.relu2(self.mlp2(out))
        out = self.mlp3(out)
        return {'preds': out}

class EdgeGCN(nn.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
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

    def forward(self, g):
        # link pred
        g.apply_edges(self.edge_link_mlp)
        # subgraph for gconv
        eids = np.where(g.edata['link'].asnumpy() > 0)
        sub_g = g.edge_subgraph(toindex(eids[0].tolist()))
        sub_g.copy_from_parent()
        x = self._box_feat_ext(sub_g.ndata['images'])
        for i, layer in enumerate(self.layers):
            x = layer(sub_g, x)
        sub_g.ndata['emb'] = x
        sub_g.apply_edges(self.edge_mlp)
        sub_g.copy_to_parent()
        return g

filehandler = logging.FileHandler('output.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

ctx = mx.gpu()
nepoch = 15
net = EdgeGCN(in_feats=1024, n_hidden=100, n_classes=50, n_layers=3, activation=nd.relu,
              box_feat_ext='mobilenet1.0', ctx=ctx)
# net.initialize(ctx=ctx)
'''
net._box_feat_ext.hybridize()
net.edge_mlp.initialize(ctx=ctx)
net.edge_link_mlp.initialize(ctx=ctx)
net.layers.initialize(ctx=ctx)
'''
net.load_parameters('params/model-14.params', ctx=ctx, ignore_extra=True)
trainer = gluon.Trainer(net.collect_params(), 'adam', 
                        {'learning_rate': 0.01, 'wd': 0.00001})
for k, v in net._box_feat_ext.collect_params().items():
    v.grad_req = 'null'

@mx.metric.register
@mx.metric.alias('auc')
class AUCMetric(mx.metric.EvalMetric):
    def __init__(self, eps=1e-12):
        super(AUCMetric, self).__init__(
            'auc')
        self.eps = eps

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        label_weight = labels[0].asnumpy()
        preds = preds[0].asnumpy()
        tmp = []
        for i in range(preds.shape[0]):
            tmp.append((label_weight[i], preds[i][1]))
        tmp = sorted(tmp, key=itemgetter(1), reverse=True)
        label_sum = label_weight.sum()
        if label_sum == 0 or label_sum == label_weight.size:
            raise Exception("AUC with one class is undefined")

        label_one_num = np.count_nonzero(label_weight)
        label_zero_num = len(label_weight) - label_one_num
        total_area = label_zero_num * label_one_num
        height = 0
        width = 0
        area = 0
        for a, _ in tmp:
            if a == 1.0:
                height += 1.0
            else:
                width += 1.0
                area += height

        self.sum_metric += area / total_area
        self.num_inst += 1

# L = gluon.loss.SoftmaxCELoss()
# L = gcv.loss.FocalLoss(num_class=2)
L_link = gluon.loss.SoftmaxCELoss()
L_rel = gluon.loss.SoftmaxCELoss()
train_metric = mx.metric.Accuracy()
train_metric_top5 = mx.metric.TopKAccuracy(5)
train_metric_f1 = mx.metric.F1()
train_metric_auc = AUCMetric()

# dataset and dataloader
vg = gcv.data.VGRelation(balancing='weight')

train_data = gluon.data.DataLoader(vg, batch_size=1, shuffle=False, num_workers=0,
                                   batchify_fn=gcv.data.dataloader.dgl_mp_batchify_fn)

save_dir = 'params'
batch_verbose_freq = 10000
for epoch in range(nepoch):
    if epoch > 0:
        break
    loss_val = 0
    tic = time.time()
    btic = time.time()
    train_metric.reset()
    train_metric_top5.reset()
    train_metric_f1.reset()
    train_metric_auc.reset()
    if epoch == 5 or epoch == 10:
        trainer.set_learning_rate(trainer.learning_rate*0.1)
    result = []
    for i, g_list in enumerate(train_data):
        if len(g_list) == 0:
            continue
        if isinstance(g_list, dgl.DGLGraph):
            G = g_list
        else:
            G = dgl.batch(g_list)
        G.ndata['images'] = G.ndata['images'].as_in_context(ctx)
        G.ndata['bbox'] = G.ndata['bbox'].as_in_context(ctx)
        G.ndata['node_class_vec'] = G.ndata['node_class_vec'].as_in_context(ctx)
        G.edata['classes'] = G.edata['classes'].as_in_context(ctx)
        link_ind = np.where(G.edata['link'].asnumpy() == 1)[0]
        G.edata['link'] = G.edata['link'].as_in_context(ctx)
        G.edata['weights'] = G.edata['weights'].expand_dims(1).as_in_context(ctx)

        # debug mode
        G = net(G)
        link_preds = G.edata['link_preds'].asnumpy()
        link_labels = G.edata['link']
        preds = G.edata['preds'].asnumpy()
        label = G.edata['classes'].asnumpy()
        bbox = G.ndata['bbox'].asnumpy()
        image = G.ndata['images'][0].asnumpy()
        img_id = G.ndata['img_id'].asnumpy()
        node_class_ids = G.ndata['node_class_ids'].asnumpy()
        inds = np.flip(np.argsort(link_preds[:,1]))
        node_ids = G.find_edges(inds)
        node_src = node_ids[0].asnumpy()
        node_dst = node_ids[1].asnumpy()
        link_preds = link_preds[inds]
        link_labels = link_labels[inds]
        preds = preds[inds]
        label = label[inds]
        node_class_dict = vg._obj_classes_dict
        edge_class_dict = vg._relations_dict
        result.append((node_src, node_dst, link_preds, link_labels, preds, label,
                       bbox, node_class_ids, node_class_dict, edge_class_dict, image, img_id))
        print(i)
        if i >= 5:
            import pickle
            with open('result.pkl', 'wb') as f:
                pickle.dump(result, f)
            break

