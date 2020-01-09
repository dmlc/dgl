import dgl
import mxnet as mx
import numpy as np
import logging, time
from operator import itemgetter
from mxnet import nd, gluon
from mxnet.gluon import nn
from dgl.utils import toindex
from dgl.nn.mxnet import GraphConv
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Pad
from gluoncv.utils import makedirs, LRSequential, LRScheduler

from model import faster_rcnn_resnet101_v1d_custom, RelDN
from utils import *
from data import *

filehandler = logging.FileHandler('output.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# Hyperparams
num_gpus = 1
batch_size = num_gpus * 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
N_relations = 50
N_objects = 150
save_dir = 'params'
makedirs(save_dir)
batch_verbose_freq = 100

net = RelDN(n_classes=N_relations, prior_pkl='freq_prior.pkl', semantic_only=True)
net.initialize(ctx=ctx)
'''
net.load_parameters('params/model-9.params', ctx=ctx)
'''

# dataset and dataloader
vg_train = VGRelationCOCO(split='val')
logger.info('data loaded!')
val_data = gluon.data.DataLoader(vg_train, batch_size=batch_size, shuffle=False, num_workers=8*num_gpus,
                                 batchify_fn=dgl_mp_batchify_fn)
n_batches = len(val_data)

detector = faster_rcnn_resnet101_v1d_custom(classes=vg_train.obj_classes,
                                            pretrained_base=False, pretrained=False,
                                            additional_output=True)
params_path = 'faster_rcnn_resnet101_v1d_custom_best.params'
detector.load_parameters(params_path, ctx=ctx, ignore_extra=True, allow_missing=True)

detector_feat = faster_rcnn_resnet101_v1d_custom(classes=vg_train.obj_classes,
                                            pretrained_base=False, pretrained=False,
                                            additional_output=True)
params_path = 'faster_rcnn_resnet101_v1d_custom_best.params'
detector_feat.load_parameters(params_path, ctx=ctx, ignore_extra=True, allow_missing=True)

def get_data_batch(g_list, img_list, ctx_list):
    if g_list is None or len(g_list) == 0:
        return None, None
    n_gpu = len(ctx_list)
    size = len(g_list)
    if size < n_gpu:
        raise Exception("too small batch")
    step = size // n_gpu
    G_list = [g_list[i*step:(i+1)*step] if i < n_gpu - 1 else g_list[i*step:size] for i in range(n_gpu)]
    img_list = [img_list[i*step:(i+1)*step] if i < n_gpu - 1 else img_list[i*step:size] for i in range(n_gpu)]

    for G_slice, ctx in zip(G_list, ctx_list):
        for G in G_slice:
            G.ndata['bbox'] = G.ndata['bbox'].as_in_context(ctx)
            G.ndata['node_class'] = G.ndata['node_class'].as_in_context(ctx)
            G.ndata['node_class_vec'] = G.ndata['node_class_vec'].as_in_context(ctx)
            G.edata['rel_class'] = G.edata['rel_class'].as_in_context(ctx)
    img_list = [img.as_in_context(ctx) for img in img_list]
    return G_list, img_list

mode = ['predcls', 'phrcls']
metric_list = []
topk_list = [20, 50, 100]
if 'predcls' in mode:
    for topk in topk_list:
        metric_list.append(PredCls(topk=topk))
if 'phrcls' in mode:
    for topk in topk_list:
        metric_list.append(PhrCls(topk=topk))
if 'sgdet' in mode:
    for topk in topk_list:
        metric_list.append(SGDet(topk=topk))
if 'sgdet+' in mode:
    for topk in topk_list:
        metric_list.append(SGDetPlus(topk=topk))
for metric in metric_list:
    metric.reset()

for i, (G_list, img_list) in enumerate(val_data):
    G_list, img_list = get_data_batch(G_list, img_list, ctx)
    if G_list is None or img_list is None:
        if (i+1) % batch_verbose_freq == 0:
            print_txt = 'Batch[%d/%d] '%\
                (i, n_batches)
            for metric in metric_list:
                metric_name, metric_val = metric.get()
                print_txt +=  '%s=%.4f '%(metric_name, metric_val)
            logger.info(print_txt)
        continue

    detector_res_list = []
    G_batch = []
    bbox_pad = Pad(axis=(0))
    # loss_cls_val = 0
    for G_slice, img in zip(G_list, img_list):
        cur_ctx = img.context
        bbox_list = [G.ndata['bbox'] for G in G_slice]
        bbox_stack = bbox_pad(bbox_list).as_in_context(cur_ctx)
        bbox, spatial_feat, cls_pred = detector((img, bbox_stack))
        g_batch = build_graph_gt_sample(G_slice, img, bbox,
                                        spatial_feat, cls_pred,# l0_w_slice,
                                        training=True, sample=False)
        rel_bbox = g_batch.edata['rel_bbox']
        batch_id = g_batch.edata['batch_id'].asnumpy()
        n_sample_edges = g_batch.number_of_edges()
        g_batch.edata['edge_feat'] = mx.nd.zeros((n_sample_edges, 49), ctx=cur_ctx)
        n_graph = len(G_slice)
        bbox_rel_list = []
        for j in range(n_graph):
            eids = np.where(batch_id == j)[0]
            if len(eids) > 0:
                bbox_rel_list.append(rel_bbox[eids])
        bbox_rel_stack = bbox_pad(bbox_rel_list).as_in_context(cur_ctx)
        _, spatial_feat_rel, _ = detector_feat((img, bbox_rel_stack))
        spatial_feat_rel_list = []
        for j in range(n_graph):
            eids = np.where(batch_id == j)[0]
            if len(eids) > 0:
                spatial_feat_rel_list.append(spatial_feat_rel[j, 0:len(eids)])
        g_batch.edata['edge_feat'] = nd.concat(*spatial_feat_rel_list, dim=0)

        G_batch.append(g_batch)

    G_batch = [net(G) for G in G_batch]

    for G_slice, G_pred, img_slice in zip(G_list, G_batch, img_list):
        for G_gt, G_pred_one in zip(G_slice, [G_pred]):
            if G_pred_one is None or G_pred_one.number_of_nodes() == 0:
                continue
            gt_objects, gt_triplet = extract_gt(G_pred, img_slice.shape[2:4])
            pred_objects, pred_triplet = extract_pred(G_pred, joint_preds=True)
            for metric in metric_list:
                if isinstance(metric, PredCls) or \
                    isinstance(metric, PhrCls) or \
                    isinstance(metric, SGDet):
                    metric.update(gt_triplet, pred_triplet)
                else:
                    metric.update((gt_objects, gt_triplet), (pred_objects, pred_triplet))
    if (i+1) % batch_verbose_freq == 0:
        print_txt = 'Batch[%d/%d] '%\
            (i, n_batches)
        for metric in metric_list:
            metric_name, metric_val = metric.get()
            print_txt +=  '%s=%.4f '%(metric_name, metric_val)
        logger.info(print_txt)
