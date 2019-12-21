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

from data import *
from utils import *
from model import EdgeGCN, faster_rcnn_resnet101_v1d_custom

num_gpus = 1
batch_size = num_gpus * 16
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
N_relations = 50
N_objects = 150

filehandler = logging.FileHandler('output.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

net = EdgeGCN(in_feats=49, n_hidden=32, n_classes=N_relations,
              n_layers=2, activation=nd.relu, pretrained_base=False, ctx=ctx)
net.load_parameters('params/model-9.params', ctx=ctx)

vg_val = VGRelation(top_frequent_rel=N_relations, top_frequent_obj=N_objects,
                    balancing='weight', split='val')
logger.info('data loaded!')
val_data = gluon.data.DataLoader(vg_val, batch_size=1, shuffle=False, num_workers=60,
                                 batchify_fn=dgl_mp_batchify_fn)

detector = faster_rcnn_resnet101_v1d_custom(classes=vg_val._obj_classes,
                                            pretrained_base=False, pretrained=False, additional_output=True)
params_path = 'faster_rcnn_resnet101_v1d_custom_0005_0.2528.params'
detector.load_parameters(params_path, ctx=ctx, ignore_extra=True, allow_missing=True)
detector.class_predictor.load_parameters('params/class_predictor-9.params', ctx=ctx)

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
            G.ndata['node_class_ids'] = G.ndata['node_class_ids'].as_in_context(ctx)
            G.ndata['node_class_vec'] = G.ndata['node_class_vec'].as_in_context(ctx)
            G.edata['classes'] = G.edata['classes'].as_in_context(ctx)
            G.edata['link'] = G.edata['link'].as_in_context(ctx)
            G.edata['weights'] = G.edata['weights'].expand_dims(1).as_in_context(ctx)
    img_list = [img.as_in_context(ctx) for img in img_list]
    return G_list, img_list

def batch_verbose(i, num_batches, metric_list, verbose_freq):
    if (i+1) % verbose_freq == 0:
        print_txt = 'Batch %d / %d\t'%(i, num_batches)
        for metric in metric_list:
            name, pred = metric.get()
            print_txt += '%s=%.4f '%(name, pred)
        logger.info(print_txt)

def validate(net, val_data, ctx, mode=['predcls'], verbose_freq=100):
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
    for i, (g_list, img_list) in enumerate(val_data):
        G_list, img_list = get_data_batch(g_list, img_list, ctx)
        if G_list is None or img_list is None:
            batch_verbose(i, len(val_data), metric_list, verbose_freq)
            continue

        detector_res_list = []
        G_pred = []
        bbox_pad = Pad(axis=(0))
        for G_slice, img in zip(G_list, img_list):
            cur_ctx = img.context
            bbox_list = [G.ndata['bbox'] for G in G_slice]
            bbox_stack = bbox_pad(bbox_list).as_in_context(cur_ctx)
            if 'sgdet' not in mode:
                bbox, spatial_feat, cls_pred = detector((img, bbox_stack))
                G_pred.append(build_graph_gt(G_slice, img, bbox, spatial_feat, cls_pred))
            else:
                ids, scores, bbox, feat, feat_ind, spatial_feat, cls_pred = detector(img)
                G_pred.append(build_graph_pred(G_slice, img, scores, bbox, feat_ind, spatial_feat, cls_pred))

        if len(G_pred) > 0:
            G_pred = [dgl.unbatch(net(G)) for G in G_pred]

        nd.waitall()
        for G_gt_slice, G_pred_slice, img_slice in zip(G_list, G_pred, img_list):
            for G_gt, G_pred in zip(G_gt_slice, G_pred_slice):
                gt_objects, gt_triplet = extract_gt(G_gt, img_slice.shape[2:4])
                pred_objects, pred_triplet = extract_pred(G_pred)
                for metric in metric_list:
                    if isinstance(metric, PredCls) or \
                       isinstance(metric, PhrCls) or \
                       isinstance(metric, SGDet):
                        metric.update(gt_triplet, pred_triplet)
                    else:
                        metric.update((gt_objects, gt_triplet), (pred_objects, pred_triplet))
        batch_verbose(i, len(val_data), metric_list, verbose_freq)
    print_txt = 'Validation Set Performance: '
    for metric in metric_list:
        name, pred = metric.get()
        print_txt += '%s=%.4f '%(name, pred)
    logger.info(print_txt)

'''
validate(net, val_data, ctx, mode=['predcls', 'phrcls'])
'''
validate(net, val_data, ctx, mode=['sgdet', 'sgdet+'], verbose_freq=100)
