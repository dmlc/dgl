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

@mx.metric.register
@mx.metric.alias('auc')
class AUCMetric(mx.metric.EvalMetric):
    def __init__(self, eps=1e-12):
        super(AUCMetric, self).__init__('auc')
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
            return

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

@mx.metric.register
@mx.metric.alias('predcls')
class PredCls(mx.metric.EvalMetric):
    def __init__(self, topk=20):
        super(PredCls, self).__init__('predcls@%d'%(topk))
        self.topk = topk

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        from operator import itemgetter
        preds = sorted(preds, key=itemgetter(0), reverse=True)
        m = min(self.topk, len(preds))
        count = 0
        for i in range(m):
            _, _, pred_sub, pred_rel, pred_ob, _, _ = preds[i]
            for gt_sub, gt_rel, gt_ob, _, _ in labels:
                if gt_sub == pred_sub and \
                   gt_rel == pred_rel and \
                   gt_ob == pred_ob:
                    count += 1
        self.sum_metric += count / len(labels)
        self.num_inst += 1

@mx.metric.register
@mx.metric.alias('phrcls')
class PhrCls(mx.metric.EvalMetric):
    def __init__(self, topk=20):
        super(PhrCls, self).__init__('phrcls@%d'%(topk))
        self.topk = topk

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        from operator import itemgetter
        preds = sorted(preds, key=itemgetter(1), reverse=True)
        m = min(self.topk, len(preds))
        count = 0
        for i in range(m):
            _, _, pred_sub, pred_rel, pred_ob, pred_sub_cls, pred_ob_cls = preds[i]
            for gt_sub, gt_rel, gt_ob, gt_sub_cls, gt_ob_cls in labels:
                if gt_sub_cls == pred_sub_cls and \
                   gt_ob_cls == pred_ob_cls and \
                   gt_sub == pred_sub and \
                   gt_rel == pred_rel and \
                   gt_ob == pred_ob:
                    count += 1
        self.sum_metric += count / len(labels)
        self.num_inst += 1

@mx.metric.register
@mx.metric.alias('sgdet')
class SGDet(mx.metric.EvalMetric):
    def __init__(self, topk=20):
        super(SGDet, self).__init__('predcls@%d'%(topk))
        self.topk = topk

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        from operator import itemgetter
        preds = sorted(preds, key=itemgetter(0), reverse=True)
        m = min(self.topk, len(preds))
        count = 0
        for i in range(m):
            score, pred_sub, pred_sub_bbox, pred_rel, pred_ob, pred_ob_bbox = preds[i]
            for gt_sub, gt_rel, gt_ob in labels:
                if gt_sub == pred_sub and \
                   gt_rel == pred_rel and \
                   gt_ob == pred_ob:
                    count += 1
        self.sum_metric += count / len(labels)
        self.num_inst += 1

def get_triplet(g):
    # gt triplet
    gt_eids = np.where(g.edata['link'].asnumpy() > 0)[0]
    gt_node_ids = g.find_edges(gt_eids)
    gt_node_sub = gt_node_ids[0].asnumpy()
    gt_node_ob = gt_node_ids[1].asnumpy()
    gt_rel = g.edata['classes'][gt_eids].asnumpy()
    gt_sub_class = g.ndata['node_class_ids'][gt_node_sub][:,0].asnumpy()
    gt_ob_class = g.ndata['node_class_ids'][gt_node_ob][:,0].asnumpy()
    gt_triplet = []
    for sub, rel, ob, sub_class, ob_class in zip(gt_node_sub, gt_rel, gt_node_ob, gt_sub_class, gt_ob_class):
        gt_triplet.append((int(sub), int(rel), int(ob), int(sub_class), int(ob_class)))

    # pred triplet
    '''
    tmp = nd.softmax(g.edata['link_preds'][:,1]).asnumpy()
    eids = tmp.argsort()[::-1]
    if len(eids) == 0:
        return gt_triplet, []
    pred_node_ids = g.find_edges(eids)
    '''
    pred_node_ids = g.edges()
    pred_node_sub = pred_node_ids[0].asnumpy()
    pred_node_ob = pred_node_ids[1].asnumpy()
    pred_link = nd.softmax(g.edata['link_preds']).asnumpy()
    pred_rel = nd.softmax(g.edata['preds']).asnumpy()
    pred_sub_prob = g.ndata['node_class_prob'][pred_node_sub].asnumpy()
    pred_ob_prob = g.ndata['node_class_prob'][pred_node_ob].asnumpy()
    pred_sub_class = pred_sub_prob.argmax(axis=1)
    pred_ob_class = pred_ob_prob.argmax(axis=1)
    pred_triplet = []

    n = g.number_of_edges()
    rel_ind = pred_rel.argmax(axis=1)
    # for rel in range(pred_rel.shape[1]):
    for i in range(n):
        rel = rel_ind[i]
        scores_pred = (pred_link[i,1] * pred_rel[i, rel])
        scores_phr = scores_pred
        scores_phr *= pred_sub_prob[i, rel]
        scores_phr *= pred_ob_prob[i, rel]
        sub = pred_node_sub[i]
        ob = pred_node_ob[i]
        sub_class = pred_sub_class[i]
        ob_class = pred_ob_class[i]
        pred_triplet.append((scores_pred, scores_phr, int(sub), int(rel), int(ob),
                             int(sub_class), int(ob_class)))
    return gt_triplet, pred_triplet

def get_triplet_sgdet(g):
    # gt triplet
    gt_eids = np.where(g.edata['link'].asnumpy() > 0)[0]
    gt_node_ids = g.find_edges(gt_eids)
    gt_node_sub = gt_node_ids[0].asnumpy()
    gt_node_ob = gt_node_ids[1].asnumpy()
    gt_rel = g.edata['classes'][gt_eids].asnumpy()
    gt_sub_class = g.ndata['node_class_ids'][gt_node_sub][:,0].asnumpy()
    gt_ob_class = g.ndata['node_class_ids'][gt_node_ob][:,0].asnumpy()
    gt_sub_bbox = g.ndata['bbox'][gt_node_sub].asnumpy()
    gt_ob_bbox = g.ndata['bbox'][gt_node_ob].asnumpy()
    gt_triplet = []
    for sub, rel, ob, sub_class, ob_class, sub_bbox, ob_bbox in zip(gt_node_sub, gt_rel, gt_node_ob, gt_sub_class, gt_ob_class, gt_sub_bbox, gt_ob_bbox):
        gt_triplet.append((int(sub), int(rel), int(ob), int(sub_class), int(ob_class),
                           sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3],
                           ob_bbox[0], ob_bbox[1], ob_bbox[2], ob_bbox[3]))

    # pred triplet
    tmp = nd.softmax(g.edata['link_preds'][:,1]).asnumpy()
    eids = tmp.argsort()[::-1]
    if len(eids) == 0:
        return gt_triplet, []
    pred_node_ids = g.find_edges(eids)
    pred_node_sub = pred_node_ids[0].asnumpy()
    pred_node_ob = pred_node_ids[1].asnumpy()
    pred_link = nd.softmax(g.edata['link_preds'][eids]).asnumpy()
    pred_rel = nd.softmax(g.edata['preds'][eids]).asnumpy()
    pred_sub_prob = g.ndata['node_class_prob'][pred_node_sub, 1:].asnumpy()
    pred_ob_prob = g.ndata['node_class_prob'][pred_node_ob, 1:].asnumpy()
    pred_sub_class = pred_sub_prob.argmax(axis=1)
    pred_ob_class = pred_ob_prob.argmax(axis=1)
    pred_sub_bbox = g.ndata['pred_bbox'][gt_node_sub].asnumpy()
    pred_ob_bbox = g.ndata['pred_bbox'][gt_node_ob].asnumpy()
    pred_triplet = []

    n = len(eids)
    rel_ind = pred_rel.argmax(axis=1)
    for i in range(n):
        rel = rel_ind[i]
        scores = (pred_link[i,1] * pred_rel[i, rel])
        scores *= pred_sub_prob[i, rel]
        scores *= pred_ob_prob[i, rel]
        sub = pred_node_sub[i]
        ob = pred_node_ob[i]
        sub_class = pred_sub_class[i]
        ob_class = pred_ob_class[i]
        sub_bbox = pred_sub_bbox[i]
        ob_bbox = pred_ob_bbox[i]
        pred_triplet.append((scores, int(sub), int(rel), int(ob), int(sub_class), int(ob_class), 
                             sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3],
                             ob_bbox[0], ob_bbox[1], ob_bbox[2], ob_bbox[3]))
    return gt_triplet, pred_triplet

def merge_res_iou(g, scores, bbox, feat_ind, spatial_feat, cls_pred, iou_thresh=0.5):
    img = g.ndata['images'][0]
    gt_bbox = g.ndata['bbox']
    gt_bbox[0,:] = [0, 0, 0, 0]
    img_size = img.shape[1:3]
    bbox[:, 0] /= img_size[1]
    bbox[:, 1] /= img_size[0]
    bbox[:, 2] /= img_size[1]
    bbox[:, 3] /= img_size[0]
    inds = np.where(scores[:,0].asnumpy() > 0)[0].tolist()
    if len(inds) == 0:
        return None
    ious = nd.contrib.box_iou(gt_bbox, bbox[inds])
    # assignment
    H, W = ious.shape
    h = H
    w = W
    assign_ind = [-1 for i in range(H)]
    assign_scores = [-1 for i in range(H)]
    while h > 0 and w > 0:
        ind = int(ious.argmax().asscalar())
        row_ind = ind // W
        col_ind = ind % W
        if ious[row_ind, col_ind].asscalar() < iou_thresh:
            break
        assign_ind[row_ind] = col_ind
        assign_scores[row_ind] = ious[row_ind, col_ind].asscalar()
        ious[row_ind, :] = -1
        ious[:, col_ind] = -1
        h -= 1
        w -= 1

    remove_inds = [i for i in range(H) if assign_ind[i] == -1]
    assign_ind = [ind for ind in assign_ind if ind > -1]
    if len(remove_inds) >= g.number_of_nodes() - 1:
        return None
    g.remove_nodes(remove_inds)
    box_ind = [inds[i] for i in assign_ind]
    roi_ind = feat_ind[box_ind].squeeze(1)
    g.ndata['pred_bbox'] = bbox[box_ind]
    g.ndata['node_feat'] = spatial_feat[roi_ind]
    g.ndata['node_class_pred'] = cls_pred[roi_ind]
    return g

def merge_res(g_slice, img, bbox, spatial_feat, cls_pred):
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]
    for i, g in enumerate(g_slice):
        n_node = g.number_of_nodes()
        g.ndata['pred_bbox'] = bbox[i, 0:n_node]
        # filter out background-class prediction
        g.ndata['node_class_pred'] = cls_pred[i, 0:n_node, 1:]
        g.ndata['node_feat'] = spatial_feat[i, 0:n_node]
    return dgl.batch(g_slice)
