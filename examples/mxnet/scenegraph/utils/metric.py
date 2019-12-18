import dgl
import mxnet as mx
import numpy as np
import logging, time
from operator import attrgetter
from mxnet import nd, gluon
from mxnet.gluon import nn
from dgl.utils import toindex
from dgl.nn.mxnet import GraphConv
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Pad

def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	iou_val = interArea / float(boxAArea + boxBArea - interArea)
 
	return iou_val

def triplet_iou_thresh(gt_triplet, pred_triplet, iou_thresh=0.5):
    sub_iou = iou(gt_triplet.sub_bbox, pred_triplet.sub_bbox)
    if sub_iou >= iou_thresh:
        ob_iou = iou(gt_triplet.ob_bbox, pred_triplet.ob_bbox)
        if ob_iou >= iou_thresh:
            return True
    return False

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
        preds = sorted(preds, key=attrgetter('score_pred'), reverse=True)
        m = min(self.topk, len(preds))
        count = 0
        for i, pred in enumerate(preds):
            if i >= m:
                break
            for label in labels:
                if label.rel_class == pred.rel_class:
                    if triplet_iou_thresh(pred, label, 0.99):
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
        preds = sorted(preds, key=attrgetter('score_phr'), reverse=True)
        m = min(self.topk, len(preds))
        count = 0
        for i, pred in enumerate(preds):
            if i >= m:
                break
            for label in labels:
                if label.rel_class == pred.rel_class and \
                   label.sub_class == pred.sub_class and \
                   label.ob_class == pred.ob_class and \
                   triplet_iou_thresh(pred, label, 0.99):
                    count += 1
        self.sum_metric += count / len(labels)
        self.num_inst += 1

@mx.metric.register
@mx.metric.alias('sgdet')
class SGDet(mx.metric.EvalMetric):
    def __init__(self, topk=20):
        super(SGDet, self).__init__('sgdet@%d'%(topk))
        self.topk = topk

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        preds = sorted(preds, key=attrgetter('score_phr'), reverse=True)
        m = min(self.topk, len(preds))
        count = 0
        for i, pred in enumerate(preds):
            if i >= m:
                break
            for label in labels:
                if label.rel_class == pred.rel_class and \
                   label.sub_class == pred.sub_class and \
                   label.ob_class == pred.ob_class and \
                   triplet_iou_thresh(pred, label):
                    count += 1
        self.sum_metric += count / len(labels)
        self.num_inst += 1

def merge_res_iou(g_slice, img_batch, scores, bbox, feat_ind, spatial_feat, cls_pred):
    n = len(g_slice)
    for i, g in enumerate(g_slice):
        img = img_batch[i]
        img_size = img.shape[1:3]

        gt_bbox = g.ndata['bbox']
        gt_bbox[:, 0] /= img_size[1]
        gt_bbox[:, 1] /= img_size[0]
        gt_bbox[:, 2] /= img_size[1]
        gt_bbox[:, 3] /= img_size[0]
        bbox[i, :, 0] /= img_size[1]
        bbox[i, :, 1] /= img_size[0]
        bbox[i, :, 2] /= img_size[1]
        bbox[i, :, 3] /= img_size[0]
        inds = np.where(scores[i, :, 0].asnumpy() > 0)[0].tolist()
        if len(inds) == 0:
            return None
        ious = nd.contrib.box_iou(gt_bbox, bbox[i, inds])
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
            assign_ind[row_ind] = col_ind
            assign_scores[row_ind] = ious[row_ind, col_ind].asscalar()
            ious[row_ind, :] = -1
            ious[:, col_ind] = -1
            h -= 1
            w -= 1

        box_ind = [inds[i] for i in assign_ind]
        roi_ind = feat_ind[i, box_ind].squeeze(1)
        g.ndata['bbox'] = gt_bbox
        g.ndata['pred_bbox'] = bbox[i, box_ind]
        g.ndata['node_feat'] = spatial_feat[i, roi_ind]
        g.ndata['node_class_pred'] = cls_pred[i, roi_ind, 1:]
    return dgl.batch(g_slice)

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

class EdgeTriplet(object):
    def __init__(self, score_pred, score_phr, rel_class, sub_class, ob_class, sub_bbox, ob_bbox):
        super(EdgeTriplet, self).__init__()
        self.score_pred = score_pred
        self.score_phr = score_phr
        self.rel_class = rel_class
        self.sub_class = sub_class
        self.ob_class = ob_class
        self.sub_bbox = sub_bbox
        self.ob_bbox = ob_bbox

    def __repr__(self):
        return self.__dict__.__repr__()


def get_gt_triplet(g, img_size):
    if g is None or g.number_of_nodes() == 0:
        return None
    gt_eids = np.where(g.edata['link'].asnumpy() > 0)[0]
    gt_node_ids = g.find_edges(gt_eids)
    gt_node_sub = gt_node_ids[0].asnumpy()
    gt_node_ob = gt_node_ids[1].asnumpy()
    gt_rel_class = g.edata['classes'][gt_eids].asnumpy()
    gt_sub_class = g.ndata['node_class_ids'][gt_node_sub][:,0].asnumpy()
    gt_ob_class = g.ndata['node_class_ids'][gt_node_ob][:,0].asnumpy()

    gt_sub_bbox = g.ndata['bbox'][gt_node_sub].asnumpy()
    gt_sub_bbox[:, 0] /= img_size[1] 
    gt_sub_bbox[:, 1] /= img_size[0] 
    gt_sub_bbox[:, 2] /= img_size[1] 
    gt_sub_bbox[:, 3] /= img_size[0] 
    gt_ob_bbox = g.ndata['bbox'][gt_node_ob].asnumpy()
    gt_ob_bbox[:, 0] /= img_size[1] 
    gt_ob_bbox[:, 1] /= img_size[0] 
    gt_ob_bbox[:, 2] /= img_size[1] 
    gt_ob_bbox[:, 3] /= img_size[0] 

    gt_triplets = []
    for rel_class, sub_class, ob_class, sub_bbox, ob_bbox in zip(gt_rel_class, gt_sub_class, gt_ob_class, gt_sub_bbox, gt_ob_bbox):
        edgetriplet = EdgeTriplet(1, 1, int(rel_class), int(sub_class), int(ob_class),
                                  (sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3]),
                                  (ob_bbox[0], ob_bbox[1], ob_bbox[2], ob_bbox[3]))
        gt_triplets.append(edgetriplet)
    return gt_triplets

def get_pred_triplet(g):
    if g is None or g.number_of_nodes() == 0:
        return None
    pred_node_ids = g.edges()
    pred_node_sub = pred_node_ids[0].asnumpy()
    pred_node_ob = pred_node_ids[1].asnumpy()

    pred_link_prob = nd.softmax(g.edata['link_preds']).asnumpy()
    pred_rel_prob = nd.softmax(g.edata['preds']).asnumpy()
    pred_rel_class = pred_rel_prob.argmax(axis=1)

    pred_sub_prob = g.ndata['node_class_prob'][pred_node_sub].asnumpy()
    pred_sub_class = pred_sub_prob.argmax(axis=1)
    pred_sub_bbox = g.ndata['pred_bbox'][pred_node_sub].asnumpy()

    pred_ob_prob = g.ndata['node_class_prob'][pred_node_ob].asnumpy()
    pred_ob_class = pred_ob_prob.argmax(axis=1)
    pred_ob_bbox = g.ndata['pred_bbox'][pred_node_ob].asnumpy()

    n = g.number_of_edges()
    pred_triplets = []
    for i in range(n):
        rel_class = pred_rel_class[i]
        score_pred = pred_link_prob[i, 1] * pred_rel_prob[i, rel_class]

        sub_class = pred_sub_class[i]
        ob_class = pred_ob_class[i]
        score_phr = score_pred
        score_phr *= pred_sub_prob[i, sub_class]
        score_phr *= pred_ob_prob[i, ob_class]

        sub_bbox = pred_sub_bbox[i]
        ob_bbox = pred_ob_bbox[i]

        edgetriplet = EdgeTriplet(score_pred, score_phr, int(rel_class), int(sub_class), int(ob_class),
                                  (sub_bbox[0], sub_bbox[1], sub_bbox[2], sub_bbox[3]),
                                  (ob_bbox[0], ob_bbox[1], ob_bbox[2], ob_bbox[3]))
        pred_triplets.append(edgetriplet)
    return pred_triplets
