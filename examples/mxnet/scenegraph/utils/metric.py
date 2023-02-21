import logging
import time
from operator import attrgetter, itemgetter

import dgl

import mxnet as mx
import numpy as np
from dgl.nn.mxnet import GraphConv
from dgl.utils import toindex
from gluoncv.data.batchify import Pad
from gluoncv.model_zoo import get_model
from mxnet import gluon, nd
from mxnet.gluon import nn


def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea < 1e-7:
        return 0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea < 1e-7:
        return 0

    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val


def object_iou_thresh(gt_object, pred_object, iou_thresh=0.5):
    obj_iou = iou(gt_object[1:5], pred_object[1:5])
    if obj_iou >= iou_thresh:
        return True
    return False


def triplet_iou_thresh(pred_triplet, gt_triplet, iou_thresh=0.5):
    sub_iou = iou(gt_triplet[5:9], pred_triplet[5:9])
    if sub_iou >= iou_thresh:
        ob_iou = iou(gt_triplet[9:13], pred_triplet[9:13])
        if ob_iou >= iou_thresh:
            return True
    return False


@mx.metric.register
@mx.metric.alias("auc")
class AUCMetric(mx.metric.EvalMetric):
    def __init__(self, name="auc", eps=1e-12):
        super(AUCMetric, self).__init__(name)
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
@mx.metric.alias("predcls")
class PredCls(mx.metric.EvalMetric):
    """Metric with ground truth object location and label"""

    def __init__(self, topk=20, iou_thresh=0.99):
        super(PredCls, self).__init__("predcls@%d" % (topk))
        self.topk = topk
        self.iou_thresh = iou_thresh

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        preds = preds[preds[:, 0].argsort()[::-1]]
        m = min(self.topk, preds.shape[0])
        count = 0
        gt_edge_num = labels.shape[0]
        label_matched = [False for label in labels]
        for i in range(m):
            pred = preds[i]
            for j in range(gt_edge_num):
                if label_matched[j]:
                    continue
                label = labels[j]
                if int(label[2]) == int(pred[2]) and triplet_iou_thresh(
                    pred, label, self.iou_thresh
                ):
                    count += 1
                    label_matched[j] = True

        total = labels.shape[0]
        self.sum_metric += count / total
        self.num_inst += 1


@mx.metric.register
@mx.metric.alias("phrcls")
class PhrCls(mx.metric.EvalMetric):
    """Metric with ground truth object location and predicted object label from detector"""

    def __init__(self, topk=20, iou_thresh=0.99):
        super(PhrCls, self).__init__("phrcls@%d" % (topk))
        self.topk = topk
        self.iou_thresh = iou_thresh

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        preds = preds[preds[:, 1].argsort()[::-1]]
        m = min(self.topk, preds.shape[0])
        count = 0
        gt_edge_num = labels.shape[0]
        label_matched = [False for label in labels]
        for i in range(m):
            pred = preds[i]
            for j in range(gt_edge_num):
                if label_matched[j]:
                    continue
                label = labels[j]
                if (
                    int(label[2]) == int(pred[2])
                    and int(label[3]) == int(pred[3])
                    and int(label[4]) == int(pred[4])
                    and triplet_iou_thresh(pred, label, self.iou_thresh)
                ):
                    count += 1
                    label_matched[j] = True
        total = labels.shape[0]
        self.sum_metric += count / total
        self.num_inst += 1


@mx.metric.register
@mx.metric.alias("sgdet")
class SGDet(mx.metric.EvalMetric):
    """Metric with predicted object information by the detector"""

    def __init__(self, topk=20, iou_thresh=0.5):
        super(SGDet, self).__init__("sgdet@%d" % (topk))
        self.topk = topk
        self.iou_thresh = iou_thresh

    def update(self, labels, preds):
        if labels is None or preds is None:
            self.num_inst += 1
            return
        preds = preds[preds[:, 1].argsort()[::-1]]
        m = min(self.topk, len(preds))
        count = 0
        gt_edge_num = labels.shape[0]
        label_matched = [False for label in labels]
        for i in range(m):
            pred = preds[i]
            for j in range(gt_edge_num):
                if label_matched[j]:
                    continue
                label = labels[j]
                if (
                    int(label[2]) == int(pred[2])
                    and int(label[3]) == int(pred[3])
                    and int(label[4]) == int(pred[4])
                    and triplet_iou_thresh(pred, label, self.iou_thresh)
                ):
                    count += 1
                    label_matched[j] = True
        total = labels.shape[0]
        self.sum_metric += count / total
        self.num_inst += 1


@mx.metric.register
@mx.metric.alias("sgdet+")
class SGDetPlus(mx.metric.EvalMetric):
    """Metric proposed by `Graph R-CNN for Scene Graph Generation`"""

    def __init__(self, topk=20, iou_thresh=0.5):
        super(SGDetPlus, self).__init__("sgdet+@%d" % (topk))
        self.topk = topk
        self.iou_thresh = iou_thresh

    def update(self, labels, preds):
        label_objects, label_triplets = labels
        pred_objects, pred_triplets = preds
        if label_objects is None or pred_objects is None:
            self.num_inst += 1
            return
        count = 0
        # count objects
        object_matched = [False for obj in label_objects]
        m = len(pred_objects)
        gt_obj_num = label_objects.shape[0]
        for i in range(m):
            pred = pred_objects[i]
            for j in range(gt_obj_num):
                if object_matched[j]:
                    continue
                label = label_objects[j]
                if int(label[0]) == int(pred[0]) and object_iou_thresh(
                    pred, label, self.iou_thresh
                ):
                    count += 1
                    object_matched[j] = True

        # count predicate and triplet
        pred_triplets = pred_triplets[pred_triplets[:, 1].argsort()[::-1]]
        m = min(self.topk, len(pred_triplets))
        gt_triplet_num = label_triplets.shape[0]
        triplet_matched = [False for label in label_triplets]
        predicate_matched = [False for label in label_triplets]
        for i in range(m):
            pred = pred_triplets[i]
            for j in range(gt_triplet_num):
                label = label_triplets[j]
                if not predicate_matched:
                    if int(label[2]) == int(pred[2]) and triplet_iou_thresh(
                        pred, label, self.iou_thresh
                    ):
                        count += label[3]
                        predicate_matched[j] = True
                if not triplet_matched[j]:
                    if (
                        int(label[2]) == int(pred[2])
                        and int(label[3]) == int(pred[3])
                        and int(label[4]) == int(pred[4])
                        and triplet_iou_thresh(pred, label, self.iou_thresh)
                    ):
                        count += 1
                        triplet_matched[j] = True
        # compute sum
        total = labels.shape[0]
        N = gt_obj_num + 2 * total
        self.sum_metric += count / N
        self.num_inst += 1


def extract_gt(g, img_size):
    """extract prediction from ground truth graph"""
    if g is None or g.number_of_nodes() == 0:
        return None, None
    gt_eids = np.where(g.edata["rel_class"].asnumpy() > 0)[0]
    if len(gt_eids) == 0:
        return None, None

    gt_class = g.ndata["node_class"][:, 0].asnumpy()
    gt_bbox = g.ndata["bbox"].asnumpy()
    gt_bbox[:, 0] /= img_size[1]
    gt_bbox[:, 1] /= img_size[0]
    gt_bbox[:, 2] /= img_size[1]
    gt_bbox[:, 3] /= img_size[0]

    gt_objects = np.vstack([gt_class, gt_bbox.transpose(1, 0)]).transpose(1, 0)

    gt_node_ids = g.find_edges(gt_eids)
    gt_node_sub = gt_node_ids[0].asnumpy()
    gt_node_ob = gt_node_ids[1].asnumpy()
    gt_rel_class = g.edata["rel_class"][gt_eids, 0].asnumpy() - 1
    gt_sub_class = gt_class[gt_node_sub]
    gt_ob_class = gt_class[gt_node_ob]

    gt_sub_bbox = gt_bbox[gt_node_sub]
    gt_ob_bbox = gt_bbox[gt_node_ob]

    n = len(gt_eids)
    gt_triplets = np.vstack(
        [
            np.ones(n),
            np.ones(n),
            gt_rel_class,
            gt_sub_class,
            gt_ob_class,
            gt_sub_bbox.transpose(1, 0),
            gt_ob_bbox.transpose(1, 0),
        ]
    ).transpose(1, 0)
    return gt_objects, gt_triplets


def extract_pred(g, topk=100, joint_preds=False):
    """extract prediction from prediction graph for validation and visualization"""
    if g is None or g.number_of_nodes() == 0:
        return None, None

    pred_class = g.ndata["node_class_pred"].asnumpy()
    pred_class_prob = g.ndata["node_class_logit"].asnumpy()
    pred_bbox = g.ndata["pred_bbox"][:, 0:4].asnumpy()

    pred_objects = np.vstack([pred_class, pred_bbox.transpose(1, 0)]).transpose(
        1, 0
    )

    score_pred = g.edata["score_pred"].asnumpy()
    score_phr = g.edata["score_phr"].asnumpy()
    score_pred_topk_eids = (-score_pred).argsort()[0:topk].tolist()
    score_phr_topk_eids = (-score_phr).argsort()[0:topk].tolist()
    topk_eids = sorted(list(set(score_pred_topk_eids + score_phr_topk_eids)))

    pred_rel_prob = g.edata["preds"][topk_eids].asnumpy()
    if joint_preds:
        pred_rel_class = pred_rel_prob[:, 1:].argmax(axis=1)
    else:
        pred_rel_class = pred_rel_prob.argmax(axis=1)

    pred_node_ids = g.find_edges(topk_eids)
    pred_node_sub = pred_node_ids[0].asnumpy()
    pred_node_ob = pred_node_ids[1].asnumpy()

    pred_sub_class = pred_class[pred_node_sub]
    pred_sub_class_prob = pred_class_prob[pred_node_sub]
    pred_sub_bbox = pred_bbox[pred_node_sub]

    pred_ob_class = pred_class[pred_node_ob]
    pred_ob_class_prob = pred_class_prob[pred_node_ob]
    pred_ob_bbox = pred_bbox[pred_node_ob]

    pred_triplets = np.vstack(
        [
            score_pred[topk_eids],
            score_phr[topk_eids],
            pred_rel_class,
            pred_sub_class,
            pred_ob_class,
            pred_sub_bbox.transpose(1, 0),
            pred_ob_bbox.transpose(1, 0),
        ]
    ).transpose(1, 0)
    return pred_objects, pred_triplets
