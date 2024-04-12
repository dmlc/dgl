#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import inspect

import numpy as np
from clustering_benchmark import ClusteringBenchmark
from utils import metrics, TextColors, Timer


def _read_meta(fn):
    labels = list()
    lb_set = set()
    with open(fn) as f:
        for lb in f.readlines():
            lb = int(lb.strip())
            labels.append(lb)
            lb_set.add(lb)
    return np.array(labels), lb_set


def evaluate(gt_labels, pred_labels, metric="pairwise"):
    if isinstance(gt_labels, str) and isinstance(pred_labels, str):
        print("[gt_labels] {}".format(gt_labels))
        print("[pred_labels] {}".format(pred_labels))
        gt_labels, gt_lb_set = _read_meta(gt_labels)
        pred_labels, pred_lb_set = _read_meta(pred_labels)

        print(
            "#inst: gt({}) vs pred({})".format(len(gt_labels), len(pred_labels))
        )
        print(
            "#cls: gt({}) vs pred({})".format(len(gt_lb_set), len(pred_lb_set))
        )

    metric_func = metrics.__dict__[metric]

    with Timer(
        "evaluate with {}{}{}".format(TextColors.FATAL, metric, TextColors.ENDC)
    ):
        result = metric_func(gt_labels, pred_labels)
    if isinstance(result, float):
        print(
            "{}{}: {:.4f}{}".format(
                TextColors.OKGREEN, metric, result, TextColors.ENDC
            )
        )
    else:
        ave_pre, ave_rec, fscore = result
        print(
            "{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}".format(
                TextColors.OKGREEN, ave_pre, ave_rec, fscore, TextColors.ENDC
            )
        )


def evaluation(pred_labels, labels, metrics):
    print("==> evaluation")
    # pred_labels = g.ndata['pred_labels'].cpu().numpy()
    max_cluster = np.max(pred_labels)
    # gt_labels_all = g.ndata['labels'].cpu().numpy()
    gt_labels_all = labels
    pred_labels_all = pred_labels
    metric_list = metrics.split(",")
    for metric in metric_list:
        evaluate(gt_labels_all, pred_labels_all, metric)
    # H and C-scores
    gt_dict = {}
    pred_dict = {}
    for i in range(len(gt_labels_all)):
        gt_dict[str(i)] = gt_labels_all[i]
        pred_dict[str(i)] = pred_labels_all[i]
    bm = ClusteringBenchmark(gt_dict)
    scores = bm.evaluate_vmeasure(pred_dict)
    fmi_scores = bm.evaluate_fowlkes_mallows_score(pred_dict)
    print(scores)
