import argparse
import logging
import time

import mxnet as mx
import numpy as np
from data import *
from gluoncv.data.batchify import Pad
from model import faster_rcnn_resnet101_v1d_custom, RelDN
from mxnet import gluon, nd
from utils import *

import dgl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Pre-trained RelDN Model."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Training with GPUs, you can specify 1,3 for example.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Total batch-size for training.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sgdet",
        help="Evaluation metric, could be 'predcls', 'phrcls', 'sgdet' or 'sgdet+'.",
    )
    parser.add_argument(
        "--pretrained-faster-rcnn-params",
        type=str,
        required=True,
        help="Path to saved Faster R-CNN model parameters.",
    )
    parser.add_argument(
        "--reldn-params",
        type=str,
        required=True,
        help="Path to saved Faster R-CNN model parameters.",
    )
    parser.add_argument(
        "--faster-rcnn-params",
        type=str,
        required=True,
        help="Path to saved Faster R-CNN model parameters.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="reldn_output.log",
        help="Path to save training logs.",
    )
    parser.add_argument(
        "--freq-prior",
        type=str,
        default="freq_prior.pkl",
        help="Path to saved frequency prior data.",
    )
    parser.add_argument(
        "--verbose-freq",
        type=int,
        default=100,
        help="Frequency of log printing in number of iterations.",
    )
    args = parser.parse_args()
    return args


args = parse_args()

filehandler = logging.FileHandler(args.log_dir)
streamhandler = logging.StreamHandler()
logger = logging.getLogger("")
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# Hyperparams
ctx = [mx.gpu(int(i)) for i in args.gpus.split(",") if i.strip()]
if ctx:
    num_gpus = len(ctx)
    assert args.batch_size % num_gpus == 0
    per_device_batch_size = int(args.batch_size / num_gpus)
else:
    ctx = [mx.cpu()]
    per_device_batch_size = args.batch_size
batch_size = args.batch_size
N_relations = 50
N_objects = 150
batch_verbose_freq = args.verbose_freq

mode = args.metric
metric_list = []
topk_list = [20, 50, 100]
if mode == "predcls":
    for topk in topk_list:
        metric_list.append(PredCls(topk=topk))
if mode == "phrcls":
    for topk in topk_list:
        metric_list.append(PhrCls(topk=topk))
if mode == "sgdet":
    for topk in topk_list:
        metric_list.append(SGDet(topk=topk))
if mode == "sgdet+":
    for topk in topk_list:
        metric_list.append(SGDetPlus(topk=topk))
for metric in metric_list:
    metric.reset()

semantic_only = False
net = RelDN(
    n_classes=N_relations,
    prior_pkl=args.freq_prior,
    semantic_only=semantic_only,
)
net.load_parameters(args.reldn_params, ctx=ctx)

# dataset and dataloader
vg_val = VGRelation(split="val")
logger.info("data loaded!")
val_data = gluon.data.DataLoader(
    vg_val,
    batch_size=len(ctx),
    shuffle=False,
    num_workers=16 * num_gpus,
    batchify_fn=dgl_mp_batchify_fn,
)
n_batches = len(val_data)

detector = faster_rcnn_resnet101_v1d_custom(
    classes=vg_val.obj_classes,
    pretrained_base=False,
    pretrained=False,
    additional_output=True,
)
params_path = args.pretrained_faster_rcnn_params
detector.load_parameters(
    params_path, ctx=ctx, ignore_extra=True, allow_missing=True
)

detector_feat = faster_rcnn_resnet101_v1d_custom(
    classes=vg_val.obj_classes,
    pretrained_base=False,
    pretrained=False,
    additional_output=True,
)
detector_feat.load_parameters(
    params_path, ctx=ctx, ignore_extra=True, allow_missing=True
)

detector_feat.features.load_parameters(args.faster_rcnn_params, ctx=ctx)


def get_data_batch(g_list, img_list, ctx_list):
    if g_list is None or len(g_list) == 0:
        return None, None
    n_gpu = len(ctx_list)
    size = len(g_list)
    if size < n_gpu:
        raise Exception("too small batch")
    step = size // n_gpu
    G_list = [
        g_list[i * step : (i + 1) * step]
        if i < n_gpu - 1
        else g_list[i * step : size]
        for i in range(n_gpu)
    ]
    img_list = [
        img_list[i * step : (i + 1) * step]
        if i < n_gpu - 1
        else img_list[i * step : size]
        for i in range(n_gpu)
    ]

    for G_slice, ctx in zip(G_list, ctx_list):
        for G in G_slice:
            G.ndata["bbox"] = G.ndata["bbox"].as_in_context(ctx)
            G.ndata["node_class"] = G.ndata["node_class"].as_in_context(ctx)
            G.ndata["node_class_vec"] = G.ndata["node_class_vec"].as_in_context(
                ctx
            )
            G.edata["rel_class"] = G.edata["rel_class"].as_in_context(ctx)
    img_list = [img.as_in_context(ctx) for img in img_list]
    return G_list, img_list


for i, (G_list, img_list) in enumerate(val_data):
    G_list, img_list = get_data_batch(G_list, img_list, ctx)
    if G_list is None or img_list is None:
        if (i + 1) % batch_verbose_freq == 0:
            print_txt = "Batch[%d/%d] " % (i, n_batches)
            for metric in metric_list:
                metric_name, metric_val = metric.get()
                print_txt += "%s=%.4f " % (metric_name, metric_val)
            logger.info(print_txt)
        continue

    detector_res_list = []
    G_batch = []
    bbox_pad = Pad(axis=(0))
    # loss_cls_val = 0
    for G_slice, img in zip(G_list, img_list):
        cur_ctx = img.context
        if mode == "predcls":
            bbox_list = [G.ndata["bbox"] for G in G_slice]
            bbox_stack = bbox_pad(bbox_list).as_in_context(cur_ctx)
            ids, scores, bbox, spatial_feat = detector(
                img, None, None, bbox_stack
            )

            node_class_list = [G.ndata["node_class"] for G in G_slice]
            node_class_stack = bbox_pad(node_class_list).as_in_context(cur_ctx)
            g_pred_batch = build_graph_validate_gt_obj(
                img,
                node_class_stack,
                bbox,
                spatial_feat,
                bbox_improvement=True,
                overlap=False,
            )
        elif mode == "phrcls":
            # use ground truth bbox
            bbox_list = [G.ndata["bbox"] for G in G_slice]
            bbox_stack = bbox_pad(bbox_list).as_in_context(cur_ctx)
            ids, scores, bbox, spatial_feat = detector(
                img, None, None, bbox_stack
            )

            g_pred_batch = build_graph_validate_gt_bbox(
                img,
                ids,
                scores,
                bbox,
                spatial_feat,
                bbox_improvement=True,
                overlap=False,
            )
        else:
            # use predicted bbox
            ids, scores, bbox, feat, feat_ind, spatial_feat = detector(img)
            g_pred_batch = build_graph_validate_pred(
                img,
                ids,
                scores,
                bbox,
                feat_ind,
                spatial_feat,
                bbox_improvement=True,
                scores_top_k=75,
                overlap=False,
            )
        if not semantic_only:
            rel_bbox = g_pred_batch.edata["rel_bbox"]
            batch_id = g_pred_batch.edata["batch_id"].asnumpy()
            n_sample_edges = g_pred_batch.number_of_edges()
            # g_pred_batch.edata['edge_feat'] = mx.nd.zeros((n_sample_edges, 49), ctx=cur_ctx)
            n_graph = len(G_slice)
            bbox_rel_list = []
            for j in range(n_graph):
                eids = np.where(batch_id == j)[0]
                if len(eids) > 0:
                    bbox_rel_list.append(rel_bbox[eids])
            bbox_rel_stack = bbox_pad(bbox_rel_list).as_in_context(cur_ctx)
            _, _, _, spatial_feat_rel = detector_feat(
                img, None, None, bbox_rel_stack
            )
            spatial_feat_rel_list = []
            for j in range(n_graph):
                eids = np.where(batch_id == j)[0]
                if len(eids) > 0:
                    spatial_feat_rel_list.append(
                        spatial_feat_rel[j, 0 : len(eids)]
                    )
            g_pred_batch.edata["edge_feat"] = nd.concat(
                *spatial_feat_rel_list, dim=0
            )

        G_batch.append(g_pred_batch)

    G_batch = [net(G) for G in G_batch]

    for G_slice, G_pred, img_slice in zip(G_list, G_batch, img_list):
        for G_gt, G_pred_one in zip(G_slice, [G_pred]):
            if G_pred_one is None or G_pred_one.number_of_nodes() == 0:
                continue
            gt_objects, gt_triplet = extract_gt(G_gt, img_slice.shape[2:4])
            pred_objects, pred_triplet = extract_pred(G_pred, joint_preds=True)
            for metric in metric_list:
                if (
                    isinstance(metric, PredCls)
                    or isinstance(metric, PhrCls)
                    or isinstance(metric, SGDet)
                ):
                    metric.update(gt_triplet, pred_triplet)
                else:
                    metric.update(
                        (gt_objects, gt_triplet), (pred_objects, pred_triplet)
                    )
    if (i + 1) % batch_verbose_freq == 0:
        print_txt = "Batch[%d/%d] " % (i, n_batches)
        for metric in metric_list:
            metric_name, metric_val = metric.get()
            print_txt += "%s=%.4f " % (metric_name, metric_val)
        logger.info(print_txt)

print_txt = "Batch[%d/%d] " % (n_batches, n_batches)
for metric in metric_list:
    metric_name, metric_val = metric.get()
    print_txt += "%s=%.4f " % (metric_name, metric_val)
logger.info(print_txt)
