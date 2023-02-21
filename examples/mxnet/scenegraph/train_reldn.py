import argparse
import logging
import time

import mxnet as mx
import numpy as np
from data import *
from gluoncv.data.batchify import Pad
from gluoncv.utils import makedirs
from model import faster_rcnn_resnet101_v1d_custom, RelDN
from mxnet import gluon, nd
from utils import *

import dgl


def parse_args():
    parser = argparse.ArgumentParser(description="Train RelDN Model.")
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
        "--epochs", type=int, default=9, help="Training epochs."
    )
    parser.add_argument(
        "--lr-reldn",
        type=float,
        default=0.01,
        help="Learning rate for RelDN module.",
    )
    parser.add_argument(
        "--wd-reldn",
        type=float,
        default=0.0001,
        help="Weight decay for RelDN module.",
    )
    parser.add_argument(
        "--lr-faster-rcnn",
        type=float,
        default=0.01,
        help="Learning rate for Faster R-CNN module.",
    )
    parser.add_argument(
        "--wd-faster-rcnn",
        type=float,
        default=0.0001,
        help="Weight decay for RelDN module.",
    )
    parser.add_argument(
        "--lr-decay-epochs",
        type=str,
        default="5,8",
        help="Learning rate decay points.",
    )
    parser.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=4000,
        help="Learning rate warm-up iterations.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="params_resnet101_v1d_reldn",
        help="Path to save model parameters.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="reldn_output.log",
        help="Path to save training logs.",
    )
    parser.add_argument(
        "--pretrained-faster-rcnn-params",
        type=str,
        required=True,
        help="Path to saved Faster R-CNN model parameters.",
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

aggregate_grad = per_device_batch_size > 1

nepoch = args.epochs
N_relations = 50
N_objects = 150
save_dir = args.save_dir
makedirs(save_dir)
batch_verbose_freq = args.verbose_freq
lr_decay_epochs = [int(i) for i in args.lr_decay_epochs.split(",")]

# Dataset and dataloader
vg_train = VGRelation(split="train")
logger.info("data loaded!")
train_data = gluon.data.DataLoader(
    vg_train,
    batch_size=len(ctx),
    shuffle=True,
    num_workers=8 * num_gpus,
    batchify_fn=dgl_mp_batchify_fn,
)
n_batches = len(train_data)

# Network definition
net = RelDN(n_classes=N_relations, prior_pkl=args.freq_prior)
net.spatial.initialize(mx.init.Normal(1e-4), ctx=ctx)
net.visual.initialize(mx.init.Normal(1e-4), ctx=ctx)
for k, v in net.collect_params().items():
    v.grad_req = "add" if aggregate_grad else "write"
net_params = net.collect_params()
net_trainer = gluon.Trainer(
    net.collect_params(),
    "adam",
    {"learning_rate": args.lr_reldn, "wd": args.wd_reldn},
)

det_params_path = args.pretrained_faster_rcnn_params
detector = faster_rcnn_resnet101_v1d_custom(
    classes=vg_train.obj_classes,
    pretrained_base=False,
    pretrained=False,
    additional_output=True,
)
detector.load_parameters(
    det_params_path, ctx=ctx, ignore_extra=True, allow_missing=True
)
for k, v in detector.collect_params().items():
    v.grad_req = "null"

detector_feat = faster_rcnn_resnet101_v1d_custom(
    classes=vg_train.obj_classes,
    pretrained_base=False,
    pretrained=False,
    additional_output=True,
)
detector_feat.load_parameters(
    det_params_path, ctx=ctx, ignore_extra=True, allow_missing=True
)
for k, v in detector_feat.collect_params().items():
    v.grad_req = "null"
for k, v in detector_feat.features.collect_params().items():
    v.grad_req = "add" if aggregate_grad else "write"
det_params = detector_feat.features.collect_params()
det_trainer = gluon.Trainer(
    detector_feat.features.collect_params(),
    "adam",
    {"learning_rate": args.lr_faster_rcnn, "wd": args.wd_faster_rcnn},
)


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


L_rel = gluon.loss.SoftmaxCELoss()

train_metric = mx.metric.Accuracy(name="rel_acc")
train_metric_top5 = mx.metric.TopKAccuracy(5, name="rel_acc_top5")
metric_list = [train_metric, train_metric_top5]


def batch_print(
    epoch, i, batch_verbose_freq, n_batches, btic, loss_rel_val, metric_list
):
    if (i + 1) % batch_verbose_freq == 0:
        print_txt = "Epoch[%d] Batch[%d/%d], time: %d, loss_rel=%.4f " % (
            epoch,
            i,
            n_batches,
            int(time.time() - btic),
            loss_rel_val / (i + 1),
        )
        for metric in metric_list:
            metric_name, metric_val = metric.get()
            print_txt += "%s=%.4f " % (metric_name, metric_val)
        logger.info(print_txt)
        btic = time.time()
        loss_rel_val = 0
    return btic, loss_rel_val


for epoch in range(nepoch):
    loss_rel_val = 0
    tic = time.time()
    btic = time.time()
    for metric in metric_list:
        metric.reset()
    if epoch == 0:
        net_trainer_base_lr = net_trainer.learning_rate
        det_trainer_base_lr = det_trainer.learning_rate
    if epoch == 5 or epoch == 8:
        net_trainer.set_learning_rate(net_trainer.learning_rate * 0.1)
        det_trainer.set_learning_rate(det_trainer.learning_rate * 0.1)
    for i, (G_list, img_list) in enumerate(train_data):
        if epoch == 0 and i < args.lr_warmup_iters:
            alpha = i / args.lr_warmup_iters
            warmup_factor = 1 / 3 * (1 - alpha) + alpha
            net_trainer.set_learning_rate(net_trainer_base_lr * warmup_factor)
            det_trainer.set_learning_rate(det_trainer_base_lr * warmup_factor)
        G_list, img_list = get_data_batch(G_list, img_list, ctx)
        if G_list is None or img_list is None:
            btic, loss_rel_val = batch_print(
                epoch,
                i,
                batch_verbose_freq,
                n_batches,
                btic,
                loss_rel_val,
                metric_list,
            )
            continue

        loss = []
        detector_res_list = []
        G_batch = []
        bbox_pad = Pad(axis=(0))
        with mx.autograd.record():
            for G_slice, img in zip(G_list, img_list):
                cur_ctx = img.context
                bbox_list = [G.ndata["bbox"] for G in G_slice]
                bbox_stack = bbox_pad(bbox_list).as_in_context(cur_ctx)
                with mx.autograd.pause():
                    ids, scores, bbox, feat, feat_ind, spatial_feat = detector(
                        img
                    )
                g_pred_batch = build_graph_train(
                    G_slice,
                    bbox_stack,
                    img,
                    ids,
                    scores,
                    bbox,
                    feat_ind,
                    spatial_feat,
                    scores_top_k=300,
                    overlap=False,
                )
                g_batch = l0_sample(g_pred_batch)
                if g_batch is None:
                    continue
                rel_bbox = g_batch.edata["rel_bbox"]
                batch_id = g_batch.edata["batch_id"].asnumpy()
                n_sample_edges = g_batch.number_of_edges()
                n_graph = len(G_slice)
                bbox_rel_list = []
                for j in range(n_graph):
                    eids = np.where(batch_id == j)[0]
                    if len(eids) > 0:
                        bbox_rel_list.append(rel_bbox[eids])
                bbox_rel_stack = bbox_pad(bbox_rel_list).as_in_context(cur_ctx)
                img_size = img.shape[2:4]
                bbox_rel_stack[:, :, 0] *= img_size[1]
                bbox_rel_stack[:, :, 1] *= img_size[0]
                bbox_rel_stack[:, :, 2] *= img_size[1]
                bbox_rel_stack[:, :, 3] *= img_size[0]
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
                g_batch.edata["edge_feat"] = nd.concat(
                    *spatial_feat_rel_list, dim=0
                )

                G_batch.append(g_batch)

            G_batch = [net(G) for G in G_batch]

            for G_pred, img in zip(G_batch, img_list):
                if G_pred is None or G_pred.number_of_nodes() == 0:
                    continue
                loss_rel = L_rel(
                    G_pred.edata["preds"],
                    G_pred.edata["rel_class"],
                    G_pred.edata["sample_weights"],
                )
                loss.append(loss_rel.sum())
                loss_rel_val += loss_rel.mean().asscalar() / num_gpus

        if len(loss) == 0:
            btic, loss_rel_val = batch_print(
                epoch,
                i,
                batch_verbose_freq,
                n_batches,
                btic,
                loss_rel_val,
                metric_list,
            )
            continue
        for l in loss:
            l.backward()
        if (i + 1) % per_device_batch_size == 0 or i == n_batches - 1:
            net_trainer.step(args.batch_size)
            det_trainer.step(args.batch_size)
            if aggregate_grad:
                for k, v in net_params.items():
                    v.zero_grad()
                for k, v in det_params.items():
                    v.zero_grad()
        for G_pred, img_slice in zip(G_batch, img_list):
            if G_pred is None or G_pred.number_of_nodes() == 0:
                continue
            link_ind = np.where(G_pred.edata["rel_class"].asnumpy() > 0)[0]
            if len(link_ind) == 0:
                continue
            train_metric.update(
                [G_pred.edata["rel_class"][link_ind]],
                [G_pred.edata["preds"][link_ind]],
            )
            train_metric_top5.update(
                [G_pred.edata["rel_class"][link_ind]],
                [G_pred.edata["preds"][link_ind]],
            )
        btic, loss_rel_val = batch_print(
            epoch,
            i,
            batch_verbose_freq,
            n_batches,
            btic,
            loss_rel_val,
            metric_list,
        )
        if (i + 1) % batch_verbose_freq == 0:
            net.save_parameters("%s/model-%d.params" % (save_dir, epoch))
            detector_feat.features.save_parameters(
                "%s/detector_feat.features-%d.params" % (save_dir, epoch)
            )
    print_txt = "Epoch[%d], time: %d, loss_rel=%.4f," % (
        epoch,
        int(time.time() - tic),
        loss_rel_val / (i + 1),
    )
    for metric in metric_list:
        metric_name, metric_val = metric.get()
        print_txt += "%s=%.4f " % (metric_name, metric_val)
    logger.info(print_txt)
    net.save_parameters("%s/model-%d.params" % (save_dir, epoch))
    detector_feat.features.save_parameters(
        "%s/detector_feat.features-%d.params" % (save_dir, epoch)
    )
