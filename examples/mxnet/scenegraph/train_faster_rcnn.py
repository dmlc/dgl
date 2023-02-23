"""Train Faster-RCNN end to end."""
import argparse
import os

# disable autotune
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
import logging
import time

import gluoncv as gcv
import mxnet as mx
import numpy as np
from data import *
from gluoncv import data as gdata, utils as gutils
from gluoncv.data.batchify import Append, FasterRCNNTrainBatchify, Tuple
from gluoncv.data.transforms.presets.rcnn import (
    FasterRCNNDefaultTrainTransform,
    FasterRCNNDefaultValTransform,
)
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.rcnn import (
    RCNNAccMetric,
    RCNNL1LossMetric,
    RPNAccMetric,
    RPNL1LossMetric,
)
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.parallel import Parallel, Parallelizable
from model import (
    faster_rcnn_resnet101_v1d_custom,
    faster_rcnn_resnet50_v1b_custom,
)
from mxnet import autograd, gluon
from mxnet.contrib import amp

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Faster-RCNN networks e2e."
    )
    parser.add_argument(
        "--network",
        type=str,
        default="resnet101_v1d",
        help="Base network name which serves as feature extraction base.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="visualgenome",
        help="Training dataset. Now support voc and coco.",
    )
    parser.add_argument(
        "--num-workers",
        "-j",
        dest="num_workers",
        type=int,
        default=8,
        help="Number of data workers, you can use larger "
        "number to accelerate data loading, "
        "if your CPU and GPUs are powerful.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Training mini-batch size."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Training with GPUs, you can specify 1,3 for example.",
    )
    parser.add_argument(
        "--epochs", type=str, default="", help="Training epochs."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from previously saved parameters if not None. "
        "For example, you can resume from ./faster_rcnn_xxx_0123.params",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Starting epoch for resuming, default is 0 for new training."
        "You can specify it to 100 for example to start from 100 epoch.",
    )
    parser.add_argument(
        "--lr",
        type=str,
        default="",
        help="Learning rate, default is 0.001 for voc single gpu training.",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="decay rate of learning rate. default is 0.1.",
    )
    parser.add_argument(
        "--lr-decay-epoch",
        type=str,
        default="",
        help="epochs at which learning rate decays. default is 14,20 for voc.",
    )
    parser.add_argument(
        "--lr-warmup",
        type=str,
        default="",
        help="warmup iterations to adjust learning rate, default is 0 for voc.",
    )
    parser.add_argument(
        "--lr-warmup-factor",
        type=float,
        default=1.0 / 3.0,
        help="warmup factor of base lr.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum, default is 0.9",
    )
    parser.add_argument(
        "--wd",
        type=str,
        default="",
        help="Weight decay, default is 5e-4 for voc",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging mini-batch interval. Default is 100.",
    )
    parser.add_argument(
        "--save-prefix", type=str, default="", help="Saving parameter prefix"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Saving parameters epoch interval, best model will always be saved.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="Epoch interval for validation, increase the number will reduce the "
        "training time if validation is slow.",
    )
    parser.add_argument(
        "--seed", type=int, default=233, help="Random seed to be fixed."
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print helpful debugging info once set.",
    )
    parser.add_argument(
        "--mixup", action="store_true", help="Use mixup training."
    )
    parser.add_argument(
        "--no-mixup-epochs",
        type=int,
        default=20,
        help="Disable mixup training if enabled in the last N epochs.",
    )

    # Norm layer options
    parser.add_argument(
        "--norm-layer",
        type=str,
        default=None,
        help="Type of normalization layer to use. "
        "If set to None, backbone normalization layer will be fixed,"
        " and no normalization layer will be used. "
        "Currently supports 'bn', and None, default is None."
        "Note that if horovod is enabled, sync bn will not work correctly.",
    )

    # FPN options
    parser.add_argument(
        "--use-fpn",
        action="store_true",
        help="Whether to use feature pyramid network.",
    )

    # Performance options
    parser.add_argument(
        "--disable-hybridization",
        action="store_true",
        help="Whether to disable hybridize the model. "
        "Memory usage and speed will decrese.",
    )
    parser.add_argument(
        "--static-alloc",
        action="store_true",
        help="Whether to use static memory allocation. Memory usage will increase.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use MXNet AMP for mixed precision training.",
    )
    parser.add_argument(
        "--horovod",
        action="store_true",
        help="Use MXNet Horovod for distributed training. Must be run with OpenMPI. "
        "--gpus is ignored when using --horovod.",
    )
    parser.add_argument(
        "--executor-threads",
        type=int,
        default=1,
        help="Number of threads for executor for scheduling ops. "
        "More threads may incur higher GPU memory footprint, "
        "but may speed up throughput. Note that when horovod is used, "
        "it is set to 1.",
    )
    parser.add_argument(
        "--kv-store",
        type=str,
        default="nccl",
        help="KV store options. local, device, nccl, dist_sync, dist_device_sync, "
        "dist_async are available.",
    )

    args = parser.parse_args()

    if args.horovod:
        if hvd is None:
            raise SystemExit(
                "Horovod not found, please check if you installed it correctly."
            )
        hvd.init()

    if args.dataset == "voc":
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = (
            args.lr_decay_epoch if args.lr_decay_epoch else "14,20"
        )
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    elif args.dataset == "visualgenome":
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = (
            args.lr_decay_epoch if args.lr_decay_epoch else "14,20"
        )
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    elif args.dataset == "coco":
        args.epochs = int(args.epochs) if args.epochs else 26
        args.lr_decay_epoch = (
            args.lr_decay_epoch if args.lr_decay_epoch else "17,23"
        )
        args.lr = float(args.lr) if args.lr else 0.01
        args.lr_warmup = args.lr_warmup if args.lr_warmup else 1000
        args.wd = float(args.wd) if args.wd else 1e-4
    return args


def get_dataset(dataset, args):
    if dataset.lower() == "voc":
        train_dataset = gdata.VOCDetection(
            splits=[(2007, "trainval"), (2012, "trainval")]
        )
        val_dataset = gdata.VOCDetection(splits=[(2007, "test")])
        val_metric = VOC07MApMetric(
            iou_thresh=0.5, class_names=val_dataset.classes
        )
    elif dataset.lower() == "coco":
        train_dataset = gdata.COCODetection(
            splits="instances_train2017", use_crowd=False
        )
        val_dataset = gdata.COCODetection(
            splits="instances_val2017", skip_empty=False
        )
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + "_eval", cleanup=True
        )
    elif dataset.lower() == "visualgenome":
        train_dataset = VGObject(
            root=os.path.join("~", ".mxnet", "datasets", "visualgenome"),
            splits="detections_train",
            use_crowd=False,
        )
        val_dataset = VGObject(
            root=os.path.join("~", ".mxnet", "datasets", "visualgenome"),
            splits="detections_val",
            skip_empty=False,
        )
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + "_eval", cleanup=True
        )
    else:
        raise NotImplementedError(
            "Dataset: {} not implemented.".format(dataset)
        )
    if args.mixup:
        from gluoncv.data.mixup import detection

        train_dataset = detection.MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric


def get_dataloader(
    net,
    train_dataset,
    val_dataset,
    train_transform,
    val_transform,
    batch_size,
    num_shards,
    args,
):
    """Get dataloader."""
    train_bfn = FasterRCNNTrainBatchify(net, num_shards)
    if hasattr(train_dataset, "get_im_aspect_ratio"):
        im_aspect_ratio = train_dataset.get_im_aspect_ratio()
    else:
        im_aspect_ratio = [1.0] * len(train_dataset)
    train_sampler = gcv.nn.sampler.SplitSortedBucketSampler(
        im_aspect_ratio,
        batch_size,
        num_parts=hvd.size() if args.horovod else 1,
        part_index=hvd.rank() if args.horovod else 0,
        shuffle=True,
    )
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(
            train_transform(
                net.short,
                net.max_size,
                net,
                ashape=net.ashape,
                multi_stage=args.use_fpn,
            )
        ),
        batch_sampler=train_sampler,
        batchify_fn=train_bfn,
        num_workers=args.num_workers,
    )
    if val_dataset is None:
        val_loader = None
    else:
        val_bfn = Tuple(*[Append() for _ in range(3)])
        short = (
            net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
        )
        # validation use 1 sample per device
        val_loader = mx.gluon.data.DataLoader(
            val_dataset.transform(val_transform(short, net.max_size)),
            num_shards,
            False,
            batchify_fn=val_bfn,
            last_batch="keep",
            num_workers=args.num_workers,
        )
    return train_loader, val_loader


def save_params(
    net, logger, best_map, current_map, epoch, save_interval, prefix
):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info(
            "[Epoch {}] mAP {} higher than current best {} saving to {}".format(
                epoch, current_map, best_map, "{:s}_best.params".format(prefix)
            )
        )
        best_map[0] = current_map
        net.save_parameters("{:s}_best.params".format(prefix))
        with open(prefix + "_best_map.log", "a") as f:
            f.write("{:04d}:\t{:.4f}\n".format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info(
            "[Epoch {}] Saving parameters to {}".format(
                epoch,
                "{:s}_{:04d}_{:.4f}.params".format(prefix, epoch, current_map),
            )
        )
        net.save_parameters(
            "{:s}_{:04d}_{:.4f}.params".format(prefix, epoch, current_map)
        )


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        if isinstance(data, (list, tuple)):
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        else:
            new_data = [data.as_in_context(ctx_list[0])]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, args):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    if not args.disable_hybridization:
        # input format is differnet than training, thus rehybridization is needed.
        net.hybridize(static_alloc=args.static_alloc)
    for i, batch in enumerate(val_data):
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(
                y.slice_axis(axis=-1, begin=5, end=6)
                if y.shape[-1] > 5
                else None
            )

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(
            det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults
        ):
            eval_metric.update(
                det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff
            )
    return eval_metric.get()


def get_lr_at_iter(alpha, lr_warmup_factor=1.0 / 3.0):
    return lr_warmup_factor * (1 - alpha) + alpha


class ForwardBackwardTask(Parallelizable):
    def __init__(
        self,
        net,
        optimizer,
        rpn_cls_loss,
        rpn_box_loss,
        rcnn_cls_loss,
        rcnn_box_loss,
        mix_ratio,
    ):
        super(ForwardBackwardTask, self).__init__()
        self.net = net
        self._optimizer = optimizer
        self.rpn_cls_loss = rpn_cls_loss
        self.rpn_box_loss = rpn_box_loss
        self.rcnn_cls_loss = rcnn_cls_loss
        self.rcnn_box_loss = rcnn_box_loss
        self.mix_ratio = mix_ratio

    def forward_backward(self, x):
        data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks = x
        with autograd.record():
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]
            (
                cls_pred,
                box_pred,
                roi,
                samples,
                matches,
                rpn_score,
                rpn_box,
                anchors,
                cls_targets,
                box_targets,
                box_masks,
                _,
            ) = net(data, gt_box, gt_label)
            # losses of rpn
            rpn_score = rpn_score.squeeze(axis=-1)
            num_rpn_pos = (rpn_cls_targets >= 0).sum()
            rpn_loss1 = (
                self.rpn_cls_loss(
                    rpn_score, rpn_cls_targets, rpn_cls_targets >= 0
                )
                * rpn_cls_targets.size
                / num_rpn_pos
            )
            rpn_loss2 = (
                self.rpn_box_loss(rpn_box, rpn_box_targets, rpn_box_masks)
                * rpn_box.size
                / num_rpn_pos
            )
            # rpn overall loss, use sum rather than average
            rpn_loss = rpn_loss1 + rpn_loss2
            # losses of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss1 = (
                self.rcnn_cls_loss(
                    cls_pred, cls_targets, cls_targets.expand_dims(-1) >= 0
                )
                * cls_targets.size
                / num_rcnn_pos
            )
            rcnn_loss2 = (
                self.rcnn_box_loss(box_pred, box_targets, box_masks)
                * box_pred.size
                / num_rcnn_pos
            )
            rcnn_loss = rcnn_loss1 + rcnn_loss2
            # overall losses
            total_loss = (
                rpn_loss.sum() * self.mix_ratio
                + rcnn_loss.sum() * self.mix_ratio
            )

            rpn_loss1_metric = rpn_loss1.mean() * self.mix_ratio
            rpn_loss2_metric = rpn_loss2.mean() * self.mix_ratio
            rcnn_loss1_metric = rcnn_loss1.mean() * self.mix_ratio
            rcnn_loss2_metric = rcnn_loss2.mean() * self.mix_ratio
            rpn_acc_metric = [
                [rpn_cls_targets, rpn_cls_targets >= 0],
                [rpn_score],
            ]
            rpn_l1_loss_metric = [[rpn_box_targets, rpn_box_masks], [rpn_box]]
            rcnn_acc_metric = [[cls_targets], [cls_pred]]
            rcnn_l1_loss_metric = [[box_targets, box_masks], [box_pred]]

            if args.amp:
                with amp.scale_loss(
                    total_loss, self._optimizer
                ) as scaled_losses:
                    autograd.backward(scaled_losses)
            else:
                total_loss.backward()

        return (
            rpn_loss1_metric,
            rpn_loss2_metric,
            rcnn_loss1_metric,
            rcnn_loss2_metric,
            rpn_acc_metric,
            rpn_l1_loss_metric,
            rcnn_acc_metric,
            rcnn_l1_loss_metric,
        )


def train(net, train_data, val_data, eval_metric, batch_size, ctx, args):
    """Training pipeline"""
    args.kv_store = (
        "device" if (args.amp and "nccl" in args.kv_store) else args.kv_store
    )
    kv = mx.kvstore.create(args.kv_store)
    net.collect_params().setattr("grad_req", "null")
    net.collect_train_params().setattr("grad_req", "write")
    optimizer_params = {
        "learning_rate": args.lr,
        "wd": args.wd,
        "momentum": args.momentum,
    }
    if args.horovod:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            "sgd",
            optimizer_params,
        )
    else:
        trainer = gluon.Trainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            "sgd",
            optimizer_params,
            update_on_kvstore=(False if args.amp else None),
            kvstore=kv,
        )

    if args.amp:
        amp.init_trainer(trainer)

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted(
        [float(ls) for ls in args.lr_decay_epoch.split(",") if ls.strip()]
    )
    lr_warmup = float(args.lr_warmup)  # avoid int division

    # TODO(zhreshold) losses?
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(
        from_sigmoid=False
    )
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.0)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1
    metrics = [
        mx.metric.Loss("RPN_Conf"),
        mx.metric.Loss("RPN_SmoothL1"),
        mx.metric.Loss("RCNN_CrossEntropy"),
        mx.metric.Loss("RCNN_SmoothL1"),
    ]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [
        rpn_acc_metric,
        rpn_bbox_metric,
        rcnn_acc_metric,
        rcnn_bbox_metric,
    ]

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + "_train.log"
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info("Trainable parameters:")
        logger.info(net.collect_train_params().keys())
    logger.info("Start training from [Epoch {}]".format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        mix_ratio = 1.0
        if not args.disable_hybridization:
            net.hybridize(static_alloc=args.static_alloc)
        rcnn_task = ForwardBackwardTask(
            net,
            trainer,
            rpn_cls_loss,
            rpn_box_loss,
            rcnn_cls_loss,
            rcnn_box_loss,
            mix_ratio=1.0,
        )
        executor = (
            Parallel(args.executor_threads, rcnn_task)
            if not args.horovod
            else None
        )
        if args.mixup:
            # TODO(zhreshold) only support evenly mixup now, target generator needs to be modified otherwise
            train_data._dataset._data.set_mixup(np.random.uniform, 0.5, 0.5)
            mix_ratio = 0.5
            if epoch >= args.epochs - args.no_mixup_epochs:
                train_data._dataset._data.set_mixup(None)
                mix_ratio = 1.0
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info(
                "[Epoch {}] Set learning rate to {}".format(epoch, new_lr)
            )
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        base_lr = trainer.learning_rate
        rcnn_task.mix_ratio = mix_ratio
        logger.info("Total Num of Batches: %d" % (len(train_data)))
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(
                    i / lr_warmup, args.lr_warmup_factor
                )
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            "[Epoch 0 Iteration {}] Set learning rate to {}".format(
                                i, new_lr
                            )
                        )
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            if executor is not None:
                for data in zip(*batch):
                    executor.put(data)
            for j in range(len(ctx)):
                if executor is not None:
                    result = executor.get()
                else:
                    result = rcnn_task.forward_backward(list(zip(*batch))[0])
                if (not args.horovod) or hvd.rank() == 0:
                    for k in range(len(metric_losses)):
                        metric_losses[k].append(result[k])
                    for k in range(len(add_losses)):
                        add_losses[k].append(result[len(metric_losses) + k])
            for metric, record in zip(metrics, metric_losses):
                metric.update(0, record)
            for metric, records in zip(metrics2, add_losses):
                for pred in records:
                    metric.update(pred[0], pred[1])
            trainer.step(batch_size)

            # update metrics
            if (
                (not args.horovod or hvd.rank() == 0)
                and args.log_interval
                and not (i + 1) % args.log_interval
            ):
                msg = ",".join(
                    [
                        "{}={:.3f}".format(*metric.get())
                        for metric in metrics + metrics2
                    ]
                )
                logger.info(
                    "[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}".format(
                        epoch,
                        i,
                        args.log_interval
                        * args.batch_size
                        / (time.time() - btic),
                        msg,
                    )
                )
                btic = time.time()

        if (not args.horovod) or hvd.rank() == 0:
            msg = ",".join(
                ["{}={:.3f}".format(*metric.get()) for metric in metrics]
            )
            logger.info(
                "[Epoch {}] Training cost: {:.3f}, {}".format(
                    epoch, (time.time() - tic), msg
                )
            )
            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                if val_data is not None:
                    map_name, mean_ap = validate(
                        net, val_data, ctx, eval_metric, args
                    )
                    val_msg = "\n".join(
                        [
                            "{}={}".format(k, v)
                            for k, v in zip(map_name, mean_ap)
                        ]
                    )
                    logger.info(
                        "[Epoch {}] Validation: \n{}".format(epoch, val_msg)
                    )
                    current_map = float(mean_ap[-1])
                else:
                    current_map = 0
            else:
                current_map = 0.0
            save_params(
                net,
                logger,
                best_map,
                current_map,
                epoch,
                args.save_interval,
                args.save_prefix,
            )


if __name__ == "__main__":
    import sys

    sys.setrecursionlimit(1100)
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    if args.amp:
        amp.init()

    # training contexts
    if args.horovod:
        ctx = [mx.gpu(hvd.local_rank())]
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(",") if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]

    # network
    kwargs = {}
    module_list = []
    if args.use_fpn:
        module_list.append("fpn")
    if args.norm_layer is not None:
        module_list.append(args.norm_layer)
        if args.norm_layer == "bn":
            kwargs["num_devices"] = len(args.gpus.split(","))

    net_name = "_".join(("faster_rcnn", *module_list, args.network, "custom"))
    args.save_prefix += net_name
    gutils.makedirs(args.save_prefix)
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    net = faster_rcnn_resnet101_v1d_custom(
        classes=train_dataset.classes,
        transfer="coco",
        pretrained_base=False,
        additional_output=False,
        per_device_batch_size=args.batch_size // len(ctx),
        **kwargs
    )
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # training data
    batch_size = (
        args.batch_size // len(ctx) if args.horovod else args.batch_size
    )
    train_data, val_data = get_dataloader(
        net,
        train_dataset,
        val_dataset,
        FasterRCNNDefaultTrainTransform,
        FasterRCNNDefaultValTransform,
        batch_size,
        len(ctx),
        args,
    )

    # training
    train(net, train_data, val_data, eval_metric, batch_size, ctx, args)
