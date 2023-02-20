"""Faster RCNN Model."""
from __future__ import absolute_import

import os
import warnings

import mxnet as mx

from gluoncv.model_zoo.faster_rcnn.rcnn_target import (
    RCNNTargetGenerator,
    RCNNTargetSampler,
)
from gluoncv.model_zoo.rcnn import RCNN
from gluoncv.model_zoo.rpn import RPN
from gluoncv.nn.feature import FPNFeatureExpander
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import SyncBatchNorm

__all__ = [
    "FasterRCNN",
    "get_faster_rcnn",
    "faster_rcnn_resnet50_v1b_coco",
    "faster_rcnn_resnet50_v1b_custom",
    "faster_rcnn_resnet101_v1d_coco",
    "faster_rcnn_resnet101_v1d_custom",
]


class FasterRCNN(RCNN):
    r"""Faster RCNN network.

    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    box_features : gluon.HybridBlock, default is None
        feature head for transforming shared ROI output (top_features) for box prediction.
        If set to None, global average pooling will be used.
    short : int, default is 600.
        Input image short side size.
    max_size : int, default is 1000.
        Maximum size of input image long side.
    min_stage : int, default is 4
        Minimum stage NO. for FPN stages.
    max_stage : int, default is 4
        Maximum stage NO. for FPN stages.
    train_patterns : str, default is None.
        Matching pattern for trainable parameters.
    nms_thresh : float, default is 0.3.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    roi_mode : str, default is align
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2, default is (14, 14)
        (height, width) of the ROI region.
    strides : int/tuple of ints, default is 16
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
        For FPN, use a tuple of ints.
    clip : float, default is None
        Clip bounding box target to this value.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    base_size : int
        The width(and height) of reference anchor box.
    scales : iterable of float, default is (8, 16, 32)
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    ratios : iterable of float, default is (0.5, 1, 2)
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    rpn_train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training of RPN.
    rpn_train_post_nms : int, default is 2000
        Return top proposal results after NMS in training of RPN.
        Will be set to rpn_train_pre_nms if it is larger than rpn_train_pre_nms.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
        Will be set to rpn_test_pre_nms if it is larger than rpn_test_pre_nms.
    rpn_nms_thresh : float, default is 0.7
        IOU threshold for NMS. It is used to remove overlapping proposals.
    rpn_num_sample : int, default is 256
        Number of samples for RPN targets.
    rpn_pos_iou_thresh : float, default is 0.7
        Anchor with IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    rpn_neg_iou_thresh : float, default is 0.3
        Anchor with IOU smaller than ``neg_iou_thresh`` is regarded as negative samples.
        Anchors with IOU in between ``pos_iou_thresh`` and ``neg_iou_thresh`` are
        ignored.
    rpn_pos_ratio : float, default is 0.5
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    rpn_box_norm : array-like of size 4, default is (1., 1., 1., 1.)
        Std value to be divided from encoded values.
    rpn_min_size : int, default is 16
        Proposals whose size is smaller than ``min_size`` will be discarded.
    per_device_batch_size : int, default is 1
        Batch size for each device during training.
    num_sample : int, default is 128
        Number of samples for RCNN targets.
    pos_iou_thresh : float, default is 0.5
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    pos_ratio : float, default is 0.25
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    max_num_gt : int, default is 300
        Maximum ground-truth number in whole training dataset. This is only an upper bound, not
        necessarily very precise. However, using a very big number may impact the training speed.
    additional_output : boolean, default is False
        ``additional_output`` is only used for Mask R-CNN to get internal outputs.
    force_nms : bool, default is False
        Appy NMS to all categories, this is to avoid overlapping detection results from different
        categories.

    Attributes
    ----------
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    num_class : int
        Number of positive categories.
    short : int
        Input image short side size.
    max_size : int
        Maximum size of input image long side.
    train_patterns : str
        Matching pattern for trainable parameters.
    nms_thresh : float
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    force_nms : bool
        Appy NMS to all categories, this is to avoid overlapping detection results
        from different categories.
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    rpn_target_generator : gluon.Block
        Generate training targets with cls_target, box_target, and box_mask.
    target_generator : gluon.Block
        Generate training targets with boxes, samples, matches, gt_label and gt_box.

    """

    def __init__(
        self,
        features,
        top_features,
        classes,
        box_features=None,
        short=600,
        max_size=1000,
        min_stage=4,
        max_stage=4,
        train_patterns=None,
        nms_thresh=0.3,
        nms_topk=400,
        post_nms=100,
        roi_mode="align",
        roi_size=(14, 14),
        strides=16,
        clip=None,
        rpn_channel=1024,
        base_size=16,
        scales=(8, 16, 32),
        ratios=(0.5, 1, 2),
        alloc_size=(128, 128),
        rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000,
        rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000,
        rpn_test_post_nms=300,
        rpn_min_size=16,
        per_device_batch_size=1,
        num_sample=128,
        pos_iou_thresh=0.5,
        pos_ratio=0.25,
        max_num_gt=300,
        additional_output=False,
        force_nms=False,
        **kwargs
    ):
        super(FasterRCNN, self).__init__(
            features=features,
            top_features=top_features,
            classes=classes,
            box_features=box_features,
            short=short,
            max_size=max_size,
            train_patterns=train_patterns,
            nms_thresh=nms_thresh,
            nms_topk=nms_topk,
            post_nms=post_nms,
            roi_mode=roi_mode,
            roi_size=roi_size,
            strides=strides,
            clip=clip,
            force_nms=force_nms,
            **kwargs
        )
        if rpn_train_post_nms > rpn_train_pre_nms:
            rpn_train_post_nms = rpn_train_pre_nms
        if rpn_test_post_nms > rpn_test_pre_nms:
            rpn_test_post_nms = rpn_test_pre_nms

        self.ashape = alloc_size[0]
        self._min_stage = min_stage
        self._max_stage = max_stage
        self.num_stages = max_stage - min_stage + 1
        if self.num_stages > 1:
            assert len(scales) == len(strides) == self.num_stages, (
                "The num_stages (%d) must match number of scales (%d) and strides (%d)"
                % (self.num_stages, len(scales), len(strides))
            )
        self._batch_size = per_device_batch_size
        self._num_sample = num_sample
        self._rpn_test_post_nms = rpn_test_post_nms
        self._target_generator = RCNNTargetGenerator(
            self.num_class, int(num_sample * pos_ratio), self._batch_size
        )
        self._additional_output = additional_output
        with self.name_scope():
            self.rpn = RPN(
                channels=rpn_channel,
                strides=strides,
                base_size=base_size,
                scales=scales,
                ratios=ratios,
                alloc_size=alloc_size,
                clip=clip,
                nms_thresh=rpn_nms_thresh,
                train_pre_nms=rpn_train_pre_nms,
                train_post_nms=rpn_train_post_nms,
                test_pre_nms=rpn_test_pre_nms,
                test_post_nms=rpn_test_post_nms,
                min_size=rpn_min_size,
                multi_level=self.num_stages > 1,
                per_level_nms=False,
            )
            self.sampler = RCNNTargetSampler(
                num_image=self._batch_size,
                num_proposal=rpn_train_post_nms,
                num_sample=num_sample,
                pos_iou_thresh=pos_iou_thresh,
                pos_ratio=pos_ratio,
                max_num_gt=max_num_gt,
            )

    @property
    def target_generator(self):
        """Returns stored target generator

        Returns
        -------
        mxnet.gluon.HybridBlock
            The RCNN target generator

        """
        return self._target_generator

    def reset_class(self, classes, reuse_weights=None):
        """Reset class categories and class predictors.

        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.
        reuse_weights : dict
            A {new_integer : old_integer} or mapping dict or {new_name : old_name} mapping dict,
            or a list of [name0, name1,...] if class names don't change.
            This allows the new predictor to reuse the
            previously trained weights specified.

        Example
        -------
        >>> net = gluoncv.model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)
        >>> # use direct name to name mapping to reuse weights
        >>> net.reset_class(classes=['person'], reuse_weights={'person':'person'})
        >>> # or use interger mapping, person is the 14th category in VOC
        >>> net.reset_class(classes=['person'], reuse_weights={0:14})
        >>> # you can even mix them
        >>> net.reset_class(classes=['person'], reuse_weights={'person':14})
        >>> # or use a list of string if class name don't change
        >>> net.reset_class(classes=['person'], reuse_weights=['person'])

        """
        super(FasterRCNN, self).reset_class(classes, reuse_weights)
        self._target_generator = RCNNTargetGenerator(
            self.num_class, self.sampler._max_pos, self._batch_size
        )

    def _pyramid_roi_feats(
        self,
        F,
        features,
        rpn_rois,
        roi_size,
        strides,
        roi_mode="align",
        roi_canonical_scale=224.0,
        eps=1e-6,
    ):
        """Assign rpn_rois to specific FPN layers according to its area
           and then perform `ROIPooling` or `ROIAlign` to generate final
           region proposals aggregated features.
        Parameters
        ----------
        features : list of mx.ndarray or mx.symbol
            Features extracted from FPN base network
        rpn_rois : mx.ndarray or mx.symbol
            (N, 5) with [[batch_index, x1, y1, x2, y2], ...] like
        roi_size : tuple
            The size of each roi with regard to ROI-Wise operation
            each region proposal will be roi_size spatial shape.
        strides : tuple e.g. [4, 8, 16, 32]
            Define the gap that ori image and feature map have
        roi_mode : str, default is align
            ROI pooling mode. Currently support 'pool' and 'align'.
        roi_canonical_scale : float, default is 224.0
            Hyperparameters for the RoI-to-FPN level mapping heuristic.
        Returns
        -------
        Pooled roi features aggregated according to its roi_level
        """
        max_stage = self._max_stage
        if self._max_stage > 5:  # do not use p6 for RCNN
            max_stage = self._max_stage - 1
        _, x1, y1, x2, y2 = F.split(rpn_rois, axis=-1, num_outputs=5)
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        roi_level = F.floor(
            4 + F.log2(F.sqrt(w * h) / roi_canonical_scale + eps)
        )
        roi_level = F.squeeze(F.clip(roi_level, self._min_stage, max_stage))
        # [2,2,..,3,3,...,4,4,...,5,5,...] ``Prohibit swap order here``
        # roi_level_sorted_args = F.argsort(roi_level, is_ascend=True)
        # roi_level = F.sort(roi_level, is_ascend=True)
        # rpn_rois = F.take(rpn_rois, roi_level_sorted_args, axis=0)
        pooled_roi_feats = []
        for i, l in enumerate(range(self._min_stage, max_stage + 1)):
            if roi_mode == "pool":
                # Pool features with all rois first, and then set invalid pooled features to zero,
                # at last ele-wise add together to aggregate all features.
                pooled_feature = F.ROIPooling(
                    features[i], rpn_rois, roi_size, 1.0 / strides[i]
                )
                pooled_feature = F.where(
                    roi_level == l, pooled_feature, F.zeros_like(pooled_feature)
                )
            elif roi_mode == "align":
                if (
                    "box_encode" in F.contrib.__dict__
                    and "box_decode" in F.contrib.__dict__
                ):
                    # TODO(jerryzcn): clean this up for once mx 1.6 is released.
                    masked_rpn_rois = F.where(
                        roi_level == l, rpn_rois, F.ones_like(rpn_rois) * -1.0
                    )
                    pooled_feature = F.contrib.ROIAlign(
                        features[i],
                        masked_rpn_rois,
                        roi_size,
                        1.0 / strides[i],
                        sample_ratio=2,
                    )
                else:
                    pooled_feature = F.contrib.ROIAlign(
                        features[i],
                        rpn_rois,
                        roi_size,
                        1.0 / strides[i],
                        sample_ratio=2,
                    )
                    pooled_feature = F.where(
                        roi_level == l,
                        pooled_feature,
                        F.zeros_like(pooled_feature),
                    )
            else:
                raise ValueError("Invalid roi mode: {}".format(roi_mode))
            pooled_roi_feats.append(pooled_feature)
        # Ele-wise add to aggregate all pooled features
        pooled_roi_feats = F.ElementWiseSum(*pooled_roi_feats)
        # Sort all pooled features by asceding order
        # [2,2,..,3,3,...,4,4,...,5,5,...]
        # pooled_roi_feats = F.take(pooled_roi_feats, roi_level_sorted_args)
        # pooled roi feats (B*N, C, 7, 7), N = N2 + N3 + N4 + N5 = num_roi, C=256 in ori paper
        return pooled_roi_feats

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, gt_box=None, gt_label=None, m_rpn_box=None):
        """Forward Faster-RCNN network.

        The behavior during training and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : type, only required during training
            The ground-truth bbox tensor with shape (B, N, 4).
        gt_label : type, only required during training
            The ground-truth label tensor with shape (B, 1, 4).

        Returns
        -------
        (ids, scores, bboxes)
            During inference, returns final class id, confidence scores, bounding
            boxes.

        """

        def _split(x, axis, num_outputs, squeeze_axis):
            x = F.split(
                x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis
            )
            if isinstance(x, list):
                return x
            else:
                return [x]

        if m_rpn_box is not None:
            manual_rpn_box = True
        else:
            manual_rpn_box = False
        feat = self.features(x)
        if not isinstance(feat, (list, tuple)):
            feat = [feat]

        # RPN proposals
        if autograd.is_training():
            if manual_rpn_box:
                rpn_box = m_rpn_box
                self.nms_thresh = 1
            else:
                (
                    rpn_score,
                    rpn_box,
                    raw_rpn_score,
                    raw_rpn_box,
                    anchors,
                ) = self.rpn(F.zeros_like(x), *feat)
                rpn_box, samples, matches = self.sampler(
                    rpn_box, rpn_score, gt_box
                )
        else:
            if manual_rpn_box:
                rpn_box = m_rpn_box
                self.nms_thresh = 1
            else:
                _, rpn_box = self.rpn(F.zeros_like(x), *feat)

        # create batchid for roi
        if not manual_rpn_box:
            num_roi = (
                self._num_sample
                if autograd.is_training()
                else self._rpn_test_post_nms
            )
            batch_size = self._batch_size if autograd.is_training() else 1
        else:
            num_roi = m_rpn_box.shape[1]
            batch_size = rpn_box.shape[0]

        with autograd.pause():
            roi_batchid = F.arange(0, batch_size)
            roi_batchid = F.repeat(roi_batchid, num_roi)
            # remove batch dim because ROIPooling require 2d input
            rpn_roi = F.concat(
                *[roi_batchid.reshape((-1, 1)), rpn_box.reshape((-1, 4))],
                dim=-1
            )
            rpn_roi = F.stop_gradient(rpn_roi)

        if self.num_stages > 1:
            # using FPN
            pooled_feat = self._pyramid_roi_feats(
                F,
                feat,
                rpn_roi,
                self._roi_size,
                self._strides,
                roi_mode=self._roi_mode,
            )
        else:
            # ROI features
            if self._roi_mode == "pool":
                pooled_feat = F.ROIPooling(
                    feat[0], rpn_roi, self._roi_size, 1.0 / self._strides
                )
            elif self._roi_mode == "align":
                pooled_feat = F.contrib.ROIAlign(
                    feat[0],
                    rpn_roi,
                    self._roi_size,
                    1.0 / self._strides,
                    sample_ratio=2,
                )
            else:
                raise ValueError("Invalid roi mode: {}".format(self._roi_mode))

        # RCNN prediction
        if self.top_features is not None:
            top_feat = self.top_features(pooled_feat)
        else:
            top_feat = pooled_feat
        if self.box_features is None:
            box_feat = F.contrib.AdaptiveAvgPooling2D(top_feat, output_size=1)
        else:
            box_feat = self.box_features(top_feat)
        cls_pred = self.class_predictor(box_feat)
        # cls_pred (B * N, C) -> (B, N, C)
        cls_pred = cls_pred.reshape((batch_size, num_roi, self.num_class + 1))
        if manual_rpn_box:
            spatial_feat = top_feat.mean(axis=1).reshape(
                (-4, rpn_box.shape[0], rpn_box.shape[1], -3)
            )
            cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
            cls_ids = cls_ids.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
            scores = scores.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
            cls_ids = _split(
                cls_ids, axis=0, num_outputs=batch_size, squeeze_axis=True
            )
            scores = _split(
                scores, axis=0, num_outputs=batch_size, squeeze_axis=True
            )
            return cls_ids, scores, rpn_box, spatial_feat

        # no need to convert bounding boxes in training, just return
        if autograd.is_training():
            (
                cls_targets,
                box_targets,
                box_masks,
                indices,
            ) = self._target_generator(
                rpn_box, samples, matches, gt_label, gt_box
            )
            box_feat = F.reshape(box_feat.expand_dims(0), (batch_size, -1, 0))
            box_pred = self.box_predictor(
                F.concat(
                    *[
                        F.take(
                            F.slice_axis(
                                box_feat, axis=0, begin=i, end=i + 1
                            ).squeeze(),
                            F.slice_axis(
                                indices, axis=0, begin=i, end=i + 1
                            ).squeeze(),
                        )
                        for i in range(batch_size)
                    ],
                    dim=0
                )
            )
            # box_pred (B * N, C * 4) -> (B, N, C, 4)
            box_pred = box_pred.reshape((batch_size, -1, self.num_class, 4))
            if self._additional_output:
                return (
                    cls_pred,
                    box_pred,
                    rpn_box,
                    samples,
                    matches,
                    raw_rpn_score,
                    raw_rpn_box,
                    anchors,
                    cls_targets,
                    box_targets,
                    box_masks,
                    top_feat,
                    indices,
                )
            return (
                cls_pred,
                box_pred,
                rpn_box,
                samples,
                matches,
                raw_rpn_score,
                raw_rpn_box,
                anchors,
                cls_targets,
                box_targets,
                box_masks,
                indices,
            )

        box_pred = self.box_predictor(box_feat)
        # box_pred (B * N, C * 4) -> (B, N, C, 4)
        box_pred = box_pred.reshape((batch_size, num_roi, self.num_class, 4))
        # cls_ids (B, N, C), scores (B, N, C)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
        # cls_ids, scores (B, N, C) -> (B, C, N) -> (B, C, N, 1)
        cls_ids = cls_ids.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        scores = scores.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        # box_pred (B, N, C, 4) -> (B, C, N, 4)
        box_pred = box_pred.transpose((0, 2, 1, 3))

        # rpn_boxes (B, N, 4) -> B * (1, N, 4)
        rpn_boxes = _split(
            rpn_box, axis=0, num_outputs=batch_size, squeeze_axis=False
        )
        # cls_ids, scores (B, C, N, 1) -> B * (C, N, 1)
        cls_ids = _split(
            cls_ids, axis=0, num_outputs=batch_size, squeeze_axis=True
        )
        scores = _split(
            scores, axis=0, num_outputs=batch_size, squeeze_axis=True
        )
        # box_preds (B, C, N, 4) -> B * (C, N, 4)
        box_preds = _split(
            box_pred, axis=0, num_outputs=batch_size, squeeze_axis=True
        )

        # per batch predict, nms, each class has topk outputs
        results = []
        # add feat index
        if self._additional_output:
            sizes = scores[0].shape[0:2]
            # ind = mx.nd.array(list(range(sizes[1])))
            ind = mx.nd.linspace(0, 999, 1000)
            ind = mx.nd.repeat(ind, repeats=sizes[0])
            ind = (
                ind.reshape(sizes[1], sizes[0])
                .transpose((1, 0))
                .expand_dims(axis=2)
            )
        for rpn_box, cls_id, score, box_pred in zip(
            rpn_boxes, cls_ids, scores, box_preds
        ):
            # box_pred (C, N, 4) rpn_box (1, N, 4) -> bbox (C, N, 4)
            bbox = self.box_decoder(box_pred, rpn_box)
            if self._additional_output:
                # res (C, N, 7)
                res = F.concat(*[cls_id, score, bbox, ind], dim=-1)
            else:
                # res (C, N, 6)
                res = F.concat(*[cls_id, score, bbox], dim=-1)
            if self.force_nms:
                # res (1, C*N, 6), to allow cross-catogory suppression
                res = res.reshape((1, -1, 0))
            # res (C, self.nms_topk, 6)
            res = F.contrib.box_nms(
                res,
                overlap_thresh=self.nms_thresh,
                topk=self.nms_topk,
                valid_thresh=0.001,
                id_index=0,
                score_index=1,
                coord_start=2,
                force_suppress=self.force_nms,
            )
            # res (C * self.nms_topk, 6)
            res = res.reshape((-3, 0))
            results.append(res)

        # result B * (C * topk, 6) -> (B, C * topk, 6)
        result = F.stack(*results, axis=0)
        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        if self._additional_output:
            feat_ind = F.slice_axis(result, axis=-1, begin=6, end=7)
            spatial_feat = (
                top_feat.mean(axis=1).expand_dims(0).reshape(batch_size, 0, -1)
            )
            return ids, scores, bboxes, feat, feat_ind, spatial_feat
        return ids, scores, bboxes


def get_faster_rcnn(
    name,
    dataset,
    pretrained=False,
    ctx=mx.cpu(),
    root=os.path.join("~", ".mxnet", "models"),
    **kwargs
):
    r"""Utility function to return faster rcnn networks.

    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The Faster-RCNN network.

    """
    net = FasterRCNN(**kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file

        full_name = "_".join(("faster_rcnn", name, dataset))
        net.load_parameters(
            get_model_file(full_name, tag=pretrained, root=root),
            ctx=ctx,
            ignore_extra=True,
            allow_missing=True,
        )
    else:
        for v in net.collect_params().values():
            try:
                v.reset_ctx(ctx)
            except ValueError:
                pass
    return net


def faster_rcnn_resnet50_v1b_coco(
    pretrained=False, pretrained_base=True, **kwargs
):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from gluoncv.data import COCODetection
    from gluoncv.model_zoo.resnetv1b import resnet50_v1b

    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(
        pretrained=pretrained_base,
        dilated=False,
        use_global_stats=True,
        **kwargs
    )
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in [
        "conv1",
        "bn1",
        "relu",
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
    ]:
        features.add(getattr(base_network, layer))
    for layer in ["layer4"]:
        top_features.add(getattr(base_network, layer))
    train_patterns = "|".join(
        [".*dense", ".*rpn", ".*down(2|3|4)_conv", ".*layers(2|3|4)_conv"]
    )
    return get_faster_rcnn(
        name="resnet50_v1b",
        dataset="coco",
        pretrained=pretrained,
        features=features,
        top_features=top_features,
        classes=classes,
        short=800,
        max_size=1333,
        train_patterns=train_patterns,
        nms_thresh=0.7,
        nms_topk=-1,
        post_nms=-1,
        roi_mode="align",
        roi_size=(14, 14),
        strides=16,
        clip=4.14,
        rpn_channel=1024,
        base_size=16,
        scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2),
        alloc_size=(128, 128),
        rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000,
        rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000,
        rpn_test_post_nms=1000,
        rpn_min_size=1,
        num_sample=128,
        pos_iou_thresh=0.5,
        pos_ratio=0.25,
        max_num_gt=3000,
        **kwargs
    )


def faster_rcnn_resnet50_v1b_custom(
    classes, transfer=None, pretrained_base=True, pretrained=False, **kwargs
):
    r"""Faster RCNN model with resnet50_v1b base network on custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Hybrid faster RCNN network.
    """
    if pretrained:
        warnings.warn(
            "Custom models don't provide `pretrained` weights, ignored."
        )
    if transfer is None:
        from gluoncv.model_zoo.resnetv1b import resnet50_v1b

        base_network = resnet50_v1b(
            pretrained=pretrained_base,
            dilated=False,
            use_global_stats=True,
            **kwargs
        )
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
        ]:
            features.add(getattr(base_network, layer))
        for layer in ["layer4"]:
            top_features.add(getattr(base_network, layer))
        train_patterns = "|".join(
            [".*dense", ".*rpn", ".*down(2|3|4)_conv", ".*layers(2|3|4)_conv"]
        )
        return get_faster_rcnn(
            name="resnet50_v1b",
            dataset="custom",
            pretrained=pretrained,
            features=features,
            top_features=top_features,
            classes=classes,
            short=600,
            max_size=1000,
            train_patterns=train_patterns,
            nms_thresh=0.7,
            nms_topk=400,
            post_nms=100,
            roi_mode="align",
            roi_size=(14, 14),
            strides=16,
            clip=4.14,
            rpn_channel=1024,
            base_size=16,
            scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2),
            alloc_size=(128, 128),
            rpn_nms_thresh=0.7,
            rpn_train_pre_nms=12000,
            rpn_train_post_nms=2000,
            rpn_test_pre_nms=6000,
            rpn_test_post_nms=300,
            rpn_min_size=16,
            num_sample=128,
            pos_iou_thresh=0.5,
            pos_ratio=0.25,
            max_num_gt=3000,
            **kwargs
        )
    else:
        from gluoncv.model_zoo import get_model

        net = get_model(
            "faster_rcnn_resnet50_v1b_" + str(transfer),
            pretrained=True,
            **kwargs
        )
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net


def faster_rcnn_resnet101_v1d_coco(
    pretrained=False, pretrained_base=True, **kwargs
):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from gluoncv.data import COCODetection
    from gluoncv.model_zoo.resnetv1b import resnet101_v1d

    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(
        pretrained=pretrained_base,
        dilated=False,
        use_global_stats=True,
        **kwargs
    )
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in [
        "conv1",
        "bn1",
        "relu",
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
    ]:
        features.add(getattr(base_network, layer))
    for layer in ["layer4"]:
        top_features.add(getattr(base_network, layer))
    train_patterns = "|".join(
        [".*dense", ".*rpn", ".*down(2|3|4)_conv", ".*layers(2|3|4)_conv"]
    )
    return get_faster_rcnn(
        name="resnet101_v1d",
        dataset="coco",
        pretrained=pretrained,
        features=features,
        top_features=top_features,
        classes=classes,
        short=800,
        max_size=1333,
        train_patterns=train_patterns,
        nms_thresh=0.5,
        nms_topk=-1,
        post_nms=100,
        roi_mode="align",
        roi_size=(14, 14),
        strides=16,
        clip=4.14,
        rpn_channel=1024,
        base_size=16,
        scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2),
        alloc_size=(128, 128),
        rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000,
        rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000,
        rpn_test_post_nms=1000,
        rpn_min_size=1,
        num_sample=128,
        pos_iou_thresh=0.5,
        pos_ratio=0.25,
        max_num_gt=3000,
        **kwargs
    )


def faster_rcnn_resnet101_v1d_custom(
    classes, transfer=None, pretrained_base=True, pretrained=False, **kwargs
):
    r"""Faster RCNN model with resnet101_v1d base network on custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Hybrid faster RCNN network.
    """
    if pretrained:
        warnings.warn(
            "Custom models don't provide `pretrained` weights, ignored."
        )
    if transfer is None:
        from gluoncv.model_zoo.resnetv1b import resnet101_v1d

        base_network = resnet101_v1d(
            pretrained=pretrained_base,
            dilated=False,
            use_global_stats=True,
            **kwargs
        )
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
        ]:
            features.add(getattr(base_network, layer))
        for layer in ["layer4"]:
            top_features.add(getattr(base_network, layer))
        train_patterns = "|".join(
            [".*dense", ".*rpn", ".*down(2|3|4)_conv", ".*layers(2|3|4)_conv"]
        )
        return get_faster_rcnn(
            name="resnet101_v1d",
            dataset="custom",
            pretrained=pretrained,
            features=features,
            top_features=top_features,
            classes=classes,
            short=600,
            max_size=1000,
            train_patterns=train_patterns,
            nms_thresh=0.5,
            nms_topk=400,
            post_nms=100,
            roi_mode="align",
            roi_size=(14, 14),
            strides=16,
            clip=4.14,
            rpn_channel=1024,
            base_size=16,
            scales=(2, 4, 8, 16, 32),
            ratios=(0.5, 1, 2),
            alloc_size=(128, 128),
            rpn_nms_thresh=0.7,
            rpn_train_pre_nms=12000,
            rpn_train_post_nms=2000,
            rpn_test_pre_nms=6000,
            rpn_test_post_nms=300,
            rpn_min_size=16,
            num_sample=128,
            pos_iou_thresh=0.5,
            pos_ratio=0.25,
            max_num_gt=3000,
            **kwargs
        )
    else:
        net = faster_rcnn_resnet101_v1d_coco(pretrained=True)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net
