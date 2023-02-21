import argparse

import gluoncv as gcv
import mxnet as mx
from data import *
from gluoncv.data.transforms import presets
from gluoncv.utilz import download
from model import faster_rcnn_resnet101_v1d_custom, RelDN
from utils import *

import dgl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo of Scene Graph Extraction."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="The image for scene graph extraction.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="",
        help="GPU id to use for inference, default is not using GPU.",
    )
    parser.add_argument(
        "--pretrained-faster-rcnn-params",
        type=str,
        default="",
        help="Path to saved Faster R-CNN model parameters.",
    )
    parser.add_argument(
        "--reldn-params",
        type=str,
        default="",
        help="Path to saved Faster R-CNN model parameters.",
    )
    parser.add_argument(
        "--faster-rcnn-params",
        type=str,
        default="",
        help="Path to saved Faster R-CNN model parameters.",
    )
    parser.add_argument(
        "--freq-prior",
        type=str,
        default="freq_prior.pkl",
        help="Path to saved frequency prior data.",
    )
    args = parser.parse_args()
    return args


args = parse_args()
if args.gpu:
    ctx = mx.gpu(int(args.gpu))
else:
    ctx = mx.cpu()

net = RelDN(n_classes=50, prior_pkl=args.freq_prior, semantic_only=False)
if args.reldn_params == "":
    download("http://data.dgl.ai/models/SceneGraph/reldn.params")
    net.load_parameters("rendl.params", ctx=ctx)
else:
    net.load_parameters(args.reldn_params, ctx=ctx)

# dataset and dataloader
vg_val = VGRelation(split="val")
detector = faster_rcnn_resnet101_v1d_custom(
    classes=vg_val.obj_classes,
    pretrained_base=False,
    pretrained=False,
    additional_output=True,
)
if args.pretrained_faster_rcnn_params == "":
    download(
        "http://data.dgl.ai/models/SceneGraph/faster_rcnn_resnet101_v1d_visualgenome.params"
    )
    params_path = "faster_rcnn_resnet101_v1d_visualgenome.params"
else:
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
if args.faster_rcnn_params == "":
    download(
        "http://data.dgl.ai/models/SceneGraph/faster_rcnn_resnet101_v1d_visualgenome.params"
    )
    detector_feat.features.load_parameters(
        "faster_rcnn_resnet101_v1d_visualgenome.params", ctx=ctx
    )
else:
    detector_feat.features.load_parameters(args.faster_rcnn_params, ctx=ctx)

# image input
if args.image:
    image_path = args.image
else:
    gcv.utils.download(
        "https://raw.githubusercontent.com/dmlc/web-data/master/"
        + "dgl/examples/mxnet/scenegraph/old-couple.png",
        "old-couple.png",
    )
    image_path = "old-couple.png"
x, img = presets.rcnn.load_test(
    args.image, short=detector.short, max_size=detector.max_size
)
x = x.as_in_context(ctx)
# detector prediction
ids, scores, bboxes, feat, feat_ind, spatial_feat = detector(x)
# build graph, extract edge features
g = build_graph_validate_pred(
    x,
    ids,
    scores,
    bboxes,
    feat_ind,
    spatial_feat,
    bbox_improvement=True,
    scores_top_k=75,
    overlap=False,
)
rel_bbox = g.edata["rel_bbox"].expand_dims(0).as_in_context(ctx)
_, _, _, spatial_feat_rel = detector_feat(x, None, None, rel_bbox)
g.edata["edge_feat"] = spatial_feat_rel[0]
# graph prediction
g = net(g)

_, preds = extract_pred(g, joint_preds=True)
preds = preds[preds[:, 1].argsort()[::-1]]

plot_sg(img, preds, detector.classes, vg_val.rel_classes, 10)
