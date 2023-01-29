import gluoncv as gcv
import numpy as np
from matplotlib import pyplot as plt


def plot_sg(img, preds, obj_classes, rel_classes, topk=1):
    """visualization of generated scene graph"""
    size = img.shape[0:2]
    box_scale = np.array([size[1], size[0], size[1], size[0]])
    topk = min(topk, preds.shape[0])
    ax = gcv.utils.viz.plot_image(img)
    for i in range(topk):
        rel = int(preds[i, 2])
        src = int(preds[i, 3])
        dst = int(preds[i, 4])
        src_name = obj_classes[src]
        dst_name = obj_classes[dst]
        rel_name = rel_classes[rel]
        src_bbox = preds[i, 5:9] * box_scale
        dst_bbox = preds[i, 9:13] * box_scale

        src_center = np.array(
            [(src_bbox[0] + src_bbox[2]) / 2, (src_bbox[1] + src_bbox[3]) / 2]
        )
        dst_center = np.array(
            [(dst_bbox[0] + dst_bbox[2]) / 2, (dst_bbox[1] + dst_bbox[3]) / 2]
        )
        rel_center = (src_center + dst_center) / 2

        line_x = np.array(
            [(src_bbox[0] + src_bbox[2]) / 2, (dst_bbox[0] + dst_bbox[2]) / 2]
        )
        line_y = np.array(
            [(src_bbox[1] + src_bbox[3]) / 2, (dst_bbox[1] + dst_bbox[3]) / 2]
        )

        ax.plot(
            line_x, line_y, linewidth=3.0, alpha=0.7, color=plt.cm.cool(rel)
        )

        ax.text(
            src_center[0],
            src_center[1],
            "{:s}".format(src_name),
            bbox=dict(alpha=0.5),
            fontsize=12,
            color="white",
        )
        ax.text(
            dst_center[0],
            dst_center[1],
            "{:s}".format(dst_name),
            bbox=dict(alpha=0.5),
            fontsize=12,
            color="white",
        )
        ax.text(
            rel_center[0],
            rel_center[1],
            "{:s}".format(rel_name),
            bbox=dict(alpha=0.5),
            fontsize=12,
            color="white",
        )
    return ax


plot_sg(img, preds, 2)
