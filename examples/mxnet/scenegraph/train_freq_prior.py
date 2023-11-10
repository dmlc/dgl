import argparse
import json
import os
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Frequenct Prior For RelDN."
    )
    parser.add_argument(
        "--overlap", action="store_true", help="Only count overlap boxes."
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="~/.mxnet/datasets/visualgenome",
        help="Only count overlap boxes.",
    )
    args = parser.parse_args()
    return args


args = parse_args()
use_overlap = args.overlap
PATH_TO_DATASETS = os.path.expanduser(args.json_path)
path_to_json = os.path.join(PATH_TO_DATASETS, "rel_annotations_train.json")


# format in y1y2x1x2
def with_overlap(boxA, boxB):
    xA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    if xB > xA:
        yA = max(boxA[0], boxB[0])
        yB = min(boxA[1], boxB[1])

        if yB > yA:
            return 1

    return 0


def box_ious(boxes):
    n = len(boxes)
    res = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            iou_val = with_overlap(boxes[i], boxes[j])
            res[i, j] = iou_val
            res[j, i] = iou_val
    return res


with open(path_to_json, "r") as f:
    tmp = f.read()
    train_data = json.loads(tmp)

fg_matrix = np.zeros((150, 150, 51), dtype=np.int64)
bg_matrix = np.zeros((150, 150), dtype=np.int64)

for _, item in train_data.items():
    gt_box_to_label = {}
    for rel in item:
        sub_bbox = rel["subject"]["bbox"]
        ob_bbox = rel["object"]["bbox"]
        sub_class = rel["subject"]["category"]
        ob_class = rel["object"]["category"]
        rel_class = rel["predicate"]

        sub_node = tuple(sub_bbox)
        ob_node = tuple(ob_bbox)
        if sub_node not in gt_box_to_label:
            gt_box_to_label[sub_node] = sub_class
        if ob_node not in gt_box_to_label:
            gt_box_to_label[ob_node] = ob_class

        fg_matrix[sub_class, ob_class, rel_class + 1] += 1

    if use_overlap:
        gt_boxes = [*gt_box_to_label]
        gt_classes = np.array([*gt_box_to_label.values()])
        iou_mat = box_ious(gt_boxes)
        cols, rows = np.where(iou_mat)
        if len(cols) and len(rows):
            for col, row in zip(cols, rows):
                bg_matrix[gt_classes[col], gt_classes[row]] += 1
        else:
            all_possib = np.ones_like(iou_mat, dtype=np.bool_)
            np.fill_diagonal(all_possib, 0)
            cols, rows = np.where(all_possib)
            for col, row in zip(cols, rows):
                bg_matrix[gt_classes[col], gt_classes[row]] += 1
    else:
        for b1, l1 in gt_box_to_label.items():
            for b2, l2 in gt_box_to_label.items():
                if b1 == b2:
                    continue
                bg_matrix[l1, l2] += 1


eps = 1e-3
bg_matrix += 1
fg_matrix[:, :, 0] = bg_matrix
pred_dist = np.log(fg_matrix / (fg_matrix.sum(2)[:, :, None] + eps) + eps)


if use_overlap:
    with open("freq_prior_overlap.pkl", "wb") as f:
        pickle.dump(pred_dist, f)
else:
    with open("freq_prior.pkl", "wb") as f:
        pickle.dump(pred_dist, f)
