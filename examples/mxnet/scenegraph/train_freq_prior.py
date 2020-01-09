import numpy as np
import json, pickle, os

PATH_TO_DATASETS = os.path.expanduser('~/.mxnet/datasets/visualgenome')
path_to_json = os.path.join(PATH_TO_DATASETS, 'rel_annotations_train.json')

with open(path_to_json, 'r') as f:
    tmp = f.read()
    train_data = json.loads(tmp)

fg_matrix = np.zeros((150, 150, 51), dtype=np.int64)
bg_matrix = np.zeros((150, 150), dtype=np.int64)

for _, item in train_data.items():
    gt_box_to_label = {}
    for rel in item:
        sub_bbox = rel['subject']['bbox']
        ob_bbox = rel['object']['bbox']
        sub_class = rel['subject']['category']
        ob_class = rel['object']['category']
        rel_class = rel['predicate']

        sub_node = tuple(sub_bbox)
        ob_node = tuple(ob_bbox)
        if sub_node not in gt_box_to_label:
            gt_box_to_label[sub_node] = sub_class
        if ob_node not in gt_box_to_label:
            gt_box_to_label[ob_node] = ob_class

        fg_matrix[sub_class, ob_class, rel_class + 1] += 1

    for b1, l1 in gt_box_to_label.items():
        for b2, l2 in gt_box_to_label.items():
            if b1 == b2:
                continue
            bg_matrix[l1, l2] += 1

eps = 1e-3
bg_matrix += 1
fg_matrix[:, :, 0] = bg_matrix
pred_dist = np.log(fg_matrix / (fg_matrix.sum(2)[:, :, None] + 1e-08) + eps)

with open('freq_prior.pkl', 'wb') as f:
    pickle.dump(pred_dist, f)