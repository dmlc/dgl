import numpy as np
import os, pickle, json

PATH_TO_DATASETS = os.path.expanduser('~/.mxnet/datasets/visualgenome')
path_to_json = os.path.join(PATH_TO_DATASETS, 'rel_annotations_val.json')

with open(path_to_json, 'r') as f:
    tmp = f.read()
    val_data = json.loads(tmp)

with open('freq_prior_overlap.pkl', 'rb') as f:
    pred_dist = pickle.load(f)

pred_cls = 0
for i, (img_id, item) in enumerate(val_data.items()):
    if img_id != '2315353.jpg':
        continue
    import pdb; pdb.set_trace()
    gt_box_to_label = {}
    gt_triplet = []
    node_id = 0
    for rel in item:
        sub_bbox = rel['subject']['bbox']
        ob_bbox = rel['object']['bbox']
        sub_class = rel['subject']['category']
        ob_class = rel['object']['category']
        rel_class = rel['predicate']

        gt_triplet.append((sub_class, ob_class, rel_class))

        sub_node = tuple(sub_bbox)
        ob_node = tuple(ob_bbox)
        if sub_node not in gt_box_to_label:
            gt_box_to_label[sub_node] = (sub_class, node_id)
            node_id += 1
        if ob_node not in gt_box_to_label:
            gt_box_to_label[ob_node] = (ob_class, node_id)
            node_id += 1

    n_node = len(gt_box_to_label)

    pred_matrix = np.zeros((n_node, n_node), dtype=np.int)
    pred_score_matrix = np.zeros((n_node, n_node), dtype=np.float32) - 1000
    sub_class_matrix = np.zeros((n_node, n_node), dtype=np.int)
    ob_class_matrix = np.zeros((n_node, n_node), dtype=np.int)
    for b1, (l1, id1) in gt_box_to_label.items():
        for b2, (l2, id2) in gt_box_to_label.items():
            if id1 == id2:
                continue
            pred_matrix[id1, id2] = pred_dist[l1, l2, 1:].argmax()
            pred_score_matrix[id1, id2] = pred_dist[l1, l2, 1:].max()
            sub_class_matrix[id1, id2] = l1
            ob_class_matrix[id1, id2] = l2

    pred_vec = pred_matrix.reshape(-1)
    pred_score_vec = pred_score_matrix.reshape(-1)
    sub_class_matrix = sub_class_matrix.reshape(-1)
    ob_class_matrix = ob_class_matrix.reshape(-1)
    m = min(100, len(pred_vec))
    topk_ind = pred_score_vec.argsort()[::-1][0:m]

    count = 0
    num_rels = len(gt_triplet)
    matched = [False for i in range(num_rels)]
    for ind in topk_ind:
        pred_sub = sub_class_matrix[ind]
        pred_ob = ob_class_matrix[ind]
        pred_rel = pred_vec[ind]
        for i in range(num_rels):
            if matched[i]:
                continue
            gt_sub, gt_ob, gt_rel = gt_triplet[i]
            if pred_sub == gt_sub and pred_ob == gt_ob and pred_rel == gt_rel:
                count += 1
                matched[i] = True

    denom = len(item)
    pred_cls += count / denom
    if (i+1) % 1000 == 0:
        print('%d\t%f'%(i, pred_cls/(i+1)))

print('Recall@100: %f'%(pred_cls/len(val_data)))
