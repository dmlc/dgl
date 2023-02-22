import dgl
import numpy as np
from mxnet import nd


def bbox_improve(bbox):
    """bbox encoding"""
    area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    return nd.concat(bbox, area.expand_dims(1))


def extract_edge_bbox(g):
    """bbox encoding"""
    src, dst = g.edges(order="eid")
    n = g.number_of_edges()
    src_bbox = g.ndata["pred_bbox"][src.asnumpy()]
    dst_bbox = g.ndata["pred_bbox"][dst.asnumpy()]
    edge_bbox = nd.zeros((n, 4), ctx=g.ndata["pred_bbox"].context)
    edge_bbox[:, 0] = nd.stack(src_bbox[:, 0], dst_bbox[:, 0]).min(axis=0)
    edge_bbox[:, 1] = nd.stack(src_bbox[:, 1], dst_bbox[:, 1]).min(axis=0)
    edge_bbox[:, 2] = nd.stack(src_bbox[:, 2], dst_bbox[:, 2]).max(axis=0)
    edge_bbox[:, 3] = nd.stack(src_bbox[:, 3], dst_bbox[:, 3]).max(axis=0)
    return edge_bbox


def build_graph_train(
    g_slice,
    gt_bbox,
    img,
    ids,
    scores,
    bbox,
    feat_ind,
    spatial_feat,
    iou_thresh=0.5,
    bbox_improvement=True,
    scores_top_k=50,
    overlap=False,
):
    """given ground truth and predicted bboxes, assign the label to the predicted w.r.t iou_thresh"""
    # match and re-factor the graph
    img_size = img.shape[2:4]
    gt_bbox[:, :, 0] /= img_size[1]
    gt_bbox[:, :, 1] /= img_size[0]
    gt_bbox[:, :, 2] /= img_size[1]
    gt_bbox[:, :, 3] /= img_size[0]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]

    n_graph = len(g_slice)
    g_pred_batch = []
    for gi in range(n_graph):
        g = g_slice[gi]
        ctx = g.ndata["bbox"].context
        inds = np.where(scores[gi, :, 0].asnumpy() > 0)[0].tolist()
        if len(inds) == 0:
            return None
        if len(inds) > scores_top_k:
            top_score_inds = (
                scores[gi, inds, 0].asnumpy().argsort()[::-1][0:scores_top_k]
            )
            inds = np.array(inds)[top_score_inds].tolist()

        n_nodes = len(inds)
        roi_ind = feat_ind[gi, inds].squeeze(axis=1)
        g_pred = dgl.DGLGraph()
        g_pred.add_nodes(
            n_nodes,
            {
                "pred_bbox": bbox[gi, inds],
                "node_feat": spatial_feat[gi, roi_ind],
                "node_class_pred": ids[gi, inds, 0],
                "node_class_logit": nd.log(scores[gi, inds, 0] + 1e-7),
            },
        )

        # iou matching
        ious = nd.contrib.box_iou(
            gt_bbox[gi], g_pred.ndata["pred_bbox"]
        ).asnumpy()
        H, W = ious.shape
        h = H
        w = W
        pred_to_gt_ind = np.array([-1 for i in range(W)])
        pred_to_gt_class_match = [0 for i in range(W)]
        pred_to_gt_class_match_id = [0 for i in range(W)]
        while h > 0 and w > 0:
            ind = int(ious.argmax())
            row_ind = ind // W
            col_ind = ind % W
            if ious[row_ind, col_ind] < iou_thresh:
                break
            pred_to_gt_ind[col_ind] = row_ind
            gt_node_class = g.ndata["node_class"][row_ind]
            pred_node_class = g_pred.ndata["node_class_pred"][col_ind]
            if gt_node_class == pred_node_class:
                pred_to_gt_class_match[col_ind] = 1
                pred_to_gt_class_match_id[col_ind] = row_ind
            ious[row_ind, :] = -1
            ious[:, col_ind] = -1
            h -= 1
            w -= 1

        n_nodes = g_pred.number_of_nodes()
        triplet = []
        adjmat = np.zeros((n_nodes, n_nodes))

        src, dst = g.all_edges(order="eid")
        eid_keys = np.column_stack([src.asnumpy(), dst.asnumpy()])
        eid_dict = {}
        for i, key in enumerate(eid_keys):
            k = tuple(key)
            if k not in eid_dict:
                eid_dict[k] = [i]
            else:
                eid_dict[k].append(i)
        ori_rel_class = g.edata["rel_class"].asnumpy()
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    if pred_to_gt_class_match[i] and pred_to_gt_class_match[j]:
                        sub_gt_id = pred_to_gt_class_match_id[i]
                        ob_gt_id = pred_to_gt_class_match_id[j]
                        eids = eid_dict[(sub_gt_id, ob_gt_id)]
                        rel_cls = ori_rel_class[eids]
                        n_edges_between = len(rel_cls)
                        for ii in range(n_edges_between):
                            triplet.append((i, j, rel_cls[ii]))
                        adjmat[i, j] = 1
                    else:
                        triplet.append((i, j, 0))
        src, dst, rel_class = tuple(zip(*triplet))
        rel_class = nd.array(rel_class, ctx=ctx).expand_dims(1)
        g_pred.add_edges(src, dst, data={"rel_class": rel_class})

        # other operations
        n_nodes = g_pred.number_of_nodes()
        n_edges = g_pred.number_of_edges()
        if bbox_improvement:
            g_pred.ndata["pred_bbox"] = bbox_improve(g_pred.ndata["pred_bbox"])
        g_pred.edata["rel_bbox"] = extract_edge_bbox(g_pred)
        g_pred.edata["batch_id"] = nd.zeros((n_edges, 1), ctx=ctx) + gi

        # remove non-overlapping edges
        if overlap:
            overlap_ious = nd.contrib.box_iou(
                g_pred.ndata["pred_bbox"][:, 0:4],
                g_pred.ndata["pred_bbox"][:, 0:4],
            ).asnumpy()
            cols, rows = np.where(overlap_ious <= 1e-7)
            if cols.shape[0] > 0:
                eids = g_pred.edge_ids(cols, rows)[2].asnumpy().tolist()
                if len(eids):
                    g_pred.remove_edges(eids)
                    if g_pred.number_of_edges() == 0:
                        g_pred = None
        g_pred_batch.append(g_pred)

    if n_graph > 1:
        return dgl.batch(g_pred_batch)
    else:
        return g_pred_batch[0]


def build_graph_validate_gt_obj(
    img, gt_ids, bbox, spatial_feat, bbox_improvement=True, overlap=False
):
    """given ground truth bbox and label, build graph for validation"""
    n_batch = img.shape[0]
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]
    ctx = img.context

    g_batch = []
    for btc in range(n_batch):
        inds = np.where(bbox[btc].sum(1).asnumpy() > 0)[0].tolist()
        if len(inds) == 0:
            continue
        n_nodes = len(inds)
        g_pred = dgl.DGLGraph()
        g_pred.add_nodes(
            n_nodes,
            {
                "pred_bbox": bbox[btc, inds],
                "node_feat": spatial_feat[btc, inds],
                "node_class_pred": gt_ids[btc, inds, 0],
                "node_class_logit": nd.zeros_like(
                    gt_ids[btc, inds, 0], ctx=ctx
                ),
            },
        )

        edge_list = []
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                edge_list.append((i, j))
        src, dst = tuple(zip(*edge_list))
        g_pred.add_edges(src, dst)
        g_pred.add_edges(dst, src)

        n_nodes = g_pred.number_of_nodes()
        n_edges = g_pred.number_of_edges()
        if bbox_improvement:
            g_pred.ndata["pred_bbox"] = bbox_improve(g_pred.ndata["pred_bbox"])
        g_pred.edata["rel_bbox"] = extract_edge_bbox(g_pred)
        g_pred.edata["batch_id"] = nd.zeros((n_edges, 1), ctx=ctx) + btc

        g_batch.append(g_pred)

    if len(g_batch) == 0:
        return None
    if len(g_batch) > 1:
        return dgl.batch(g_batch)
    return g_batch[0]


def build_graph_validate_gt_bbox(
    img,
    ids,
    scores,
    bbox,
    spatial_feat,
    gt_ids=None,
    bbox_improvement=True,
    overlap=False,
):
    """given ground truth bbox, build graph for validation"""
    n_batch = img.shape[0]
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]
    ctx = img.context

    g_batch = []
    for btc in range(n_batch):
        id_btc = scores[btc][:, :, 0].argmax(0)
        score_btc = scores[btc][:, :, 0].max(0)
        inds = np.where(bbox[btc].sum(1).asnumpy() > 0)[0].tolist()
        if len(inds) == 0:
            continue
        n_nodes = len(inds)
        g_pred = dgl.DGLGraph()
        g_pred.add_nodes(
            n_nodes,
            {
                "pred_bbox": bbox[btc, inds],
                "node_feat": spatial_feat[btc, inds],
                "node_class_pred": id_btc,
                "node_class_logit": nd.log(score_btc + 1e-7),
            },
        )

        edge_list = []
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                edge_list.append((i, j))
        src, dst = tuple(zip(*edge_list))
        g_pred.add_edges(src, dst)
        g_pred.add_edges(dst, src)

        n_nodes = g_pred.number_of_nodes()
        n_edges = g_pred.number_of_edges()
        if bbox_improvement:
            g_pred.ndata["pred_bbox"] = bbox_improve(g_pred.ndata["pred_bbox"])
        g_pred.edata["rel_bbox"] = extract_edge_bbox(g_pred)
        g_pred.edata["batch_id"] = nd.zeros((n_edges, 1), ctx=ctx) + btc

        g_batch.append(g_pred)

    if len(g_batch) == 0:
        return None
    if len(g_batch) > 1:
        return dgl.batch(g_batch)
    return g_batch[0]


def build_graph_validate_pred(
    img,
    ids,
    scores,
    bbox,
    feat_ind,
    spatial_feat,
    bbox_improvement=True,
    scores_top_k=50,
    overlap=False,
):
    """given predicted bbox, build graph for validation"""
    n_batch = img.shape[0]
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]
    ctx = img.context

    g_batch = []
    for btc in range(n_batch):
        inds = np.where(scores[btc, :, 0].asnumpy() > 0)[0].tolist()
        if len(inds) == 0:
            continue
        if len(inds) > scores_top_k:
            top_score_inds = (
                scores[btc, inds, 0].asnumpy().argsort()[::-1][0:scores_top_k]
            )
            inds = np.array(inds)[top_score_inds].tolist()
        n_nodes = len(inds)
        roi_ind = feat_ind[btc, inds].squeeze(axis=1)

        g_pred = dgl.DGLGraph()
        g_pred.add_nodes(
            n_nodes,
            {
                "pred_bbox": bbox[btc, inds],
                "node_feat": spatial_feat[btc, roi_ind],
                "node_class_pred": ids[btc, inds, 0],
                "node_class_logit": nd.log(scores[btc, inds, 0] + 1e-7),
            },
        )

        edge_list = []
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                edge_list.append((i, j))
        src, dst = tuple(zip(*edge_list))
        g_pred.add_edges(src, dst)
        g_pred.add_edges(dst, src)

        n_nodes = g_pred.number_of_nodes()
        n_edges = g_pred.number_of_edges()
        if bbox_improvement:
            g_pred.ndata["pred_bbox"] = bbox_improve(g_pred.ndata["pred_bbox"])
        g_pred.edata["rel_bbox"] = extract_edge_bbox(g_pred)
        g_pred.edata["batch_id"] = nd.zeros((n_edges, 1), ctx=ctx) + btc

        g_batch.append(g_pred)

    if len(g_batch) == 0:
        return None
    if len(g_batch) > 1:
        return dgl.batch(g_batch)
    return g_batch[0]
