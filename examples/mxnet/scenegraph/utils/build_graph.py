import dgl
from mxnet import nd
import numpy as np
from .sampling import l0_sample

def build_complete_graph(n_nodes):
    g = dgl.DGLGraph()
    g.add_nodes(n_nodes)
    edge_list = []
    for i in range(n_nodes-1):
        for j in range(i+1, n_nodes):
            edge_list.append((i, j))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g

def build_graph_with_reference(g_ref):
    n_nodes = g_ref.number_of_nodes()
    g = dgl.DGLGraph()
    g.add_nodes(n_nodes)
    gt_eids = g_ref.edges(order='eid')
    g.add_edges(gt_eids[0], gt_eids[1])
    return g

def bbox_improve(bbox):
    area = (bbox[:,2] - bbox[:,0]) * (bbox[:,3] - bbox[:,1])
    return nd.concat(bbox, area.expand_dims(1))

def extract_edge_bbox(g):
    src, dst = g.edges(order='eid')
    n = g.number_of_edges()
    src_bbox = g.ndata['pred_bbox'][src]
    dst_bbox = g.ndata['pred_bbox'][dst]
    edge_bbox = nd.zeros((n, 4), ctx=g.ndata['pred_bbox'].context)
    edge_bbox[:,0] = nd.stack(src_bbox[:,0], dst_bbox[:,0]).min(axis=0)
    edge_bbox[:,1] = nd.stack(src_bbox[:,1], dst_bbox[:,1]).min(axis=0)
    edge_bbox[:,2] = nd.stack(src_bbox[:,2], dst_bbox[:,2]).max(axis=0)
    edge_bbox[:,3] = nd.stack(src_bbox[:,3], dst_bbox[:,3]).max(axis=0)
    return edge_bbox

def build_graph_gt(g_slice, img, bbox, spatial_feat, cls_pred, l0_w_slice=None,
                   training=False,
                   bbox_improvement=True, batchify=True):
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]

    n_graph = len(g_slice)
    g_batch = []
    for i in range(n_graph):
        n_nodes = g_slice[i].number_of_nodes()
        g = build_graph_with_reference(g_slice[i])
        if bbox_improvement:
            bbox_improved = bbox_improve(bbox[i, 0:n_nodes])
            g.ndata['pred_bbox'] = bbox_improved
        else:
            g.ndata['pred_bbox'] = bbox[i, 0:n_nodes]
        g.ndata['node_feat'] = spatial_feat[i, 0:n_nodes]
        g.ndata['node_class_pred'] = cls_pred[i, 0:n_nodes, 1:]
        if training:
            g.edata['rel_class'] = g_slice[i].edata['rel_class']
        if l0_w_slice is not None:
            g.edata['sample'] = l0_w_slice[i]
        g_batch.append(g)
    if batchify:
        return dgl.batch(g_batch)
    else:
        return g_batch

def build_graph_gt_rel(g_batch, img, bbox, spatial_feat):
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]

    n_graph = len(g_batch)
    for i in range(n_graph):
        g = g_batch[i]
        n_edges = g.number_of_edges()
        if 'sample' in g.edata:
            ctx = g.edata['sample'].context
            sample_inds = np.where(g.edata['sample'] > 0)[0]
            n_samples = len(sample_inds)
            g.edata['rel_bbox'] = nd.zeros((n_edges, 5), ctx=ctx)
            g.edata['rel_bbox'][sample_inds] = bbox_improve(bbox[i, 0:n_samples])
            g.edata['edge_feat'] = nd.zeros((n_edges, spatial_feat.shape[2]), ctx=ctx)
            g.edata['edge_feat'][sample_inds] = spatial_feat[i, 0:n_samples]
        else:
            g.edata['rel_bbox'] = bbox_improve(bbox[i, 0:n_edges])
            g.edata['edge_feat'] = spatial_feat[i, 0:n_edges]
    return dgl.batch(g_batch)

def build_graph_pred(g_slice, img, scores, bbox, feat_ind, spatial_feat, cls_pred, node_topk=300):
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]

    n_graph = len(g_slice)
    g_batch = []
    for i in range(n_graph):
        inds = np.where(scores[i, :, 0].asnumpy() > 0)[0]
        if len(inds) == 0:
            g = None
        if len(inds) > node_topk:
            topk_inds = scores[i, inds, 0].argsort()[::-1][0:node_topk].asnumpy().astype(np.int)
            inds = inds[topk_inds]
        n_nodes = len(inds)
        g = build_graph_with_reference(g_slice[i])
        g.ndata['pred_bbox'] = bbox[i, inds]
        roi_ind = feat_ind[i, inds].squeeze(1)
        g.ndata['node_feat'] = spatial_feat[i, roi_ind]
        g.ndata['node_class_pred'] = cls_pred[i, roi_ind, 1:]
        g_batch.append(g)
    return dgl.batch(g_batch)

def build_graph_gt_sample(g_slice, img, bbox, spatial_feat, cls_pred,
                          training=False, bbox_improvement=True, sample=True):
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]

    n_graph = len(g_slice)
    for i in range(n_graph):
        g = g_slice[i]
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        if bbox_improvement:
            bbox_improved = bbox_improve(bbox[i, 0:n_nodes])
            g.ndata['pred_bbox'] = bbox_improved
        else:
            g.ndata['pred_bbox'] = bbox[i, 0:n_nodes]
        g.ndata['node_feat'] = spatial_feat[i, 0:n_nodes]
        g.ndata['node_class_pred'] = cls_pred[i, 0:n_nodes, 1:]
        g.edata['rel_bbox'] = extract_edge_bbox(g)
        g.edata['batch_id'] = nd.zeros((n_edges, 1), ctx = g.edata['rel_bbox'].context) + i
    g_batch = dgl.batch(g_slice)
    if sample:
        sub_g = l0_sample(g_batch)
    else:
        sub_g = g_batch
    return sub_g
