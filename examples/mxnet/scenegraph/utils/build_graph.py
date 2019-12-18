import dgl
import numpy as np

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

def build_graph_gt(g_slice, img, bbox, spatial_feat, cls_pred):
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]

    n_graph = len(g_slice)
    g_batch = []
    for i in range(n_graph):
        n_nodes = g_slice[i].number_of_nodes()
        g = build_complete_graph(n_nodes)
        g.ndata['pred_bbox'] = bbox[i, 0:n_nodes]
        g.ndata['node_feat'] = spatial_feat[i, 0:n_nodes]
        g.ndata['node_class_pred'] = cls_pred[i, 0:n_nodes, 1:]
        g_batch.append(g)
    return dgl.batch(g_batch)

def build_graph_pred(g_slice, img, scores, bbox, feat_ind, spatial_feat, cls_pred):
    img_size = img.shape[2:4]
    bbox[:, :, 0] /= img_size[1]
    bbox[:, :, 1] /= img_size[0]
    bbox[:, :, 2] /= img_size[1]
    bbox[:, :, 3] /= img_size[0]

    n_graph = len(g_slice)
    g_batch = []
    for i in range(n_graph):
        inds = np.where(scores[i, :, 0].asnumpy() > 0)[0].tolist()
        if len(inds) == 0:
            g = None
        n_nodes = len(inds)
        g = build_complete_graph(n_nodes)
        g.ndata['pred_bbox'] = bbox[i, inds]
        roi_ind = feat_ind[i, inds].squeeze(1)
        g.ndata['node_feat'] = spatial_feat[i, roi_ind]
        g.ndata['node_class_pred'] = cls_pred[i, roi_ind, 1:]
        g_batch.append(g)
    return dgl.batch(g_batch)
