import dgl
import mxnet as mx
import numpy as np
import logging, time, pickle
from mxnet import nd, gluon

from utils import *
from data import *

filehandler = logging.FileHandler('output.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# Hyperparams
ctx = [mx.cpu()]
batch_size = 1
N_relations = 50
N_objects = 150
batch_verbose_freq = 100

# dataset and dataloader
vg_train = VGRelationCOCO(split='val')
logger.info('data loaded!')
val_data = gluon.data.DataLoader(vg_train, batch_size=batch_size, shuffle=False, num_workers=8,
                                   batchify_fn=dgl_mp_batchify_fn)
n_batches = len(val_data)

with open('freq_prior.pkl', 'rb') as f:
    freq_prior = pickle.load(f)

metric_list = []
for k in [20, 50, 100]:
    metric_list.append(PredCls(topk=k))
# metric = PredCls(topk=100)
for metric in metric_list:
    metric.reset()

for i, (G_list, img_list) in enumerate(val_data):
    if G_list is None or len(G_list) == 0:
        continue
    G = dgl.batch(G_list)
    img_size = img_list.shape[2:4]
    # eids = np.where(G.edata['rel_class'].asnumpy() > 0)[0]
    # src, dst = G.find_edges(eids)
    src, dst = G.all_edges(order='eid')
    src_ind = G.ndata['node_class'][src,0].asnumpy().astype(int)
    dst_ind = G.ndata['node_class'][dst,0].asnumpy().astype(int)
    prob = nd.array(freq_prior[src_ind, dst_ind])
    G.ndata['node_class_prob'] = G.ndata['node_class_vec']
    G.ndata['pred_bbox'] = G.ndata['bbox'].copy()
    G.ndata['pred_bbox'][:, 0] /= img_size[1]
    G.ndata['pred_bbox'][:, 1] /= img_size[0]
    G.ndata['pred_bbox'][:, 2] /= img_size[1]
    G.ndata['pred_bbox'][:, 3] /= img_size[0]
    G.edata['preds'] = prob
    G.edata['score_pred'] = G.edata['preds'][:,1:].max(axis=1)
    G.edata['score_phr'] = G.edata['preds'][:,1:].max(axis=1)

    gt_objects, gt_triplet = extract_gt(G, img_size)
    pred_objects, pred_triplet = extract_pred(G, joint_preds=True)

    for metric in metric_list:
        metric.update(gt_triplet, pred_triplet)

    if (i+1) % batch_verbose_freq == 0:
        print_txt = 'Batch[%d/%d] '%(i, n_batches)
        for metric in metric_list:
            name, acc = metric.get()
            print_txt += '%s=%.6f'%(name, acc)
        # name, acc = metric.get()
        # print('Batch[%d/%d] %s=%f'%(i, n_batches, name, acc))
        print(print_txt)




    '''
    pos_eids = np.where(G.edata['rel_class'].asnumpy() > 0)[0]
    if len(pos_eids) > 0:
        src, dst = G.find_edges(pos_eids.tolist())
        src_id = G.ndata['node_class'][src, 0].asnumpy().astype(int)
        dst_id = G.ndata['node_class'][dst, 0].asnumpy().astype(int)
        rel_id = G.edata['rel_class'][pos_eids, 0].asnumpy().astype(int)
        np.add.at(pos_count, (src_id, dst_id, rel_id), 1)

    neg_eids = np.where(G.edata['rel_class'].asnumpy() == 0)[0]
    if len(neg_eids) > 0:
        src, dst = G.find_edges(neg_eids.tolist())
        src_id = G.ndata['node_class'][src, 0].asnumpy().astype(int)
        dst_id = G.ndata['node_class'][dst, 0].asnumpy().astype(int)
        rel_id = G.edata['rel_class'][neg_eids, 0].asnumpy().astype(int)
        np.add.at(neg_count, (src_id, dst_id, rel_id), 1)

total_count = (pos_count + neg_count).sum(axis=2, keepdims=True)
freq_prior = np.log(pos_count / (total_count + 1e-8) + 1e-8)

with open('freq_prior.pkl', 'wb') as f:
    pickle.dump(freq_prior, f)

total_prior = freq_prior.mean()
total_count_average = total_count.mean()
bayesian_freq_prior = (pos_count + total_prior * total_count_average) / (total_count + total_count_average)
with open('bayesian_freq_prior.pkl', 'wb') as f:
    pickle.dump(bayesian_freq_prior, f)

    '''