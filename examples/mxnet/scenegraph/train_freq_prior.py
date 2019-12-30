import dgl
import mxnet as mx
import numpy as np
import logging, time, pickle
from mxnet import nd, gluon

from data import *

filehandler = logging.FileHandler('output.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# Hyperparams
ctx = [mx.cpu()]
batch_size = 32
N_relations = 50
N_objects = 150
batch_verbose_freq = 1

# dataset and dataloader
vg_train = VGRelation(top_frequent_rel=N_relations, top_frequent_obj=N_objects,
                      balancing='weight', split='train')
logger.info('data loaded!')
train_data = gluon.data.DataLoader(vg_train, batch_size=batch_size, shuffle=False, num_workers=60,
                                   batchify_fn=dgl_mp_batchify_fn)
n_batches = len(train_data)

pos_count = np.zeros((N_objects, N_objects, N_relations + 1))
neg_count = np.zeros((N_objects, N_objects, N_relations + 1))
freq_prior = np.zeros((N_objects, N_objects, N_relations + 1))

print(n_batches)

for i, (G_list, img_list) in enumerate(train_data):
    if G_list is None or len(G_list) == 0:
        continue
    if (i+1) % batch_verbose_freq == 0:
        print(i)
    G = dgl.batch(G_list)
    pos_eids = np.where(G.edata['link'].asnumpy() > 0)[0]
    if len(pos_eids) > 0:
        src, dst = G.find_edges(pos_eids.tolist())
        src_id = G.ndata['node_class_ids'][src,0].asnumpy().astype(int)
        dst_id = G.ndata['node_class_ids'][dst,0].asnumpy().astype(int)
        rel_id = (G.edata['classes'] + G.edata['link'])[pos_eids].asnumpy().astype(int)
        np.add.at(pos_count, (src_id, dst_id, rel_id), 1)

    neg_eids = np.where(G.edata['link'].asnumpy() == 0)[0]
    if len(neg_eids) > 0:
        src, dst = G.find_edges(neg_eids.tolist())
        src_id = G.ndata['node_class_ids'][src,0].asnumpy().astype(int)
        dst_id = G.ndata['node_class_ids'][dst,0].asnumpy().astype(int)
        rel_id = (G.edata['classes'] + G.edata['link'])[neg_eids].asnumpy().astype(int)
        np.add.at(neg_count, (src_id, dst_id, rel_id), 1)

total_count = (pos_count + neg_count).sum(axis=2, keepdims=True)
freq_prior = pos_count / (total_count + 1e-7)

with open('freq_prior.pkl', 'wb') as f:
    pickle.dump(freq_prior, f)

total_prior = freq_prior.mean()
total_count_average = total_count.mean()
bayesian_freq_prior = (pos_count + total_prior * total_count_average) / (total_count + total_count_average)
with open('bayesian_freq_prior.pkl', 'wb') as f:
    pickle.dump(bayesian_freq_prior, f)
