import argparse, time, os, pickle
import random

from utils.deduce import get_edge_dist
import numpy as np
import shutil

import dgl
import torch
import torch.optim as optim

from models import LANDER
from dataset import LanderDataset
from utils import evaluation, decode, build_next_level, stop_iterating

###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--model_filename', type=str, default='lander.pth')
parser.add_argument('--faiss_gpu', action='store_true')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--output_filename', type=str, default='data/features.pkl')

# HyperParam
parser.add_argument('--knn_k', type=int, default=10)
parser.add_argument('--levels', type=int, default=1)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--threshold', type=str, default='prob')
parser.add_argument('--metrics', type=str, default='pairwise,bcubed,nmi')
parser.add_argument('--early_stop', action='store_true')

# Model
parser.add_argument('--hidden', type=int, default=512)
parser.add_argument('--num_conv', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--gat', action='store_true')
parser.add_argument('--gat_k', type=int, default=1)
parser.add_argument('--balance', action='store_true')
parser.add_argument('--use_cluster_feat', action='store_true')
parser.add_argument('--use_focal_loss', action='store_true')
parser.add_argument('--use_gt', action='store_true')

# Subgraph
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--mode', type=str, default="1head")
parser.add_argument('--midpoint', type=str, default="false")
parser.add_argument('--linsize', type=int, default=29011)
parser.add_argument('--uinsize', type=int, default=18403)
parser.add_argument('--inclasses', type=int, default=948)
parser.add_argument('--thresh', type=float, default=1.0)

args = parser.parse_args()
print(args)
MODE = args.mode
linsize = args.linsize
uinsize = args.uinsize
inclasses = args.inclasses

###########################
# Environment Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

##################
# Data Preparation
with open(args.data_path, 'rb') as f:
    loaded_data = pickle.load(f)
    path2idx, features, pred_labels, labels, masks = loaded_data

idx2path = {v: k for k, v in path2idx.items()}
gtlabels = labels
orifeatures = features
orilabels = gtlabels

if MODE == "selectbyindegree":
    lastusim = np.where(masks == 1)
    masks[lastusim] = 2
    selectedidx = np.where(masks != 0)
    features = features[selectedidx]
    labels = gtlabels[selectedidx]
    selectmasks = masks[selectedidx]
    print("filtered features:", len(features))
    print("mask0:", len(np.where(masks == 0)[0]))
    print("mask1:", len(np.where(masks == 1)[0]))
    print("mask2:", len(np.where(masks == 2)[0]))

global_features = features.copy() # global features
dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                        levels=1, faiss_gpu=False)
g = dataset.gs[0]
g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
global_labels = labels.copy()
ids = np.arange(g.number_of_nodes())
global_edges = ([], [])
global_peaks = np.array([], dtype=np.long)
global_edges_len = len(global_edges[0])
global_num_nodes = g.number_of_nodes()
global_in_degrees = g.in_degrees()

global_densities = g.ndata['density'][:linsize]
global_densities = np.sort(global_densities)
xs = np.arange(len(global_densities))

fanouts = [args.knn_k-1 for i in range(args.num_conv + 1)]
sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
# fix the number of edges
test_loader = dgl.dataloading.NodeDataLoader(
    g, torch.arange(g.number_of_nodes()), sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers
)

##################
# Model Definition
if not args.use_gt:
    feature_dim = g.ndata['features'].shape[1]
    model = LANDER(feature_dim=feature_dim, nhid=args.hidden,
                   num_conv=args.num_conv, dropout=args.dropout,
                   use_GAT=args.gat, K=args.gat_k,
                   balance=args.balance,
                   use_cluster_feat=args.use_cluster_feat,
                   use_focal_loss=args.use_focal_loss)
    model.load_state_dict(torch.load(args.model_filename))
    model = model.to(device)
    model.eval()

# number of edges added is the indicator for early stopping
num_edges_add_last_level = np.Inf
##################################
# Predict connectivity and density
for level in range(args.levels):
    print("level:", level)
    if not args.use_gt:
        total_batches = len(test_loader)
        for batch, minibatch in enumerate(test_loader):
            input_nodes, sub_g, bipartites = minibatch
            sub_g = sub_g.to(device)
            bipartites = [b.to(device) for b in bipartites]
            with torch.no_grad():
                output_bipartite = model(bipartites)
            global_nid = output_bipartite.dstdata[dgl.NID]
            global_eid = output_bipartite.edata['global_eid']
            g.ndata['pred_den'][global_nid] = output_bipartite.dstdata['pred_den'].to('cpu')
            g.edata['prob_conn'][global_eid] = output_bipartite.edata['prob_conn'].to('cpu')
            torch.cuda.empty_cache()
            if (batch + 1) % 10 == 0:
                print('Batch %d / %d for inference' % (batch, total_batches))

    new_pred_labels, peaks,\
        global_edges, global_pred_labels, global_peaks = decode(g, args.tau, args.threshold, args.use_gt,
                                                                ids, global_edges, global_num_nodes,
                                                                global_peaks)
    if level == 0:
        global_pred_densities = g.ndata['pred_den']
        global_densities = g.ndata['density']
        g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))

    ids = ids[peaks]
    new_global_edges_len = len(global_edges[0])
    num_edges_add_this_level = new_global_edges_len - global_edges_len
    if stop_iterating(level, args.levels, args.early_stop, num_edges_add_this_level, num_edges_add_last_level, args.knn_k):
        break
    global_edges_len = new_global_edges_len
    num_edges_add_last_level = num_edges_add_this_level

    # build new dataset
    features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                          global_features, global_pred_labels, global_peaks)
    # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.
    dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                            levels=1, faiss_gpu=False, cluster_features = cluster_features)
    g = dataset.gs[0]
    g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
    g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
    test_loader = dgl.dataloading.NodeDataLoader(
        g, torch.arange(g.number_of_nodes()), sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

if MODE == "selectbyindegree":
    thresh = args.thresh
    global_in_degrees = np.array(global_in_degrees).astype(int)
    sortidx = np.argsort(global_in_degrees)
    print(sortidx)
    selectnum = int(len(orilabels) * args.thresh)
    selectidx = sortidx[len(sortidx) - selectnum:]
    notselectidx = sortidx[:len(sortidx) - selectnum]
    print("select:", len(selectidx))
    print("not select:", len(notselectidx))
    global_pred_labels[notselectidx] = -1

    global_pred_labels_new = np.zeros_like(orilabels)
    global_pred_labels_new[:] = -1
    Tidx = np.where(masks != 2)

    l_in_gt = orilabels[Tidx]
    l_in_features = orifeatures[Tidx]
    l_in_gt_new = np.zeros_like(l_in_gt)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i

    global_pred_labels_new[Tidx] = l_in_gt_new
    global_pred_labels[selectidx] = global_pred_labels[selectidx] + len(l_in_unique)
    global_pred_labels_new[selectedidx] = global_pred_labels
    global_pred_labels = global_pred_labels_new

    linunique = np.unique(global_pred_labels[Tidx])
    uunique = np.unique(global_pred_labels[selectedidx])
    allnique = np.unique(global_pred_labels)

    global_masks = np.zeros_like(masks)
    global_masks[:] = 1
    global_masks[np.array(selectedidx[0])[notselectidx]] = 2
    Tidx = np.where(masks != 2)
    global_masks[Tidx] = 0
    global_gt_labels = orilabels



print("##################### L_in ########################")
print(linsize)
if len(global_pred_labels) >= linsize:
    evaluation(global_pred_labels[:linsize], global_gt_labels[:linsize], args.metrics)
else:
    print("No samples in L_in!")
print("##################### U_in ########################")
uinidx = np.where(global_pred_labels[linsize:linsize+uinsize] != -1)[0]
uinidx = uinidx+linsize
print(len(uinidx))
if len(uinidx):
    evaluation(global_pred_labels[uinidx], global_gt_labels[uinidx], args.metrics)
else:
    print("No samples in U_in!")
print("##################### U_out ########################")
uoutidx = np.where(global_pred_labels[linsize+uinsize:] != -1)[0]
uoutidx = uoutidx+linsize+uinsize
print(len(uoutidx))
if len(uoutidx):
    evaluation(global_pred_labels[uoutidx], global_gt_labels[uoutidx], args.metrics)
else:
    print("No samples in U_out!")
print("##################### U ########################")
uidx = np.where(global_pred_labels[linsize:] != -1)[0]
uidx = uidx+linsize
print(len(uidx))
if len(uidx):
    evaluation(global_pred_labels[uidx], global_gt_labels[uidx], args.metrics)
else:
    print("No samples in U!")
print("##################### L+U ########################")
luidx = np.where(global_pred_labels != -1)[0]
print(len(luidx))
evaluation(global_pred_labels[luidx], global_gt_labels[luidx], args.metrics)
selectedlinfile = open("/home/ubuntu/code/Smooth_AP/data/Inaturalist/"+"lin_train_set12.txt", 'w')
for s in luidx:
    selectedlinfile.write(idx2path[s]+'\n')
selectedlinfile.close()
print("##################### new selected samples ########################")
sidx = np.where(global_masks==1)[0]
print(len(sidx))
if len(sidx) != 0:
    evaluation(global_pred_labels[sidx], global_gt_labels[sidx], args.metrics)
print("##################### not selected samples ########################")
nsidx = np.where(global_masks==2)[0]
print(len(nsidx))
if len(nsidx) != 0:
    evaluation(global_pred_labels[nsidx], global_gt_labels[nsidx], args.metrics)

print(global_pred_labels.shape)
print(gtlabels.shape)


with open(args.output_filename, 'wb') as f:
    print(orifeatures.shape)
    print(global_pred_labels.shape)
    print(global_gt_labels.shape)
    print(global_masks.shape)
    pickle.dump([path2idx, orifeatures, global_pred_labels, global_gt_labels, global_masks], f)
