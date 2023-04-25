import argparse, os, pickle, time
import random
import sys

sys.path.append("..")

import shutil

import dgl
import numpy as np
import seaborn
import torch
import torch.optim as optim
from dataset import LanderDataset

from matplotlib import pyplot as plt

from models import LANDER
from utils import build_next_level, decode, evaluation, stop_iterating
from utils.deduce import get_edge_dist

STATISTIC = False

###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_filename", type=str, default="lander.pth")
parser.add_argument("--faiss_gpu", action="store_true")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--output_filename", type=str, default="data/features.pkl")

# HyperParam
parser.add_argument("--knn_k", type=int, default=10)
parser.add_argument("--levels", type=int, default=1)
parser.add_argument("--tau", type=float, default=0.5)
parser.add_argument("--threshold", type=str, default="prob")
parser.add_argument("--metrics", type=str, default="pairwise,bcubed,nmi")
parser.add_argument("--early_stop", action="store_true")

# Model
parser.add_argument("--hidden", type=int, default=512)
parser.add_argument("--num_conv", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--gat", action="store_true")
parser.add_argument("--gat_k", type=int, default=1)
parser.add_argument("--balance", action="store_true")
parser.add_argument("--use_cluster_feat", action="store_true")
parser.add_argument("--use_focal_loss", action="store_true")
parser.add_argument("--use_gt", action="store_true")

# Subgraph
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--mode", type=str, default="1head")
parser.add_argument("--midpoint", type=str, default="false")
parser.add_argument("--linsize", type=int, default=29011)
parser.add_argument("--uinsize", type=int, default=18403)
parser.add_argument("--inclasses", type=int, default=948)
parser.add_argument("--thresh", type=float, default=1.0)

parser.add_argument("--draw", type=str, default="false")
parser.add_argument(
    "--density_distance_pkl", type=str, default="density_distance.pkl"
)
parser.add_argument(
    "--density_lindistance_jpg", type=str, default="density_lindistance.jpg"
)

args = parser.parse_args()
print(args)
MODE = args.mode
linsize = args.linsize
uinsize = args.uinsize
inclasses = args.inclasses

if args.draw == "false":
    args.draw = False
elif args.draw == "true":
    args.draw = True

###########################
# Environment Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

##################
# Data Preparation
with open(args.data_path, "rb") as f:
    loaded_data = pickle.load(f)
    path2idx, features, pred_labels, labels, masks = loaded_data

idx2path = {v: k for k, v in path2idx.items()}
gtlabels = labels

orifeatures = features
orilabels = gtlabels

if MODE == "selectbydensity":
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
elif MODE == "recluster":
    selectedidx = np.where(masks == 1)
    features = features[selectedidx]
    labels = gtlabels[selectedidx]
    labelspred = pred_labels[selectedidx]
    selectmasks = masks[selectedidx]
    gtlabels = gtlabels[selectedidx]
    print("filtered features:", len(features))
else:
    selectedidx = np.where(masks != 0)
    features = features[selectedidx]
    labels = gtlabels[selectedidx]
    labelspred = pred_labels[selectedidx]
    selectmasks = masks[selectedidx]
    gtlabels = gtlabels[selectedidx]
    print("filtered features:", len(features))

global_features = features.copy()  # global features
dataset = LanderDataset(
    features=features, labels=labels, k=args.knn_k, levels=1, faiss_gpu=False
)
g = dataset.gs[0]
g.ndata["pred_den"] = torch.zeros((g.num_nodes()))
g.edata["prob_conn"] = torch.zeros((g.num_edges(), 2))
global_labels = labels.copy()
ids = np.arange(g.num_nodes())
global_edges = ([], [])
global_peaks = np.array([], dtype=np.long)
global_edges_len = len(global_edges[0])
global_num_nodes = g.num_nodes()

global_densities = g.ndata["density"][:linsize]
global_densities = np.sort(global_densities)
xs = np.arange(len(global_densities))

fanouts = [args.knn_k - 1 for i in range(args.num_conv + 1)]
sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
# fix the number of edges
test_loader = dgl.dataloading.DataLoader(
    g,
    torch.arange(g.num_nodes()),
    sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
)

##################
# Model Definition
if not args.use_gt:
    feature_dim = g.ndata["features"].shape[1]
    model = LANDER(
        feature_dim=feature_dim,
        nhid=args.hidden,
        num_conv=args.num_conv,
        dropout=args.dropout,
        use_GAT=args.gat,
        K=args.gat_k,
        balance=args.balance,
        use_cluster_feat=args.use_cluster_feat,
        use_focal_loss=args.use_focal_loss,
    )
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
            global_eid = output_bipartite.edata["global_eid"]
            g.ndata["pred_den"][global_nid] = output_bipartite.dstdata[
                "pred_den"
            ].to("cpu")
            g.edata["prob_conn"][global_eid] = output_bipartite.edata[
                "prob_conn"
            ].to("cpu")
            torch.cuda.empty_cache()
            if (batch + 1) % 10 == 0:
                print("Batch %d / %d for inference" % (batch, total_batches))

    (
        new_pred_labels,
        peaks,
        global_edges,
        global_pred_labels,
        global_peaks,
    ) = decode(
        g,
        args.tau,
        args.threshold,
        args.use_gt,
        ids,
        global_edges,
        global_num_nodes,
        global_peaks,
    )
    if level == 0:
        global_pred_densities = g.ndata["pred_den"]
        global_densities = g.ndata["density"]
        g.edata["prob_conn"] = torch.zeros((g.num_edges(), 2))

    ids = ids[peaks]
    new_global_edges_len = len(global_edges[0])
    num_edges_add_this_level = new_global_edges_len - global_edges_len
    if stop_iterating(
        level,
        args.levels,
        args.early_stop,
        num_edges_add_this_level,
        num_edges_add_last_level,
        args.knn_k,
    ):
        break
    global_edges_len = new_global_edges_len
    num_edges_add_last_level = num_edges_add_this_level

    # build new dataset
    features, labels, cluster_features = build_next_level(
        features,
        labels,
        peaks,
        global_features,
        global_pred_labels,
        global_peaks,
    )
    # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.
    dataset = LanderDataset(
        features=features,
        labels=labels,
        k=args.knn_k,
        levels=1,
        faiss_gpu=False,
        cluster_features=cluster_features,
    )
    g = dataset.gs[0]
    g.ndata["pred_den"] = torch.zeros((g.num_nodes()))
    g.edata["prob_conn"] = torch.zeros((g.num_edges(), 2))
    test_loader = dgl.dataloading.DataLoader(
        g,
        torch.arange(g.num_nodes()),
        sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

if MODE == "selectbydensity":
    thresh = args.thresh
    global_pred_densities = np.array(global_pred_densities).astype(float)
    global_densities = np.array(global_densities).astype(float)
    distance = np.abs(global_pred_densities - global_densities)
    print("densities shape", global_pred_densities.shape)
    print(global_pred_densities.max(), global_pred_densities.min())

    selectidx = np.where(global_pred_densities > thresh)[0]
    selected_pred_densities = global_pred_densities[selectidx]
    selected_densities = global_densities[selectidx]
    selected_distance = np.abs(selected_pred_densities - selected_densities)
    print(np.mean(selected_distance))
    print("number of selected samples:", len(selectidx))

    notselectidx = np.where(global_pred_densities <= thresh)
    print("not selected:", len(notselectidx[0]))
    global_pred_labels[notselectidx] = -1

    global_pred_labels_new = np.zeros_like(orilabels)
    global_pred_labels_new[:] = -1
    Tidx = np.where(masks != 2)
    print("T:", len(Tidx[0]))

    l_in_gt = orilabels[Tidx]
    l_in_features = orifeatures[Tidx]
    l_in_gt_new = np.zeros_like(l_in_gt)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i
    print("len(l_in_unique)", len(l_in_unique))

    if args.draw:
        prototypes = np.zeros((len(l_in_unique), features.shape[1]))
        for i in range(len(l_in_unique)):
            idx = np.where(l_in_gt_new == i)
            prototypes[i] = np.mean(l_in_features[idx], axis=0)

        similarity_matrix = torch.mm(
            torch.from_numpy(global_features.astype(np.float32)),
            torch.from_numpy(prototypes.astype(np.float32)).t(),
        )
        similarity_matrix = (1 - similarity_matrix) / 2
        minvalues, selected_pred_labels = torch.min(similarity_matrix, 1)
        # far-close ratio
        closeidx = np.where(minvalues < 0.15)
        faridx = np.where(minvalues >= 0.15)
        print("far:", len(faridx[0]))
        print("close:", len(closeidx[0]))

        cutidx = np.where(global_pred_densities >= 0.5)
        draw_minvalues = minvalues[cutidx]
        draw_densities = global_pred_densities[cutidx]
        with open(args.density_distance_pkl, "wb") as f:
            pickle.dump((global_pred_densities, minvalues), f)
        print("dumped.")
        plt.clf()
        fig, ax = plt.subplots()
        import random

        if len(draw_densities) > 10000:
            samples_idx = random.sample(range(len(draw_minvalues)), 10000)
            ax.plot(
                draw_densities[random],
                draw_minvalues[random],
                color="tab:blue",
                marker="*",
                linestyle="None",
                markersize=1,
            )
        else:
            ax.plot(
                draw_densities[random],
                draw_minvalues[random],
                color="tab:blue",
                marker="*",
                linestyle="None",
                markersize=1,
            )
        plt.savefig(args.density_lindistance_jpg)

    global_pred_labels_new[Tidx] = l_in_gt_new
    global_pred_labels[selectidx] = global_pred_labels[selectidx] + len(
        l_in_unique
    )
    global_pred_labels_new[selectedidx] = global_pred_labels

    global_pred_labels = global_pred_labels_new
    linunique = np.unique(global_pred_labels[Tidx])
    uunique = np.unique(global_pred_labels[selectedidx])
    allnique = np.unique(global_pred_labels)
    print("labels")
    print(len(linunique), len(uunique), len(allnique))

    global_masks = np.zeros_like(masks)
    global_masks[:] = 1
    global_masks[np.array(selectedidx[0])[notselectidx]] = 2
    Tidx = np.where(masks != 2)
    global_masks[Tidx] = 0
    print("mask0", len(np.where(global_masks == 0)[0]))
    print("mask1", len(np.where(global_masks == 1)[0]))
    print("mask2", len(np.where(global_masks == 2)[0]))
    print("all", len(masks), len(orilabels), len(orifeatures))

    global_gt_labels = orilabels

if MODE == "recluster":
    global_pred_labels_new = np.zeros_like(orilabels)
    global_pred_labels_new[:] = -1
    Tidx = np.where(masks == 0)
    print("T:", len(Tidx[0]))

    l_in_gt = orilabels[Tidx]
    l_in_features = orifeatures[Tidx]
    l_in_gt_new = np.zeros_like(l_in_gt)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i
    print("len(l_in_unique)", len(l_in_unique))

    global_pred_labels_new[Tidx] = l_in_gt_new
    print(len(global_pred_labels))
    print(len(selectedidx[0]))
    global_pred_labels_new[selectedidx[0]] = global_pred_labels + len(
        l_in_unique
    )
    global_pred_labels = global_pred_labels_new
    global_masks = masks
    print("mask0", len(np.where(global_masks == 0)[0]))
    print("mask1", len(np.where(global_masks == 1)[0]))
    print("mask2", len(np.where(global_masks == 2)[0]))
    print("all", len(masks), len(orilabels), len(orifeatures))
    global_gt_labels = orilabels

if MODE == "donothing":
    global_masks = masks
    pass

print("##################### L_in ########################")
print(linsize)
if len(global_pred_labels) >= linsize:
    evaluation(
        global_pred_labels[:linsize], global_gt_labels[:linsize], args.metrics
    )
else:
    print("No samples in L_in!")
print("##################### U_in ########################")
uinidx = np.where(global_pred_labels[linsize : linsize + uinsize] != -1)[0]
uinidx = uinidx + linsize
print(len(uinidx))
if len(uinidx):
    evaluation(
        global_pred_labels[uinidx], global_gt_labels[uinidx], args.metrics
    )
else:
    print("No samples in U_in!")
print("##################### U_out ########################")
uoutidx = np.where(global_pred_labels[linsize + uinsize :] != -1)[0]
uoutidx = uoutidx + linsize + uinsize
print(len(uoutidx))
if len(uoutidx):
    evaluation(
        global_pred_labels[uoutidx], global_gt_labels[uoutidx], args.metrics
    )
else:
    print("No samples in U_out!")
print("##################### U ########################")
uidx = np.where(global_pred_labels[linsize:] != -1)[0]
uidx = uidx + linsize
print(len(uidx))
if len(uidx):
    evaluation(global_pred_labels[uidx], global_gt_labels[uidx], args.metrics)
else:
    print("No samples in U!")
print("##################### L+U ########################")
luidx = np.where(global_pred_labels != -1)[0]
print(len(luidx))
evaluation(global_pred_labels[luidx], global_gt_labels[luidx], args.metrics)
print("##################### new selected samples ########################")
sidx = np.where(global_masks == 1)[0]
print(len(sidx))
if len(sidx) != 0:
    evaluation(global_pred_labels[sidx], global_gt_labels[sidx], args.metrics)
print("##################### not selected samples ########################")
nsidx = np.where(global_masks == 2)[0]
print(len(nsidx))
if len(nsidx) != 0:
    evaluation(global_pred_labels[nsidx], global_gt_labels[nsidx], args.metrics)

with open(args.output_filename, "wb") as f:
    print(orifeatures.shape)
    print(global_pred_labels.shape)
    print(global_gt_labels.shape)
    print(global_masks.shape)
    pickle.dump(
        [
            path2idx,
            orifeatures,
            global_pred_labels,
            global_gt_labels,
            global_masks,
        ],
        f,
    )
