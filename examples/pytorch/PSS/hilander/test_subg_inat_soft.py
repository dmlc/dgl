import argparse, time, os, pickle
from utils.deduce import get_edge_dist
import numpy as np
import shutil

import dgl
import torch
import torch.optim as optim

from models import LANDER
from dataset import LanderDataset, LanderDatasetSoft
from utils import evaluation, decode, build_next_level, stop_iterating

# MODE in ["uingt_uoutgt", "uingt", "uingt+uoutreducedcluster", "uingt+uoutsplitcluster", "splituinuout",
#          "oraclereduceclusters", "recluster", "oracle+cscore+hscore", "splituinuoutfeatures"]

STATISTIC = False

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
parser.add_argument('--soft', type=str, default="true")

args = parser.parse_args()
print(args)

if args.soft == 'true':
    args.soft = True
elif args.soft == "false":
    args.soft = False
else:
    print("Invalid argument soft choice.")

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

with open("data/groundtruthlabels.pkl", 'rb') as gtf:
    gtlabels = pickle.load(gtf)

orifeatures = features
orilabels = gtlabels

if MODE == "selectbydensity":
    # selectedidx = np.where(masks == 2)
    selectedidx = np.where(masks != 0)
    features = features[selectedidx]
    labels = labels[selectedidx]
    selectmasks = masks[selectedidx]
    print("filtered features:", len(features))
    print("mask0:", len(np.where(masks == 0)[0]))
    print("mask1:", len(np.where(masks == 1)[0]))
    print("mask2:", len(np.where(masks == 2)[0]))

elif MODE == "newsplituinuout":
    features = features[linsize + uinsize:]
    labels = labels[linsize + uinsize:]

elif MODE == "onlyuincluster":
    features = features[:linsize + uinsize]
    labels = labels[:linsize + uinsize]

else:
    selectedidx = np.where(pred_labels != -1)
    # selectedidx = np.where(masks != 2)
    features = features[selectedidx]
    labels = labels[selectedidx]
    selectmasks = masks[selectedidx]
    print("filtered features:", len(features))

global_features = features.copy()  # global features
dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                        levels=1, faiss_gpu=args.faiss_gpu)
g = dataset.gs[0]
g.ndata['pred_den'] = torch.zeros((g.number_of_nodes()))
g.edata['prob_conn'] = torch.zeros((g.number_of_edges(), 2))
global_labels = labels.copy()
ids = np.arange(g.number_of_nodes())
global_edges = ([], [])
global_peaks = np.array([], dtype=np.long)
global_edges_len = len(global_edges[0])
global_num_nodes = g.number_of_nodes()

# srcfeature = g.ndata['features']
# dstfeature = g.ndata['features']
# norm = g.ndata['norm']
# distances = torch.cdist(srcfeature, dstfeature) / norm
# distances = np.array(torch.flatten(distances))
idx1 = np.where(np.array(g.edata['labels_conn'] == 1))
idx2 = np.where(np.array(g.edata['labels_conn'] == 0))
dis1 = np.array(g.edata['raw_affine'])[idx1]
dis2 = np.array(g.edata['raw_affine'])[idx2]
print("dis1.shape", dis1.shape)
print("dis2.shape", dis2.shape)

from matplotlib import pyplot as plt
import seaborn

plt.clf()
fig = seaborn.histplot(data=dis1, stat="probability", color="skyblue", multiple="layer", kde=True, label="same labels")
fig = seaborn.histplot(data=dis2, stat="probability", color="orange", multiple="layer", kde=True,
                       label="different labels")
histfig = fig.get_figure()
plt.legend()
histfig.savefig("./edge_distances.png")

fanouts = [args.knn_k - 1 for i in range(args.num_conv + 1)]
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

    new_pred_labels, peaks, \
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
    if stop_iterating(level, args.levels, args.early_stop, num_edges_add_this_level, num_edges_add_last_level,
                      args.knn_k):
        break
    global_edges_len = new_global_edges_len
    num_edges_add_last_level = num_edges_add_this_level

    # build new dataset
    features, labels, cluster_features = build_next_level(features, labels, peaks,
                                                          global_features, global_pred_labels, global_peaks)
    # After the first level, the number of nodes reduce a lot. Using cpu faiss is faster.
    dataset = LanderDataset(features=features, labels=labels, k=args.knn_k,
                                levels=1, faiss_gpu=False, cluster_features=cluster_features)
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
print(global_pred_labels.shape)
print(global_labels.shape)
print(args.metrics)

from collections import Counter

cnts = Counter(global_pred_labels)
res = cnts.values()
plt.clf()
clusterfig = seaborn.histplot(data=res, stat="probability", color="skyblue", multiple="layer", kde=True,
                              label="same labels")
clusterhistfig = clusterfig.get_figure()
plt.legend()
clusterhistfig.savefig("./clustersize.png")

print("##################### U_in ########################")
insize = 3550
evaluation(global_pred_labels[linsize:linsize + insize], global_labels[linsize:linsize + insize], args.metrics)
print("##################### U_out ########################")
evaluation(global_pred_labels[linsize + insize:], global_labels[linsize + insize:], args.metrics)
print("##################### U ########################")
evaluation(global_pred_labels[linsize:], global_labels[linsize:], args.metrics)
print("##################### L+U ########################")
evaluation(global_pred_labels, global_labels, args.metrics)


if MODE == "selectbydensity":
    thresh = args.thresh
    global_pred_densities = np.array(global_pred_densities).astype(float)
    global_densities = np.array(global_densities).astype(float)
    distance = np.abs(global_pred_densities - global_densities)
    print("densities shape", global_pred_densities.shape)
    print(global_pred_densities.max(), global_pred_densities.min())

    plt.clf()
    densityfig = seaborn.histplot(data=global_pred_densities, stat="probability", color="skyblue", multiple="layer",
                                  kde=True, label="densities")
    densityfig = densityfig.get_figure()
    plt.legend()
    densityfig.savefig("./density_pred.png")

    plt.clf()
    densityfig1 = seaborn.histplot(data=global_densities, stat="probability", color="skyblue", multiple="layer",
                                   kde=True, label="densities")
    densityfig1 = densityfig1.get_figure()
    plt.legend()
    densityfig1.savefig("./density_gt.png")

    selectidx = np.where(global_pred_densities > thresh)[0]
    selected_pred_densities = global_pred_densities[selectidx]
    selected_densities = global_densities[selectidx]
    selected_distance = np.abs(selected_pred_densities - selected_densities)
    print(np.mean(selected_distance))
    print("number of selected samples:", len(selectidx))
    uinidx = np.where(selectidx < uinsize)[0]
    uoutidx = np.where(selectidx >= uinsize)[0]
    print("number of Uin selected samples:", len(uinidx))
    print("number of Uout selected samples:", len(uoutidx))

    plt.clf()
    densityfig2 = seaborn.histplot(data=selected_densities, stat="probability", color="skyblue", multiple="layer",
                                   kde=True, label="densities")
    densityfig2 = densityfig2.get_figure()
    plt.legend()
    densityfig2.savefig("./density_gt_selected.png")

    newselectidx = np.where(distance < 0.1)
    newselected_pred_densities = global_pred_densities[newselectidx]
    newselected_densities = global_densities[newselectidx]
    plt.clf()
    densityfig3 = seaborn.histplot(data=newselected_pred_densities, stat="probability", color="skyblue",
                                   multiple="layer", kde=True, label="densities")
    densityfig3 = densityfig3.get_figure()
    plt.legend()
    densityfig3.savefig("./density_pred_selectedbydis.png")

    plt.clf()
    densityfig4 = seaborn.histplot(data=newselected_densities, stat="probability", color="skyblue", multiple="layer",
                                   kde=True, label="densities")
    densityfig4 = densityfig4.get_figure()
    plt.legend()
    densityfig4.savefig("./density_gt_selectedbydis.png")

    notselectidx = np.where(global_pred_densities <= thresh)
    print("not selected:", len(notselectidx[0]))
    global_pred_labels[notselectidx] = -1

    global_pred_labels_new = np.zeros_like(orilabels)
    global_pred_labels_new[:] = -1
    Tidx = np.where(masks != 2)
    print("T:", len(Tidx[0]))
    global_pred_labels_new[Tidx] = orilabels[Tidx]
    global_pred_labels_new[selectedidx] = global_pred_labels
    global_pred_labels = global_pred_labels_new

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

if MODE == "PL":
    l_in_gt = orilabels[:linsize]
    l_in_features = orifeatures[:linsize]
    l_in_gt_new = np.zeros_like(l_in_gt)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i

    u_gt = orilabels[linsize:]
    u_gt_new = np.zeros_like(u_gt)
    for i in range(len(l_in_unique)):
        u = l_in_unique[i]
        u_idx = np.where(u_gt == u)
        u_gt_new[u_idx] = i

    prototypes = np.zeros((inclasses, l_in_features.shape[1]))
    for i in range(inclasses):
        idx = np.where(l_in_gt_new == i)
        prototypes[i] = np.mean(l_in_features[idx], axis=0)

    similarity_matrix = torch.mm(torch.from_numpy(orifeatures[linsize:].astype(np.float32)), \
                                 torch.from_numpy(prototypes.astype(np.float32)).t())
    maxvalues, u_pred_labels = torch.max(similarity_matrix, 1)

    global_pred_labels[:linsize] = l_in_gt_new
    global_pred_labels[linsize:] = u_pred_labels
    global_gt_labels = orilabels
    global_masks = masks

## select samples using similarity threshold DONE
if MODE == "selectUsim":
    l_in_gt = orilabels[:linsize]
    l_in_features = orifeatures[:linsize]
    l_in_gt_new = np.zeros_like(l_in_gt)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i

    u_in_gt = orilabels[linsize:linsize + uinsize]
    u_in_gt_new = np.zeros_like(u_in_gt)
    for i in range(len(l_in_unique)):
        u_in = l_in_unique[i]
        u_in_idx = np.where(u_in_gt == u_in)
        u_in_gt_new[u_in_idx] = i

    prototypes = np.zeros((inclasses, l_in_features.shape[1]))
    for i in range(inclasses):
        idx = np.where(l_in_gt_new == i)
        prototypes[i] = np.mean(l_in_features[idx], axis=0)

    similarity_matrix = torch.mm(torch.from_numpy(orifeatures[linsize:].astype(np.float32)), \
                                 torch.from_numpy(prototypes.astype(np.float32)).t())
    maxvalues, u_pred_labels = torch.max(similarity_matrix, 1)

    selected = []
    uincnt = 0
    for i in range(len(maxvalues)):
        if maxvalues[i] < thresh:
            u_pred_labels[i] = -1
        else:
            selected.append(i)
            if i < uinsize:
                uincnt += 1
    print("selected samples in U_in:", uincnt)
    print("selected samples:", len(selected))

    correct = 0
    for i in selected:
        if u_pred_labels[i] != -1 and i < uinsize and u_pred_labels[i] == u_in_gt_new[i]:
            correct += 1
    print("Acc:", 1.0 * correct / uincnt)

    global_pred_labels = np.concatenate((l_in_gt_new, u_pred_labels))

    plt.clf()
    uinmax = np.array(maxvalues)[:uinsize]
    uoutmax = np.array(maxvalues)[uinsize:]
    fig1 = seaborn.histplot(data=uinmax, stat="probability", color="skyblue", multiple="layer", kde=True, label="Uin")
    fig1 = seaborn.histplot(data=uoutmax, stat="probability", color="orange", multiple="layer", kde=True, label="Uout")
    # fig = seaborn.histplot(data=dis11, stat="probability", color="blue", multiple="layer", kde=True, label="same labels (margin)")
    # fig = seaborn.histplot(data=dis21, stat="probability", color="red", multiple="layer", kde=True, label="different labels (margin)")
    histfig1 = fig1.get_figure()
    plt.legend()
    histfig1.savefig("./u_maxprob.png")

    plt.clf()
    uinsim = np.array(similarity_matrix)[:uinsize]
    # uinXs = [[i for i in range(inclasses)] for _ in range(len(uinsim))]
    # plt.plot(uinXs, uinsim, color='skyblue', alpha=0.3)
    # plt.savefig("./uinlogits.png")

    plt.clf()
    uoutsim = np.array(similarity_matrix)[uinsize:]
    # uoutXs = [[i for i in range(inclasses)] for _ in range(len(uoutsim))]
    # plt.plot(uoutXs, uoutsim, color='orange', alpha=0.3)
    # plt.savefig("./uoutlogits.png")

    from scipy.stats import entropy

    plt.clf()
    from sklearn.preprocessing import normalize

    uinsimidx = np.where(uinsim != -1)
    uoutsimidx = np.where(uoutsim != -1)
    minuin = np.min(uinsim[uinsimidx], axis=1)
    minuout = np.min(uoutsim[uoutsimidx], axis=1)
    uinsimnorm = normalize(uinsim - minuin[:, None], axis=1)
    uoutsimnorm = normalize(uoutsim - minuout[:, None], axis=1)
    uinentropy = np.array(entropy(uinsimnorm, axis=1))
    uoutentropy = np.array(entropy(uoutsimnorm, axis=1))
    print(uinentropy)
    print(uoutentropy)

    fig2 = seaborn.histplot(data=uinentropy, stat="probability", color="skyblue", multiple="layer", kde=True,
                            label="Uin")
    fig2 = seaborn.histplot(data=uoutentropy, stat="probability", color="orange", multiple="layer", kde=True,
                            label="Uout")
    histfig2 = fig2.get_figure()
    plt.legend()
    histfig2.savefig("./entropyselected.png")

    plt.clf()
    import matplotlib.patheffects as PathEffects


    def fashion_scatter(x, colors):
        # choose a color palette with seaborn.
        num_classes = len(np.unique(colors))
        palette = np.array(seaborn.color_palette("hls", num_classes))

        # create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # add the labels for each digit corresponding to the label
        txts = []

        for i in range(num_classes):
            # Position of each label at median of data points.

            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        return f, ax, sc, txts


    from sklearn import manifold

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(orifeatures)
    f, ax, sc, txts = fashion_scatter(X_tsne, orilabels)
    f.savefig("./tsne.png")

## use oracle to split U_in and U_out, use pseudo label for U_in and cluster label for U_out
## DONE
if MODE == "splituinuout":
    l_in_gt = orilabels[:linsize]
    l_in_features = orifeatures[:linsize]
    l_in_gt_new = np.zeros_like(l_in_gt)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i

    prototypes = np.zeros((inclasses, l_in_features.shape[1]))
    for i in range(inclasses):
        idx = np.where(l_in_gt_new == i)
        prototypes[i] = np.mean(l_in_features[idx], axis=0)

    similarity_matrix = torch.mm(torch.from_numpy(orifeatures[linsize:linsize + uinsize].astype(np.float32)), \
                                 torch.from_numpy(prototypes.astype(np.float32)).t())
    uin_maxvalues, uin_pred_labels = torch.max(similarity_matrix, 1)

    uout_pred_labels_new = np.zeros_like(global_pred_labels[linsize + uinsize:])
    uout_pred_labels_new[:] = -1
    uout_pred_labels = global_pred_labels[linsize + uinsize:]
    uout_pred_uniques = np.unique(uout_pred_labels)
    cnt = inclasses
    for newi in uout_pred_uniques:
        idx = np.where(uout_pred_labels == newi)
        if (idx[0].shape[0] != 1):
            uout_pred_labels_new[idx] = cnt
            cnt += 1

    global_pred_labels[:linsize] = l_in_gt_new
    global_pred_labels[linsize:linsize + uinsize] = uin_pred_labels
    global_pred_labels[linsize + uinsize:] = uout_pred_labels_new

if MODE == "newsplituinuout":
    l_in_gt = orilabels[:linsize]
    l_in_features = orifeatures[:linsize]
    prototypes = np.zeros((948, l_in_features.shape[1]))
    for i in range(948):
        idx = np.where(l_in_gt == i)
        prototypes[i] = np.mean(l_in_features[idx])

    from scipy.spatial import distance

    similarity_matrix = torch.tensor(distance.cdist(orifeatures[linsize:linsize + uinsize], prototypes, 'cosine'))
    u_in_pred_labels = torch.argmax(similarity_matrix, dim=1)

    u_out_pred_labels = np.zeros_like(global_pred_labels)
    u_out_unique = np.unique(global_pred_labels)
    for u_out_label in u_out_unique:
        idx = np.where(global_pred_labels == u_out_label)
        u_out_pred_labels[idx] = u_out_label + 948

    global_pred_labels = np.concatenate((l_in_gt, u_in_pred_labels, u_out_pred_labels))
    global_features = orifeatures

## U_in gt + U_out pseudo
if MODE == "uingt+uoutcluster":
    global_pred_labels[linsize:linsize + uinsize] = global_labels[linsize:linsize + uinsize]
    uout_pred_labels = global_pred_labels[linsize + uinsize:]
    uout_pred_labels_new = np.zeros_like(uout_pred_labels)
    uout_pred_uniques = np.unique(uout_pred_labels)
    cnt = inclasses
    for newi in uout_pred_uniques:
        idx = np.where(uout_pred_labels == newi)
        uout_pred_labels_new[idx] = cnt
        cnt += 1
    global_pred_labels[linsize + uinsize:] = uout_pred_labels_new
    global_pred_labels[:linsize] = global_labels[:linsize]

## U_in gt + U_out pseudo nonsingle
if MODE == "uingt+uoutclusternonsingle":
    global_pred_labels[linsize:linsize + uinsize] = global_labels[linsize:linsize + uinsize]
    uout_pred_labels = global_pred_labels[linsize + uinsize:]
    uout_pred_labels_new = np.zeros_like(uout_pred_labels)
    uout_pred_uniques = np.unique(uout_pred_labels)
    cnt = inclasses
    for newi in uout_pred_uniques:
        idx = np.where(uout_pred_labels == newi)
        if (idx[0].shape[0] != 1):
            uout_pred_labels_new[idx] = cnt
            cnt += 1
    global_pred_labels[:linsize] = global_labels[:linsize]
    global_pred_labels[linsize + uinsize:] = uout_pred_labels_new

## 2head DONE
if MODE == "2head":
    l_in_labels = global_pred_labels[:linsize]
    l_in_gt = global_labels[:linsize]
    l_in_gt_new = np.zeros_like(l_in_gt)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i

    u_pred_labels = global_pred_labels[linsize:]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    u_unique = np.unique(u_pred_labels)
    new_id = inclasses
    for u_l in u_unique:
        idx = np.where(l_in_labels == u_l)
        u_pred_labels_new[idx2] = new_id
        new_id += 1
    global_pred_labels[linsize:] = u_pred_labels_new
    global_pred_labels[:linsize] = l_in_gt_new

## 1head DONE
if MODE == "1head":
    Tidx = np.where(selectmasks == 0)
    T_labels = global_pred_labels[Tidx]
    T_gt = orilabels[Tidx]
    T_gt_new = np.zeros_like(T_gt)
    T_unique = np.unique(T_gt)
    for i in range(len(T_unique)):
        T = T_unique[i]
        T_idx = np.where(T_gt == T)
        T_gt_new[T_idx] = i

    Uidx = np.where(selectmasks != 0)
    u_pred_labels = global_pred_labels[Uidx]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    u_unique = np.unique(u_pred_labels)
    new_id = len(T_unique)
    for u_l in u_unique:
        idx = np.where(T_labels == u_l)
        gts = T_gt_new[idx]
        idx2 = np.where(u_pred_labels == u_l)
        if len(gts) != 0:
            gt_l = np.argmax(np.bincount(gts))
            u_pred_labels_new[idx2] = gt_l
        else:
            u_pred_labels_new[idx2] = new_id
            new_id += 1
    global_pred_labels[Uidx] = u_pred_labels_new
    global_pred_labels[Tidx] = T_gt_new

    uidx = np.where(masks == 2)
    U_gt = orilabels[uidx]
    U_gt_new = np.zeros_like(masks[uidx])
    U_unique = np.unique(U_gt)
    for U_l in U_unique:
        idx2 = np.where(U_gt == U_l)
        U_gt_new[idx2] = new_id
        new_id += 1

    global_pred_labels_new = np.zeros_like(orilabels)
    global_pred_labels_new[:] = -1
    global_pred_labels_new[selectedidx] = global_pred_labels
    global_masks = masks

    global_gt_labels = np.zeros_like(orilabels)
    global_gt_labels[selectedidx] = global_pred_labels
    global_gt_labels[uidx] = U_gt_new
    global_pred_labels = global_pred_labels_new

if MODE == "1headsoft":
    Tidx = np.where(selectmasks == 0)
    T_labels = global_pred_labels[Tidx]
    T_gt = orilabels[Tidx]
    T_gt_new = np.zeros_like(T_gt)
    T_unique = np.unique(T_gt)
    # get unique labels for U
    Uidx = np.where(selectmasks != 0)
    u_pred_labels = global_pred_labels[Uidx]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    u_unique = np.unique(u_pred_labels)
    # init soft labels
    class_num = len(T_unique) + len(u_unique)
    soft_labels = np.zeros(len(global_pred_labels), class_num)
    # T soft labels
    for i in range(len(T_unique)):
        T = T_unique[i]
        T_idx = np.where(T_gt == T)
        T_gt_new[T_idx] = i
        soft_labels[T_idx][i] = 1
    # U soft labels
    new_id = len(T_unique)
    for u_l in u_unique:
        idx = np.where(T_labels == u_l)
        gts = T_gt_new[idx]
        idx2 = np.where(u_pred_labels == u_l)
        softlabel_idx = len(T_labels) + idx2
        if len(gts) != 0:
            cluster_size = len(idx[0])
            cnts = Counter(gts)
            cnts = cnts.items()
            for ori_cnt in cnts:
                soft_labels[softlabel_idx][ori_cnt[0]] = ori_cnt[1] * 1.0 / cluster_size
            soft_labels[softlabel_idx][new_id] = 1.0 - np.sum(soft_labels[softlabel_idx])
            new_id += 1
        else:
            soft_labels[softlabel_idx][new_id] = 1
            new_id += 1

    uidx = np.where(masks == 2)
    U_gt = orilabels[uidx]
    U_gt_new = np.zeros_like(masks[uidx])
    U_unique = np.unique(U_gt)
    u_class_num = class_num + len(U_unique)
    U_gt_new_softlabels = np.zeros(len(uidx[0]), u_class_num)
    for U_l in U_unique:
        idx2 = np.where(U_gt == U_l)
        U_gt_new[idx2] = new_id
        U_gt_new_softlabels[new_id] = 1
        new_id += 1

    soft_labels = np.pad(soft_labels, (0, len(U_unique)), 'constant', constant_values=(0,0))
    global_pred_labels = soft_labels
    global_masks = masks

    global_gt_labels = np.zeros_like(len(orilabels), class_num)
    global_gt_labels[selectedidx] = global_pred_labels
    global_gt_labels[uidx] = U_gt_new_softlabels


if MODE == "1headsofttmp":
    Tidx = np.where(selectmasks == 0)
    T_labels = global_pred_labels[Tidx]
    T_gt = orilabels[Tidx]
    T_gt_new = np.zeros_like(T_gt)
    T_unique = np.unique(T_gt)
    for i in range(len(T_unique)):
        T = T_unique[i]
        T_idx = np.where(T_gt == T)
        T_gt_new[T_idx] = i

    # get unique labels for U
    Uidx = np.where(selectmasks != 0)
    u_pred_labels = global_pred_labels[Uidx]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    u_unique = np.unique(u_pred_labels)
    # init soft labels
    class_num = len(T_unique) + len(u_unique)
    soft_labels = np.zeros((len(global_pred_labels), class_num))
    # T soft labels
    for i in range(len(T_unique)):
        T = T_unique[i]
        T_idx = np.where(T_gt == T)
        T_gt_new[T_idx] = i
        soft_labels[T_idx, i] = 1

    # U soft labels
    new_id = len(T_unique)
    for u_l in u_unique:
        idx = np.where(T_labels == u_l)[0]
        gts = T_gt_new[idx]
        idx2 = np.where(u_pred_labels == u_l)[0]
        softlabel_idx = idx2 + len(T_labels)
        if len(gts) != 0:
            cluster_size = len(gts)
            cnts = Counter(gts)
            cnts = cnts.items()
            for ori_cnt in cnts:
                soft_labels[softlabel_idx, ori_cnt[0]] = (ori_cnt[1] * 1.0) / cluster_size
            soft_labels[softlabel_idx, new_id] = 1.0 - np.sum(soft_labels[softlabel_idx])/len(softlabel_idx)

            u_pred_labels_new[idx2] = np.argmax(soft_labels[softlabel_idx])
            new_id += 1
        else:
            soft_labels[softlabel_idx, new_id] = 1
            u_pred_labels_new[idx2] = new_id
            new_id += 1
    global_pred_labels[Uidx] = u_pred_labels_new
    global_pred_labels[Tidx] = T_gt_new

    uidx = np.where(masks == 2)
    U_gt = orilabels[uidx]
    U_gt_new = np.zeros_like(masks[uidx])
    U_unique = np.unique(U_gt)
    for U_l in U_unique:
        idx2 = np.where(U_gt == U_l)
        U_gt_new[idx2] = new_id
        new_id += 1

    global_pred_labels_new = np.zeros_like(orilabels)
    global_pred_labels_new[:] = -1
    global_pred_labels_new[selectedidx] = global_pred_labels
    global_masks = masks

    global_gt_labels = np.zeros_like(orilabels)
    global_gt_labels[selectedidx] = global_pred_labels
    global_gt_labels[uidx] = U_gt_new
    global_pred_labels = global_pred_labels_new

    soft_labels_new = np.zeros((len(orilabels), class_num))
    soft_labels_new[selectedidx] = soft_labels
    soft_labels = soft_labels_new

    print(soft_labels.shape)
    print(global_gt_labels.shape)
    print(global_pred_labels.shape)
    print(global_masks.shape)

    sel = np.where(global_masks!=2)[0]
    print(len(sel))
    print(soft_labels[sel].max(1))
    print(global_pred_labels[sel])


if MODE == "1headnotfix":
    T_labels = global_pred_labels[:linsize]
    T_gt = orilabels[:linsize]
    T_gt_new = np.zeros_like(T_gt)
    T_unique = np.unique(T_gt)
    for i in range(len(T_unique)):
        T = T_unique[i]
        T_idx = np.where(T_gt == T)
        T_gt_new[T_idx] = i

    u_pred_labels = global_pred_labels[linsize:]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    u_unique = np.unique(u_pred_labels)
    new_id = len(T_unique)
    for u_l in u_unique:
        idx = np.where(T_labels == u_l)
        gts = T_gt_new[idx]
        idx2 = np.where(u_pred_labels == u_l)
        if len(gts) != 0:
            gt_l = np.argmax(np.bincount(gts))
            u_pred_labels_new[idx2] = gt_l
        else:
            u_pred_labels_new[idx2] = new_id
            new_id += 1
    global_pred_labels[linsize:] = u_pred_labels_new
    global_pred_labels[:linsize] = T_gt_new

    uidx = np.where(masks == 2)
    U_gt = orilabels[uidx]
    U_gt_new = np.zeros_like(masks[uidx])
    U_unique = np.unique(U_gt)
    for U_l in U_unique:
        idx2 = np.where(U_gt == U_l)
        U_gt_new[idx2] = new_id
        new_id += 1

    global_pred_labels_new = np.zeros_like(orilabels)
    global_pred_labels_new[:] = -1
    global_pred_labels_new[selectedidx] = global_pred_labels
    global_masks = masks

    global_gt_labels = np.zeros_like(orilabels)
    global_gt_labels[selectedidx] = global_pred_labels
    global_gt_labels[uidx] = U_gt_new
    global_pred_labels = global_pred_labels_new

if MODE == "onlyuincluster":
    uin_pred_labels = global_pred_labels[linsize:linsize + uinsize]
    uin_pred_labels_new = np.zeros_like(uin_pred_labels)
    l_in_labels = global_pred_labels[:linsize]
    l_in_gt = orilabels[:linsize]
    l_in_gt_new = np.zeros_like(l_in_gt)
    uin_unique = np.unique(uin_pred_labels)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i

    new_id = inclasses
    for u_l in uin_unique:
        idx = np.where(l_in_labels == u_l)
        gts = l_in_gt_new[idx]
        idx2 = np.where(uin_pred_labels == u_l)
        if len(gts) != 0:
            gt_l = np.argmax(np.bincount(gts))
            uin_pred_labels_new[idx2] = gt_l
        else:
            uin_pred_labels_new[idx2] = new_id
            new_id += 1

    new_global_pred_labels = np.zeros_like(orilabels)
    new_global_pred_labels[linsize:linsize + uinsize] = uin_pred_labels_new
    new_global_pred_labels[:linsize] = l_in_gt_new
    new_global_pred_labels[linsize + uinsize:] = -1
    global_pred_labels = new_global_pred_labels
    global_features = orifeatures

    u_in_gt = global_labels[linsize:linsize + uinsize]
    u_in_gt_new = np.zeros_like(u_in_gt)
    for i in range(len(l_in_unique)):
        u_in = l_in_unique[i]
        u_in_idx = np.where(u_in_gt == u_in)
        u_in_gt_new[u_in_idx] = i

    correct = 0
    for i in range(len(uin_pred_labels_new)):
        if uin_pred_labels_new[i] == u_in_gt_new[i]:
            correct += 1
    print("Acc:", 1.0 * correct / linsize)

## recluster DONE
if MODE == "recluster":
    u_pred_labels = global_pred_labels[linsize:]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    u_unique = np.unique(u_pred_labels)

    l_in_labels = global_pred_labels[:linsize]
    l_in_gt = global_labels[:linsize]
    l_in_gt_new = np.zeros_like(l_in_gt)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i

    new_id = inclasses
    for u_l in u_unique:
        idx = np.where(l_in_labels == u_l)
        gts = l_in_gt_new[idx]
        idx2 = np.where(u_pred_labels == u_l)
        if len(gts) != 0:
            gt_l = np.argmax(np.bincount(gts))
            u_pred_labels_new[idx2] = gt_l
        else:
            u_pred_labels_new[idx2] = new_id
            new_id += 1
    global_pred_labels[linsize:] = u_pred_labels_new
    global_pred_labels[:linsize] = l_in_gt_new
    new_pred_labels = np.zeros_like(orilabels)
    new_pred_labels[:] = -1
    new_pred_labels[selectedidx] = global_pred_labels
    global_pred_labels = new_pred_labels

if MODE == "recluster-reduce":
    new_global_pred_labels = np.zeros_like(global_pred_labels)
    new_global_pred_labels[:] = -1
    unique = np.unique(global_pred_labels)
    global_label = 0
    for label in unique:
        idx = np.where(global_pred_labels == label)
        cluster_gt = global_labels[idx]
        cluster_gt_unique = np.unique(cluster_gt)

        cnts = Counter(cluster_gt)
        max_cluster_gt = cnts.most_common(1)[0][0]
        gt_idx = np.where(cluster_gt == max_cluster_gt)
        # print("gt_idx", gt_idx)
        # print(global_node_densities[idx])
        new_idx = idx[0][gt_idx[0]]
        new_global_pred_labels[new_idx] = global_label
        global_label += 1

    u_pred_labels = new_global_pred_labels[linsize:]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    l_in_labels = new_global_pred_labels[:linsize]
    l_in_gt = global_labels[:linsize]
    u_unique = np.unique(u_pred_labels)
    new_id = inclasses
    for u_l in u_unique:
        if u_l == -1:
            continue
        else:
            idx = np.where(l_in_labels == u_l)
            gts = l_in_gt[idx]
            idx2 = np.where(u_pred_labels == u_l)
            if len(gts) != 0:
                gt_l = np.argmax(np.bincount(gts))
                u_pred_labels_new[idx2] = gt_l
            else:
                u_pred_labels_new[idx2] = new_id
                new_id += 1
    new_global_pred_labels[linsize:] = u_pred_labels_new
    new_global_pred_labels[:linsize] = global_labels[:linsize]
    new_pred_labels = np.zeros_like(orilabels)
    new_pred_labels[:] = -1
    new_pred_labels[selectedidx] = new_global_pred_labels
    global_pred_labels = new_pred_labels
    print(len(global_pred_labels))
    print(len(orilabels))

if MODE == "1headnonsingle":
    u_pred_labels = global_pred_labels[linsize:]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    l_in_labels = global_pred_labels[:linsize]
    l_in_gt = global_labels[:linsize]
    u_unique = np.unique(u_pred_labels)
    new_id = inclasses
    for u_l in u_unique:
        idx = np.where(l_in_labels == u_l)
        gts = l_in_gt[idx]
        idx2 = np.where(u_pred_labels == u_l)
        if len(gts) != 0:
            gt_l = np.argmax(np.bincount(gts))
            u_pred_labels_new[idx2] = gt_l
        else:
            u_pred_labels_new[idx2] = new_id
            new_id += 1
    global_pred_labels[linsize:] = u_pred_labels_new
    global_pred_labels[:linsize] = global_labels[:linsize]

    u_pred_labels = global_pred_labels[linsize:]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    u_pred_uniques = np.unique(u_pred_labels)
    cnt = inclasses
    for newi in u_pred_uniques:
        idx = np.where(uout_pred_labels == newi)
        if (idx[0].shape[0] != 1):
            u_pred_labels_new[idx] = cnt
            cnt += 1
    global_pred_labels[:linsize] = global_labels[:linsize]
    global_pred_labels[linsize:] = uout_pred_labels_new

## oracle split clusters
if MODE == "splitclusters":
    new_global_pred_labels = np.zeros_like(global_pred_labels)
    unique = np.unique(global_pred_labels)
    global_label = 1
    for label in unique:
        idx = np.where(global_pred_labels == label)
        cluster_gt = global_labels[idx]
        cluster_gt_unique = np.unique(cluster_gt)
        for cluster_gt in cluster_gt_unique:
            gt_idx = np.where(cluster_gt == cluster_gt)
            new_idx = idx[0][gt_idx[0]]
            new_global_pred_labels[new_idx] = global_label
            global_label += 1

    u_pred_labels = new_global_pred_labels[linsize:]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    l_in_labels = new_global_pred_labels[:linsize]
    l_in_gt = global_labels[:linsize]
    u_unique = np.unique(u_pred_labels)
    new_id = inclasses
    for u_l in u_unique:
        idx = np.where(l_in_labels == u_l)
        gts = l_in_gt[idx]
        idx2 = np.where(u_pred_labels == u_l)
        if len(gts) != 0:
            gt_l = np.argmax(np.bincount(gts))
            u_pred_labels_new[idx2] = gt_l
        else:
            u_pred_labels_new[idx2] = new_id
            new_id += 1
    new_global_pred_labels[linsize:] = u_pred_labels_new
    new_global_pred_labels[:linsize] = global_labels[:linsize]

    global_pred_labels = new_global_pred_labels

if MODE == "uingt+uoutreducedcluster" or MODE == "uingt+uoutsplitcluster":
    new_global_pred_labels = np.zeros_like(global_pred_labels)
    new_global_pred_labels[:] = -1
    uout_unique = np.unique(global_pred_labels[linsize + uinsize:])
    global_label = inclasses
    # print(g.ndata['pred_den'].shape)
    # print(global_pred_labels.shape)
    # print(global_node_densities.shape)
    for label in uout_unique:
        idx = np.where(global_pred_labels == label)
        cluster_gt = global_labels[idx]
        cluster_gt_unique = np.unique(cluster_gt)

        if MODE == "uingt+uoutsplitcluster":
            for cluster_gt in cluster_gt_unique:
                gt_idx = np.where(cluster_gt == cluster_gt)
                new_idx = idx[0][gt_idx[0]]
                new_global_pred_labels[new_idx] = global_label
                global_label += 1

        if MODE == "uingt+uoutreducedcluster":
            cnts = Counter(cluster_gt)
            max_cluster_gt = cnts.most_common(1)[0][0]
            gt_idx = np.where(cluster_gt == max_cluster_gt)
            # print("gt_idx", gt_idx)
            # print(global_node_densities[idx])
            new_idx = idx[0][gt_idx[0]]
            new_global_pred_labels[new_idx] = global_label
            global_label += 1

    new_global_pred_labels[linsize + uinsize:] = new_global_pred_labels[linsize + uinsize:]
    new_global_pred_labels[:linsize + uinsize] = global_labels[:linsize + uinsize]
    global_pred_labels = new_global_pred_labels

if MODE == "oracle+cscore+hscore":
    new_global_pred_labels = np.zeros_like(global_pred_labels)
    new_global_pred_labels[:] = -1
    unique = np.unique(global_pred_labels)
    global_label = 0
    for label in unique:
        idx = np.where(global_pred_labels == label)
        cluster_gt = global_labels[idx]
        cluster_gt_unique = np.unique(cluster_gt)

        cnts = Counter(cluster_gt)
        max_cluster_gt = cnts.most_common(1)[0][0]
        gt_idx = np.where(cluster_gt == max_cluster_gt)
        # print("gt_idx", gt_idx)
        # print(global_node_densities[idx])
        new_idx = idx[0][gt_idx[0]]
        new_global_pred_labels[new_idx] = global_label
        global_label += 1

    u_pred_labels = new_global_pred_labels[linsize:]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    l_in_labels = new_global_pred_labels[:linsize]
    l_in_gt = global_labels[:linsize]
    l_in_gt_new = np.zeros_like(l_in_gt)
    u_unique = np.unique(u_pred_labels)
    l_in_unique = np.unique(l_in_gt)

    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i

    new_id = inclasses
    for u_l in u_unique:
        if u_l == -1:
            continue
        else:
            idx = np.where(l_in_labels == u_l)
            gts = l_in_gt_new[idx]
            idx2 = np.where(u_pred_labels == u_l)
            if len(gts) != 0:
                gt_l = np.argmax(np.bincount(gts))
                u_pred_labels_new[idx2] = gt_l
            else:
                u_pred_labels_new[idx2] = new_id
                new_id += 1
    new_global_pred_labels[linsize:] = u_pred_labels_new
    new_global_pred_labels[:linsize] = l_in_gt_new
    global_pred_labels = new_global_pred_labels

    remained = np.where(global_pred_labels != -1)
    print("remained:", remained)

    # bring back c-score
    from sklearn.metrics.cluster import completeness_score

    cth = 0.85
    unique_labels = np.unique(global_labels)
    for label in unique_labels:
        idxs = np.where(global_labels == label)
        pred_labels_in_idxs = np.unique(global_pred_labels[idxs])
        if len(pred_labels_in_idxs) < 10:
            if pred_labels_in_idxs[0] == -1 and len(pred_labels_in_idxs) > 1:
                global_pred_labels[idxs] = pred_labels_in_idxs[1]
            else:
                global_pred_labels[idxs] = pred_labels_in_idxs[0]

        # pred_cnts = Counter(pred_labels_in_idxs)
        # max_cluster_pred = pred_cnts.most_common(1)[0][0]
        # for pred in pred_labels_in_idxs:
        #     new_idx = np.where(global_pred_labels[idxs] == pred)
        #     if len(new_idx[0]) < 6:
        #         global_pred_labels[idxs][new_idx] = max_cluster_pred

    global_pred_labels[:linsize] = l_in_gt_new

if MODE == "oraclesplitclusters" or MODE == "oraclereduceclusters":
    new_global_pred_labels = np.zeros_like(global_pred_labels)
    new_global_pred_labels[:] = -1
    unique = np.unique(global_pred_labels)
    global_label = 0
    # print(g.ndata['pred_den'].shape)
    # print(global_pred_labels.shape)
    # print(global_node_densities.shape)
    for label in unique:
        idx = np.where(global_pred_labels == label)
        cluster_gt = global_labels[idx]
        cluster_gt_unique = np.unique(cluster_gt)

        if MODE == "oraclesplitclusters":
            for cluster_gt in cluster_gt_unique:
                gt_idx = np.where(cluster_gt == cluster_gt)
                new_idx = idx[0][gt_idx[0]]
                new_global_pred_labels[new_idx] = global_label
                global_label += 1

        if MODE == "oraclereduceclusters":
            cnts = Counter(cluster_gt)
            max_cluster_gt = cnts.most_common(1)[0][0]
            gt_idx = np.where(cluster_gt == max_cluster_gt)
            # print("gt_idx", gt_idx)
            # print(global_node_densities[idx])
            new_idx = idx[0][gt_idx[0]]
            new_global_pred_labels[new_idx] = global_label
            global_label += 1

    u_pred_labels = new_global_pred_labels[linsize:]
    u_pred_labels_new = np.zeros_like(u_pred_labels)
    u_pred_labels_new[:] = -1
    l_in_labels = new_global_pred_labels[:linsize]
    l_in_gt = global_labels[:linsize]
    l_in_gt_new = np.zeros_like(l_in_gt)
    u_unique = np.unique(u_pred_labels)
    l_in_unique = np.unique(l_in_gt)
    for i in range(len(l_in_unique)):
        l_in = l_in_unique[i]
        l_in_idx = np.where(l_in_gt == l_in)
        l_in_gt_new[l_in_idx] = i

    print(u_unique)
    new_id = inclasses
    for u_l in u_unique:
        if u_l == -1:
            continue
        else:
            idx = np.where(l_in_labels == u_l)
            gts = l_in_gt_new[idx]
            idx2 = np.where(u_pred_labels == u_l)
            if len(gts) != 0:
                gt_l = np.argmax(np.bincount(gts))
                u_pred_labels_new[idx2] = gt_l
            else:
                u_pred_labels_new[idx2] = new_id
                new_id += 1
    new_global_pred_labels[linsize:] = u_pred_labels_new
    new_global_pred_labels[:linsize] = l_in_gt_new
    global_pred_labels = new_global_pred_labels

if MODE == "uingt+uoutgt":
    global_pred_labels = global_labels

if MODE == "uingt":
    global_pred_labels[:linsize + uinsize] = global_labels[:linsize + uinsize]
    global_pred_labels[linsize + uinsize:] = -1

if MODE == "donothing":
    pass

##################################################
################ STASTIC ANALYSIS ################
##################################################
if STATISTIC:
    statistic_f = open("statistics.txt", 'w')
    clusters = np.unique(global_pred_labels)
    cluster_size_dict = {}
    title = "cluster_id|cluster size|GT labels|#GT labels|#in-classes|#Lin samples|#Uin samples|#Uout samples|Lin samples%|Uin samples%|Uout samples%\n"
    statistic_f.write(title)
    for c in clusters:
        clusteridx = np.where(global_pred_labels == c)[0]
        gt = global_labels[clusteridx]
        gt_labels = np.unique(gt)
        gt_in_classes = np.where(gt_labels < inclasses)[0]
        L_in_idx = np.where(clusteridx < linsize)[0]
        gt_labels = list(map(str, gt_labels))
        U_in_idx = np.where((clusteridx >= linsize) & (clusteridx < linsize + uinsize))[0]
        line = str(c) + "|" + str(len(clusteridx)) + "|" + ','.join(gt_labels) + \
               "|" + str(len(gt_labels)) + "|" + str(len(gt_in_classes)) + "|" + str(len(L_in_idx)) + \
               "|" + str(len(U_in_idx)) + "|" + str(len(clusteridx) - len(L_in_idx) - len(U_in_idx)) + \
               "|" + str(len(L_in_idx) * 1.0 / len(clusteridx)) + "|" + str(len(U_in_idx) * 1.0 / len(clusteridx)) + \
               "|" + str((len(clusteridx) - len(L_in_idx) - len(U_in_idx)) * 1.0 / len(clusteridx)) + '\n'
        # print(line)
        if len(clusteridx) not in cluster_size_dict.keys():
            cluster_size_dict[len(clusteridx)] = [[len(gt_labels)], [len(gt_in_classes)], [len(L_in_idx)],
                                                  [len(U_in_idx)], [len(clusteridx) - len(L_in_idx) - len(U_in_idx)]]
        else:
            cluster_size_dict[len(clusteridx)][0].append(len(gt_labels))
            cluster_size_dict[len(clusteridx)][1].append(len(gt_in_classes))
            cluster_size_dict[len(clusteridx)][2].append(len(L_in_idx))
            cluster_size_dict[len(clusteridx)][3].append(len(U_in_idx))
            cluster_size_dict[len(clusteridx)][4].append(len(clusteridx) - len(L_in_idx) - len(U_in_idx))
        statistic_f.write(line)
    statistic_f.close()

    statistic_mean_f = open("statistic_mean.txt", 'w')
    title_mean = "cluster size|#GT labels|#in-classes|#Lin samples|#Uin samples|#Uout samples|Lin samples%|Uin samples%|Uout samples%\n"
    statistic_mean_f.write(title_mean)
    for cs in cluster_size_dict:
        gt_size = sum(cluster_size_dict[cs][0]) * 1.0 / len(cluster_size_dict[cs][0])
        in_classes_size = sum(cluster_size_dict[cs][1]) * 1.0 / len(cluster_size_dict[cs][1])
        lin_samples_size = sum(cluster_size_dict[cs][2]) * 1.0 / len(cluster_size_dict[cs][2])
        uin_samples_size = sum(cluster_size_dict[cs][3]) * 1.0 / len(cluster_size_dict[cs][3])
        uout_samples_size = sum(cluster_size_dict[cs][4]) * 1.0 / len(cluster_size_dict[cs][4])
        lin_samples_pct = lin_samples_size / cs
        uin_samples_pct = uin_samples_size / cs
        uout_samples_pct = uout_samples_size / cs
        line = str(cs) + "|" + str(gt_size) + "|" + str(in_classes_size) + "|" + str(lin_samples_size) + "|" + str(
            uin_samples_size) + \
               "|" + str(uout_samples_size) + "|" + str(lin_samples_pct) + "|" + str(uin_samples_pct) + "|" + str(
            uout_samples_pct) + '\n'
        statistic_mean_f.write(line)
    statistic_mean_f.close()

    L_in_clusters = np.unique(global_pred_labels[:linsize])
    U_in_clusters = np.unique(global_pred_labels[linsize:linsize + uinsize])
    U_out_clusters = np.unique(global_pred_labels[linsize + uinsize:])

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3, venn2

    my_dpi = 150
    plt.figure(figsize=(600 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    g = venn3(subsets=[set(L_in_clusters), set(U_in_clusters), set(U_out_clusters)],
              set_labels=('L_in', 'U_in', 'U_out'),
              set_colors=("#01a2d9", "#31A354", "#c72e29"),
              alpha=0.5,
              normalize_to=1.0,
              )
    # g=venn2(subsets = [set(U_in_clusters), set(U_out_clusters)],
    #         set_labels = ('U_in', 'U_out'),
    #         set_colors=("#31A354", "#c72e29"),
    #         alpha=0.5,
    #         normalize_to=1.0,
    #     )
    plt.savefig("venn.png")

if MODE == "reducedgt":
    global_features = orifeatures
    global_pred_labels = np.zeros_like(orilabels)
    global_pred_labels[:] = -1
    global_pred_labels[selectedidx] = global_labels

print("##################### L_in ########################")
print(linsize)
evaluation(global_pred_labels[:linsize], gtlabels[:linsize], args.metrics)
print("##################### U_in ########################")
uinidx = np.where(global_pred_labels[linsize:linsize + uinsize] != -1)[0]
uinidx = uinidx + linsize
print(len(uinidx))
if len(uinidx):
    evaluation(global_pred_labels[uinidx], gtlabels[uinidx], args.metrics)
else:
    print("No samples in U_in!")
print("##################### U_out ########################")
uoutidx = np.where(global_pred_labels[linsize + uinsize:] != -1)[0]
uoutidx = uoutidx + linsize + uinsize
print(len(uoutidx))
if len(uoutidx):
    evaluation(global_pred_labels[uoutidx], gtlabels[uoutidx], args.metrics)
else:
    print("No samples in U_out!")
print("##################### U ########################")
uidx = np.where(global_pred_labels[linsize:] != -1)[0]
uidx = uidx + linsize
print(len(uidx))
if len(uidx):
    evaluation(global_pred_labels[uidx], gtlabels[uidx], args.metrics)
else:
    print("No samples in U!")
print("##################### L+U ########################")
luidx = np.where(global_pred_labels != -1)[0]
print(len(luidx))
evaluation(global_pred_labels[luidx], gtlabels[luidx], args.metrics)
print("##################### new selected samples ########################")
sidx = np.where(global_masks == 1)[0]
print(len(sidx))
if len(sidx) != 0:
    evaluation(global_pred_labels[sidx], gtlabels[sidx], args.metrics)

with open(args.output_filename, 'wb') as f:
    pickle.dump([path2idx, orifeatures, soft_labels, global_pred_labels, global_gt_labels, global_masks], f, protocol=4)
