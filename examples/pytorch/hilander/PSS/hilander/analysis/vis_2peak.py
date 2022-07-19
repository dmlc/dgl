import pickle as pkl
import numpy as np
import torch

import seaborn
import matplotlib.pyplot as plt

# hyper parameters
inclasses = 948
linsize = 29011
uinsize = 18403
allclasses = 5690

# load data path
'''
iter 0 -> iter 1: inat_hilander_l_smoothap_train_selectbydensity_expand_0.8_0.9_iter0.pkl
iter 1 -> iter 2: inat_hilander_l_smoothap_train_selectbydensity_expand_0.8_0.9_iter1.pkl
iter 2 -> iter 3: inat_hilander_l_smoothap_train_selectbydensity_expand_0.8_0.9_iter2.pkl
'''
pklpath = "/home/ubuntu/code/hilander/hilander/data/inat_hilander_l_smoothap_train_selectbydensity_expand_0.4_0.9_iter1.pkl"
with open(pklpath, 'rb') as f:
    loaded_data = pkl.load(f)
    path2idx, features, pred_labels, labels, masks = loaded_data
idx2path = {v: k for k, v in path2idx.items()}


# get lin, selected samples, not selected samples indexes
lin_index = np.where(masks == 0)[0]
selected_index = np.where(masks == 1)[0]
notselected_index = np.where(masks == 2)[0]
u_index = np.where(masks != 0)[0]
print("lin size:", len(lin_index))
print("selected size:", len(selected_index))
print("notselected size:", len(notselected_index))
print("u size:", len(u_index))

# lin samples
lin_features = features[lin_index]
lin_labels = pred_labels[lin_index]
lin_gt_labels = labels[lin_index]

# lin_features = features[:linsize]
# lin_labels = pred_labels[:linsize]
# lin_gt_labels = labels[:linsize]


l_gt_new = np.zeros_like(lin_labels)
unique = np.unique(lin_gt_labels)
for i in range(len(unique)):
    cls = unique[i]
    cls_idx = np.where(lin_gt_labels == cls)
    l_gt_new[cls_idx] = i
print("len(unique)", len(unique))
lin_labels = l_gt_new
print(lin_labels)

# selected samples
selected_features = features[selected_index]
selected_labels = pred_labels[selected_index]
selected_gt_labels = labels[selected_index]

# notselected samples
notselected_features = features[notselected_index]
notselected_labels = pred_labels[notselected_index]
notselected_gt_labels = labels[notselected_index]

# u samples
u_features = features[u_index]
u_labels = pred_labels[u_index]
u_gt_labels = labels[u_index]

gt_new = np.zeros_like(labels)
unique = np.unique(labels)
for i in range(len(unique)):
    cls = unique[i]
    cls_idx = np.where(labels == cls)
    gt_new[cls_idx] = i
print("len(unique)", len(unique))
u_labels = gt_new[u_index]

# Lin prototype
prototypes = np.zeros((inclasses, u_features.shape[1]))
for i in range(inclasses):
    idx = np.where(lin_labels == i)
    prototypes[i] = np.mean(lin_features[idx], axis=0)

selected_similarity_matrix = torch.mm(torch.from_numpy(selected_features.astype(np.float32)),
                                      torch.from_numpy(prototypes.astype(np.float32)).t())
# similarity_matrix = (1 - similarity_matrix) / 2
selected_minvalues, selected_pred_labels = torch.max(selected_similarity_matrix, 1)
selected_minvalues = np.array(selected_minvalues)

notselected_similarity_matrix = torch.mm(torch.from_numpy(notselected_features.astype(np.float32)),
                                         torch.from_numpy(prototypes.astype(np.float32)).t())
# similarity_matrix = (1 - similarity_matrix) / 2
notselected_minvalues, notselected_pred_labels = torch.max(notselected_similarity_matrix, 1)
notselected_minvalues = np.array(notselected_minvalues)
print(selected_minvalues)
print(notselected_minvalues)

plt.clf()
fig = seaborn.histplot(data=selected_minvalues, stat="probability", color="skyblue", multiple="layer", kde=True,
                       label="U_selected - nearest L_in prototypes")
fig = seaborn.histplot(data=notselected_minvalues, stat="probability", color="orange", multiple="layer", kde=True,
                       label="U_unselected - nearest L_in prototypes")

mean = float(np.mean(selected_minvalues))
std = float(np.std(selected_minvalues))
plt.axvline(mean, color='blue', linestyle='-')
fig.text(mean, 0, '%.2f' % mean, c='blue')
plt.axvline(mean - std, color='blue', linestyle='--')
fig.text(mean - std, 0, '%.2f' % (mean - std), c='blue')
plt.axvline(mean + std, color='blue', linestyle='--')
fig.text(mean + std, 0, '%.2f' % (mean + std), c='blue')

mean = float(np.mean(notselected_minvalues))
std = float(np.std(notselected_minvalues))
plt.axvline(mean, color='red', linestyle='-')
fig.text(mean, 0, '%.2f' % mean, c='blue')
plt.axvline(mean - std, color='red', linestyle='--')
fig.text(mean - std, 0, '%.2f' % (mean - std), c='blue')
plt.axvline(mean + std, color='red', linestyle='--')
fig.text(mean + std, 0, '%.2f' % (mean + std), c='blue')

histfig = fig.get_figure()
plt.legend()
histfig.savefig("./analysis/selected_notselected_distances.jpg")