import pickle as pkl
import numpy as np
import random

# hyper parameters
inclasses = 948

# load data from pickle file
pklpath = "/home/ubuntu/code/hilander/hilander/data/inat_hilander_l_smoothap_train_selectbydensity_0.8_0.9_iter1.pkl"
with open(pklpath, 'rb') as f:
    loaded_data = pkl.load(f)
    path2idx, features, pred_labels, labels, masks = loaded_data
idx2path = {v: k for k, v in path2idx.items()}

# get lin and selected samples indexes
lin_index = np.where(masks == 0)[0]
selected_index = np.where(masks == 1)[0]
notselected_index = np.where(masks == 2)[0]
print("lin size:", len(lin_index))
print("selected size:", len(selected_index))
print("notselected size:", len(notselected_index))
lin_index_sample = random.choices(lin_index, k=int(len(lin_index)*0.1))
selected_index_sample = random.choices(selected_index, k=int(len(selected_index)*0.1))
notselected_index_sample = random.choices(notselected_index, k=int(len(notselected_index)*0.1))

# lin samples
lin_features = features[lin_index]
lin_labels = pred_labels[lin_index]
lin_gt_labels = labels[lin_index]

# selected samples
selected_features = features[selected_index]
selected_labels = pred_labels[selected_index]
selected_gt_labels = labels[selected_index]

# notselected samples
notselected_features = features[notselected_index]
notselected_labels = pred_labels[notselected_index]
notselected_gt_labels = labels[notselected_index]

# lin prototypes
prototypes = np.zeros((inclasses, lin_features.shape[1]))
for i in range(inclasses):
    idx = np.where(lin_labels == i)
    prototypes[i] = np.mean(lin_features[idx], axis=0)

# lin label distribution
lin_distribution = {}
lin_distribution_super = {}
for i in range(len(lin_labels)):
    superclass = idx2path[i].split('/')[-3]
    label = idx2path[i].split('/')[-2]
    key = superclass+' '+label
    if key not in lin_distribution.keys():
        lin_distribution[key] = 0
    lin_distribution[key] += 1

    if superclass not in lin_distribution_super.keys():
        lin_distribution_super[superclass] = 0
    lin_distribution_super[superclass] += 1

'''tsne visualization'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

features_embedded = TSNE(n_components=2, init='random').fit_transform(features)
lin_features_embedded = features_embedded[lin_index]
selected_features_embedded = features_embedded[selected_index]
notselected_features_embedded = features_embedded[notselected_index]

plt.clf()
plt.scatter(notselected_features_embedded[:,0], notselected_features_embedded[:,1],
            color='#c2c2c2', alpha=0.5, s=0.5, label="U_notselected")
plt.legend()
plt.savefig("./analysis/unotselected_tsne.jpg")

plt.clf()
plt.scatter(selected_features_embedded[:,0], selected_features_embedded[:,1],
            alpha=0.5, s=0.5, label="U_selected")
plt.legend()
plt.savefig("./analysis/uselected_tsne.jpg")

plt.clf()
plt.scatter(lin_features_embedded[:,0], lin_features_embedded[:,1], alpha=0.5, c='orange', s=0.5, label="L_in")
plt.legend()
plt.savefig("./analysis/lin_tsne.jpg")