from typing import Dict, List

import numpy as np
import torch

from dgl.graphbolt.impl.torch_based_feature_store import DiskBasedFeature

mmap_mode = "r+"
path = "/home/ubuntu/dgl/dgl/datasets/ogb-lsc-mag240m/preprocessed/features/inst-feat.npy"
tt = torch.as_tensor(np.load(path, mmap_mode=mmap_mode))
print(tt.shape)
s = tt.index_select(0, torch.tensor([10, 1003, 1004, 10000]))  #
print(s)
print(s.size()[1:])


t = DiskBasedFeature(path)
print(t.size())
