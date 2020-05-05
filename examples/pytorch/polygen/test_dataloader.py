from modules import *
from loss import *
from optims import *
from dataset import *
from modules.config import *
#from modules.viz import *
import numpy as np
import argparse
import torch
from functools import partial
import torch.distributed as dist
from torch.utils.data import DataLoader

if __name__ == '__main__':
    np.random.seed(1111)
    file_list_path = 'all_file_list_filtered.txt'
    #dataset = VertexDataset(file_list_path, 'train')
    #loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=collate_vertexgraphs)
    dataset = FaceDataset(file_list_path, 'val', 'cpu')
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16, collate_fn=collate_facegraphs)
    loader_iter = iter(loader)
    #for batch_ndx, a in enumerate(loader):
    batch_idx = 0
    while True:
        try:
            batch = loader_iter.next()
        except:
            loader_iter = iter(loader)
            batch = loader_iter.next()
        print (batch_idx)
        batch_idx += 1
