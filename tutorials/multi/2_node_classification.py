"""
Single Machine Multi-GPU Minibatch Node Classification
======================================================

In this tutorial, you will learn how to use multiple GPUs in training a
graph neural network (GNN) for node classification.

(Time estimate: 8 minutes)

This tutorial assumes that you have read the :doc:`Training GNN with Neighbor
Sampling for Node Classification <../large/L1_large_node_classification>`
tutorial. It also assumes that you know the basics of training general
models with multi-GPU with ``DistributedDataParallel``.

.. note::

   See `this tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__
   from PyTorch for general multi-GPU training with ``DistributedDataParallel``.  Also,
   see the first section of :doc:`the multi-GPU graph classification
   tutorial <1_graph_classification>`
   for an overview of using ``DistributedDataParallel`` with DGL.

"""


######################################################################
# Loading Dataset
# ---------------
# 
# OGB already prepared the data as a ``DGLGraph`` object. The following code is
# copy-pasted from the :doc:`Training GNN with Neighbor Sampling for Node
# Classification <../large/L1_large_node_classification>`
# tutorial.
# 

import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import sklearn.metrics

dataset = DglNodePropPredDataset('ogbn-arxiv')

graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
graph.ndata['label'] = node_labels[:, 0]

node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()

idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']    # Test node IDs, not used in the tutorial though.


######################################################################
# Defining Model
# --------------
# 
# The model will be again identical to the :doc:`Training GNN with Neighbor
# Sampling for Node Classification <../large/L1_large_node_classification>`
# tutorial.
# 

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h


######################################################################
# Defining Training Procedure
# ---------------------------
# 
# The training procedure will be slightly different from what you saw
# previously, in the sense that you will need to
#
# * Initialize a distributed training context with ``torch.distributed``.
# * Wrap your model with ``torch.nn.parallel.DistributedDataParallel``.
# * Add a ``use_ddp=True`` argument to the DGL dataloader you wish to run
#   together with DDP.
# 
# You will also need to wrap the training loop inside a function so that
# you can spawn subprocesses to run it.
# 

def run(proc_id, devices):
    # Initialize distributed training context.
    dev_id = devices[proc_id]
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(
            backend='nccl', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    
    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    sampler = dgl.dataloading.NeighborSampler([4, 4])
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to NodeDataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=True,       # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=1024,    # Per-device batch size.
                            # The effective batch size is this number times the number of GPUs.
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        device=device,
        use_ddp=False,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    model = Model(num_features, 128, num_classes).to(device)
    # Wrap the model with distributed data parallel module.
    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)
    
    # Define optimizer
    opt = torch.optim.Adam(model.parameters())
    
    best_accuracy = 0
    best_model_path = './model.pt'
    
    # Copied from previous tutorial with changes highlighted.
    for epoch in range(10):
        model.train()

        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata['feat']
                labels = mfgs[-1].dstdata['label']

                predictions = model(mfgs, inputs)

                loss = F.cross_entropy(predictions, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())

                tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)

        model.eval()

        # Evaluate on only the first GPU.
        if proc_id == 0:
            predictions = []
            labels = []
            with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = mfgs[0].srcdata['feat']
                    labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), best_model_path)

        # Note that this tutorial does not train the whole model to the end.
        break


######################################################################
# Spawning Trainer Processes
# --------------------------
# 
# A typical scenario for multi-GPU training with DDP is to replicate the
# model once per GPU, and spawn one trainer process per GPU.
#
# Normally, DGL maintains only one sparse matrix representation (usually COO)
# for each graph, and will create new formats when some APIs are called for
# efficiency.  For instance, calling ``in_degrees`` will create a CSC
# representation for the graph, and calling ``out_degrees`` will create a
# CSR representation.  A consequence is that if a graph is shared to
# trainer processes via copy-on-write *before* having its CSC/CSR
# created, each trainer will create its own CSC/CSR replica once ``in_degrees``
# or ``out_degrees`` is called.  To avoid this, you need to create
# all sparse matrix representations beforehand using the ``create_formats_``
# method:
# 

graph.create_formats_()


######################################################################
# Then you can spawn the subprocesses to train with multiple GPUs.
# 
# 
# .. code:: python
#
#    # Say you have four GPUs.
#    if __name__ == '__main__':
#        num_gpus = 4
#        import torch.multiprocessing as mp
#        mp.spawn(run, args=(list(range(num_gpus)),), nprocs=num_gpus)

# Thumbnail credits: Stanford CS224W Notes
# sphinx_gallery_thumbnail_path = '_static/blitz_1_introduction.png'
