"""
Distributed Node Classification
===============================

In this tutorial, we will walk through the steps of performing distributed GNN training
for a node classification task. To understand distributed GNN training, you need to
read the tutorial of multi-GPU training first. This tutorial is developed on top of
multi-GPU training by providing extra steps for partitioning a graph, modifying the training script
and setting up the environment for distributed training.


Partition a graph
-----------------

In this tutorial, we will use `OGBN products graph <https://ogb.stanford.edu/docs/nodeprop/#ogbn-products>`_
as an example to illustrate the graph partitioning. Let's first load the graph into a DGL graph.
Here we store the node labels as node data in the DGL Graph.




.. code-block:: python


    import os
    os.environ['DGLBACKEND'] = 'pytorch'
    import dgl
    import torch as th
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(name='ogbn-products')
    graph, labels = data[0]
    labels = labels[:, 0]
    graph.ndata['labels'] = labels


We need to split the data into training/validation/test set during the graph partitioning.
Because this is a node classification task, the training/validation/test sets contain node IDs.
We recommend users to convert them as boolean arrays, in which True indicates the existence
of the node ID in the set. In this way, we can store them as node data. After the partitioning,
the boolean arrays will be stored with the graph partitions.




.. code-block:: python


    splitted_idx = data.get_idx_split()
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask


Then we call the `partition_graph` function to partition the graph with
`METIS <http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_ and save the partitioned results
in the specified folder. **Note**: `partition_graph` runs on a single machine with a single thread.
You can go to `our user guide <https://docs.dgl.ai/en/latest/guide/distributed-preprocessing.html#distributed-partitioning>`_
to see more information on distributed graph partitioning.

The code below shows an example of invoking the partitioning algorithm and generate four partitions.
The partitioned results are stored in a folder called `4part_data`. While partitioning a graph,
we allow users to specify how to balance the partitions. By default, the algorithm balances the number
of nodes in each partition as much as possible. However, this balancing strategy is not sufficient
for distributed GNN training because some partitions may have many more training nodes than other partitions
or some partitions may have more edges than others. As such, `partition_graph` provides two additional arguments
`balance_ntypes` and `balance_edges` to enforce more balancing criteria. For example, we can use the training mask
to balance the number of training nodes in each partition, as shown in the example below. We can also turn on
the `balance_edges` flag to ensure that all partitions have roughly the same number of edges.




.. code-block:: python


    dgl.distributed.partition_graph(graph, graph_name='ogbn-products', num_parts=4,
                                    out_path='4part_data',
                                    balance_ntypes=graph.ndata['train_mask'],
                                    balance_edges=True)


When partitioning a graph, DGL shuffles node IDs and edge IDs so that nodes/edges assigned to
a partition have contiguous IDs. This is necessary for DGL to maintain the mappings of global
node/edge IDs and partition IDs. If a user needs to map the shuffled node/edge IDs to their original IDs,
they can turn on the `return_mapping` flag of `partition_graph`, which returns a vector for the node ID mapping
and edge ID mapping. Below shows an example of using the ID mapping to save the node embeddings after
distributed training. This is a common use case when users want to use the trained node embeddings
in their downstream task. Below let's assume that the trained node embeddings are stored in the `node_emb` tensor,
which is indexed by the shuffled node IDs. We shuffle the embeddings again and store them in
the `orig_node_emb` tensor, which is indexed by the original node IDs.




.. code-block:: python


    nmap, emap = dgl.distributed.partition_graph(graph, graph_name='ogbn-products',
                                                 num_parts=4,
                                                 out_path='4part_data',
                                                 balance_ntypes=graph.ndata['train_mask'],
                                                 balance_edges=True,
                                                 return_mapping=True)
    orig_node_emb = th.zeros(node_emb.shape, dtype=node_emb.dtype)
    orig_node_emb[nmap] = node_emb


Distributed training script
---------------------------

The distributed training script is very similar to multi-GPU training script with just a few modifications.
It also relies on the Pytorch distributed component to exchange gradients and update model parameters.
The distributed training script only contains the code of the trainers.

Initialize network communication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distributed GNN training requires to access the partitioned graph structure and node/edge features
as well as aggregating the gradients of model parameters from multiple trainers. DGL's distributed
component is responsible for accessing the distributed graph structure and distributed node features
and edge features while Pytorch distributed is responsible for exchanging the gradients of model parameters.
As such, we need to initialize both DGL and Pytorch distributed components at the beginning of the training script.

We need to call DGL's initialize function to initialize the trainers' network communication and
connect with DGL's servers at the very beginning of the distributed training script. This function
has an argument that accepts the path to the cluster configuration file.




.. code-block:: python

    import dgl
    import torch as th
    dgl.distributed.initialize(ip_config='ip_config.txt')


The configuration file `ip_config.txt` has the following format:

.. code-block:: shell

  ip_addr1 [port1]
  ip_addr2 [port2]

Each row is a machine. The first column is the IP address and the second column is the port for
connecting to the DGL server on the machine. The port is optional and the default port is 30050.

After initializing DGL's network communication, a user can initialize Pytorch's distributed communication.




.. code-block:: python

    th.distributed.init_process_group(backend='gloo')


Reference to the distributed graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DGL's servers load the graph partitions automatically. After the servers load the partitions,
trainers connect to the servers and can start to reference to the distributed graph in the cluster as below.




.. code-block:: python

    g = dgl.distributed.DistGraph('ogbn-products')


As shown in the code, we refer to a distributed graph by its name. This name is basically the one passed
to the `partition_graph` function as shown in the section above.

Get training and validation node IDs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For distributed training, each trainer can run its own set of training nodes.
The training nodes of the entire graph are stored in a distributed tensor as the `train_mask` node data,
which was constructed before we partitioned the graph. Each trainer can invoke `node_split` to its set
of training nodes. The `node_split` function splits the full training set evenly and returns
the training nodes, majority of which are stored in the local partition, to ensure good data locality.




.. code-block:: python

    train_nid = dgl.distributed.node_split(g.ndata['train_mask'])


We can split the validation nodes in the same way as above. In this case, each trainer gets
a different set of validation nodes.




.. code-block:: python

    valid_nid = dgl.distributed.node_split(g.ndata['val_mask'])


Define a GNN model
^^^^^^^^^^^^^^^^^^

For distributed training, we define a GNN model exactly in the same way as
`mini-batch training <https://doc.dgl.ai/guide/minibatch.html#>`_ or
`full-graph training <https://doc.dgl.ai/guide/training-node.html#guide-training-node-classification>`_.
The code below defines the GraphSage model.




.. code-block:: python

    import torch.nn as nn
    import torch.nn.functional as F
    import dgl.nn as dglnn
    import torch.optim as optim

    class SAGE(nn.Module):
        def __init__(self, in_feats, n_hidden, n_classes, n_layers):
            super().__init__()
            self.n_layers = n_layers
            self.n_hidden = n_hidden
            self.n_classes = n_classes
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))

        def forward(self, blocks, x):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                x = layer(block, x)
                if l != self.n_layers - 1:
                    x = F.relu(x)
            return x

    num_hidden = 256
    num_labels = len(th.unique(g.ndata['labels'][0:g.num_nodes()]))
    num_layers = 2
    lr = 0.001
    model = SAGE(g.ndata['feat'].shape[1], num_hidden, num_labels, num_layers)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


For distributed training, we need to convert the model into a distributed model with
Pytorch's `DistributedDataParallel`.




.. code-block:: python

    model = th.nn.parallel.DistributedDataParallel(model)


Distributed mini-batch sampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use the same :class:`~dgl.dataloading.pytorch.DistNodeDataLoader`, the distributed counterpart
of :class:`~dgl.dataloading.pytorch.NodeDataLoader`, to create a distributed mini-batch sampler for
node classification.




.. code-block:: python

    sampler = dgl.dataloading.MultiLayerNeighborSampler([25,10])
    train_dataloader = dgl.dataloading.DistNodeDataLoader(
                                 g, train_nid, sampler, batch_size=1024,
                                 shuffle=True, drop_last=False)
    valid_dataloader = dgl.dataloading.DistNodeDataLoader(
                                 g, valid_nid, sampler, batch_size=1024,
                                 shuffle=False, drop_last=False)


Training loop
^^^^^^^^^^^^^

The training loop for distributed training is also exactly the same as the single-process training.




.. code-block:: python

    import sklearn.metrics
    import numpy as np

    for epoch in range(10):
        # Loop over the dataloader to sample mini-batches.
        losses = []
        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # Load the input features as well as output labels
                batch_inputs = g.ndata['feat'][input_nodes]
                batch_labels = g.ndata['labels'][seeds]

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.detach().cpu().numpy())
                optimizer.step()

        # validation
        predictions = []
        labels = []
        with th.no_grad(), model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(valid_dataloader):
                inputs = g.ndata['feat'][input_nodes]
                labels.append(g.ndata['labels'][seeds].numpy())
                predictions.append(model(blocks, inputs).argmax(1).numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            accuracy = sklearn.metrics.accuracy_score(labels, predictions)
            print('Epoch {}: Validation Accuracy {}'.format(epoch, accuracy))


Set up distributed training environment
---------------------------------------

After partitioning a graph and preparing the training script, we now need to set up
the distributed training environment and launch the training job. Basically, we need to
create a cluster of machines and upload both the training script and the partitioned data
to each machine in the cluster. A recommended solution of sharing the training script and
the partitioned data in the cluster is to use NFS (Network File System).

For any users who are not familiar with NFS, below is a small tutorial of setting up NFS
in an existing cluster.

NFS server side setup (ubuntu only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, install essential libs on the storage server

.. code-block:: shell

  sudo apt-get install nfs-kernel-server

Below we assume the user account is ubuntu and we create a directory of workspace in the home directory.

.. code-block:: shell

  mkdir -p /home/ubuntu/workspace

We assume that the all servers are under a subnet with ip range 192.168.0.0 to 192.168.255.255.
We need to add the following line to `/etc/exports`

.. code-block:: shell

  /home/ubuntu/workspace  192.168.0.0/16(rw,sync,no_subtree_check)

Then restart NFS, the setup on server side is finished.

.. code-block:: shell

  sudo systemctl restart nfs-kernel-server

For configuration details, please refer to NFS ArchWiki (https://wiki.archlinux.org/index.php/NFS).

NFS client side setup (ubuntu only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use NFS, clients also require to install essential packages

.. code-block:: shell

  sudo apt-get install nfs-common

You can either mount the NFS manually

.. code-block:: shell

  mkdir -p /home/ubuntu/workspace
  sudo mount -t nfs <nfs-server-ip>:/home/ubuntu/workspace /home/ubuntu/workspace

or add the following line to `/etc/fstab` so the folder will be mounted automatically

.. code-block:: shell

  <nfs-server-ip>:/home/ubuntu/workspace   /home/ubuntu/workspace   nfs   defaults    0 0

Then run

.. code-block:: shell

  mount -a

Now go to `/home/ubuntu/workspace` and save the training script and the partitioned data in the folder.

SSH Access
^^^^^^^^^^

The launch script accesses the machines in the cluster via SSH. Users should follow the instruction
in `this document <https://linuxize.com/post/how-to-setup-passwordless-ssh-login/>`_ to set up
the passwordless SSH login on every machine in the cluster. After setting up the passwordless SSH,
users need to authenticate the connection to each machine and add their key fingerprints to `~/.ssh/known_hosts`.
This can be done automatically when we ssh to a machine for the first time.

Launch the distributed training job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After everything is ready, we can now use the launch script provided by DGL to launch the distributed
training job in the cluster. We can run the launch script on any machine in the cluster. 

.. code-block:: shell

  python3 ~/workspace/dgl/tools/launch.py \
  --workspace ~/workspace/ \
  --num_trainers 1 \
  --num_samplers 0 \
  --num_servers 1 \
  --part_config 4part_data/ogbn-products.json \
  --ip_config ip_config.txt \
  "python3 train_dist.py"

If we split the graph into four partitions as demonstrated at the beginning of the tutorial, the cluster has to have four machines. The command above launches one trainer and one server on each machine in the cluster. `ip_config.txt` lists the IP addresses of all machines in the cluster as follows:

.. code-block:: shell

  ip_addr1
  ip_addr2
  ip_addr3
  ip_addr4

Sample neighbors with `GraphBolt`
----------------------------------

Since DGL 2.0, we have introduced a new dataloading framework
`GraphBolt <https://doc.dgl.ai/stochastic_training/index.html>`_ in
which sampling is highly improved compared to previous implementations in DGL.
As a result, we've introduced `GraphBolt` to distributed training to improve
the performance of distributed sampling. What's more, the graph partitions
could be much smaller than before, which is beneficial for the loading speed
and memory usage during distributed training.

Graph partitioning
^^^^^^^^^^^^^^^^^^^

In order to benefit from `GraphBolt` for distributed sampling, we need to
convert partitions from `DGL` format to `GraphBolt` format. This can be done by
`dgl.distributed.dgl_partition_to_graphbolt` function. Alternatively, we can use
`dgl.distributed.partition_graph` function to generate partitions in `GraphBolt`
format directly.

1. Convert partitions from `DGL` format to `GraphBolt` format.

.. code-block:: python
  
    part_config = "4part_data/ogbn-products.json"
    dgl.distributed.dgl_partition_to_graphbolt(part_config)

The new partitions will be stored in the same directory as the original
partitions.

2. Generate partitions in `GraphBolt` format directly. Just set the
`use_graphbolt` flag to `True` in `partition_graph` function.

.. code-block:: python
  
    dgl.distributed.partition_graph(graph, graph_name='ogbn-products', num_parts=4,
                                    out_path='4part_data',
                                    balance_ntypes=graph.ndata['train_mask'],
                                    balance_edges=True,
                                    use_graphbolt=True)

Enable `GraphBolt` sampling in the training script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Just set the `use_graphbolt` flag to `True` in `dgl.distributed.initialize`
function. This is the only change needed in the training script to enable
`GraphBolt` sampling.

.. code-block:: python

    dgl.distributed.initialize('ip_config.txt', use_graphbolt=True)

  
"""
