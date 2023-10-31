## DistDGL with GraphBolt(Homograph + Node Classification)

### How to partition graph

#### Partition from original dataset with `dgl.distributed.partition_graph()`

```
DGL_HOME=/home/ubuntu/workspace/dgl_2 DGL_LIBRARY_PATH=$DGL_HOME/build PYTHONPATH=tests:$DGL_HOME/python:tests/python/pytorch/graphbolt:$PYTHONPATH python3 examples/distributed/graphsage/partition_graph.py --dataset ogbn-products --num_parts 2 --balance_train --balance_edges --graphbolt
```

#### Convert existing partitions into GraphBolt formats

```
DGL_LIBRARY_PATH=$DGL_HOME/build PYTHONPATH=tests:$DGL_HOME/python:tests/python/pytorch/graphbolt:$PYTHONPATH python3 -c "from dgl.distributed import convert_dgl_partition_to_csc_sampling_graph as f;f('data/ogbn-products.json')"
```

#### Partition sizes compared between GraphBolt and DistDGL

`csc_sampling_graph.tar` is the GraphBolt partitions.
`graph.dgl` is the original DistDGL partitions, namely, DGLGraph.

###### ogbn-products
homogeneous, ~2.4M nodes, ~123.7M edges(reverse edges are added), 2 parts.

| DGL(GB) | GraphBolt w/o EIDs(MB) | GraphBolt w/ EIDs(MB) |
| --- | ------------------ | ----------------- |
| 1.6/1.7 | 258/272 | 502/530 |

```
-rw-rw-r-- 1 ubuntu ubuntu 258M Oct 31 01:56 homo_data/part0/csc_sampling_graph.tar
-rw-rw-r-- 1 ubuntu ubuntu 502M Oct 31 04:45 homo_data/part0/csc_sampling_graph_eids.tar
-rw-rw-r-- 1 ubuntu ubuntu   24 Oct 31 00:51 homo_data/part0/edge_feat.dgl
-rw-rw-r-- 1 ubuntu ubuntu 1.6G Oct 31 00:51 homo_data/part0/graph.dgl
-rw-rw-r-- 1 ubuntu ubuntu 501M Oct 31 00:51 homo_data/part0/node_feat.dgl
-rw-rw-r-- 1 ubuntu ubuntu 272M Oct 31 01:56 homo_data/part1/csc_sampling_graph.tar
-rw-rw-r-- 1 ubuntu ubuntu 530M Oct 31 04:45 homo_data/part1/csc_sampling_graph_eids.tar
-rw-rw-r-- 1 ubuntu ubuntu   24 Oct 31 00:51 homo_data/part1/edge_feat.dgl
-rw-rw-r-- 1 ubuntu ubuntu 1.7G Oct 31 00:51 homo_data/part1/graph.dgl
-rw-rw-r-- 1 ubuntu ubuntu 460M Oct 31 00:51 homo_data/part1/node_feat.dgl
```

### Train with GraphBolt partitions
just append `--graphbolt`.

```
python3 /home/ubuntu/workspace/dgl_2/tools/launch.py \
    --workspace /home/ubuntu/workspace/dgl_2/examples/distributed/graphsage/ \
    --num_trainers 4 \
    --num_servers 2 \
    --num_samplers 0 \
    --part_config /home/ubuntu/workspace/dgl_2/homo_data/ogbn-products.json \
    --ip_config /home/ubuntu/workspace/ip_config.txt \
    "DGL_LIBRARY_PATH=/home/ubuntu/workspace/dgl_2/build PYTHONPATH=tests:/home/ubuntu/workspace/dgl_2/python:tests/python/pytorch/graphbolt:$PYTHONPATH python3 node_classification.py --graph_name ogbn-products --ip_config /home/ubuntu/workspace/ip_config.txt --num_epochs 3 --eval_every 2 --graphbolt"
```

#### Results
`g4dn.metal` x 2, `ogbn-products`.

DistDGL with GraphBolt takes less time for sampling(from **1.8283s** to **1.4470s**) and for whole epoch(from **4.9259s** to **4.4898s**) while keeping comparable accuracies in validation and test.

##### DistDGL

```
Part 0, Epoch Time(s): 4.9648, sample+data_copy: 1.8283, forward: 0.2912, backward: 1.1307, update: 0.0232, #seeds: 24577, #inputs: 4136843

Summary of node classification(GraphSAGE): GraphName ogbn-products | TrainEpochTime(mean) 4.9259 | TestAccuracy 0.6213
```

##### DistDGL with GraphBolt

```
Part 0, Epoch Time(s): 4.4826, sample+data_copy: 1.4470, forward: 0.2517, backward: 0.9081, update: 0.0175, #seeds: 24577, #inputs: 41369
80

Summary of node classification(GraphSAGE): GraphName ogbn-products | TrainEpochTime(mean) 4.4898 | TestAccuracy 0.6174
```

---------------------------------------


## Distributed training

This is an example of training GraphSage in a distributed fashion. Before training, please install some python libs by pip:

```
pip3 install ogb
```

**Requires PyTorch 1.12.0+ to work.**

To train GraphSage, it has five steps:

### Step 0: Setup a Distributed File System
* You may skip this step if your cluster already has folder(s) synchronized across machines.

To perform distributed training, files and codes need to be accessed across multiple machines. A distributed file system would perfectly handle the job (i.e., NFS, Ceph).

#### Server side setup
Here is an example of how to setup NFS. First, install essential libs on the storage server

```
sudo apt-get install nfs-kernel-server
```

Below we assume the user account is `ubuntu` and we create a directory of `workspace` in the home directory.

```
mkdir -p /home/ubuntu/workspace
```

We assume that the all servers are under a subnet with ip range `192.168.0.0` to `192.168.255.255`. The exports configuration needs to be modifed to

```
sudo vim /etc/exports
# add the following line
/home/ubuntu/workspace  192.168.0.0/16(rw,sync,no_subtree_check)
```

The server's internal ip can be checked  via `ifconfig` or `ip`. If the ip does not begin with `192.168`, then you may use

```
/home/ubuntu/workspace  10.0.0.0/8(rw,sync,no_subtree_check)
/home/ubuntu/workspace  172.16.0.0/12(rw,sync,no_subtree_check)
```

Then restart NFS, the setup on server side is finished.

```
sudo systemctl restart nfs-kernel-server
```

For configraution details, please refer to [NFS ArchWiki](https://wiki.archlinux.org/index.php/NFS).

#### Client side setup

To use NFS, clients also require to install essential packages

```
sudo apt-get install nfs-common
```

You can either mount the NFS manually

```
mkdir -p /home/ubuntu/workspace
sudo mount -t nfs <nfs-server-ip>:/home/ubuntu/workspace /home/ubuntu/workspace
```

or edit the fstab so the folder will be mounted automatically

```
# vim /etc/fstab
## append the following line to the file
<nfs-server-ip>:/home/ubuntu/workspace   /home/ubuntu/workspace   nfs   defaults	0 0
```

Then run `mount -a`.

Now go to `/home/ubuntu/workspace` and clone the DGL Github repository.

### Step 1: set IP configuration file.

User need to set their own IP configuration file `ip_config.txt` before training. For example, if we have four machines in current cluster, the IP configuration
could like this:

```
172.31.19.1
172.31.23.205
172.31.29.175
172.31.16.98
```

Users need to make sure that the master node (node-0) has right permission to ssh to all the other nodes without password authentication.
[This link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/) provides instructions of setting passwordless SSH login.

### Step 2: partition the graph.

The example provides a script to partition some builtin graphs such as Reddit and OGB product graph.
If we want to train GraphSage on 4 machines, we need to partition the graph into 4 parts.

In this example, we partition the ogbn-products graph into 4 parts with Metis on node-0. The partitions are balanced with respect to
the number of nodes, the number of edges and the number of labelled nodes.

```
python3 partition_graph.py --dataset ogbn-products --num_parts 4 --balance_train --balance_edges
```

This script generates partitioned graphs and store them in the directory called `data`.


### Step 3: Launch distributed jobs

DGL provides a script to launch the training job in the cluster. `part_config` and `ip_config`
specify relative paths to the path of the workspace.

The command below launches one process per machine for both sampling and training.

```
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 1 \
--num_samplers 0 \
--num_servers 1 \
--part_config data/ogbn-products.json \
--ip_config ip_config.txt \
"python3 node_classification.py --graph_name ogbn-products --ip_config ip_config.txt --num_epochs 30 --batch_size 1000"
```

By default, this code will run on CPU. If you have GPU support, you can just add a `--num_gpus` argument in user command:

```
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/graphsage/dist/ \
--num_trainers 4 \
--num_samplers 0 \
--num_servers 1 \
--part_config data/ogbn-products.json \
--ip_config ip_config.txt \
"python3 node_classification.py --graph_name ogbn-products --ip_config ip_config.txt --num_epochs 30 --batch_size 1000 --num_gpus 4"
```
