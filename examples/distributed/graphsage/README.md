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
--workspace ~/workspace/dgl/examples/distributed/graphsage/ \
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
--workspace ~/workspace/dgl/examples/distributed/graphsage/ \
--num_trainers 4 \
--num_samplers 0 \
--num_servers 1 \
--part_config data/ogbn-products.json \
--ip_config ip_config.txt \
"python3 node_classification.py --graph_name ogbn-products --ip_config ip_config.txt --num_epochs 30 --batch_size 1000 --num_gpus 4"
```

Unsupervised training(train with link prediction dataloader).

```
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/distributed/graphsage/ \
--num_trainers 1 \
--num_samplers 0 \
--num_servers 1 \
--part_config data/ogbn-products.json \
--ip_config ip_config.txt \
"python3 node_classification_unsupervised.py --graph_name ogbn-products --ip_config ip_config.txt --num_epochs 30 --batch_size 1000 --remove_edge"
```

### Running with GraphBolt

In order to run with `GraphBolt`, we need to partition graph into `GraphBolt` data formats.Please note that both `DGL` and `GraphBolt` partitions are saved together.

If we have already partitioned into `DGL` format, just convert them directly like below:

```
    python3 -c "import dgl; dgl.distributed.dgl_partition_to_graphbolt('ogbn-products.json')"
```

Or partition from scratch like this:

```
python3 partition_graph.py --dataset ogbn-products --num_parts 2 --balance_train --balance_edges --use_graphbolt
```

#### Partition sizes compared to DGL

Compared to `DGL`, `GraphBolt` partitions are much smaller(reduced to **16%** and **19%** for `ogbn-products` and `ogbn-papers100M` respectively).

`ogbn-products`

| Data Formats |         File Name            | Part 0 | Part 1 |
| ------------ | ---------------------------- | ------ | ------ |
| DGL          | graph.dgl                    | 1.5GB  | 1.6GB  |
| GraphBolt    | fused_csc_sampling_graph.pt  | 255MB  | 265MB  |

`ogbn-papers100M`

| Data Formats |         File Name            | Part 0 | Part 1 |
| ------------ | ---------------------------- | ------ | ------ |
| DGL          | graph.dgl                    | 23GB   | 22GB   |
| GraphBolt    | fused_csc_sampling_graph.pt  | 4.4GB  | 4.1GB  |

Then run example with `--use_graphbolt`.

```
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/distributed/graphsage/ \
--num_trainers 4 \
--num_samplers 0 \
--num_servers 2 \
--part_config data/ogbn-products.json \
--ip_config ip_config.txt \
"python3 node_classification.py --graph_name ogbn-products --ip_config ip_config.txt --num_epochs 10 --use_graphbolt"
```

#### Performance compared to `DGL`

Compared to `DGL`, `GraphBolt`'s sampler works faster(reduced to **80%** and **77%** for `ogbn-products` and `ogbn-papers100M` respectively). `Min` and `Max` are statistics of all trainers on all nodes(machines).

As for RAM usage, the shared memory(measured by **shared** field of `free` command) usage is decreased due to smaller graph partitions in `GraphBolt` though the peak memory used by processes(measured by **used** field of `free` command) does not decrease.

`ogbn-products`

| Data Formats | Sample Time Per Epoch (CPU) |      Test Accuracy (10 epochs)   |  shared | used (peak) |
| ------------ | --------------------------- | -------------------------------- |  -----  | ---- |
|     DGL      | Min: 1.2884s, Max: 1.4159s  | Min: 64.38%, Max: 70.42%         |  2.4GB  | 7.8GB|
|   GraphBolt  | Min: 1.0589s, Max: 1.1400s  | Min: 61.68%, Max: 71.23%         |  1.1GB  | 7.8GB|


`ogbn-papers100M`

| Data Formats | Sample Time Per Epoch (CPU) |      Test Accuracy (10 epochs)   |  shared | used (peak) |
| ------------ | --------------------------- | -------------------------------- |  -----  | ---- |
|     DGL      | Min: 5.5570s, Max: 6.1900s  | Min: 29.12%, Max: 34.33%         |  84GB   | 43GB |
|   GraphBolt  | Min: 4.5046s, Max: 4.7718s  | Min: 29.11%, Max: 33.49%         |  67GB   | 43GB |
