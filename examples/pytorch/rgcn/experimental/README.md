## DistDGL with GraphBolt partitions and sampling

### How to partition graph

#### Partition from original dataset with `dgl.distributed.partition_graph()`

```
DGL_HOME=/home/ubuntu/workspace/dgl_2 DGL_LIBRARY_PATH=$DGL_HOME/build PYTHONPATH=tests:$DGL_HOME/python:tests/python/pytorch/graphbolt:$PYTHONPATH python3 examples/pytorch/rgcn/experimental/partition_graph.py --dataset ogbn-mag --num_parts 2 --balance_train --balance_edges --graphbolt
```

#### Convert existing partitions into GraphBolt formats

```
import dgl
part_config = "./data/ogbn-mag.json"
dgl.distributed.convert_dgl_partition_to_csc_sampling_graph(
    part_config,
    store_orig_nids=True,
    store_etypes=True,
)
```

#### Partition sizes compared between GraphBolt and DistDGL

`csc_sampling_graph.tar` is the GraphBolt partitions.
`graph.dgl` is the original DistDGL partitions, namely, DGLGraph.

```
-rw-rw-r-- 1 ubuntu ubuntu 231M Oct 26 01:51 data/part0/csc_sampling_graph.tar
-rw-rw-r-- 1 ubuntu ubuntu   24 Oct 26 01:51 data/part0/edge_feat.dgl
-rw-rw-r-- 1 ubuntu ubuntu 701M Oct 26 01:51 data/part0/graph.dgl
-rw-rw-r-- 1 ubuntu ubuntu 182M Oct 26 01:51 data/part0/node_feat.dgl
-rw-rw-r-- 1 ubuntu ubuntu 235M Oct 26 01:51 data/part1/csc_sampling_graph.tar
-rw-rw-r-- 1 ubuntu ubuntu   24 Oct 26 01:51 data/part1/edge_feat.dgl
-rw-rw-r-- 1 ubuntu ubuntu 711M Oct 26 01:51 data/part1/graph.dgl
-rw-rw-r-- 1 ubuntu ubuntu 187M Oct 26 01:51 data/part1/node_feat.dgl
```

### Train with GraphBolt partitions
just append `--graphbolt`.

```
python3 /home/ubuntu/workspace/dgl_2/tools/launch.py \
    --workspace /home/ubuntu/workspace/dgl_2/examples/pytorch/rgcn/experimental/ \
    --num_trainers 4 \
    --num_servers 2 \
    --num_samplers 0 \
    --part_config /home/ubuntu/workspace/dgl_2/data/ogbn-mag.json \
    --ip_config /home/ubuntu/workspace/ip_config.txt \
    "DGL_LIBRARY_PATH=/home/ubuntu/workspace/dgl_2/build PYTHONPATH=tests:/home/ubuntu/workspace/dgl_2/python:tests/python/pytorch/graphbolt:$PYTHONPATH python3 entity_classify_dist.py --graph-name ogbn-mag --dataset ogbn-mag --ip-config /home/ubuntu/workspace/ip_config.txt --fanout='25,10' --batch-size 1024  --n-hidden 64 --lr 0.01 --eval-batch-size 1024 --low-mem --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 1 --layer-norm --sparse-embedding --sparse-lr 0.06  --graphbolt"
```

#### Results
`g4dn.metal` x 2, `ogbn-mag`.

DistDGL with GraphBolt takes less time for sampling(from **73s** to **11s**) and for whole epoch(from **178s** to **70s**) while keeping comparable accuracies in validation and test.

##### DistDGL

```
Epoch Time(s): 177.6757, sample: 73.0354, data copy: 27.7802, forward: 2.4314, backward: 63.2740, update: 11.1546, #train: 78696, #inp
ut: 34579790

Val Acc 0.4618, Test Acc 0.4485, time: 16.9179
```

##### DistDGL with GraphBolt

fanout = [25, 10], call `gb.sample_etype_neighbors()` correctly. [**Default**]

```
Epoch Time(s): 70.3498, sample: 10.6339, data copy: 8.9492, forward: 2.6577, backward: 36.1793, update: 11.9295, #train: 78696, #input
: 34559464

Val Acc 0.4572, Test Acc 0.4498, time: 3.5830
```

fanout = [25, 10], multiplied by `num_etypes`. [**Deprecated**]

```
[3]Epoch Time(s): 137.0454, sample: 27.0914, data copy: 32.2842, forward: 3.5588, backward: 60.5921, update: 13.5188, #train: 78696, #inp
ut: 76402212

Val Acc 0.4648, Test Acc 0.4498, time: 10.4527
```

fanout = [25, 10], not multiplied by `num_etypes`. [**Deprecated**]

```
Epoch Time(s): 32.7923, sample: 5.4970, data copy: 4.9976, forward: 2.4069, backward: 15.4529, update: 4.4377, #train: 78696, #input: 
18370936

Val Acc 0.3901, Test Acc 0.3844, time: 2.2284
```

---------------------------------------

## Distributed training

This is an example of training RGCN node classification in a distributed fashion. Currently, the example train RGCN graphs with input node features. The current implementation follows ../rgcn/entity_claasify_mp.py.

Before training, install python libs by pip:

```bash
pip3 install ogb pyarrow
```

To train RGCN, it has four steps:

### Step 0: Setup a Distributed File System
* You may skip this step if your cluster already has folder(s) synchronized across machines.

To perform distributed training, files and codes need to be accessed across multiple machines. A distributed file system would perfectly handle the job (i.e., NFS, Ceph).

#### Server side setup
Here is an example of how to setup NFS. First, install essential libs on the storage server
```bash
sudo apt-get install nfs-kernel-server
```

Below we assume the user account is `ubuntu` and we create a directory of `workspace` in the home directory.
```bash
mkdir -p /home/ubuntu/workspace
```

We assume that the all servers are under a subnet with ip range `192.168.0.0` to `192.168.255.255`. The exports configuration needs to be modifed to

```bash
sudo vim /etc/exports
# add the following line
/home/ubuntu/workspace  192.168.0.0/16(rw,sync,no_subtree_check)
```

The server's internal ip can be checked  via `ifconfig` or `ip`. If the ip does not begin with `192.168`, then you may use
```bash
# for ip range 10.0.0.0 – 10.255.255.255
/home/ubuntu/workspace  10.0.0.0/8(rw,sync,no_subtree_check)
# for ip range 172.16.0.0 – 172.31.255.255
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

User need to set their own IP configuration file `ip_config.txt` before training. For example, if we have four machines in current cluster, the IP configuration could like this:

```bash
172.31.0.1
172.31.0.2
172.31.0.3
172.31.0.4
```

Users need to make sure that the master node (node-0) has right permission to ssh to all the other nodes without password authentication.
[This link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/) provides instructions of setting passwordless SSH login.

### Step 2: partition the graph.

The example provides a script to partition some builtin graphs such as ogbn-mag graph.
If we want to train RGCN on 4 machines, we need to partition the graph into 4 parts.

In this example, we partition the ogbn-mag graph into 4 parts with Metis. The partitions are balanced with respect to
the number of nodes, the number of edges and the number of labelled nodes.
```bash
python3 partition_graph.py --dataset ogbn-mag --num_parts 4 --balance_train --balance_edges
```

### Step 3: Launch distributed jobs

DGL provides a script to launch the training job in the cluster. `part_config` and `ip_config`
specify relative paths to the path of the workspace.

The command below launches one training process on each machine and each training process has 4 sampling processes.

```bash
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/rgcn/experimental/ \
--num_trainers 1 \
--num_servers 1 \
--num_samplers 4 \
--part_config data/ogbn-mag.json \
--ip_config ip_config.txt \
"python3 entity_classify_dist.py --graph-name ogbn-mag --dataset ogbn-mag --fanout='25,25' --batch-size 1024  --n-hidden 64 --lr 0.01 --eval-batch-size 1024  --low-mem --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --layer-norm --ip-config ip_config.txt  --sparse-embedding --sparse-lr 0.06 --num_gpus 1"
```

We can get the performance score at the second epoch:
```
Val Acc 0.4323, Test Acc 0.4255, time: 128.0379
```

The command below launches the same distributed training job using dgl distributed DistEmbedding
```bash
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/pytorch/rgcn/experimental/ \
--num_trainers 1 \
--num_servers 1 \
--num_samplers 4 \
--part_config data/ogbn-mag.json \
--ip_config ip_config.txt \
"python3 entity_classify_dist.py --graph-name ogbn-mag --dataset ogbn-mag --fanout='25,25' --batch-size 1024  --n-hidden 64 --lr 0.01 --eval-batch-size 1024  --low-mem --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --layer-norm --ip-config ip_config.txt  --sparse-embedding --sparse-lr 0.06 --num_gpus 1 --dgl-sparse"
```

We can get the performance score at the second epoch:
```
Val Acc 0.4410, Test Acc 0.4282, time: 32.5274
```

**Note:** if you are using conda or other virtual environments on the remote machines, you need to replace `python3` in the command string (i.e. the last argument) with the path to the Python interpreter in that environment.

## Partition a graph with ParMETIS

It has four steps to partition a graph with ParMETIS for DGL's distributed training.
More details about the four steps are explained in our
[user guide](https://doc.dgl.ai/guide/distributed-preprocessing.html).

### Step 1: write the graph into files.

The graph structure should be written as a node file and an edge file. The node features and edge features
can be written as DGL tensors. `write_mag.py` shows an example of writing the OGB MAG graph into files.

As `pm_dglpart` cannot handle self-loops and duplicate edges correctly, these edges are removed and stored
into `mag_removed_edges.txt` when calling `write_mag.py`. When converting ParMETIS outputs into DGLGraph
in next steps, `mag_removed_edges.txt` should be passed in. Refer to Step 3 for more details.

```bash
python3 write_mag.py
```

### Step 2: partition the graph with ParMETIS
Run the program called `pm_dglpart` in ParMETIS to read the node file and the edge file output in Step 1
to partition the graph.

```bash
pm_dglpart mag 2
```
This partitions the graph into two parts with a single process.

```
mpirun -np 4 pm_dglpart mag 2
```
This partitions the graph into eight parts with four processes.

```
mpirun --hostfile hostfile -np 4 pm_dglpart mag 2
```
This partitions the graph into eight parts with four processes on multiple machines.
`hostfile` specifies the IPs of the machines; one line for a machine. The input files
should reside in the machine where the command line runs. Each process will write
the partitions to files in the local machine. For simplicity, we recommend users to
write the files on NFS.

### Step 3: Convert the ParMETIS partitions into DGLGraph

DGL provides a tool called `convert_partition.py` to load one partition at a time and convert it into a DGLGraph
and save it into a file. As mentioned in Step 1, please pass `mag_removed_edges.txt` if any self-loops and
duplicate edges are removed.

```bash
python3 ~/workspace/dgl/tools/convert_partition.py --input-dir . --graph-name mag --schema mag.json --num-parts 2 --num-node-weights 4 --output outputs --removed-edges mag_removed_edges.txt
```

### Step 4: Read node data and edge data for each partition

This shows an example of reading node data and edge data of each partition and saving them into files located in the same directory as the DGLGraph file.

```bash
python3 get_mag_data.py
```

### Step 5: Verify the partition result (Optional)

```bash
python3 verify_mag_partitions.py
```

## Distributed code runs in the standalone mode

The standalone mode is mainly used for development and testing. The procedure to run the code is much simpler.

### Step 1: graph construction.
When testing the standalone mode of the training script, we should construct a graph with one partition.
```bash
python3 partition_graph.py --dataset ogbn-mag --num_parts 1
```

### Step 2: run the training script
```bash
DGL_DIST_MODE=standalone python3 entity_classify_dist.py --graph-name ogbn-mag  --dataset ogbn-mag --fanout='25,25' --batch-size 512 --n-hidden 64 --lr 0.01 --eval-batch-size 128 --low-mem --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --layer-norm --ip-config ip_config.txt --conf-path 'data/ogbn-mag.json' --standalone  --sparse-embedding  --sparse-lr 0.06
```
