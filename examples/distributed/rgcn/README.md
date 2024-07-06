## Distributed training

This is an example of training RGCN node classification in a distributed fashion. Currently, the example train RGCN graphs with input node features.

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
# for ip range 10.0.0.0 - 10.255.255.255
/home/ubuntu/workspace  10.0.0.0/8(rw,sync,no_subtree_check)
# for ip range 172.16.0.0 - 172.31.255.255
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
```

Users need to make sure that the master node (node-0) has right permission to ssh to all the other nodes without password authentication.
[This link](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/) provides instructions of setting passwordless SSH login.

### Step 2: partition the graph.

The example provides a script to partition some builtin graphs such as ogbn-mag graph.
If we want to train RGCN on 2 machines, we need to partition the graph into 2 parts.

In this example, we partition the ogbn-mag graph into 2 parts with Metis. The partitions are balanced with respect to the number of nodes, the number of edges and the number of labelled nodes.

```bash
python3 partition_graph.py --dataset ogbn-mag --num_parts 2 --balance_train --balance_edges
```

If we want to train RGCN with `GraphBolt`, we need to append `--use_graphbolt` to generate partitions in `GraphBolt` format.

```bash
python3 partition_graph.py --dataset ogbn-mag --num_parts 2 --balance_train --balance_edges --use_graphbolt
```

If we have already partitioned into `DGL` format, just convert them directly like below:

```
    python3 -c "import dgl; dgl.distributed.dgl_partition_to_graphbolt('ogbn-products.json')"
```


### Step 3: Launch distributed jobs

DGL provides a script to launch the training job in the cluster. `part_config` and `ip_config`
specify relative paths to the path of the workspace.

The command below launches 4 training processes on each machine as we'd like to utilize 4 GPUs for training.

```bash
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/distributed/rgcn/ \
--num_trainers 4 \
--num_servers 2 \
--num_samplers 0 \
--part_config data/ogbn-mag.json \
--ip_config ip_config.txt \
"python3 node_classification.py --graph-name ogbn-mag --dataset ogbn-mag --fanout='25,25' --batch-size 1024  --n-hidden 64 --lr 0.01 --eval-batch-size 1024  --low-mem --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --layer-norm --ip-config ip_config.txt --num_gpus 4"
```

If we want to train RGCN with `GraphBolt`, we need to append `--use_graphbolt`.

```bash
python3 ~/workspace/dgl/tools/launch.py \
--workspace ~/workspace/dgl/examples/distributed/rgcn/ \
--num_trainers 4 \
--num_servers 2 \
--num_samplers 0 \
--part_config data/ogbn-mag.json \
--ip_config ip_config.txt \
"python3 node_classification.py --graph-name ogbn-mag --dataset ogbn-mag --fanout='25,25' --batch-size 1024  --n-hidden 64 --lr 0.01 --eval-batch-size 1024  --low-mem --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --layer-norm --ip-config ip_config.txt --num_gpus 4 --use_graphbolt"
```

**Note:** if you are using conda or other virtual environments on the remote machines, you need to replace `python3` in the command string (i.e. the last argument) with the path to the Python interpreter in that environment.


## Comparison between `DGL` and `GraphBolt`

### Partition sizes

Compared to `DGL`, `GraphBolt` partitions are reduced to **19%** for `ogbn-mag`.

`ogbn-mag`

| Data Formats |         File Name            | Part 0 | Part 1 |
| ------------ | ---------------------------- | ------ | ------ |
| DGL          | graph.dgl                    | 714MB  | 716MB  |
| GraphBolt    | fused_csc_sampling_graph.pt  | 137MB  | 136MB  |


### Performance

Compared to `DGL`, `GraphBolt`'s sampler works faster(reduced to **16%** `ogbn-mag`). `Min` and `Max` are statistics of all trainers on all nodes(machines).

As for RAM usage, the shared memory(measured by **shared** field of `free` command) usage decreases due to smaller graph partitions in `GraphBolt`. The peak memory used by processes(measured by **used** field of `free` command) decreases as well.

`ogbn-mag`

| Data Formats | Sample Time Per Epoch (CPU) |  Test Accuracy (3 epochs) | shared | used (peak) | CPU Util |
| ------------ | --------------------------- | ------------------------- |  -----  | ---- | ----- |
|     DGL      | Min: 48.2s, Max: 91.4s      |            42.76%         |  1.3GB  | 9.2GB| 10.4% |
|   GraphBolt  | Min: 9.2s, Max: 11.9s       |            42.46%         |  742MB  | 5.9GB| 18.1% |


## Demonstrate and profile sampling for Link Prediction task

### DGL

```
python3 ~/workspace/dgl/tools/launch.py \
    --workspace ~/workspace/dgl/examples/distributed/rgcn/ \
    --num_trainers 4 \
    --num_servers 2 \
    --num_samplers 0 \
    --part_config ~/data/ogbn_mag_lp/ogbn-mag.json \
    --ip_config ~/workspace/ip_config.txt \
    "python3 lp_perf.py --fanout='25,25' --batch-size 1024  --n-epochs 1 --graph-name ogbn-mag --ip-config ~/workspace/ip_config.txt --num_gpus 4 --remove_edge"
```

### GraphBolt

In order to sample with `GraphBolt`, we need to convert partitions into `GraphBolt` formats with below command.

```
python3 -c "import dgl;dgl.distributed.dgl_partition_to_graphbolt('/home/ubuntu/workspace/data/ogbn_mag_lp/ogbn-mag.json', store_eids=True, graph_formats='coo')"
```

Then train with appended `--use_graphbolt`.

```
python3 ~/workspace/dgl/tools/launch.py \
    --workspace ~/workspace/dgl/examples/distributed/rgcn/ \
    --num_trainers 4 \
    --num_servers 2 \
    --num_samplers 0 \
    --part_config ~/data/ogbn_mag_lp/ogbn-mag.json \
    --ip_config ~/workspace/ip_config.txt \
    "python3 lp_perf.py --fanout='25,25' --batch-size 1024  --n-epochs 1 --graph-name ogbn-mag --ip-config ~/workspace/ip_config.txt --num_gpus 4 --remove_edge --use_graphbolt"
```

### Partition sizes

Compared to `DGL`, `GraphBolt` partitions are reduced to **72%** for `ogbn-mag`.

#### ogbn-mag

| Data Formats |         File Name            | Part 0 | Part 1 |
| ------------ | ---------------------------- | ------ | ------ |
| DGL          | graph.dgl                    | 714MB  | 716MB  |
| GraphBolt    | fused_csc_sampling_graph.pt  | 512MB  | 514MB  |

### Performance Comparison

#### Major used parameters

1. 2 nodes(g4dn.metal), 4 trainers, 2 servers per node. Sample on main process.
2. 2 layers.
3. fanouts = 25, 25 for all edge types.
4. batch_size = 1024.
5. seed edge IDs are all edges of ("author", "writes", "paper"), ~7M in total.
6. ratio of negative sampler = 3.
7. exclude = "reverse_types".

#### ogbn-mag

Compared to `DGL`, sampling with `GraphBolt` is reduced to **15%**. As for the overhead of `exclude`, it's about **5%** in this test. This number could be higher if larger `fanout` or `batch size` is applied.

The time shown below is the mean sampling time per iteration(60 iters in total, slowest rank). Unit: seconds

| Data Formats | No Exclude | Exclude |
| ------------ | ---------- | ------- |
| DGL          |   6.50     |   6.86  |
| GraphBolt    |   0.95     |   1.00  |
