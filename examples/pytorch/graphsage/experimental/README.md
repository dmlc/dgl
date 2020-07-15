## Distributed training

This is an example of training GraphSage in a distributed fashion. To train GraphSage, it has four steps:

### Step 1: partition the graph.

The example provides a script to partition some builtin graphs such as Reddit and OGB product graph.
If we want to train GraphSage on 4 machines, we need to partition the graph into 4 parts.

We need to load some function from the parent directory.
```bash
export PYTHONPATH=$PYTHONPATH:..
```

In this example, we partition the OGB product graph into 4 parts with Metis. The partitions are balanced with respect to
the number of nodes, the number of edges and the number of labelled nodes.
```bash
# partition graph
python3 partition_graph.py --dataset ogb-product --num_parts 4 --balance_train --balance_edges
```

### Step 2: copy the partitioned data to the cluster

When copying data to the cluster, we recommend users to copy the partitioned data to NFS so that all worker machines
will be able to access the partitioned data.

### Step 3: run servers

To perform actual distributed training (running training jobs in multiple machines), we need to run
a server on each machine. Before running the servers, we need to update `ip_config.txt` with the right IP addresses.

On each of the machines, set the following environment variables.

```bash
export DGL_ROLE=server
export DGL_IP_CONFIG=ip_config.txt
export DGL_CONF_PATH=data/ogb-product.json
export DGL_NUM_CLIENT=4
```

```bash
# run server on machine 0
export DGL_SERVER_ID=0
python3 train_dist.py

# run server on machine 1
export DGL_SERVER_ID=1
python3 train_dist.py

# run server on machine 2
export DGL_SERVER_ID=2
python3 train_dist.py

# run server on machine 3
export DGL_SERVER_ID=3
python3 train_dist.py
```

### Step 4: run trainers
We run a trainer process on each machine. Here we use Pytorch distributed. We need to use pytorch distributed launch to run each trainer process.
Pytorch distributed requires one of the trainer process to be the master. Here we use the first machine to run the master process.

```bash
# set the DistGraph in distributed mode
export DGL_DIST_MODE="distributed"
# run client on machine 0
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.31.16.250" --master_port=1234 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --batch-size 1000 --lr 0.1
# run client on machine 1
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=1 --master_addr="172.31.16.250" --master_port=1234 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --batch-size 1000 --lr 0.1
# run client on machine 2
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=2 --master_addr="172.31.16.250" --master_port=1234 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --batch-size 1000 --lr 0.1
# run client on machine 3
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=3 --master_addr="172.31.16.250" --master_port=1234 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --batch-size 1000 --lr 0.1
```

## Distributed code runs in the standalone mode

The standalone mode is mainly used for development and testing. The procedure to run the code is much simpler.

### Step 1: graph construction.

When testing the standalone mode of the training script, we should construct a graph with one partition.
```bash
python3 partition_graph.py --dataset ogb-product --num_parts 1
```

### Step 2: run the training script

```bash
python3 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --batch-size 1000 --conf_path data/ogb-product.json --standalone
```

Note: please ensure that all environment variables shown above are unset if they were set for testing distributed training.
