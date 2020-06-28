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

We need to run a server on each machine. Before running the servers, we need to update `ip_config.txt` with the right IP addresses.

```bash
# run server on machine 0
python3 train_dist.py --server --graph-name ogb-product --id 0 --num-client 4 --conf_path data/ogb-product.json --ip_config ip_config.txt
# run server on machine 1
python3 train_dist.py --server --graph-name ogb-product --id 1 --num-client 4 --conf_path data/ogb-product.json --ip_config ip_config.txt
# run server on machine 2
python3 train_dist.py --server --graph-name ogb-product --id 2 --num-client 4 --conf_path data/ogb-product.json --ip_config ip_config.txt
# run server on machine 3
python3 train_dist.py --server --graph-name ogb-product --id 3 --num-client 4 --conf_path data/ogb-product.json --ip_config ip_config.txt
```

### Step 4: run trainers
We run a trainer process on each machine. Here we use Pytorch distributed. We need to use pytorch distributed launch to run each trainer process.
Pytorch distributed requires one of the trainer process to be the master. Here we use the first machine to run the master process.

```bash
# run client on machine 0
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr="172.31.16.250" --master_port=1234 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 4 --batch-size 1000 --lr 0.1
# run client on machine 1
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=1 --master_addr="172.31.16.250" --master_port=1234 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 4 --batch-size 1000 --lr 0.1
# run client on machine 2
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=2 --master_addr="172.31.16.250" --master_port=1234 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 4 --batch-size 1000 --lr 0.1
# run client on machine 3
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=3 --master_addr="172.31.16.250" --master_port=1234 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 4 --batch-size 1000 --lr 0.1
```
